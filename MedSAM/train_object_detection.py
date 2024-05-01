# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from detection.build_image_encoder import build_MedSAM_image_encoder
from detection.models import DetectionModel
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import time
import matplotlib.patches as patches
from torchvision.ops import box_iou, nms, generalized_box_iou_loss
from utils import *
from detection.unet import MyUNet

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/KidneyData/npy_1024_train/MRI_kidney",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-i2",
    "--val_npy_path",
    type=str,
    default='',
    help="path to validation npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B-Kidney-Detection")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="checkpoint/MedSAM/medsam_vit_b.pth"
                # "-checkpoint", type=str, default="checkpoint/SAM/sam_vit_b_01ec64.pth"
)
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=11)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=4)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )


# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

class KidneyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=0):
        self.data_root = data_root
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path = os.path.join(data_root, "gts")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]

        self.bbox_shift = bbox_shift
        print(f"Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), normalized to [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img = np.transpose(img, (2, 0, 1))
        assert (
                np.max(img) <= 1.0 and np.min(img) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # labels [0, 1, 2], (256, 256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
                "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )

        bboxes = []
        labels = []
        for class_id in [1, 2]: # kidney label
            class_mask = np.uint8(gt == class_id)
            if class_mask.sum() > 0:
                y_indices, x_indices = np.where(class_mask > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = class_mask.shape
                x_min = max(0, x_min - self.bbox_shift)
                x_max = min(W, x_max + self.bbox_shift)
                y_min = max(0, y_min - self.bbox_shift)
                y_max = min(H, y_max + self.bbox_shift)
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id - 1)  # 0 is left kidney, 1 is right kidney
        bboxes = np.array(bboxes)
        labels = np.array(labels)

        # Rotate the image counterclockwise by 90 degrees
        img = np.rot90(img, k=1,
                       axes=(1, 2)).copy()  # k=1 for 90 degrees, axes=(1,2) because img is in (C, H, W) format

        # Rotate the bounding boxes
        # Calculate the new coordinates after rotation
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            new_x_min, new_y_min = y_min, img.shape[2] - x_max
            new_x_max, new_y_max = y_max, img.shape[2] - x_min

            new_x_min = max(0, new_x_min - self.bbox_shift)
            new_x_max = min(img.shape[2], new_x_max + self.bbox_shift)
            new_y_min = max(0, new_y_min - self.bbox_shift)
            new_y_max = min(img.shape[1], new_y_max + self.bbox_shift)

            new_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
            # new_bboxes.append([new_x_min, new_y_min + 80, new_x_max - 50, new_y_max - 50])

        new_bboxes = np.array(new_bboxes)

        return (
            torch.tensor(img).float(),
            torch.tensor(new_bboxes).float(),
            torch.tensor(labels).long(),
            img_name
        )

'''
def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='mean'):
    """
    Implementation of Generalized IoU Loss.
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss
'''


def main_detection():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    image_encoder = build_MedSAM_image_encoder(args.model_type, checkpoint=args.checkpoint)
    # image_encoder = MyUNet()

    num_classes = 2
    detection_model = DetectionModel(
        image_encoder=image_encoder,
        num_classes=num_classes,
    ).to(device)

    detection_model.train()

    optimizer = optim.AdamW(detection_model.parameters(), args.lr, weight_decay=args.weight_decay)

    classification_loss = nn.CrossEntropyLoss()
    box_regression_loss = nn.SmoothL1Loss()
    # box_regression_loss = generalized_box_iou_loss
    confidence_loss = nn.BCELoss()

    train_dataset = KidneyDataset(args.tr_npy_path)
    print("Number of training samples: ", len(train_dataset))
    val_dataset = KidneyDataset(args.val_npy_path)
    print("Number of training samples: ", len(val_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_iou = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        epoch_conf_loss = 0

        for step, (images, gt_boxes, labels, img_name) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            images = images.to(device)
            gt_boxes = gt_boxes.to(device)
            labels = labels.to(device)  # [batch_size, num_gt_boxes]

            # show_img_and_bbox(images[0].cpu(), gt_boxes[0].cpu(), labels[0].cpu(), img_name)

            class_logits, pred_boxes, conf_scores = detection_model(images)
            # Match predicted boxes to ground truth boxes based on IoU
            ious = box_iou(pred_boxes[0], gt_boxes[0])
            iou_max, iou_max_idx = ious.max(dim=1)

            # Filter out indices where IoU is below the threshold
            conf_labels = iou_max

            cls_loss = classification_loss(class_logits[0], labels[0][iou_max_idx].long())
            box_reg_loss = box_regression_loss(pred_boxes[0], gt_boxes[0][iou_max_idx])  # SmoothL1
            conf_loss = confidence_loss(conf_scores[0], conf_labels)

            total_loss = cls_loss + box_reg_loss + conf_loss
            total_loss.backward()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_box_reg_loss += box_reg_loss.item()
            epoch_conf_loss += conf_loss.item()

            optimizer.step()

        epoch_loss /= len(train_dataloader)
        epoch_cls_loss /= len(train_dataloader)
        epoch_box_reg_loss /= len(train_dataloader)
        epoch_conf_loss /= len(train_dataloader)

        print(f"Epoch: {epoch}, Total Loss: {epoch_loss:.8f}, "
              f"Class Loss: {epoch_cls_loss:.8f}, Box Reg Loss: {epoch_box_reg_loss:.8f}, "
              f"Conf Loss: {epoch_conf_loss:.8f}")
        
        # No valid set, save the model
        if args.val_npy_path == '':
            if epoch % 1 == 0:
                model_path = join(model_save_path, f"model_epoch_{epoch}.pth")
                torch.save(detection_model.state_dict(), model_path)
        else: # have validation set
            detection_model.eval()
            val_box_reg_loss = 0
            with torch.no_grad():
                ious = []
                for images, gt_boxes, labels, img_name in val_dataloader:
                    images = images.to(device)
                    gt_boxes = gt_boxes.to(device)
                    labels = labels.to(device)

                    class_preds, pred_boxes, conf_scores = detection_model(images)

                    pred_boxes = pred_boxes[0]
                    conf_scores = conf_scores[0]
                    class_preds = class_preds[0]

                    iou = box_iou(pred_boxes, gt_boxes[0])
                    iou_max, iou_max_idx = iou.max(dim=1)
                    box_reg_loss = box_regression_loss(pred_boxes, gt_boxes[0][iou_max_idx])  # SmoothL1
                    val_box_reg_loss += box_reg_loss.item()

                    class_scores = torch.softmax(class_preds, dim=-1)
                    _, class_preds = class_scores.max(dim=-1)

                    # Apply NMS
                    keep_indices = nms(pred_boxes, conf_scores, iou_threshold=0.5)

                    class_preds = class_preds[keep_indices]
                    pred_boxes = pred_boxes[keep_indices]
                    conf_scores = conf_scores[keep_indices]

                    ious_temp, class_preds = calculate_iou(pred_boxes, class_preds, gt_boxes[0], labels[0])

                    ious.extend(ious_temp.cpu().numpy())

                val_box_reg_loss /= len(val_dataloader)
                mean_iou = np.mean(ious)
                print(f"Epoch: {epoch}, Validation Box Reg Loss: {val_box_reg_loss:.8f} Validation Average IoU: {mean_iou:.8f}")

                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_model_path = join(model_save_path, f"best_model_epoch_{epoch}.pth")
                    torch.save(detection_model.state_dict(), best_model_path)
                    print(f"New best model saved with IoU: {best_iou:.8f}")
                if epoch % 20 == 0:
                    model_path = join(model_save_path, f"model_epoch_{epoch}.pth")
                    torch.save(detection_model.state_dict(), model_path)

if __name__ == "__main__":
    main_detection()
