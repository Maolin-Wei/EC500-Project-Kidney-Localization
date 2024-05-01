# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detection.build_image_encoder import build_MedSAM_image_encoder
from torch.utils.data import DataLoader, Dataset
from detection.models import DetectionModel
from detection.unet import MyUNet
import glob
from tqdm import tqdm
join = os.path.join
from torchvision.ops import box_iou, nms
from sklearn.metrics import precision_score, recall_score
from utils import *


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
        img = np.load(join(self.img_path, img_name), "r", allow_pickle=True)  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img = np.transpose(img, (2, 0, 1))
        assert (np.max(img) <= 1.0 and np.min(img) >= 0.0), "image should be normalized to [0, 1]"
        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)  # labels [0, 1, 2], (256, 256)
        assert img_name == os.path.basename(self.gt_path_files[index]), ("img gt name error" + self.gt_path_files[index] + self.npy_files[index])

        bboxes = []
        labels = []
        for class_id in [1, 2]:  # 0 is left kidney, 1 is right kidney
            class_mask = np.uint8(gt == class_id)
            if class_mask.sum() > 0:  # If there is an object of this class
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
                labels.append(class_id - 1)
        bboxes = np.array(bboxes)
        labels = np.array(labels)

        # Rotate the image counterclockwise by 90 degrees
        img = np.rot90(img, k=1, axes=(1, 2)).copy()  # k=1 for 90 degrees, axes=(1,2) because img is in (C, H, W) format

        # Rotate the bounding boxes
        # Calculate the new coordinates after rotation
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            new_x_min, new_y_min = y_min, img.shape[2] - x_max
            new_x_max, new_y_max = y_max, img.shape[2] - x_min

            # Here we consider bbox_shift if it's relevant after rotation
            new_x_min = max(0, new_x_min - self.bbox_shift)
            new_x_max = min(img.shape[2], new_x_max + self.bbox_shift)
            new_y_min = max(0, new_y_min - self.bbox_shift)
            new_y_max = min(img.shape[1], new_y_max + self.bbox_shift)

            new_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

        new_bboxes = np.array(new_bboxes)

        return (
            torch.tensor(img).float(),
            torch.tensor(new_bboxes).float(),
            torch.tensor(labels).long(),
            img_name
        )


def precision_recall(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall


def load_model(model_path, model_type, device):
    image_encoder = None
    if model_type == 'MedSAM':
        image_encoder = build_MedSAM_image_encoder('vit_b', checkpoint=None)
    elif model_type == 'UNet':
        image_encoder = MyUNet()
    detection_model = DetectionModel(image_encoder=image_encoder, num_classes=2)
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model = detection_model.to(device)
    detection_model.eval()
    return detection_model


def visualize_predictions_and_gt(img, class_preds, pred_bboxes, conf_scores, gt_bboxes, gt_labels):
    fig, ax = plt.subplots(1)
    img = img.transpose(1, 2, 0)  # Convert shape from (C, H, W) to (H, W, C)
    ax.imshow(img)

    # Plot predicted bounding boxes
    for bbox, class_pred, conf_score in zip(pred_bboxes, class_preds, conf_scores):
        if conf_score > 0.5:  # Threshold for visualization
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], f'Pred: {class_pred}, Conf: {conf_score:.2f}', color='red', va='bottom',
                    ha='left')

    # Plot ground truth bounding boxes
    for bbox, label in zip(gt_bboxes, gt_labels):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], f'GT: {label}', color='green', va='top', ha='left')

    plt.axis('off')
    plt.title("Prediction and Ground Truth")
    plt.tight_layout(pad=0)
    # plt.show()

    return fig


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type = 'MedSAM' # MedSAM or UNet
    # model_path = 'work_dir/Model4_UNet2/best_model_epoch_7.pth'
    model_path = 'work_dir/Model4_MedSAM2/best_model_epoch_2.pth'
    data_root = 'data/KidneyData/npy_1024_test_all_slices/MRI_kidney/'
    output_dir = f'./work_dir/output_{model_type}/'
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, model_type, device)
    test_dataset = KidneyDataset(data_root)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_ious = []
    precisions = []
    recalls = []
    temp_boxes = []

    img_count = 0
    max_pred_boxes = {0: 0.0, 1: 0.0}
    max_iou = {0: 0.0, 1: 0.0}
    max_iou_conf = {0: 0.0, 1: 0.0}
    img_list = []
    img_slice_num = 18
    for step, (images, gt_boxes, labels, img_name) in enumerate(tqdm(test_dataloader)):
        images = images.to(device)
        gt_boxes = gt_boxes.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            class_preds, pred_boxes, conf_scores = model(images)

            pred_boxes = pred_boxes[0]
            conf_scores = conf_scores[0]
            class_preds = class_preds[0]

            class_scores = torch.softmax(class_preds, dim=-1)
            _, class_preds = class_scores.max(dim=-1)

            # Apply NMS
            keep_indices = nms(pred_boxes, conf_scores, iou_threshold=0.5)
            class_preds = class_preds[keep_indices]
            pred_boxes = pred_boxes[keep_indices]
            conf_scores = conf_scores[keep_indices]

            ious_temp, _ = calculate_iou(pred_boxes, class_preds, gt_boxes[0], labels[0])
            class_preds = class_preds.cpu().numpy()
            ious_temp = ious_temp.cpu().numpy()
            pred_boxes = pred_boxes.cpu().numpy()
            conf_scores = conf_scores.cpu().numpy()

            # get the max iou bounding boxes for each kedney
            if max_iou[class_preds[0]] < ious_temp[0]:
                max_iou[class_preds[0]] = ious_temp[0]
                max_pred_boxes[class_preds[0]] = pred_boxes[0]
                max_iou_conf[class_preds[0]] = conf_scores[0]
            if max_iou[class_preds[1]] < ious_temp[1]:
                max_iou[class_preds[1]] = ious_temp[1]
                max_pred_boxes[class_preds[1]] = pred_boxes[1]
                max_iou_conf[class_preds[1]] = conf_scores[0]
        img_list.append(images[0].cpu().numpy())

        # use the max iou bounding boxes for all slices
        img_count += 1
        if img_count % img_slice_num == 0:
            all_ious.append(max_iou[0])
            all_ious.append(max_iou[1])
            pred_boxes = [max_pred_boxes[class_preds[0]], max_pred_boxes[class_preds[1]]]
            conf_scores = [max_iou_conf[class_preds[0]], max_iou_conf[class_preds[1]]]
            for i in range(img_count):
                fig = visualize_predictions_and_gt(img_list[i], class_preds, pred_boxes, conf_scores, gt_boxes[0].cpu().numpy(), labels[0].cpu().numpy())
                fig.savefig(f'{output_dir}/{img_name[0][:-8]}_slice{i}.png')
                # fig.show()
                plt.close(fig)

            img_count = 0
            max_pred_boxes = {0: 0.0, 1: 0.0}
            max_iou = {0: 0.0, 1: 0.0}
            img_list = []

    mean_iou = np.mean(all_ious)

    print(f"Mean IoU: {mean_iou}")
