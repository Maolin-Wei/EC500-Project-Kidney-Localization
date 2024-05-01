import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
join = os.path.join


def visulize_npy_image(file_path):
    '''
    Load and visualize the NPY image
    '''
    # Load the image file
    img = np.load(file_path)
    img = np.rot90(img, k=1).copy()
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.title('Image')
    plt.show()

def calculate_iou(pred_boxes, pred_labels, gt_boxes, gt_labels):
    ious = []
    pred_classes = []
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        # Find IoUs only for boxes with the same label
        gt_same_label_mask = (gt_labels == pred_label)
        if not gt_same_label_mask.any():
            continue  # No ground truth box with the same label, skip
        gt_same_label_boxes = gt_boxes[gt_same_label_mask]

        # Calculate IoU for boxes with the same label
        for gt_box in gt_same_label_boxes:
            # Calculate area of intersection
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)

            # Calculate area of union
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union = pred_area + gt_area - intersection

            # Calculate IoU
            iou = intersection / union if union > 0 else 0
            if iou > 0:
                ious.append(iou)
                pred_classes.append(pred_labels[gt_same_label_mask][0].item())

    return torch.tensor(ious), torch.tensor(pred_classes)


def calculate_iou_3d(pred_boxes, pred_labels, gt_boxes, gt_labels):
    ious = []
    pred_classes = []
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        # Find IoUs only for boxes with the same label
        gt_same_label_mask = (gt_labels == pred_label)
        if not gt_same_label_mask.any():
            continue  # No ground truth box with the same label, skip
        gt_same_label_boxes = gt_boxes[gt_same_label_mask]

        # Calculate IoU for boxes with the same label
        for gt_box in gt_same_label_boxes:
            # Calculate volume of intersection
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            z1 = max(pred_box[2], gt_box[2])
            x2 = min(pred_box[3], gt_box[3])
            y2 = min(pred_box[4], gt_box[4])
            z2 = min(pred_box[5], gt_box[5])
            intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)

            # Calculate volume of union
            pred_volume = (pred_box[3] - pred_box[0]) * (pred_box[4] - pred_box[1]) * (pred_box[5] - pred_box[2])
            gt_volume = (gt_box[3] - gt_box[0]) * (gt_box[4] - gt_box[1]) * (gt_box[5] - gt_box[2])
            union = pred_volume + gt_volume - intersection

            # Calculate IoU
            iou = intersection / union if union > 0 else 0
            if iou > 0:
                ious.append(iou)
                pred_classes.append(pred_label.item())

    return torch.tensor(ious), torch.tensor(pred_classes)


def normalize_bboxes(bboxes, image_width, image_height):
    normalized_bboxes = np.zeros_like(bboxes, dtype=np.float32)
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        normalized_bboxes[i, 0] = x_min / image_width
        normalized_bboxes[i, 1] = y_min / image_height
        normalized_bboxes[i, 2] = width / image_width
        normalized_bboxes[i, 3] = height / image_height
    return normalized_bboxes


def denormalize_bboxes(normalized_bboxes, image_width, image_height):
    denormalized_bboxes = np.zeros_like(normalized_bboxes, dtype=np.int32)
    for i, bbox in enumerate(normalized_bboxes):
        x_min_norm, y_min_norm, width_norm, height_norm = bbox
        x_min = int(x_min_norm * image_width)
        y_min = int(y_min_norm * image_height)
        width = int(width_norm * image_width)
        height = int(height_norm * image_height)
        x_max = x_min + width
        y_max = y_min + height
        denormalized_bboxes[i] = [x_min, y_min, x_max, y_max]
    return denormalized_bboxes


def show_img_and_bbox(img, bboxes, labels, img_name):
    """
    Display the image with bounding boxes.

    :param img: The image tensor (C, H, W)
    :param bboxes: The bounding boxes tensor (N, 4)
    :param labels: The labels tensor (N)
    :param img_name: The name of the image
    """
    # Convert tensor to numpy array and transpose to (H, W, C)
    img = img.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()
    for bbox, label in zip(bboxes, labels):
        # Draw a rectangle on the image
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'{label.item()}', color='white', fontsize=12,
                 bbox=dict(facecolor='red', alpha=0.5))
    plt.title(f"Image: {img_name}")
    plt.axis('off')
    plt.show()


def show_img_and_bbox_2(img, gt_boxes, labels, pred_boxes, img_name):
    """
    Display the image with ground truth and predicted bounding boxes.

    :param img: The image tensor (C, H, W)
    :param gt_boxes: The ground truth bounding boxes tensor (N, 4)
    :param labels: The labels tensor (N)
    :param pred_boxes: The predicted bounding boxes tensor (M, 4)
    :param img_name: The name of the image
    """
    # Convert tensor to numpy array and transpose to (H, W, C)
    img = img.permute(1, 2, 0).cpu().numpy()

    # Create figure and axis
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Display the image
    plt.imshow(img)

    # Plot ground truth bounding boxes
    for bbox, label in zip(gt_boxes, labels):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'GT: {label.item()}', color='white', fontsize=12,
                 bbox=dict(facecolor='green', alpha=0.5))

    # Plot predicted bounding boxes
    for bbox in pred_boxes:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x_min, y_min, 'Pred', color='white', fontsize=12,
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.title(f"Image: {img_name}")
    plt.axis('off')
    plt.show()


def box_iou_3d(boxes1, boxes2):
    """
    Calculate 3D IoU for each predicted box against each ground truth box.
    Args:
        boxes1 (torch.Tensor): Predicted boxes with shape [num_pred_boxes, 6]
        boxes2 (torch.Tensor): Ground truth boxes with shape [num_gt_boxes, 6]
    Returns:
        torch.Tensor: IoU matrix with shape [num_pred_boxes, num_gt_boxes]
    """
    # Number of boxes
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    # Expand boxes to compute intersections and unions across all boxes
    boxes1 = boxes1.unsqueeze(1).expand(num_boxes1, num_boxes2, 6)
    boxes2 = boxes2.unsqueeze(0).expand(num_boxes1, num_boxes2, 6)

    # Calculate intersection coordinates
    max_xyz = torch.min(boxes1[..., 3:], boxes2[..., 3:])
    min_xyz = torch.max(boxes1[..., :3], boxes2[..., :3])
    inter_dims = torch.clamp(max_xyz - min_xyz, min=0)

    # Intersection volume
    inter_vol = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]

    # Volumes of each box
    vol1 = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
    vol2 = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 4] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

    # Union volume
    union_vol = vol1 + vol2 - inter_vol

    # Compute IoU
    iou = inter_vol / union_vol

    return iou


def nms_3d(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) for 3D bounding boxes.

    Args:
        boxes (torch.Tensor): Bounding boxes, shape [num_boxes, 6] [x_min, y_min, z_min, x_max, y_max, z_max]
        scores (torch.Tensor): Confidence scores for each box, shape [num_boxes]
        iou_threshold (float): Threshold for IoU to determine overlap

    Returns:
        List[int]: Indices of boxes that are kept after NMS.
    """
    # Sort the indices of the scores in descending order (higher score = keep first)
    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        # Take the index of the box with the highest score and add it to the keep list
        current = idxs[0]
        keep.append(current.item())

        # Break if this is the last box
        if idxs.size(0) == 1:
            break

        # Calculate IoU between the current box and all other boxes
        current_box = boxes[current].unsqueeze(0)
        remaining_boxes = boxes[idxs[1:]]
        ious = box_iou_3d(current_box, remaining_boxes)

        # Find the indices of boxes that are less than the threshold
        idxs = idxs[1:][ious < iou_threshold]

    return keep