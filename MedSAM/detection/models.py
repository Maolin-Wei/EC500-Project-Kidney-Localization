import torch.nn as nn
import torch
import numpy as np


class AnchorGenerator:
    def __init__(self, anchor_sizes, aspect_ratios, scales, center=[(512, 512)], img_size=1024):
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.center = center
        self.img_size = img_size

    def generate_anchors(self):
        anchors = []
        for size in self.anchor_sizes:
            for scale in self.scales:
                scaled_size = size * scale
                for ratio in self.aspect_ratios:
                    width = scaled_size * np.sqrt(ratio)
                    height = scaled_size / np.sqrt(ratio)
                    for center in self.center:
                        x_min = max(0, center[0] - width / 2)
                        y_min = max(0, center[1] - height / 2)
                        x_max = min(self.img_size, center[0] + width / 2)
                        y_max = min(self.img_size, center[1] + height / 2)
                        anchors.append([x_min, y_min, x_max, y_max])
        return torch.tensor(anchors, dtype=torch.float32)


def apply_deltas_to_anchors(anchors, deltas):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # Apply deltas
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


class DetectionHead(nn.Module):
    def __init__(self, in_features, num_classes, num_boxes=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.classifier = nn.Linear(in_features, self.num_classes * num_boxes)
        self.box_predictor = nn.Linear(in_features, 4 * num_boxes)

        self.confidence_predictor = nn.Sequential(
            nn.Linear(in_features, num_boxes),
            nn.Sigmoid()
        )

    def forward(self, x):
        class_logits = self.classifier(x)
        class_logits = class_logits.view(-1, self.num_boxes, self.num_classes)  # Shape: [batch_size, num_boxes, num_classes]

        bounding_boxes = self.box_predictor(x)
        bounding_boxes = bounding_boxes.view(-1, self.num_boxes, 4)  # Shape: [batch_size, num_boxes, 4]

        conf_scores = self.confidence_predictor(x)
        conf_scores = conf_scores.view(-1, self.num_boxes)

        return class_logits, bounding_boxes, conf_scores


class DetectionModel(nn.Module):
    def __init__(self, image_encoder, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.image_encoder = image_encoder
        anchor_generator = AnchorGenerator(anchor_sizes=[512], aspect_ratios=[0.5, 1], scales=[1, 1.5, 2])
        self.anchors = anchor_generator.generate_anchors()
        self.num_anchors = self.anchors.shape[0]
        self.detection_head = DetectionHead(256, self.num_classes, self.num_anchors)


    def forward(self, images):
        image_embedding = self.image_encoder(images)  # (B, C, H, W)
        features = torch.mean(image_embedding, dim=[2, 3])  # Global Average Pooling (B, C)
        class_logits, pred_bboxes, conf_scores = self.detection_head(features)

        box_deltas = pred_bboxes.reshape(-1, 4)  # Flatten for processing
        adjusted_boxes = apply_deltas_to_anchors(self.anchors.repeat(images.size(0), 1, 1).reshape(-1, 4).to(images.device), box_deltas)
        adjusted_boxes = adjusted_boxes.view(-1, self.num_anchors, 4)  # Reshape back

        return class_logits, adjusted_boxes, conf_scores


if __name__ == '__main__':
    anchor_generator = AnchorGenerator(anchor_sizes=[128, 256, 512], aspect_ratios=[0.5, 1, 2], scales=[1])
    anchors = anchor_generator.generate_anchors()
    print(anchors)

