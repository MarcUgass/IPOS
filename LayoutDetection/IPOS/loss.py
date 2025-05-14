import torch
import torch.nn as nn

# Assume S, B, C are defined (e.g., S=7, B=2, C=20 for VOC)
# Assume lambda_coord = 5.0
# Assume lambda_noobj = 0.5

class YOLOLoss(nn.Module):
    def __init__(self, S, B, C, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='sum') # Sum of squared errors

    def compute_iou(self, box1, box2):
        # box1, box2: [x, y, w, h]
        # Convert to [x1, y1, x2, y2] for easier IOU calculation
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        intersection_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union_area = box1_area + box2_area - intersection_area + 1e-6 # Add epsilon for stability

        return intersection_area / union_area

    def forward(self, predictions, target):
        # predictions: tensor of shape (batch_size, S, S, B*5 + C)
        # target: tensor of shape (batch_size, S, S, B*5 + C) or a list of ground truths per image

        # Initialize losses
        localization_loss = 0.0
        confidence_loss_obj = 0.0
        confidence_loss_noobj = 0.0
        classification_loss = 0.0

        # Reshape predictions for easier access
        # pred_boxes will be (batch, S, S, B, 4) -> x,y,w,h
        # pred_conf will be (batch, S, S, B, 1) -> confidence
        # pred_class_probs will be (batch, S, S, C) -> class probabilities

        # For simplicity, let's assume target is structured similarly or iterate through batch
        for batch_idx in range(predictions.size(0)):
            pred = predictions[batch_idx] # Shape (S, S, B*5 + C)
            gt = target[batch_idx]      # Shape (S, S, 5 + C) - assuming one GT box per cell for simplicity here

            for i in range(self.S): # Iterate over grid cells (rows)
                for j in range(self.S): # Iterate over grid cells (cols)
                    # gt_box_coords: [x, y, w, h]
                    # gt_confidence: 1 if object present, 0 otherwise
                    # gt_class_label: one-hot encoded class label or index

                    # I_i_obj: Indicator if object is in grid cell (i,j) in ground truth
                    # This is 1 if gt_confidence for cell (i,j) is 1, 0 otherwise
                    I_i_obj = gt[i, j, 4] # Assuming 5th element in GT is objectness

                    if I_i_obj: # If an object is present in this grid cell
                        # --- Identify Responsible Predictor ---
                        best_iou = 0
                        responsible_predictor_idx = 0
                        gt_box_cell = gt[i, j, :4] # Ground truth box [x,y,w,h] for this cell

                        for b in range(self.B):
                            # pred_box_cell: predicted box [x,y,w,h] by predictor b in cell (i,j)
                            pred_box_cell = pred[i, j, (b*5):(b*5 + 4)]
                            iou = self.compute_iou(pred_box_cell, gt_box_cell)
                            if iou > best_iou:
                                best_iou = iou
                                responsible_predictor_idx = b

                        # Now use the responsible_predictor_idx for loss calculation
                        pred_resp_box = pred[i, j, (responsible_predictor_idx*5):(responsible_predictor_idx*5 + 4)]
                        pred_resp_conf = pred[i, j, (responsible_predictor_idx*5 + 4)]
                        pred_cell_class_probs = pred[i, j, (self.B*5):]

                        # --- 1. Localization Loss (for the responsible predictor) ---
                        # (x, y) coordinates [cite: 98]
                        xy_loss = self.mse_loss(pred_resp_box[:2], gt_box_cell[:2])
                        # (w, h) coordinates - using square root as per paper [cite: 93, 98]
                        # Ensure no negative values before sqrt, add small epsilon for stability
                        pred_wh_sqrt = torch.sqrt(torch.clamp(pred_resp_box[2:4], min=1e-6))
                        gt_wh_sqrt = torch.sqrt(torch.clamp(gt_box_cell[2:4], min=1e-6))
                        wh_loss = self.mse_loss(pred_wh_sqrt, gt_wh_sqrt)
                        localization_loss += (xy_loss + wh_loss)

                        # --- 2. Confidence Loss (object present, for the responsible predictor) --- [cite: 98]
                        # Target confidence here is the IOU [cite: 50]
                        confidence_loss_obj += self.mse_loss(pred_resp_conf, best_iou) # best_iou or fixed 1.0 * IOU

                        # --- 3. Classification Loss (if object present in cell) --- [cite: 98, 99]
                        gt_cell_class_probs = gt[i, j, 5:(5+self.C)] # Ground truth class probabilities
                        classification_loss += self.mse_loss(pred_cell_class_probs, gt_cell_class_probs)

                    else: # If no object is present in this grid cell
                        # --- 4. Confidence Loss (no object present, for ALL predictors in this cell) --- [cite: 98]
                        for b in range(self.B):
                            pred_box_conf = pred[i, j, (b*5 + 4)]
                            # Target confidence is 0 [cite: 49]
                            confidence_loss_noobj += self.mse_loss(pred_box_conf, torch.zeros_like(pred_box_conf))

        # Total Loss calculation with lambda weights [cite: 90, 98]
        total_loss = (
            self.lambda_coord * localization_loss +
            confidence_loss_obj +
            self.lambda_noobj * confidence_loss_noobj +
            classification_loss
        )

        return total_loss / predictions.size(0) # Average loss over batch