# train_yolo_publaynet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


import sys
import requests
import tarfile
import json
import numpy as np
from os import path
from PIL import Image
# from PIL import ImageFont, ImageDraw # Not needed for training script
from glob import glob
# from matplotlib import pyplot as plt # Not needed for training script

# --- 1. Download and Prepare PubLayNet Example Data (from your hola.py) ---
print("Downloading and extracting PubLayNet example data...")
fname = 'examples.tar.gz'
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + fname
if not path.exists(fname):
    r = requests.get(url)
    open(fname , 'wb').write(r.content)

data_extracted_path = "examples/"
if not path.exists(data_extracted_path):
    tar = tarfile.open(fname)
    tar.extractall()
    tar.close()
print("Data preparation complete.")

# --- 2. YOLOv1 Model Definition (IposNet from your model.py) ---
class IposNet(nn.Module):
    def __init__(self, S=7, B=2, C=5): # Default C=5 for PubLayNet example
        super(IposNet, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        leaky_relu_slope = 0.1
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(128, 256 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=2, padding=1), nn.LeakyReLU(leaky_relu_slope)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope)
        )

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.drop_out = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        
        out = torch.flatten(out, start_dim=1)
        
        out = self.fc1(out)
        out = nn.LeakyReLU(0.1)(out)
        out = self.drop_out(out)
        
        out = self.fc2(out)
        
        # Reshape to (N, S, S, B*5+C)
        out = out.reshape(-1, self.S, self.S, (self.B * 5 + self.C))
        return out

# --- 3. PubLayNet Dataset Class ---
class PubLayNetDataset(Dataset):
    def __init__(self, json_file, root_dir, S=7, B=2, C=5, img_size=448, transform=None):
        with open(json_file, 'r') as fp:
            self.data = json.load(fp)
        
        self.root_dir = root_dir
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        self.transform = transform

        self.images_info = {} # {id: {file_name, width, height}}
        for img_data in self.data['images']:
            self.images_info[img_data['id']] = {
                'file_name': img_data['file_name'],
                'width': img_data['width'],
                'height': img_data['height']
            }
        
        self.annotations_by_image_id = {} # {image_id: [list of annotations]}
        for ann in self.data['annotations']:
            if ann['image_id'] not in self.annotations_by_image_id:
                self.annotations_by_image_id[ann['image_id']] = []
            self.annotations_by_image_id[ann['image_id']].append(ann)
        
        # Filter out images that don't have annotations (if any)
        self.image_ids = [img_id for img_id in self.images_info.keys() if img_id in self.annotations_by_image_id]
        
        # Map category_id to 0-indexed class_idx
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.data['categories']}
        self.cat_name_to_idx = {name: i for i, name in enumerate(self.cat_id_to_name.values())}
        
        # Verify C matches number of categories
        if C != len(self.cat_name_to_idx):
            print(f"Warning: Provided C={C} does not match number of categories found: {len(self.cat_name_to_idx)}")
            self.C = len(self.cat_name_to_idx) # Override C

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]
        img_path = path.join(self.root_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = img_info['width'], img_info['height']
        
        annotations = self.annotations_by_image_id[image_id]

        if self.transform:
            # Apply transforms that affect both image and bboxes first (e.g. resize)
            # For simplicity, we'll just resize. More complex augmentations need careful bbox handling.
            image = TF.resize(image, (self.img_size, self.img_size))
        
        # Target tensor: (S, S, B*5 + C)
        # B*5 = (x, y, w, h, conf) for each box
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for ann in annotations:
            bbox = ann['bbox'] # [x_min, y_min, width, height] (COCO format)
            category_id = ann['category_id']
            class_name = self.cat_id_to_name[category_id]
            class_idx = self.cat_name_to_idx[class_name]

            # Convert bbox to [x_center, y_center, width, height] relative to image (0-1)
            x_center_abs = bbox[0] + bbox[2] / 2
            y_center_abs = bbox[1] + bbox[3] / 2
            
            x_center_norm = x_center_abs / original_w
            y_center_norm = y_center_abs / original_h
            w_norm = bbox[2] / original_w
            h_norm = bbox[3] / original_h

            # Determine grid cell
            grid_x_idx = int(x_center_norm * self.S)
            grid_y_idx = int(y_center_norm * self.S)

            # Ensure indices are within bounds (can happen with objects at exact edges)
            grid_x_idx = min(grid_x_idx, self.S - 1)
            grid_y_idx = min(grid_y_idx, self.S - 1)

            # Coordinates relative to the cell (0-1)
            x_cell = (x_center_norm * self.S) - grid_x_idx
            y_cell = (y_center_norm * self.S) - grid_y_idx

            # If cell is not already taken by another object (for the first predictor)
            if target[grid_y_idx, grid_x_idx, 4] == 0: # Check confidence of 1st box
                # Box 1 (responsible predictor)
                target[grid_y_idx, grid_x_idx, 0] = x_cell
                target[grid_y_idx, grid_x_idx, 1] = y_cell
                target[grid_y_idx, grid_x_idx, 2] = torch.sqrt(torch.tensor(w_norm, dtype=torch.float32)) # Store sqrt(w)
                target[grid_y_idx, grid_x_idx, 3] = torch.sqrt(torch.tensor(h_norm, dtype=torch.float32)) # Store sqrt(h)
                target[grid_y_idx, grid_x_idx, 4] = 1.0  # Confidence for object presence

                # Class probabilities (one-hot)
                target[grid_y_idx, grid_x_idx, self.B * 5 + class_idx] = 1.0
        
        if self.transform: # Apply tensor conversion and normalization
            image = self.transform(image)
            
        return image, target

# --- 4. YOLO Loss Function ---
class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=5, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum') # Sum squared error

    def forward(self, predictions, target):
        # predictions: (batch_size, S, S, B*5+C)
        # target: (batch_size, S, S, B*5+C)
        
        # --- Identify masks ---
        # obj_ij_mask: True for cells (i,j) where the 1st predictor is responsible for an object
        obj_ij_mask = target[..., 4] > 0  # Based on confidence of 1st box in target
        # noobj_ij_mask: True for cells (i,j) where the 1st predictor is NOT responsible
        noobj_ij_mask = target[..., 4] == 0

        # --- Localization Loss (x, y, w, h) ---
        # Only for the responsible predictor (box 1 in our target generation) in cells with objects
        
        # xy loss
        pred_xy_resp = predictions[..., 0:2][obj_ij_mask]
        target_xy_resp = target[..., 0:2][obj_ij_mask]
        loss_xy = self.mse(pred_xy_resp, target_xy_resp)
        
        # wh loss (predictions are sqrt(w), sqrt(h); target also stores sqrt(w), sqrt(h))
        # Add small epsilon to prevent nan if w/h is 0, though target w/h should be >0 for obj
        pred_wh_sqrt_resp = torch.sign(predictions[..., 2:4][obj_ij_mask]) * torch.sqrt(torch.abs(predictions[..., 2:4][obj_ij_mask]) + 1e-6)
        target_wh_sqrt_resp = target[..., 2:4][obj_ij_mask] # Already sqrt in target
        loss_wh = self.mse(pred_wh_sqrt_resp, target_wh_sqrt_resp)
        
        coord_loss = self.lambda_coord * (loss_xy + loss_wh)

        # --- Confidence Loss ---
        # Object confidence loss (for responsible predictor in object cells)
        pred_conf_obj = predictions[..., 4][obj_ij_mask]
        target_conf_obj = target[..., 4][obj_ij_mask] # Should be 1.0
        loss_conf_obj = self.mse(pred_conf_obj, target_conf_obj)

        # No-object confidence loss
        # Box 1 in no-object cells
        pred_conf_noobj_b1 = predictions[..., 4][noobj_ij_mask]
        target_conf_noobj_b1 = target[..., 4][noobj_ij_mask] # Should be 0.0
        loss_conf_noobj_b1 = self.mse(pred_conf_noobj_b1, target_conf_noobj_b1)
        
        # Box 2 (always no-object in our target generation, so target confidence is 0 for all cells)
        pred_conf_noobj_b2 = predictions[..., 9] # Confidence of 2nd box
        target_conf_noobj_b2 = torch.zeros_like(pred_conf_noobj_b2) # Target is always 0
        loss_conf_noobj_b2 = self.mse(pred_conf_noobj_b2, target_conf_noobj_b2)
        
        noobj_conf_loss = self.lambda_noobj * (loss_conf_noobj_b1 + loss_conf_noobj_b2)
        
        confidence_loss = loss_conf_obj + noobj_conf_loss

        # --- Classification Loss ---
        # Only for cells with objects
        pred_class_probs = predictions[..., self.B*5:][obj_ij_mask]
        target_class_probs = target[..., self.B*5:][obj_ij_mask]
        class_loss = self.mse(pred_class_probs, target_class_probs)
        
        # Total loss (normalize by batch size)
        total_loss = (coord_loss + confidence_loss + class_loss) / predictions.size(0)
        
        return total_loss, loss_xy/predictions.size(0), loss_wh/predictions.size(0), loss_conf_obj/predictions.size(0), (loss_conf_noobj_b1 + loss_conf_noobj_b2)/predictions.size(0), class_loss/predictions.size(0)


# --- 5. Training Setup and Loop ---
if __name__ == '__main__':
    # Hyperparameters
    LEARNING_RATE = 2e-5 # Often needs careful tuning
    # LEARNING_RATE = 1e-4 # Paper starts with 10^-2, then 10^-3, then 10^-4. 135 epochs total.
                         # This is a much smaller dataset, so a smaller LR might be safer initially.
    BATCH_SIZE = 4 # Keep small for example data and memory
    NUM_EPOCHS = 100 # For the example data, more epochs might be needed to see anything
    IMG_SIZE = 448
    S = 7
    B = 2
    # C will be determined by the dataset
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Data transformations
    # Standard normalization for pre-trained models, good practice anyway
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    json_file_path = path.join(data_extracted_path, 'samples.json')
    
    # Create a temporary PubLayNetDataset to determine C
    # This is a bit hacky for C, but works for this example
    temp_dataset_for_C = PubLayNetDataset(json_file=json_file_path, root_dir=data_extracted_path, S=S, B=B, C=1, img_size=IMG_SIZE)
    NUM_CLASSES = temp_dataset_for_C.C 
    print(f"Number of classes detected: {NUM_CLASSES}")

    train_dataset = PubLayNetDataset(json_file=json_file_path, root_dir=data_extracted_path, S=S, B=B, C=NUM_CLASSES, img_size=IMG_SIZE, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last if batch_size doesn't divide dataset size

    # Initialize model, loss, optimizer
    model = IposNet(S=S, B=B, C=NUM_CLASSES).to(DEVICE)
    criterion = YOLOLoss(S=S, B=B, C=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005) # Closer to paper

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_loss_xy, epoch_loss_wh, epoch_loss_conf_obj, epoch_loss_conf_noobj, epoch_loss_class = 0,0,0,0,0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss, l_xy, l_wh, l_conf_obj, l_conf_noobj, l_class = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_xy += l_xy.item()
            epoch_loss_wh += l_wh.item()
            epoch_loss_conf_obj += l_conf_obj.item()
            epoch_loss_conf_noobj += l_conf_noobj.item()
            epoch_loss_class += l_class.item()

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or batch_idx == len(train_loader) -1 : # Print more often for small dataset
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] COMPLETE ---")
        print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")
        print(f"  Avg XY Loss: {epoch_loss_xy/len(train_loader):.4f}, Avg WH Loss: {epoch_loss_wh/len(train_loader):.4f}")
        print(f"  Avg Conf Obj Loss: {epoch_loss_conf_obj/len(train_loader):.4f}, Avg Conf NoObj Loss: {epoch_loss_conf_noobj/len(train_loader):.4f}")
        print(f"  Avg Class Loss: {epoch_loss_class/len(train_loader):.4f}\n")

        # Potentially save model checkpoint here
        # if (epoch + 1) % 10 == 0:
        #    torch.save(model.state_dict(), f"yolo_publaynet_epoch_{epoch+1}.pth")

    print("Training finished.")