# train_yolo_publaynet_tb_val.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter # TensorBoard

import sys
import requests
import tarfile
import json
import numpy as np
from os import path, makedirs
from PIL import Image
from glob import glob
import time # For unique TensorBoard run names

# --- 1. Download and Prepare PubLayNet Example Data ---
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

# --- 2. YOLOv1 Model Definition (IposNet) ---
class IposNet(nn.Module):
    def __init__(self, S=7, B=2, C=5):
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
        out = out.reshape(-1, self.S, self.S, (self.B * 5 + self.C))
        return out

# --- 3. PubLayNet Dataset Class ---
class PubLayNetDataset(Dataset):
    def __init__(self, json_file, root_dir, S=7, B=2, C=5, img_size=448, transform=None, is_train=True): # Added is_train
        with open(json_file, 'r') as fp:
            self.data = json.load(fp)
        
        self.root_dir = root_dir
        self.S = S
        self.B = B
        self.C = C # Initial C, might be overridden
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train # To handle potential future augmentations specific to training

        self.images_info = {}
        for img_data in self.data['images']:
            self.images_info[img_data['id']] = {
                'file_name': img_data['file_name'],
                'width': img_data['width'],
                'height': img_data['height']
            }
        
        self.annotations_by_image_id = {}
        for ann in self.data['annotations']:
            if ann['image_id'] not in self.annotations_by_image_id:
                self.annotations_by_image_id[ann['image_id']] = []
            self.annotations_by_image_id[ann['image_id']].append(ann)
        
        self.image_ids = [img_id for img_id in self.images_info.keys() if img_id in self.annotations_by_image_id]
        
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.data['categories']}
        # Sort category names to ensure consistent mapping if C is overridden
        sorted_cat_names = sorted(list(self.cat_id_to_name.values()))
        self.cat_name_to_idx = {name: i for i, name in enumerate(sorted_cat_names)}
        
        self.actual_num_classes = len(self.cat_name_to_idx)
        if C != self.actual_num_classes:
            print(f"Warning: Provided C={C} does not match number of categories found: {self.actual_num_classes}. Using {self.actual_num_classes}.")
            self.C = self.actual_num_classes


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]
        img_path = path.join(self.root_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = img_info['width'], img_info['height']
        
        annotations = self.annotations_by_image_id[image_id]

        # Resize transform for both image and bbox calculations
        current_img_w, current_img_h = original_w, original_h
        if self.transform: # If a PIL-based resize is part of transform (before ToTensor)
            # For simplicity, we apply resize and then ToTensor/Normalize
            # More complex augmentations require careful handling
            image = TF.resize(image, (self.img_size, self.img_size))
            current_img_w, current_img_h = self.img_size, self.img_size # After resize
        
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            # Handle cases where category_id might not be in our map (if dataset is inconsistent)
            # This shouldn't happen with the provided example but good for robustness
            if category_id not in self.cat_id_to_name:
                print(f"Warning: category_id {category_id} not found in categories map. Skipping annotation.")
                continue
            class_name = self.cat_id_to_name[category_id]
            if class_name not in self.cat_name_to_idx:
                print(f"Warning: class_name {class_name} not found in name_to_idx map. Skipping annotation.")
                continue

            class_idx = self.cat_name_to_idx[class_name]

            x_center_abs = bbox[0] + bbox[2] / 2
            y_center_abs = bbox[1] + bbox[3] / 2
            
            # Bbox values relative to original image dimensions
            x_center_norm_orig = x_center_abs / original_w
            y_center_norm_orig = y_center_abs / original_h
            w_norm_orig = bbox[2] / original_w
            h_norm_orig = bbox[3] / original_h

            # These normalized values remain the same even after resizing the image
            # as long as the aspect ratio is preserved (which TF.resize to square does not)
            # For YOLO, we care about these values relative to the *input image to the network*
            # which is self.img_size x self.img_size.
            # Since we resize, the effective bbox coordinates need to be scaled
            # to the *new* image dimensions if we were doing augmentations that change them.
            # Here, it's simpler: the normalized coords w.r.t original image are used
            # to place the object in the grid.
            
            grid_x_idx = int(x_center_norm_orig * self.S)
            grid_y_idx = int(y_center_norm_orig * self.S)

            grid_x_idx = min(grid_x_idx, self.S - 1)
            grid_y_idx = min(grid_y_idx, self.S - 1)

            x_cell = (x_center_norm_orig * self.S) - grid_x_idx
            y_cell = (y_center_norm_orig * self.S) - grid_y_idx

            if target[grid_y_idx, grid_x_idx, 4] == 0:
                target[grid_y_idx, grid_x_idx, 0] = x_cell
                target[grid_y_idx, grid_x_idx, 1] = y_cell
                target[grid_y_idx, grid_x_idx, 2] = torch.sqrt(torch.tensor(w_norm_orig, dtype=torch.float32))
                target[grid_y_idx, grid_x_idx, 3] = torch.sqrt(torch.tensor(h_norm_orig, dtype=torch.float32))
                target[grid_y_idx, grid_x_idx, 4] = 1.0
                target[grid_y_idx, grid_x_idx, self.B * 5 + class_idx] = 1.0
        
        # Apply ToTensor and Normalize transforms after PIL operations
        if self.transform:
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
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, target):
        obj_ij_mask = target[..., 4] > 0
        noobj_ij_mask = target[..., 4] == 0

        # Localization Loss
        pred_xy_resp = predictions[..., 0:2][obj_ij_mask]
        target_xy_resp = target[..., 0:2][obj_ij_mask]
        loss_xy = self.mse(pred_xy_resp, target_xy_resp)
        
        pred_wh_sqrt_resp = torch.sign(predictions[..., 2:4][obj_ij_mask]) * torch.sqrt(torch.abs(predictions[..., 2:4][obj_ij_mask]) + 1e-6)
        target_wh_sqrt_resp = target[..., 2:4][obj_ij_mask]
        loss_wh = self.mse(pred_wh_sqrt_resp, target_wh_sqrt_resp)
        coord_loss = self.lambda_coord * (loss_xy + loss_wh)

        # Confidence Loss
        pred_conf_obj = predictions[..., 4][obj_ij_mask]
        target_conf_obj = target[..., 4][obj_ij_mask]
        loss_conf_obj = self.mse(pred_conf_obj, target_conf_obj)

        pred_conf_noobj_b1 = predictions[..., 4][noobj_ij_mask]
        target_conf_noobj_b1 = target[..., 4][noobj_ij_mask]
        loss_conf_noobj_b1 = self.mse(pred_conf_noobj_b1, target_conf_noobj_b1)
        
        pred_conf_noobj_b2 = predictions[..., 9]
        target_conf_noobj_b2 = torch.zeros_like(pred_conf_noobj_b2).to(pred_conf_noobj_b2.device)
        loss_conf_noobj_b2 = self.mse(pred_conf_noobj_b2, target_conf_noobj_b2)
        noobj_conf_loss = self.lambda_noobj * (loss_conf_noobj_b1 + loss_conf_noobj_b2)
        confidence_loss = loss_conf_obj + noobj_conf_loss

        # Classification Loss
        pred_class_probs = predictions[..., self.B*5:][obj_ij_mask]
        target_class_probs = target[..., self.B*5:][obj_ij_mask]
        class_loss = self.mse(pred_class_probs, target_class_probs)
        
        total_loss = (coord_loss + confidence_loss + class_loss) / predictions.size(0)
        
        # Return individual components for logging
        return total_loss, coord_loss/predictions.size(0), confidence_loss/predictions.size(0), class_loss/predictions.size(0), \
               loss_xy/predictions.size(0), loss_wh/predictions.size(0), \
               loss_conf_obj/predictions.size(0), noobj_conf_loss/predictions.size(0)

# --- 5. Validation Function ---
def validate_model(model, val_loader, criterion, device, S, B, C):
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    val_coord_loss, val_conf_loss, val_class_loss = 0.0, 0.0, 0.0
    val_loss_xy, val_loss_wh, val_loss_conf_obj, val_loss_conf_noobj = 0.0,0.0,0.0,0.0


    with torch.no_grad(): # Disable gradient calculations
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss, coord, conf, cls, l_xy, l_wh, l_conf_obj, l_conf_noobj = criterion(predictions, targets)
            
            val_loss += loss.item()
            val_coord_loss += coord.item()
            val_conf_loss += conf.item()
            val_class_loss += cls.item()
            val_loss_xy += l_xy.item()
            val_loss_wh += l_wh.item()
            val_loss_conf_obj += l_conf_obj.item()
            val_loss_conf_noobj += l_conf_noobj.item()


    avg_val_loss = val_loss / len(val_loader)
    avg_val_coord_loss = val_coord_loss / len(val_loader)
    avg_val_conf_loss = val_conf_loss / len(val_loader)
    avg_val_class_loss = val_class_loss / len(val_loader)
    
    avg_val_loss_xy = val_loss_xy / len(val_loader)
    avg_val_loss_wh = val_loss_wh / len(val_loader)
    avg_val_loss_conf_obj = val_loss_conf_obj / len(val_loader)
    avg_val_loss_conf_noobj = val_loss_conf_noobj / len(val_loader)


    print(f"\nValidation Results:")
    print(f"  Avg Val Loss: {avg_val_loss:.4f}")
    print(f"  Avg Val Coord Loss: {avg_val_coord_loss:.4f} (XY: {avg_val_loss_xy:.4f}, WH: {avg_val_loss_wh:.4f})")
    print(f"  Avg Val Conf Loss: {avg_val_conf_loss:.4f} (Obj: {avg_val_loss_conf_obj:.4f}, NoObj: {avg_val_loss_conf_noobj:.4f})")
    print(f"  Avg Val Class Loss: {avg_val_class_loss:.4f}\n")
    
    return avg_val_loss, avg_val_coord_loss, avg_val_conf_loss, avg_val_class_loss, \
           avg_val_loss_xy, avg_val_loss_wh, avg_val_loss_conf_obj, avg_val_loss_conf_noobj


# --- 6. Training Setup and Loop ---
if __name__ == '__main__':
    # Hyperparameters
    LEARNING_RATE = 2e-5 
    BATCH_SIZE = 4 
    NUM_EPOCHS = 100
    IMG_SIZE = 448
    S = 7
    B = 2
    # C will be determined
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # TensorBoard setup
    run_name = f"yolo_publaynet_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = path.join('runs', run_name)
    makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Data transformations
    # Train transform (can include augmentations later)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize first
        # Add augmentations here if desired, e.g., ColorJitter, RandomHorizontalFlip
        # Ensure augmentations are compatible with bbox handling if they change geometry significantly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Validation transform (typically no augmentations, just resize, ToTensor, Normalize)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    json_file_path = path.join(data_extracted_path, 'samples.json')
    
    # Determine NUM_CLASSES
    temp_dataset_for_C = PubLayNetDataset(json_file=json_file_path, root_dir=data_extracted_path, S=S, B=B, C=1, img_size=IMG_SIZE)
    NUM_CLASSES = temp_dataset_for_C.actual_num_classes 
    print(f"Number of classes detected: {NUM_CLASSES}")

    # For this example, we'll use the same small dataset for train and "validation"
    # In a real scenario, you'd split your data or have a separate validation set.
    train_dataset = PubLayNetDataset(json_file=json_file_path, root_dir=data_extracted_path, S=S, B=B, C=NUM_CLASSES, img_size=IMG_SIZE, transform=train_transform, is_train=True)
    val_dataset = PubLayNetDataset(json_file=json_file_path, root_dir=data_extracted_path, S=S, B=B, C=NUM_CLASSES, img_size=IMG_SIZE, transform=val_transform, is_train=False)
    
    # Ensure dataset is not empty
    if len(train_dataset) == 0:
        print("Training dataset is empty. Exiting.")
        sys.exit(1)
    if len(val_dataset) == 0:
        print("Validation dataset is empty. Consider disabling validation or providing data.")
        # sys.exit(1) # Or just skip validation

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True if len(train_dataset) > BATCH_SIZE else False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


    # Initialize model, loss, optimizer
    model = IposNet(S=S, B=B, C=NUM_CLASSES).to(DEVICE)
    criterion = YOLOLoss(S=S, B=B, C=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (optional, but often helpful)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    print("Starting training...")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        epoch_train_coord_loss, epoch_train_conf_loss, epoch_train_class_loss = 0,0,0
        epoch_train_loss_xy, epoch_train_loss_wh, epoch_train_loss_conf_obj, epoch_train_loss_conf_noobj = 0,0,0,0


        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss, coord, conf, cls, l_xy, l_wh, l_conf_obj, l_conf_noobj = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_coord_loss += coord.item()
            epoch_train_conf_loss += conf.item()
            epoch_train_class_loss += cls.item()
            epoch_train_loss_xy += l_xy.item()
            epoch_train_loss_wh += l_wh.item()
            epoch_train_loss_conf_obj += l_conf_obj.item()
            epoch_train_loss_conf_noobj += l_conf_noobj.item()


            # Log batch loss to TensorBoard
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or batch_idx == len(train_loader) -1 :
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
        
        # Calculate average losses for the epoch
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        avg_epoch_train_coord_loss = epoch_train_coord_loss / len(train_loader)
        avg_epoch_train_conf_loss = epoch_train_conf_loss / len(train_loader)
        avg_epoch_train_class_loss = epoch_train_class_loss / len(train_loader)
        avg_epoch_train_loss_xy = epoch_train_loss_xy / len(train_loader)
        avg_epoch_train_loss_wh = epoch_train_loss_wh / len(train_loader)
        avg_epoch_train_loss_conf_obj = epoch_train_loss_conf_obj / len(train_loader)
        avg_epoch_train_loss_conf_noobj = epoch_train_loss_conf_noobj / len(train_loader)


        # Log epoch average losses to TensorBoard
        writer.add_scalar('Loss/train_epoch_avg', avg_epoch_train_loss, epoch)
        writer.add_scalar('Loss_Components/train_coord_epoch_avg', avg_epoch_train_coord_loss, epoch)
        writer.add_scalar('Loss_Components/train_conf_epoch_avg', avg_epoch_train_conf_loss, epoch)
        writer.add_scalar('Loss_Components/train_class_epoch_avg', avg_epoch_train_class_loss, epoch)
        writer.add_scalar('Loss_Breakdown/train_xy_epoch_avg', avg_epoch_train_loss_xy, epoch)
        writer.add_scalar('Loss_Breakdown/train_wh_epoch_avg', avg_epoch_train_loss_wh, epoch)
        writer.add_scalar('Loss_Breakdown/train_conf_obj_epoch_avg', avg_epoch_train_loss_conf_obj, epoch)
        writer.add_scalar('Loss_Breakdown/train_conf_noobj_epoch_avg', avg_epoch_train_loss_conf_noobj, epoch)


        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] Training Complete ---")
        print(f"Average Train Epoch Loss: {avg_epoch_train_loss:.4f}")

        # Validation step
        if len(val_loader) > 0: # Only validate if val_loader is not empty
            avg_val_loss, avg_val_coord, avg_val_conf, avg_val_class, \
            avg_val_xy, avg_val_wh, avg_val_conf_obj, avg_val_conf_noobj = validate_model(model, val_loader, criterion, DEVICE, S, B, NUM_CLASSES)
            
            # Log validation losses to TensorBoard
            writer.add_scalar('Loss/val_epoch_avg', avg_val_loss, epoch)
            writer.add_scalar('Loss_Components/val_coord_epoch_avg', avg_val_coord, epoch)
            writer.add_scalar('Loss_Components/val_conf_epoch_avg', avg_val_conf, epoch)
            writer.add_scalar('Loss_Components/val_class_epoch_avg', avg_val_class, epoch)
            writer.add_scalar('Loss_Breakdown/val_xy_epoch_avg', avg_val_xy, epoch)
            writer.add_scalar('Loss_Breakdown/val_wh_epoch_avg', avg_val_wh, epoch)
            writer.add_scalar('Loss_Breakdown/val_conf_obj_epoch_avg', avg_val_conf_obj, epoch)
            writer.add_scalar('Loss_Breakdown/val_conf_noobj_epoch_avg', avg_val_conf_noobj, epoch)
            
            # if scheduler: scheduler.step(avg_val_loss) # For ReduceLROnPlateau

        # Potentially save model checkpoint here based on validation loss
        # best_val_loss = float('inf')
        # if avg_val_loss < best_val_loss:
        #    best_val_loss = avg_val_loss
        #    torch.save(model.state_dict(), path.join(log_dir, f"yolo_best_epoch_{epoch+1}.pth"))
        #    print(f"Saved new best model at epoch {epoch+1} with val_loss: {best_val_loss:.4f}")

        print("-" * 50)

    writer.close() # Close the TensorBoard writer
    print("Training finished.")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={path.abspath('runs')}")