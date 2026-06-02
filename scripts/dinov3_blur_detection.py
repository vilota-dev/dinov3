import os
import sys
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np

# Add your dinov3 root directory to path if running as a standalone script
root_dir = "/home/vilota/mingjie/dinov3"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dinov3.hub.backbones import dinov3_vith16plus, Weights

# ==========================================
# 1. CLASS MAPPINGS
# ==========================================
# Reverse the mapping to get the string label from the predicted integer ID
ID_TO_LABEL = {
    0: 'o',  # ok
    1: 'sf', # slightly far
    2: 'mf', # medium far
    3: 'xf', # extremely far
    4: 'sn', # slightly near
    5: 'mn', # medium near
    6: 'xn'  # extremely near
}
NUM_CLASSES = len(ID_TO_LABEL)

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return x + self.block(x)


class MaskedEdgeBlurDetector(nn.Module):
    def __init__(self, backbone_name="dinov3_vith16plus", num_classes=7, weights_path=None):
        super().__init__()
        
        # Load the backbone
        weights_arg = weights_path if weights_path else Weights.LVD1689M
        self.backbone = dinov3_vith16plus(pretrained=True, weights=weights_arg)
        
        embed_dim = self.backbone.embed_dim
        linear_input_dim = 2 * embed_dim 
        
        # The trained linear head
        self.linear_head = nn.Sequential(
            nn.Linear(linear_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            ResidualBlock(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # Multi-class output
        )

    def forward(self, x, patch_mask):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]       
        patch_tokens = features["x_norm_patchtokens"] 
        
        mask_weights = patch_mask.float().unsqueeze(-1)
        weighted_patches = patch_tokens * mask_weights
        
        summed_patches = weighted_patches.sum(dim=1) 
        valid_patch_count = mask_weights.sum(dim=1) + 1e-6
        masked_patch_mean = summed_patches / valid_patch_count 
        
        linear_input = torch.cat([cls_token, masked_patch_mean], dim=1) 
        logits = self.linear_head(linear_input)
        return logits 

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_horizontal_patch_mask(image_path, image_size=224, patch_size=16, threshold=50):
    """Generates the OpenCV horizontal edge mask."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    img = cv2.resize(img, (image_size, image_size))
    sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
    abs_sobel_y = np.absolute(sobel_y)
    sobel_8u = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
    
    _, binary_mask = cv2.threshold(sobel_8u, threshold, 255, cv2.THRESH_BINARY)
    mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    patch_mask_2d = F.max_pool2d(mask_tensor, kernel_size=patch_size, stride=patch_size)
    patch_mask_flat = patch_mask_2d.view(-1) > 0 
    
    return patch_mask_flat

def predict_image_high_precision(image_path, model, device, transform, OK_THRESHOLD=0.9):
    """
    Runs an image through the model. 
    Requires extreme confidence to call it 'o' (OK). Otherwise, flags it for review.
    """
    pil_image = Image.open(image_path).convert("RGB")
    pixel_values = transform(pil_image).unsqueeze(0).to(device)
    
    patch_mask = get_horizontal_patch_mask(image_path).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values, patch_mask)
        probabilities = torch.softmax(logits, dim=1)[0] # Extract the first batch sample array
        
    # Index 0 corresponds to 'o' (OK) based on the class map
    ok_probability = probabilities[0].item()
    
    # Check if the selection successfully meets your strict high-precision criteria
    if ok_probability >= OK_THRESHOLD:
        final_prediction = 'o'
        confidence = ok_probability
        action_required = "Pass (Confirmed OK)"
    else:
        # If confidence falls below your gate threshold, find the most probable alternative defect category
        defect_probs = probabilities[1:]
        predicted_defect_idx = torch.argmax(defect_probs).item() + 1 # Add 1 to align with the global map index
        
        final_prediction = ID_TO_LABEL[predicted_defect_idx]
        confidence = probabilities[predicted_defect_idx].item()
        action_required = "Review (Flagged for Double Check)"
        
    return final_prediction, confidence, action_required

# ==========================================
# 4. MAIN BATCH EXECUTION & PANDAS CSV EXPORT
# ==========================================
import pandas as pd

if __name__ == "__main__":
    
    # --- Configuration Paths ---
    BASE_DINO_WEIGHTS = "/home/vilota/mingjie/dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    TRAINED_HEAD_WEIGHTS = "/home/vilota/mingjie/dinov3/scripts/v4_pth/dino_classifier_head_multiclass_epoch_30.pth"
    
    # Folder containing all the unseen images (can have nested subfolders)
    UNSEEN_DIR = "/home/vilota/566-qa-2/618D/processed_images"
    
    # Output CSV file path
    OUTPUT_CSV = "consolidated_batch_predictions.csv"
    
    # Set confidence parameters for filtering out pristine images safely
    CONFIDENCE_GATE = 0.9 # Require 90% certainty to let an OK pass unchecked
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on {device}...")
    
    # 1. Initialize the Model
    model = MaskedEdgeBlurDetector(
        backbone_name="dinov3_vith16plus", 
        num_classes=NUM_CLASSES,
        weights_path=BASE_DINO_WEIGHTS
    ).to(device)
    
    # 2. Load Your Trained Classifier Head
    print(f"Loading trained head from {TRAINED_HEAD_WEIGHTS}...")
    try:
        model.linear_head.load_state_dict(torch.load(TRAINED_HEAD_WEIGHTS, map_location=device))
        print("✓ Trained weights loaded successfully.\n")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        sys.exit(1)

    # 3. Setup transforms
    dino_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # 4. Find all images recursively
    print(f"Scanning directory: {UNSEEN_DIR} ...")
    valid_extensions = {'.bmp', '.png', '.jpg', '.jpeg'}
    image_paths = [p for p in Path(UNSEEN_DIR).rglob('*') if p.is_file() and p.suffix.lower() in valid_extensions]
    
    if not image_paths:
        print(f"No images found in {UNSEEN_DIR}. Exiting.")
        sys.exit(0)
        
    print(f"Found {len(image_paths)} images. Starting high-precision batch prediction...\n")
    
    # List to store raw dictionaries before pandas grouping aggregation
    raw_predictions_list = []
    
    # 5. Run inference and extract tracking metrics
    for i, img_path in enumerate(image_paths, 1):
        try:
            # Parse filename format (Example: "processed_1359-3.png" or "1359-3.bmp")
            filename_stem = img_path.stem
            # Clean out the "processed_" prefix if it exists from your sampler script
            clean_stem = filename_stem.replace("processed_", "")
            
            if '-' in clean_stem:
                sn_part, pos_part = clean_stem.split('-')[:2]
                sn = sn_part.strip()
                pos_num = int(pos_part.strip())
            else:
                print(f"[{i}/{len(image_paths)}] ⚠️ Skipping {img_path.name}: Name does not follow SN-pos convention.")
                continue

            # Run through high precision decision gate
            label, conf, action = predict_image_high_precision(
                image_path=str(img_path), 
                model=model, 
                device=device, 
                transform=dino_transform,
                OK_THRESHOLD=CONFIDENCE_GATE
            )
            
            raw_predictions_list.append({
                "SN": sn,
                "Position": pos_num,
                "Prediction": label,
                "Confidence": f"{round(conf * 100, 2)}%",
                "Action": action
            })
            
            print(f"[{i}/{len(image_paths)}] SN: {sn} | Pos: {pos_num} -> {label} ({conf:.1%}) | {action}")
            
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] ✗ Error processing {img_path.name}: {e}")

    # ==========================================
    # 6. PANDAS PIVOT & CONSOLIDATION LOGIC
    # ==========================================
    if raw_predictions_list:
        print(f"\nProcessing pivot transformation across serial numbers...")
        df_raw = pd.DataFrame(raw_predictions_list)
        
        # Unique list of all serial numbers found
        unique_sns = df_raw["SN"].unique()
        consolidated_rows = []
        
        target_positions = [1, 3, 5, 7, 9]
        
        for sn in unique_sns:
            # Filter rows specifically matching the loop target SN
            sn_subset = df_raw[df_raw["SN"] == sn]
            
            # Base data container for our single output row
            sn_row_entry = {"SN": sn}
            
            # Flag to track if any position requires manual intervention
            any_review_needed = False
            
            for pos in target_positions:
                # Find matching record inside positions
                pos_match = sn_subset[sn_subset["Position"] == pos]
                
                if not pos_match.empty:
                    pred_val = pos_match.iloc[0]["Prediction"]
                    conf_val = pos_match.iloc[0]["Confidence"]
                    action_val = pos_match.iloc[0]["Action"]
                    
                    sn_row_entry[f"pos {pos} predict"] = pred_val
                    sn_row_entry[f"pos {pos} confidence"] = conf_val
                    
                    if "Review" in action_val:
                        any_review_needed = True
                else:
                    # Fallback defaults if position index image file was missing from SSD directory
                    sn_row_entry[f"pos {pos} predict"] = "Missing"
                    sn_row_entry[f"pos {pos} confidence"] = "N/A"
            
            # Enforce global macro check status rules 
            sn_row_entry["Action Required"] = (
                "Review (Flagged for Double Check)" if any_review_needed else "Pass (Confirmed OK)"
            )
            consolidated_rows.append(sn_row_entry)
            
        # Reorder columns cleanly to ensure ordered presentation structure
        column_order = ["SN"]
        for pos in target_positions:
            column_order.extend([f"pos {pos} predict", f"pos {pos} confidence"])
        column_order.append("Action Required")
        
        df_final = pd.DataFrame(consolidated_rows)[column_order]
        
        # Export final output matrix straight to CSV format
        df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"✓ Consolidated matrix saved successfully to: {OUTPUT_CSV}")