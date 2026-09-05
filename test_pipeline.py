import glob
import os
import torch
from torch.utils.data import DataLoader
from dataset import dataset as BraTSDataset  # Alias your dataset class
from model import Unet3D

def verify_pipeline():
    # 1. Update this to the folder containing your individual patient sub-folders
    data_root = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    
    # Collect all patient directory paths
    patient_dirs = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    
    if len(patient_dirs) == 0:
        print(f"No patient folders found in {data_root}. Check your path.")
        return

    print(f"[1/4] Found {len(patient_dirs)} patient folders. Initializing Dataset...")
    brats_dataset = BraTSDataset(patient_dirs)
    
    # 2. Wrap in DataLoader
    print("[2/4] Initializing DataLoader...")
    loader = DataLoader(brats_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Pull a single sample
    print("[3/4] Loading single batch from disk...")
    images, masks = next(iter(loader))
    print(f"Loaded Image Tensor Shape: {images.shape}")  # Expect: [1, 4, 128, 128, 128]
    print(f"Loaded Mask Tensor Shape:  {masks.shape}")   # Expect: [1, 3, 128, 128, 128]
    
    # 4. Model Forward Pass
    print("[4/4] Passing volume through Unet3D...")
    model = Unet3D(in_channels=4, out_channels=3)
    model.eval()
    
    with torch.no_grad():
        output = model(images)
        
    print("\n--- PIPELINE VERIFICATION SUCCESSFUL ---")
    print(f"Model Output Shape:       {output.shape}")
    print(f"Matches Target Mask:     {output.shape == masks.shape}")

if __name__ == "__main__":
    verify_pipeline()