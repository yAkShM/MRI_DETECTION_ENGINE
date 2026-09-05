import os
import glob
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

def normalize_channel(slice_data):
    """Z-score normalization strictly within non-zero brain tissue."""
    mask = slice_data > 0
    if np.any(mask):
        mean = slice_data[mask].mean()
        std = slice_data[mask].std()
        slice_data[mask] = (slice_data[mask] - mean) / (std + 1e-8)
    return slice_data

def preprocess_and_save(raw_data_dir, output_dir):
    """
    Loads raw multi-modal NIfTI files, applies normalization, crops to [128, 128, 128],
    and serializes directly to fast .pt binary files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    patient_dirs = [
        os.path.join(raw_data_dir, d) for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ]
    
    print(f"Starting offline preprocessing for {len(patient_dirs)} patients...")
    print(f"Target Output Directory: {output_dir}")

    for p_dir in tqdm(patient_dirs, desc="Preprocessing Scans"):
        p_name = os.path.basename(p_dir)
        save_path = os.path.join(output_dir, f"{p_name}.pt")

        # Skip if already preprocessed
        if os.path.exists(save_path):
            continue

        try:
            # 1. Locate all 4 modalities and segmentation mask
            flair_path = glob.glob(os.path.join(p_dir, "*_flair.nii*"))[0]
            t1_path    = glob.glob(os.path.join(p_dir, "*_t1.nii*"))[0]
            t1ce_path  = glob.glob(os.path.join(p_dir, "*_t1ce.nii*"))[0]
            t2_path    = glob.glob(os.path.join(p_dir, "*_t2.nii*"))[0]
            seg_path   = glob.glob(os.path.join(p_dir, "*_seg.nii*"))[0]

            # 2. Load NIfTI and cast to float32
            flair = nib.load(flair_path).get_fdata().astype(np.float32)
            t1    = nib.load(t1_path).get_fdata().astype(np.float32)
            t1ce  = nib.load(t1ce_path).get_fdata().astype(np.float32)
            t2    = nib.load(t2_path).get_fdata().astype(np.float32)
            seg   = nib.load(seg_path).get_fdata().astype(np.uint8)

            # 3. Z-Score normalization per modality
            flair = normalize_channel(flair)
            t1    = normalize_channel(t1)
            t1ce  = normalize_channel(t1ce)
            t2    = normalize_channel(t2)

            # 4. Transpose from (H, W, D) -> (D, H, W)
            flair = np.transpose(flair, (2, 0, 1))
            t1    = np.transpose(t1, (2, 0, 1))
            t1ce  = np.transpose(t1ce, (2, 0, 1))
            t2    = np.transpose(t2, (2, 0, 1))
            seg   = np.transpose(seg, (2, 0, 1))

            # 5. Stack multi-modal image tensor: [4, D, H, W]
            image = np.stack([flair, t1, t1ce, t2], axis=0)

            # 6. Decouple BraTS masks into binary sub-regions: [3, D, H, W]
            # WT: Whole Tumor (labels 1, 2, 4)
            # TC: Tumor Core (labels 1, 4)
            # ET: Enhancing Tumor (label 4)
            wt = (seg == 1) | (seg == 2) | (seg == 4)
            tc = (seg == 1) | (seg == 4)
            et = (seg == 4)
            mask = np.stack([wt, tc, et], axis=0).astype(np.float32)

            # 7. Convert to torch tensors
            image_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask)

            # 8. Center-crop from (155, 240, 240) -> (128, 128, 128)
            image_tensor = image_tensor[:, 13:141, 56:184, 56:184]
            mask_tensor  = mask_tensor[:, 13:141, 56:184, 56:184]

            # 9. Save as a single fast binary dictionary
            torch.save({
                "image": image_tensor.half(),  # Save in FP16 to halve disk size & I/O
                "mask": mask_tensor.byte()     # Save masks as uint8 (0 or 1)
            }, save_path)

        except Exception as e:
            print(f"Error processing {p_name}: {e}")

    print("\n Offline preprocessing finished successfully.")

if __name__ == "__main__":
    RAW_DATA_PATH = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    PROCESSED_PATH = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed"

    preprocess_and_save(RAW_DATA_PATH, PROCESSED_PATH)
