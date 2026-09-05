import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def diagnostic_eda(scan_path, mask_path):
    """
    Executes the 4 Core Engineering Requirements for 3D Medical EDA
    on a target BraTS 2020 patient case.
    """
    print("\n" + "="*50)
    print("      STAGE 1: 3D MEDICAL DATA AUDIT ENGINE      ")
    print("="*50)

    # -------------------------------------------------------------------------
    # REQUIREMENT 1: Spatial & Geometric Audit
    # -------------------------------------------------------------------------
    print("\n[RUNNING] Requirement A: Spatial & Geometric Audit...")
    
    
    scan_volume = nib.load(scan_path)
    
    
    scan_data = scan_volume.get_fdata()
    voxel_spacing = scan_volume.header.get_zooms()
    
    
    depth, height, width = scan_data.shape
    mid_d, mid_h, mid_w = depth // 2, height // 2, width // 2
    
    print(f"  -> Raw Volume Matrix Shape (D x H x W) : {scan_data.shape}")
    print(f"  -> Physical Voxel Resolution (mm)     : {voxel_spacing}")
    print(f"  -> Computed Spatial Center Index       : ({mid_d}, {mid_h}, {mid_w})")

    # -------------------------------------------------------------------------
    # REQUIREMENT 2: Intensity Range & Data Quality Audit
    # -------------------------------------------------------------------------
    print("\n[RUNNING] Requirement B: Intensity Range & Noise Audit...")
    
    min_val = np.min(scan_data)
    max_val = np.max(scan_data)
    mean_val = np.mean(scan_data)
    std_val = np.std(scan_data)
    
    
    nan_count = np.isnan(scan_data).sum()
    
   
    empty_voxels = np.sum(scan_data <= 0.05)
    air_percent = (empty_voxels / scan_data.size) * 100
    
    print(f"  -> Raw Signal Intensity Range          : [{min_val:.2f}, {max_val:.2f}]")
    print(f"  -> Distribution Metrics (Mean / Std)   : {mean_val:.2f} / {std_val:.2f}")
    print(f"  -> Corrupt Elements Detected (NaNs)    : {nan_count}")
    print(f"  -> Unproductive Background Air Volume  : {air_percent:.2f}%")

    # -------------------------------------------------------------------------
    # REQUIREMENT C: Target / Label Audit
    # -------------------------------------------------------------------------
    print("\n[RUNNING] Requirement C: Ground Truth Target Audit...")
    
    mask_volume = nib.load(mask_path)
    mask_data = mask_volume.get_fdata()
    
   
    assert mask_data.shape == scan_data.shape, "CRITICAL ERROR: Scan and Mask geometries do not align!"
    
  
    unique_labels = np.unique(mask_data)
    tumor_voxels = np.sum(mask_data > 0)
    class_imbalance_ratio = (tumor_voxels / scan_data.size) * 100
    
    print(f"  -> Geometric Alignment Verification    : MATCHED SUCCESS")
    print(f"  -> Unique Radiologist Labels Detected  : {unique_labels}  (Expect: [0. 1. 2. 4.])")
    print(f"  -> Total Malignant Tumor Voxels        : {tumor_voxels}")
    print(f"  -> Pathological Class Imbalance Footprint: {class_imbalance_ratio:.4f}% of total grid")

    # -------------------------------------------------------------------------
    # REQUIREMENT D: Visual Inspection Layout (Multi-Planar Slicing)
    # -------------------------------------------------------------------------
    print("\n[RUNNING] Requirement D: Generating Multi-Planar Visualizations...")
    
    
    axial_scan = scan_data[mid_d, :, :]
    axial_mask = mask_data[mid_d, :, :]
    
    coronal_scan = scan_data[:, mid_h, :]
    coronal_mask = mask_data[:, mid_h, :]
    
    sagittal_scan = scan_data[:, :, mid_w]
    sagittal_mask = mask_data[:, :, mid_w]
    
    views = [
        (axial_scan, axial_mask, "Axial (Top-Down)"),
        (coronal_scan, coronal_mask, "Coronal (Face-to-Face)"),
        (sagittal_scan, sagittal_mask, "Sagittal (Side-Profile)")
    ]
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"BraTS 2020 Multi-Planar Structural View\nFile: {os.path.basename(scan_path)}", fontsize=14, fontweight='bold')
    
    for idx, (scan_slice, mask_slice, plane_name) in enumerate(views):
        
        axes[idx].imshow(np.rot90(scan_slice), cmap="gray")
        
        
        if mask_slice is not None and np.sum(mask_slice) > 0:
            masked_tumor = np.ma.masked_where(mask_slice == 0, mask_slice)
            axes[idx].imshow(np.rot90(masked_tumor), cmap="Reds", alpha=0.5)
            
        axes[idx].set_title(plane_name, fontsize=12, fontweight='bold')
        axes[idx].axis("off")
        
    plt.tight_layout()
    output_filename = "braTS_diagnostic_visual_profile.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"  -> Multi-Planar Plot Rendered successfully!")
    print(f"  -> Saved structural diagnostic map as: '{output_filename}'")
    print("\n" + "="*50)
    print("               AUDIT PIPELINE COMPLETE           ")
    print("="*50 + "\n")


if __name__ == "__main__":
   

    BASE_PATIENT_DIR = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    
    # Construct paths matching the BraTS 4-modality structural naming standard
    DATASET_SCAN = os.path.join(BASE_PATIENT_DIR, r"BraTS20_Training_001\BraTS20_Training_001_flair.nii")
    DATASET_MASK = os.path.join(BASE_PATIENT_DIR, r"BraTS20_Training_001\BraTS20_Training_001_seg.nii")
    
    # Verification check to intercept dead string paths before execution loop
    if not os.path.exists(DATASET_SCAN) or not os.path.exists(DATASET_MASK):
        print("CRITICAL ERROR: Target NIfTI files not located at the specified paths.")
        print(f"   Please inspect your path string configuration:\n   Looking for: {DATASET_SCAN}")
    else:
        diagnostic_eda(DATASET_SCAN, DATASET_MASK)