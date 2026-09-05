import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Unet3D

# Select GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, checkpoint_path):
    """
    Loads saved model weights from the training checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model


def create_rgb_mask(mask_slice):

    c,h,w = mask_slice.shape

    rgb = np.zeros((h,w,3), dtype=np.float32)

    rgb[..., 0] = mask_slice[0]  # Red   -> WT
    rgb[..., 1] = mask_slice[1]  # Green -> TC
    rgb[..., 2] = mask_slice[2]  # Blue  -> ET
    
    return rgb

def visualize_prediction(patient_pt_path, checkpoint_path, output_image_path="prediction_result.png"):

    model = Unet3D(in_channels=4, out_channels=3).to(DEVICE)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    data = torch.load(patient_pt_path, map_location="cpu", weights_only=True)
    image = data["image"].float().unsqueeze(0).to(DEVICE)
    gt_mask = data["mask"].float().cpu().numpy()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            logits = model(image)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float().squeeze(0).cpu().numpy()  # Shape: [3, 128, 128, 128]

    # 4. Dynamically locate the axial slice with the largest tumor area
    wt_slice_sums = gt_mask[0].sum(axis=(1, 2))
    best_slice_idx = int(np.argmax(wt_slice_sums))

    # Fallback to center slice if scan has minimal pathology
    if wt_slice_sums[best_slice_idx] == 0:
        best_slice_idx = 64

    print(f"Targeting Axial Slice Index: {best_slice_idx}")

    # 5. Extract 2D modalities and generate RGB overlays
    mri_flair = image[0, 0, best_slice_idx].cpu().numpy()
    mri_t1ce = image[0, 2, best_slice_idx].cpu().numpy()

    gt_slice_rgb = create_rgb_mask(gt_mask[:, best_slice_idx, :, :])
    pred_slice_rgb = create_rgb_mask(pred_mask[:, best_slice_idx, :, :])

    # 6. Construct radiological comparison grid
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(mri_flair, cmap="gray")
    axes[0].set_title("MRI: FLAIR", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mri_t1ce, cmap="gray")
    axes[1].set_title("MRI: T1ce", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(mri_flair, cmap="gray")
    axes[2].imshow(gt_slice_rgb, alpha=0.6)
    axes[2].set_title("Ground Truth\n(Red: WT, Green: TC, Blue: ET)", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(mri_flair, cmap="gray")
    axes[3].imshow(pred_slice_rgb, alpha=0.6)
    axes[3].set_title("Model Prediction\n(Red: WT, Green: TC, Blue: ET)", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Visual assessment saved to: {output_image_path}")

if __name__ == "__main__":
    # --- Paths Configuration ---
    CHECKPOINT_PATH = r"checkpoints/best_model.pth"
    PROCESSED_DIR = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed"
    OUTPUT_IMAGE_NAME = "prediction_comparison.png"

    # Locate available preprocessed patient scans
    patient_files = glob.glob(os.path.join(PROCESSED_DIR, "*.pt"))
    
    if not patient_files:
        raise FileNotFoundError(f"No preprocessed .pt files located in {PROCESSED_DIR}")

    # Select a sample scan (first patient in directory)
    sample_patient_path = patient_files[0]
    patient_id = os.path.splitext(os.path.basename(sample_patient_path))[0]
    
    print(f"\nProcessing Diagnostic Assessment for: {patient_id}")
    visualize_prediction(
        patient_pt_path=sample_patient_path,
        checkpoint_path=CHECKPOINT_PATH,
        output_image_path=OUTPUT_IMAGE_NAME
    )

