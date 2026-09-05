import os
# Prevent CUDA memory fragmentation on limited VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import glob
import torch
from torch.utils.data import DataLoader
from dataset import dataset as BraTSDataset
from model import Unet3D
from loss import BCEDiceLoss
from metrics import MetricTracker

# --- 1. System & Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing pipeline on device: {DEVICE}")


def get_patient_splits(data_root, train_ratio=0.8, seed=42):
    all_files = glob.glob(os.path.join(data_root, "*.pt"))
    
    random.seed(seed)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total Preprocessed Patients: {len(all_files)} | Train: {len(train_files)} | Validation: {len(val_files)}")
    return train_files, val_files


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Executes one full training epoch under Automatic Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)

        # Updated modern torch.amp.autocast syntax
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    """
    Evaluates validation set without tracking gradients.
    """
    model.eval()
    running_loss = 0.0
    tracker = MetricTracker()

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item()
            tracker.update(outputs, masks)

    metrics = tracker.get_results()
    metrics['val_loss'] = running_loss / len(loader)
    return metrics


if __name__ == "__main__":
    # --- Configuration Hyperparameters ---
    DATA_ROOT = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed"
    EPOCHS = 50
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4

    # --- Data Pipelines ---
    train_dirs, val_dirs = get_patient_splits(DATA_ROOT)

    train_dataset = BraTSDataset(train_dirs)
    val_dataset = BraTSDataset(val_dirs)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=(DEVICE.type == "cuda")
    )

    # --- Model, Loss, Optimizer & Modern Scaler Initialization ---
    model = Unet3D(in_channels=4, out_channels=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Modern torch.amp.GradScaler syntax
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    # --- Checkpoint Setup ---
    best_mean_dice = 0.0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nStarting Training Pipeline...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_metrics = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] (LR: {current_lr:.6f}) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Mean Dice: {val_metrics['Mean_Dice']:.4f} "
              f"(WT: {val_metrics['Dice_WT']:.4f}, TC: {val_metrics['Dice_TC']:.4f}, ET: {val_metrics['Dice_ET']:.4f})")

        # Save checkpoint if Mean Dice improves
        if val_metrics["Mean_Dice"] > best_mean_dice:
            best_mean_dice = val_metrics["Mean_Dice"]
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_dice": best_mean_dice,
            }, save_path)
            print(f"  >>> Checkpoint saved: New best Mean Dice = {best_mean_dice:.4f} at {save_path}")