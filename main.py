import argparse
import os
import glob
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluation(data_dir, checkpoint_path, batch_size=1):
    """
    Runs full evaluation over preprocessed validation .pt files and computes global metrics.
    """
    from model import Unet3D
    from dataset import BraTSDataset
    from metrics import MetricTracker
    from torch.utils.data import DataLoader

    print(f"\n--- Running Full Dataset Evaluation on {DEVICE} ---")
    val_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not val_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    val_split = val_files[int(len(val_files) * 0.8):]
    val_dataset = BraTSDataset(val_split)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Unet3D(in_channels=4, out_channels=3).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                outputs = model(images)
            tracker.update(outputs, masks)

    metrics = tracker.compute()
    print("\n--- Final Evaluation Metrics ---")
    print(f"Mean Dice:           {metrics['Mean_Dice']:.4f}")
    print(f"Whole Tumor (WT):    {metrics['Dice_WT']:.4f}")
    print(f"Tumor Core (TC):     {metrics['Dice_TC']:.4f}")
    print(f"Enhancing Tumor (ET):{metrics['Dice_ET']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Brain Tumor Segmentation (BraTS) Pipeline Orchestrator"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "eval", "visualize", "quantify"],
        help="Pipeline stage: 'preprocess', 'train', 'eval', 'visualize', or 'quantify'"
    )
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
        help="Path to raw BraTS dataset"
    )
    
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed",
        help="Path to preprocessed .pt files"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model weights checkpoint"
    )
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    
    parser.add_argument(
        "--patient_id",
        type=str,
        default="BraTS20_Training_001",
        help="Patient identifier (e.g., BraTS20_Training_001)"
    )
    parser.add_argument(
        "--output_img",
        type=str,
        default="prediction_comparison.png",
        help="Filename for visual inference output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "preprocess":
        from preprocess import preprocess_and_save
        print("Starting offline data preprocessing pipeline...")
        preprocess_and_save(args.raw_data_dir, args.processed_dir)

    elif args.mode == "train":
        # Executes train.py directly
        import train
        if hasattr(train, "train_model"):
            train.train_model(args.processed_dir, args.epochs, args.batch_size, args.lr)
        elif hasattr(train, "main"):
            train.main()
        else:
            print("Please run 'python train.py' directly to start full training.")

    elif args.mode == "eval":
        run_evaluation(
            data_dir=args.processed_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size
        )

    elif args.mode == "visualize":
        from visualize import visualize_prediction
        patient_path = os.path.join(args.processed_dir, f"{args.patient_id}.pt")
        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"Target patient scan not found: {patient_path}")
        
        print(f"Running visual inference on: {args.patient_id}")
        visualize_prediction(
            patient_pt_path=patient_path,
            checkpoint_path=args.checkpoint,
            output_image_path=args.output_img
        )

    elif args.mode == "quantify":
        from volumetry import quantify_patient
        patient_path = os.path.join(args.processed_dir, f"{args.patient_id}.pt")
        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"Target patient scan not found: {patient_path}")

        out_json = f"{args.patient_id}_volumetry.json"
        print(f"Calculating clinical volumetry for: {args.patient_id}")
        report = quantify_patient(
            patient_pt_path=patient_path,
            checkpoint_path=args.checkpoint,
            output_json_path=out_json
        )
        print(f"\n--- Clinical Volumetry: {args.patient_id} ---")
        print(f"  Whole Tumor (WT):    {report['volumes_cm3']['whole_tumor']} cm³")
        print(f"  Tumor Core (TC):     {report['volumes_cm3']['tumor_core']} cm³")
        print(f"  Enhancing Tumor (ET):{report['volumes_cm3']['enhancing_tumor']} cm³")
        print(f"  Peritumoral Edema:   {report['volumes_cm3']['peritumoral_edema']} cm³")
        print(f"  Peak Axial Slice:    Index {report['clinical_metrics']['peak_axial_slice_idx']}")


if __name__ == "__main__":
    main()