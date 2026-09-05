import os
import json
import numpy as np
import torch

def compute_tumor_volumes(pred_mask, voxel_spacing=(1.0, 1.0, 1.0)):

    if isinstance(pred_mask, torch.Tensor):

        pred_mask = pred_mask.detach().cpu().numpy()

    pred_mask = (pred_mask>0.5).astype(np.uint8)

    voxel_volume_mm3 = voxel_spacing[0]*voxel_spacing[1]*voxel_spacing[2]
    voxel_volume_cm3 = voxel_volume_mm3/1000.0

    wt_voxels = int(np.sum(pred_mask[0]))
    tc_voxels = int(np.sum(pred_mask[1]))
    et_voxels = int(np.sum(pred_mask[2]))

    vol_wt = wt_voxels*voxel_volume_cm3
    vol_tc = wt_voxels*voxel_volume_cm3
    vol_et = et_voxels*voxel_volume_cm3

    vol_edema = max(0.0, vol_wt-vol_tc)

    et_to_tc = (vol_et/vol_tc * 100.0) if vol_tc>0 else 0.0
    tc_to_wt = (vol_tc/vol_wt * 100.0) if vol_wt>0 else 0.0

    wt_slice_count = pred_mask[0].sum(axis=(1,2))
    peak_slice = int(np.argmax(wt_slice_count)) if wt_voxels>0 else -1

    return {
        "volumes_cm3": {
            "whole_tumor": round(vol_wt, 3),
            "tumor_core": round(vol_tc, 3),
            "enhancing_tumor": round(vol_et, 3),
            "peritumoral_edema": round(vol_edema, 3)
        },
        "voxel_counts": {
            "wt": wt_voxels,
            "tc": tc_voxels,
            "et": et_voxels
        },
        "clinical_metrics": {
            "enhancing_fraction_pct": round(et_to_tc, 2),
            "core_to_whole_ratio_pct": round(tc_to_wt, 2),
            "peak_axial_slice_idx": peak_slice
        }
    }

def quantify_patient(patient_pt_path, checkpoint_path, output_json_path=None, device="cuda"):

    from model import Unet3D

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    data = torch.load(patient_pt_path, map_location=dev, weights_only=False)
    image = data["image"].unsqueeze(0).to(dev).float()

    model = Unet3D(in_channels=4, out_channels=3).to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
            logits = model(image)
            probs = torch.sigmoid(logits)

    
    binary_mask = (probs[0] > 0.5).byte()
    results = compute_tumor_volumes(binary_mask)

    
    patient_id = data.get("patient_id", os.path.splitext(os.path.basename(patient_pt_path))[0])
    results["patient_id"] = patient_id

    
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Volumetric clinical report saved to: {output_json_path}")

    return results


if __name__ == "__main__":
    # Test block on a sample patient
    test_pt = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed\BraTS20_Training_001.pt"
    ckpt = "checkpoints/best_model.pth"

    if os.path.exists(test_pt) and os.path.exists(ckpt):
        report = quantify_patient(test_pt, ckpt, output_json_path="volumetric_report.json")
        print("\n--- Clinical Volumetry Telemetry ---")
        print(json.dumps(report, indent=4))
    else:
        print("Verification notice: Test scan or checkpoint path not found. Check local paths.")

