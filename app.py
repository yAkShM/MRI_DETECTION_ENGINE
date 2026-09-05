import os
import glob
import json
import tempfile
import numpy as np
import torch
import streamlit as st
from PIL import Image
from fpdf import FPDF

from model import Unet3D
from volumetry import compute_tumor_volumes

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="BraTS 3D Volumetric Tumor Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = r"C:\Users\HP-PC\OneDrive\Desktop\BRATS\BraTS2020_Processed"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ---------------------------------------------------------
# 1. Critical Caching Engines (Prevents GPU Memory Leaks)
# ---------------------------------------------------------
@st.cache_resource
def load_model(checkpoint_path):
    """Loads 3D U-Net weights into VRAM strictly ONCE."""
    model = Unet3D(in_channels=4, out_channels=3).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

@st.cache_data
def run_patient_inference(patient_pt_path):
    """
    Runs 3D forward pass ONCE per patient scan.
    Returns:
        image_np: [4, 128, 128, 128] float32 array
        pred_mask: [3, 128, 128, 128] uint8 binary array
        telemetry: clinical metrics dictionary
    """
    model = load_model(CHECKPOINT_PATH)
    data = torch.load(patient_pt_path, map_location=DEVICE, weights_only=False)
    image_tensor = data["image"].unsqueeze(0).to(DEVICE).float()  # [1, 4, 128, 128, 128]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)

    pred_mask = (probs[0] > 0.5).byte().cpu().numpy()  # [3, 128, 128, 128]
    image_np = data["image"].float().cpu().numpy()      # [4, 128, 128, 128]
    telemetry = compute_tumor_volumes(pred_mask)
    return image_np, pred_mask, telemetry

# ---------------------------------------------------------
# 2. Vectorized Alpha Compositor
# ---------------------------------------------------------
def create_composite_overlay(base_slice, wt_slice, tc_slice, et_slice, show_wt, show_tc, show_et, alpha):
    """Alpha-blends RGB color channels onto grayscale MRI without contrast clipping."""
    # Min-max scale background to [0, 1] for visual display
    b_min, b_max = base_slice.min(), base_slice.max()
    norm_base = (base_slice - b_min) / (b_max - b_min + 1e-8)
    rgb = np.stack([norm_base] * 3, axis=-1)  # [H, W, 3]

    # Red: Whole Tumor (WT), Green: Tumor Core (TC), Blue: Enhancing (ET)
    if show_wt and np.any(wt_slice):
        mask = wt_slice.astype(bool)
        rgb[mask] = rgb[mask] * (1 - alpha) + np.array([1.0, 0.1, 0.1]) * alpha
    if show_tc and np.any(tc_slice):
        mask = tc_slice.astype(bool)
        rgb[mask] = rgb[mask] * (1 - alpha) + np.array([0.1, 0.9, 0.1]) * alpha
    if show_et and np.any(et_slice):
        mask = et_slice.astype(bool)
        rgb[mask] = rgb[mask] * (1 - alpha) + np.array([0.1, 0.4, 1.0]) * alpha

    return np.clip(rgb, 0.0, 1.0)

def generate_pdf_report(telemetry, composite_img, patient_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, text="BraTS 3D Volumetric Tumor Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, text=f"Patient ID: {patient_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=f"Malignancy Index (% ET/TC): {telemetry['clinical_metrics']['enhancing_fraction_pct']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=f"Surgical Ratio (% TC/WT): {telemetry['clinical_metrics']['core_to_whole_ratio_pct']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, text="Volume Metrics:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, text=f"- Whole Tumor (WT): {telemetry['volumes_cm3']['whole_tumor']} cm3", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=f"- Tumor Core (TC): {telemetry['volumes_cm3']['tumor_core']} cm3", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=f"- Enhancing (ET): {telemetry['volumes_cm3']['enhancing_tumor']} cm3", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, text=f"- Peritumoral Edema: {telemetry['volumes_cm3']['peritumoral_edema']} cm3", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Save composite as temp image and add to PDF
    img_uint8 = (composite_img * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        pdf.image(tmp.name, w=100)
    
    # PDF output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
    
    return pdf_bytes

# ---------------------------------------------------------
# 3. UI Controls & Telemetry Dashboard
# ---------------------------------------------------------
st.title("3D Volumetric Tumor Viewer")
st.caption(f"Inference Engine: 3D U-Net | Active Accelerator: {DEVICE.type.upper()}")

# Locate patient scans
patient_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.pt")))
if not patient_files:
    st.error(f"No .pt files found in {PROCESSED_DIR}")
    st.stop()

patient_names = [os.path.splitext(os.path.basename(f))[0] for f in patient_files]

with st.sidebar:
    st.header("Patient Selection")
    selected_name = st.selectbox("Scan ID", patient_names)
    selected_pt = os.path.join(PROCESSED_DIR, f"{selected_name}.pt")

    st.divider()
    st.header("Modality & Orientation")
    modality_idx = st.radio("MRI Sequence", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["FLAIR", "T1", "T1ce", "T2"][x], index=0)
    orientation = st.selectbox("Plane View", ["Axial (Top-Down)", "Coronal (Front)", "Sagittal (Side)"])

    st.divider()
    st.header("Overlay Masks")
    show_wt = st.checkbox("Whole Tumor (Red)", value=True)
    show_tc = st.checkbox("Tumor Core (Green)", value=True)
    show_et = st.checkbox("Enhancing Tumor (Blue)", value=True)
    alpha = st.slider("Mask Opacity (Alpha)", 0.1, 0.9, 0.45, 0.05)

# Run cached inference
image_np, pred_mask, telemetry = run_patient_inference(selected_pt)

# Telemetry KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Whole Tumor (WT)", f"{telemetry['volumes_cm3']['whole_tumor']} cm³")
col2.metric("Tumor Core (TC)", f"{telemetry['volumes_cm3']['tumor_core']} cm³")
col3.metric("Enhancing (ET)", f"{telemetry['volumes_cm3']['enhancing_tumor']} cm³")
col4.metric("Peritumoral Edema", f"{telemetry['volumes_cm3']['peritumoral_edema']} cm³")

# Slicing Navigation
peak_slice = telemetry["clinical_metrics"]["peak_axial_slice_idx"]
default_slice = peak_slice if (peak_slice != -1 and orientation.startswith("Axial")) else 64

col_btn, col_slider = st.columns([1, 4])
with col_btn:
    st.write("")
    if st.button("📍 Snap to Peak"):
        st.session_state["slice_slider"] = peak_slice

with col_slider:
    slice_idx = st.slider("Depth Navigation (0 - 127)", 0, 127, value=default_slice, key="slice_slider")

# Extract 2D slices based on orientation
if orientation.startswith("Axial"):
    base_slice = image_np[modality_idx, slice_idx, :, :]
    wt_slice   = pred_mask[0, slice_idx, :, :]
    tc_slice   = pred_mask[1, slice_idx, :, :]
    et_slice   = pred_mask[2, slice_idx, :, :]
elif orientation.startswith("Coronal"):
    base_slice = image_np[modality_idx, :, slice_idx, :]
    wt_slice   = pred_mask[0, :, slice_idx, :]
    tc_slice   = pred_mask[1, :, slice_idx, :]
    et_slice   = pred_mask[2, :, slice_idx, :]
else:  # Sagittal
    base_slice = image_np[modality_idx, :, :, slice_idx]
    wt_slice   = pred_mask[0, :, :, slice_idx]
    tc_slice   = pred_mask[1, :, :, slice_idx]
    et_slice   = pred_mask[2, :, :, slice_idx]

composite = create_composite_overlay(base_slice, wt_slice, tc_slice, et_slice,
                                     show_wt, show_tc, show_et, alpha)

# Display main viewport
col_img, col_info = st.columns([3, 2])
with col_img:
    st.image(composite[::-1, :, :], use_container_width=True)

with col_info:
    st.subheader("Diagnostic Telemetry")
    st.write(f"**Current Slice Index:** `{slice_idx} / 127`")
    st.write(f"**Peak Axial Slice Index:** `{peak_slice}`")
    st.write(f"**Malignancy Index (% ET/TC):** `{telemetry['clinical_metrics']['enhancing_fraction_pct']}%`")
    st.write(f"**Surgical Ratio (% TC/WT):** `{telemetry['clinical_metrics']['core_to_whole_ratio_pct']}%`")
    
    st.divider()
    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(telemetry, indent=4),
        file_name=f"{selected_name}_report.json",
        mime="application/json"
    )
    
    pdf_bytes = generate_pdf_report(telemetry, composite[::-1, :, :], selected_name)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"{selected_name}_report.pdf",
        mime="application/pdf"
    )