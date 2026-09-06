import io
import os
import time
import torch
import numpy as np
import zipfile
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Tuple

from model import Unet3D
from volumetry import compute_tumor_volumes
from dicom_parser import load_and_process_dicom_series

# Schemas
class VolumetryMetrics(BaseModel):
    volume_wt_cm3: float
    volume_tc_cm3: float
    volume_et_cm3: float
    malignancy_ratio: float
    peak_slice_idx: int
    inference_latency_ms: float

class DicomMetadata(BaseModel):
    patient_id: str
    series_description: str
    pixel_spacing: Tuple[float, float]
    slice_thickness: float
    calculated_voxel_volume_mm3: float

class DicomResponse(BaseModel):
    metadata: DicomMetadata
    volumetry: VolumetryMetrics

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    vram_allocated_mb: Optional[float]
    pacs_listener_active: Optional[bool] = None

# Global variables for model and device
model = None
device = None
pacs_listener = None

import json
def on_pacs_series_complete(staging_dir, study_uid, series_uid):
    print(f"PACS: Series {series_uid} fully received. Processing...")
    try:
        from dicom_parser import load_and_process_dicom_series
        tensor, metadata, effective_spacing = load_and_process_dicom_series(staging_dir)
        
        binary_mask, latency_ms = _run_inference(tensor)
        metrics_dict = compute_tumor_volumes(binary_mask, voxel_spacing=effective_spacing)
        
        vols = metrics_dict["volumes_cm3"]
        clinical = metrics_dict["clinical_metrics"]
        
        results = {
            "metadata": metadata,
            "volumetry": {
                "volume_wt_cm3": vols["whole_tumor"],
                "volume_tc_cm3": vols["tumor_core"],
                "volume_et_cm3": vols["enhancing_tumor"],
                "malignancy_ratio": clinical["enhancing_fraction_pct"] / 100.0,
                "peak_slice_idx": clinical["peak_axial_slice_idx"],
                "inference_latency_ms": latency_ms
            },
            "study_uid": study_uid,
            "series_uid": series_uid
        }
        
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", f"{series_uid}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"PACS: Results saved to {out_path}")
        
    except Exception as e:
        print(f"PACS Error processing series {series_uid}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, pacs_listener
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    print(f"Loading model onto {device}...")
    model = Unet3D(in_channels=4, out_channels=3).to(device)
    
    checkpoint_path = "checkpoints/best_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}")
        # Continuing anyway for the sake of starting up, might want to fail fast in prod.
    
    # Start PACS listener
    from pacs_listener import PACSListener
    try:
        pacs_listener = PACSListener(on_series_complete_callback=on_pacs_series_complete)
        pacs_listener.start()
    except Exception as e:
        print(f"Warning: Could not start PACS listener: {e}")
    
    yield
    
    # Clean up
    print("Shutting down and cleaning up model...")
    if pacs_listener:
        pacs_listener.stop()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="3D Brain Tumor Segmentation API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    vram_mb = None
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
    pacs_active = False
    if pacs_listener is not None:
        pacs_active = pacs_listener.is_running
        
    return HealthResponse(
        status="ok",
        device=str(device),
        model_loaded=model is not None,
        vram_allocated_mb=vram_mb,
        pacs_listener_active=pacs_active
    )

def _run_inference(tensor: torch.Tensor):
    """Internal function to run the inference pass."""
    tensor = tensor.to(device).float()
    
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
        
    if tensor.shape[1] != 4 or tensor.shape[2:] != (128, 128, 128):
        raise ValueError(f"Expected shape [4, 128, 128, 128] or [1, 4, 128, 128, 128], got {tensor.shape}")

    start_time = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(tensor)
            probs = torch.sigmoid(logits)
            binary_mask = (probs[0] > 0.5).byte()
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return binary_mask, latency_ms

@app.post("/predict/pt", response_model=VolumetryMetrics)
async def predict_volumetry(file: UploadFile = File(...)):
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Only .pt files are supported.")
        
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        data = torch.load(buffer, map_location=device, weights_only=False)
        
        if isinstance(data, dict) and "image" in data:
            tensor = data["image"]
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            raise HTTPException(status_code=400, detail="Invalid .pt file content. Expected a tensor or a dict with 'image'.")
            
        binary_mask, latency_ms = _run_inference(tensor)
        
        # Calculate volumetry
        metrics_dict = compute_tumor_volumes(binary_mask)
        
        # Clean up input tensor and buffer
        del tensor
        del data
        del buffer
        del contents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Parse metrics
        vols = metrics_dict["volumes_cm3"]
        clinical = metrics_dict["clinical_metrics"]
        
        return VolumetryMetrics(
            volume_wt_cm3=vols["whole_tumor"],
            volume_tc_cm3=vols["tumor_core"],
            volume_et_cm3=vols["enhancing_tumor"],
            malignancy_ratio=clinical["enhancing_fraction_pct"] / 100.0,
            peak_slice_idx=clinical["peak_axial_slice_idx"],
            inference_latency_ms=latency_ms
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/pt/mask")
async def predict_mask(file: UploadFile = File(...)):
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Only .pt files are supported.")
        
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        data = torch.load(buffer, map_location=device, weights_only=False)
        
        if isinstance(data, dict) and "image" in data:
            tensor = data["image"]
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            raise HTTPException(status_code=400, detail="Invalid .pt file content. Expected a tensor or a dict with 'image'.")
            
        binary_mask, _ = _run_inference(tensor)
        
        # Serialize the output tensor
        out_buffer = io.BytesIO()
        torch.save(binary_mask.cpu(), out_buffer)
        out_buffer.seek(0)
        
        # Clean up
        del tensor
        del data
        del buffer
        del contents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return StreamingResponse(
            out_buffer, 
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file.filename.replace('.pt', '_mask.pt')}"}
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/dicom", response_model=DicomResponse)
async def predict_dicom(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files containing DICOM series are supported.")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            contents = await file.read()
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(contents)
                
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # Process DICOM
            tensor, metadata, effective_spacing = load_and_process_dicom_series(temp_dir)
            
            # Inference
            binary_mask, latency_ms = _run_inference(tensor)
            
            # Volumetry calculation using the effective spacing
            metrics_dict = compute_tumor_volumes(binary_mask, voxel_spacing=effective_spacing)
            
            vols = metrics_dict["volumes_cm3"]
            clinical = metrics_dict["clinical_metrics"]
            
            volumetry = VolumetryMetrics(
                volume_wt_cm3=vols["whole_tumor"],
                volume_tc_cm3=vols["tumor_core"],
                volume_et_cm3=vols["enhancing_tumor"],
                malignancy_ratio=clinical["enhancing_fraction_pct"] / 100.0,
                peak_slice_idx=clinical["peak_axial_slice_idx"],
                inference_latency_ms=latency_ms
            )
            
            dicom_meta = DicomMetadata(**metadata)
            
            return DicomResponse(
                metadata=dicom_meta,
                volumetry=volumetry
            )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
