import os
import pydicom
import numpy as np
import torch
import scipy.ndimage

def load_and_process_dicom_series(dir_path):
    """
    Reads DICOM files from dir_path, sorts them, extracts metadata, 
    resamples to 128x128x128, and normalizes them.
    Returns:
        tensor: Shape [4, 128, 128, 128], dtype torch.float32
        metadata: Dict with parsed DICOM metadata
        effective_voxel_spacing: Tuple of (z, y, x) spacing in mm for the resampled grid
    """
    slices = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('.dcm') or not '.' in file:
                try:
                    ds = pydicom.dcmread(os.path.join(root, file))
                    if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'PixelData'):
                        slices.append(ds)
                except Exception as e:
                    # Not a valid DICOM or unreadable
                    pass
                    
    if not slices:
        raise ValueError("No valid DICOM slices found in the provided directory.")

    # Sort slices by ImagePositionPatient
    # Calculate normal vector of the slice from ImageOrientationPatient, or fallback to Z-axis
    if hasattr(slices[0], 'ImageOrientationPatient'):
        iop = slices[0].ImageOrientationPatient
        row_cosine = np.array(iop[:3])
        col_cosine = np.array(iop[3:])
        normal = np.cross(row_cosine, col_cosine)
    else:
        normal = np.array([0, 0, 1])

    # Sort strictly by projection of ImagePositionPatient on the normal vector
    slices.sort(key=lambda s: np.dot(np.array(s.ImagePositionPatient), normal))

    # Extract metadata from the first slice
    ref_slice = slices[0]
    
    if hasattr(ref_slice, 'PixelSpacing'):
        spacing_y, spacing_x = map(float, ref_slice.PixelSpacing)
    else:
        spacing_y, spacing_x = 1.0, 1.0

    if hasattr(ref_slice, 'SliceThickness'):
        slice_thickness = float(ref_slice.SliceThickness)
    else:
        if len(slices) > 1:
            pos1 = np.array(slices[0].ImagePositionPatient)
            pos2 = np.array(slices[1].ImagePositionPatient)
            slice_thickness = float(np.linalg.norm(pos2 - pos1))
        else:
            slice_thickness = 1.0

    patient_id = getattr(ref_slice, 'PatientID', 'Unknown')
    series_desc = getattr(ref_slice, 'SeriesDescription', 'Unknown')
    original_voxel_volume_mm3 = spacing_x * spacing_y * slice_thickness

    metadata = {
        'patient_id': patient_id,
        'series_description': series_desc,
        'pixel_spacing': (spacing_x, spacing_y),
        'slice_thickness': slice_thickness,
        'calculated_voxel_volume_mm3': original_voxel_volume_mm3
    }

    # Volume Construction
    volume = []
    for s in slices:
        img2d = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, 'RescaleSlope', 1.0))
        intercept = float(getattr(s, 'RescaleIntercept', 0.0))
        img2d = img2d * slope + intercept
        volume.append(img2d)

    # Convert to 3D NumPy array: Shape [D, H, W]
    volume3d = np.stack(volume)
    D, H, W = volume3d.shape

    # Resample to [128, 128, 128]
    target_shape = (128, 128, 128)
    zoom_factors = (
        target_shape[0] / D,
        target_shape[1] / H,
        target_shape[2] / W
    )
    
    # Resample
    resampled_volume = scipy.ndimage.zoom(volume3d, zoom_factors, order=1)
    
    # Calculate effective voxel spacing for the resampled grid
    eff_spacing_z = slice_thickness / zoom_factors[0]
    eff_spacing_y = spacing_y / zoom_factors[1]
    eff_spacing_x = spacing_x / zoom_factors[2]
    effective_voxel_spacing = (eff_spacing_z, eff_spacing_y, eff_spacing_x)

    # Foreground Intensity Normalization (Z-score over non-zero voxels)
    mask = resampled_volume > 0
    if np.any(mask):
        mean_val = np.mean(resampled_volume[mask])
        std_val = np.std(resampled_volume[mask])
        if std_val > 0:
            resampled_volume[mask] = (resampled_volume[mask] - mean_val) / std_val
        else:
            resampled_volume[mask] = resampled_volume[mask] - mean_val

    # Modality Channeling
    # 3D U-Net requires [4, 128, 128, 128]
    tensor = torch.tensor(resampled_volume, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).repeat(4, 1, 1, 1)

    return tensor, metadata, effective_voxel_spacing
