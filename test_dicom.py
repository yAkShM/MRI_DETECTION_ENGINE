import os
import zipfile
import tempfile
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from fastapi.testclient import TestClient

from api import app

def create_synthetic_dicom_series(temp_dir, num_slices=5):
    os.makedirs(temp_dir, exist_ok=True)
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    spacing = [1.0, 1.0]
    thickness = 2.0
    
    for i in range(num_slices):
        filename = os.path.join(temp_dir, f"slice_{i:03d}.dcm")
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        ds.PatientID = "TestPatient123"
        ds.SeriesDescription = "Synthetic Series"
        ds.PixelSpacing = spacing
        ds.SliceThickness = thickness
        ds.ImagePositionPatient = [0.0, 0.0, float(i * thickness)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        ds.Rows = 128
        ds.Columns = 128
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        
        # Dummy pixel data
        pixel_array = np.zeros((128, 128), dtype=np.int16)
        pixel_array[60:68, 60:68] = 1000
        
        ds.PixelData = pixel_array.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        
        ds.save_as(filename)

def test_dicom_pipeline():
    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_dir = os.path.join(temp_dir, "dicoms")
        create_synthetic_dicom_series(dicom_dir, num_slices=5)
        
        zip_path = os.path.join(temp_dir, "test_series.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, _, files in os.walk(dicom_dir):
                for file in files:
                    zf.write(os.path.join(root, file), arcname=file)
                    
        print("Zip created. Starting API test...")
        with TestClient(app) as client:
            with open(zip_path, "rb") as f:
                response = client.post(
                    "/predict/dicom",
                    files={"file": ("test_series.zip", f, "application/zip")}
                )
                
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
        assert response.status_code == 200, f"Failed with {response.status_code}"
        
        data = response.json()
        assert "metadata" in data
        assert "volumetry" in data
        
        metadata = data["metadata"]
        assert metadata["patient_id"] == "TestPatient123"
        assert metadata["series_description"] == "Synthetic Series"
        assert metadata["slice_thickness"] == 2.0
        
        volumetry = data["volumetry"]
        print("Success! Test passed.")
        print(f"Volumetry: {volumetry}")
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    test_dicom_pipeline()
