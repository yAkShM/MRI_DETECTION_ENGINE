import os
import time
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from pynetdicom import AE, StoragePresentationContexts

def create_dummy_slice(i, num_slices=3):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = UID(f"1.2.3.{i}")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = FileDataset(f"dummy_{i}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "PACS_TEST_001"
    ds.StudyInstanceUID = "9.9.9"
    ds.SeriesInstanceUID = "9.9.9.1"
    ds.SOPInstanceUID = f"1.2.3.{i}"
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    
    ds.SeriesDescription = "Test PACS Push"
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    ds.ImagePositionPatient = [0.0, 0.0, float(i * 2.0)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    
    ds.Rows = 128
    ds.Columns = 128
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    
    pixel_array = np.zeros((128, 128), dtype=np.int16)
    pixel_array[50:70, 50:70] = 500  # Fake tumor
    ds.PixelData = pixel_array.tobytes()
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    return ds

def test_pacs_push():
    ae = AE()
    ae.requested_contexts = StoragePresentationContexts
    
    print("Initiating Association with BRAIN_AI_SCP on port 11112...")
    assoc = ae.associate('127.0.0.1', 11112, ae_title=b'BRAIN_AI_SCP')
    
    if assoc.is_established:
        print("Association established. Sending slices...")
        for i in range(3):
            ds = create_dummy_slice(i)
            status = assoc.send_c_store(ds)
            if status:
                print(f"Slice {i} sent successfully (Status: {status.Status})")
            else:
                print(f"Failed to send slice {i}")
                
        assoc.release()
        print("Association released.")
    else:
        print("Failed to establish association. Is api.py running?")
        return
        
    print("Waiting 5 seconds for debounce and inference...")
    time.sleep(5)
    
    # Check if JSON was generated
    out_file = os.path.join("output", "9.9.9.1_results.json")
    if os.path.exists(out_file):
        print(f"Success! Output JSON found at {out_file}")
        with open(out_file, "r") as f:
            print(f.read())
    else:
        print(f"Failed: Output JSON {out_file} not found!")

if __name__ == "__main__":
    test_pacs_push()
