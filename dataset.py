import os
import glob
import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, file_paths):
        """
        file_paths: list of paths to preprocessed .pt files
        """
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location="cpu", weights_only=True)
        
        # Cast back to float32 for training computation
        image = data["image"].float()
        mask = data["mask"].float()

        return image, mask
