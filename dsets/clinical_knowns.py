import json

import torch
from torch.utils.data import Dataset


class ClinicalKnownsDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        known_loc = f"{data_dir}/knowns.json"

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
