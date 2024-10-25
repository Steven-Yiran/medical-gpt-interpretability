import json

from torch.utils.data import Dataset

class ClinicalDiseaseDataset(Dataset):
    def __init__(self, data_dir: str):
        pqa_loc = f"{data_dir}/disease_pqal.json"

        with open(pqa_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class ClinicalMedicineDataset(Dataset):
    def __init__(self, data_dir: str):
        pqa_loc = f"{data_dir}/medicine_pqal.json"

        with open(pqa_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]