import json

from torch.utils.data import Dataset

class ClinicalICDDiseaseDataset(Dataset):
    def __init__(self, data_dir: str):
        data_loc = f"{data_dir}/pqal_grouped_by_icd.json"

        with open(data_loc, "r") as f:
            self.icd_group = json.load(f)

        group_sizes = {k: len(v) for k, v in self.icd_group.items()}
        self.total_samples = sum(group_sizes.values())
        print(f"Loaded dataset with {self.total_samples} samples")

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        