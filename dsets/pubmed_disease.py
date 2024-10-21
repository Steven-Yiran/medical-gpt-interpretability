import json

from torch.utils.data import Dataset

class ClinicalDiseaseDataset(Dataset):
    def __init__(self, data_dir: str, with_context: bool):
        self.with_context = with_context
        pqa_loc = f"{data_dir}/pqa_disease.json"

        with open(pqa_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        subject_arr = self.data[idx]["subject"]
        self.data[idx]["subject"] = subject_arr[0]

        prompt = "Question: " + self.data[idx]["question"]
        if self.with_context:
            prompt += " Context: " + self.data[idx]["context"] \
                + "the answer to the question given the context is"
        else:
            prompt += " the answer to the question is"
        self.data[idx]["prompt"] = prompt

        return self.data[idx]

        