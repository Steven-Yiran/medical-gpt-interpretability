import json

from torch.utils.data import Dataset

class ClinicalICDDiseaseDataset(Dataset):
    def __init__(self, data_dir: str, with_context: bool = False):
        data_loc = f"{data_dir}/pqal_grouped_by_icd.json"
        self.data = []
        self.with_context = with_context

        with open(data_loc, "r") as f:
            self.icd_group = json.load(f)

        self.data = self.flatten_data(self.icd_group)
        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def flatten_data(self, data_dict):
        data = []
        not_found = 0
        for icd, group in data_dict.items():
            for item in group:
                entry = {}
                entry["icd_group"] = icd

                if self.with_context:
                    raise NotImplementedError("Context is not implemented yet")
                else:
                    template = "Question: {} The answer to the question is"
                    entry["prompt"] = template.format(item["QUESTION"].lower())

                entry["subject"] = ""
                for s in item["subject"]:
                    if s in entry["prompt"]:
                        entry["subject"] = s
                        break
                if entry["subject"] == "":
                    not_found += 1
                    continue

                entry["attribute"] = item["final_decision"]
                data.append(entry)
        print(f"Could not find subjects in {not_found} entries")
        return data
        