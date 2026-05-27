import torch
from torch.utils.data import Dataset

class DrugResponsePointDataset(Dataset):
    """
    Drug combination response dataset class
    Dataset: Data sample list, drug feature dictionary
    return: Drug characteristics, concentration coordinates, response values, and boundaries for a single sample
    """
    def __init__(self, data_list, drug_features_dict):
        self.data_list = data_list
        self.drug_features_dict = drug_features_dict
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        combo_name = sample['combo_name']
        try:
            drug1_name, drug2_name = combo_name.split('_')
            features1 = self.drug_features_dict[drug1_name.strip()]
            features2 = self.drug_features_dict[drug2_name.strip()]
            drug_features = torch.stack([features1, features2], dim=1)
        except (KeyError, ValueError):
            return None
        return {
            "drug_features": drug_features.float(),
            "sparse_coords_log": torch.from_numpy(sample["sparse_coords_log"]).float(),
            "sparse_y": torch.from_numpy(sample["sparse_y"]).float(),
            "log_coord_bounds": torch.from_numpy(sample["log_coord_bounds"]).float()
        }