import torch
from torch.utils.data import Dataset

class QFormerAlignmentDataset(Dataset):
    def __init__(self, filename: str):
        obj = torch.load(filename, map_location="cpu")
        self.samples = obj["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "u": torch.tensor(s["u"], dtype=torch.long),
            "i_pos": torch.tensor(s["i_pos"], dtype=torch.long),
            "i_negs": torch.tensor(s["i_negs"], dtype=torch.long),
            "instruction": s["instruction"],
            "item_text": s["item_text"],
        }
