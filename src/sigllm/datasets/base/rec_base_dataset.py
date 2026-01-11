"""Base abstractions for recommendation datasets."""

from torch.utils.data import Dataset
import pandas as pd

from sigllm.datasets.configs.rec_dataset_config import RecDatasetConfig

class RecBaseDataset(Dataset):
    """Placeholder class for shared dataset behavior."""

    def __init__(self, config: RecDatasetConfig) -> None:
        ann_path = config.ann_path

        if (ann_path is None) or (not ann_path.exists()):
            raise ValueError(f"Annotation path {ann_path} does not exist.")
        
        self.annotation = pd.read_pickle(ann_path.with_suffix(".pkl")).values

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        return self.annotation[idx]
    
    def convert_title_list(self, titles):
        titles_ = []
        for x in titles:
            if len(x)>0:
                titles_.append("\""+ x + "\"")
        if len(titles_)>0:
            return ", ".join(titles_)
        else:
            return "unkow"