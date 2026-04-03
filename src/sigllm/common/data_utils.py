
import torch
from typing import List
import logging
import random as random
import webdataset as wds


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, cuda_enabled=True):
    if not cuda_enabled:
        return sample

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets

class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: List[wds.DataPipeline]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                logging.info("One of the datapipeline doesn't define ratio and set to 1 automatically.")

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)
