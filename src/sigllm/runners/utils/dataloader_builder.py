import torch
import logging
import webdataset as wds
from torch.utils.data import DataLoader, DistributedSampler
from sigllm.common.dist_utils import get_world_size, get_rank
from sigllm.common.data_utils import ChainDataset
from sigllm.common.dataloader_utils import MultiIterLoader, PrefetchLoader, IterLoader

def build_dataloaders(datasets, config, train_splits, use_distributed, use_dist_eval_sampler):
    """
    Build dataloaders for training and validation.
    """
    logging.info(
        "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
    )

    # print dataset statistics after concatenation/chaining
    for split_name in datasets:
        if isinstance(datasets[split_name], tuple) or isinstance(
            datasets[split_name], list
        ):
            # mixed wds.DataPipeline and torch.utils.data.Dataset
            num_records = sum(
                [
                    len(d)
                    if not type(d) in [wds.DataPipeline, ChainDataset]
                    else 0
                    for d in datasets[split_name]
                ]
            )

        else:
            if hasattr(datasets[split_name], "__len__"):
                # a single map-style dataset
                num_records = len(datasets[split_name])
            else:
                # a single wds.DataPipeline
                num_records = -1
                logging.info(
                    "Only a single wds.DataPipeline dataset, no __len__ attribute."
                )

        if num_records >= 0:
            logging.info(
                "Loaded {} records for {} split from the dataset.".format(
                    num_records, split_name
                )
            )

    # create dataloaders
    split_names = sorted(datasets.keys())

    datasets_list = [datasets[split] for split in split_names]
    is_trains = [split in train_splits for split in split_names]

    batch_sizes = [
        config.run_cfg.batch_size_train
        if split == "train"
        else config.run_cfg.batch_size_eval
        for split in split_names
    ]

    collate_fns = []
    for dataset in datasets_list:
        if isinstance(dataset, tuple) or isinstance(dataset, list):
            collate_fns.append([getattr(d, "collater", None) for d in dataset])
        else:
            collate_fns.append(getattr(dataset, "collater", None))
    
    loaders = create_loaders(
        datasets=datasets_list,
        num_workers=config.run_cfg.num_workers,
        batch_sizes=batch_sizes,
        is_trains=is_trains,
        collate_fns=collate_fns,
        use_distributed=use_distributed,
        use_dist_eval_sampler=use_dist_eval_sampler
    )

    dataloaders = {k: v for k, v in zip(split_names, loaders)}
    return dataloaders

def create_loaders(
    datasets,
    num_workers,
    batch_sizes,
    is_trains,
    collate_fns,
    use_distributed,
    use_dist_eval_sampler,
    dataset_ratios=None,
):
    """
    Create dataloaders for training and validation.
    """

    def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
        # create a single dataloader for each split
        if isinstance(dataset, ChainDataset) or isinstance(
            dataset, wds.DataPipeline
        ):
            # wds.WebdDataset instance are chained together
            # webdataset.DataPipeline has its own sampler and collate_fn
            loader = iter(
                DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
        else:
            # map-style dataset are concatenated together
            # setup distributed sampler
            if use_distributed:
                sampler = DistributedSampler(
                    dataset,
                    shuffle=is_train,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                )
                if not use_dist_eval_sampler:
                    # e.g. retrieval evaluation
                    sampler = sampler if is_train else None
            else:
                sampler = None

            loader = DataLoader(
                dataset,
                batch_size=bsz,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=sampler is None and is_train,
                collate_fn=collate_fn,
                drop_last=True if is_train else False,
            )
            loader = PrefetchLoader(loader)

            if is_train:
                loader = IterLoader(loader, use_distributed=use_distributed)

        return loader

    loaders = []

    for dataset, bsz, is_train, collate_fn in zip(
        datasets, batch_sizes, is_trains, collate_fns
    ):
        if isinstance(dataset, list) or isinstance(dataset, tuple):
            if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                dataset_ratios = [d.sample_ratio for d in dataset]
            loader = MultiIterLoader(
                loaders=[
                    _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                    for i, d in enumerate(dataset)
                ],
                ratios=dataset_ratios,
            )
        else:
            loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

        loaders.append(loader)

    return loaders
