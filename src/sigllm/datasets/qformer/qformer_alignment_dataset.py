"""Dataset for Q-Former alignment (stage 1)."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sigllm.common.logging_utils import NotebookLogger
from sigllm.datasets.base.rec_base_dataset import RecBaseDataset

LOGGER = NotebookLogger.rich_logger("sigllm.qformer_alignment_dataset")


def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


class QFormerAlignmentDataset(RecBaseDataset):
    def __init__(self, config, filename: str, neg_k: int = 5) -> None:
        self.neg_k = neg_k

        dataset_config = config
        ann_path = dataset_config.build_info.storage / filename
        if (ann_path is None) or (not ann_path.exists()):
            raise ValueError(f"Annotation path {ann_path} does not exist.")

        df = pd.read_pickle(ann_path.with_suffix(".pkl")).reset_index(drop=True)
        pos_df = df[df["label"] == 1]
        if pos_df.empty:
            raise ValueError("No positive samples found for alignment stage.")

        # Cache lookups for sampling
        self.annotation = df
        self.user_pos_indices = pos_df.groupby("uid").apply(lambda x: x.index.tolist()).to_dict()
        self.users = list(self.user_pos_indices.keys())
        self.all_items = set(df["iid"].unique().tolist())

        log_step("data path", str(ann_path))
        log_step("users with positives", str(len(self.users)))
        log_step("item vocab", str(len(self.all_items)))

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = random.choice(self.users)
        pos_indices = self.user_pos_indices[user]
        pos_row = self.annotation.iloc[random.choice(pos_indices)]

        pos_item = int(pos_row["iid"])
        title = str(pos_row["title"])
        history_set = set(pos_row["his"]) if "his" in pos_row else set()
        banned_items = history_set | {pos_item}

        candidate_pool = list(self.all_items - banned_items)
        if len(candidate_pool) == 0:
            raise ValueError(f"No negatives available for user {user}.")

        replace_flag = len(candidate_pool) < self.neg_k
        neg_items = np.random.choice(candidate_pool, size=self.neg_k, replace=replace_flag)

        sample = {
            "u": torch.tensor(user, dtype=torch.long),
            "i_pos": torch.tensor(pos_item, dtype=torch.long),
            "i_negs": torch.tensor(neg_items, dtype=torch.long),
            "text": f"Title: {title}",
        }
        return sample
