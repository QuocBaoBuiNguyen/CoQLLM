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


TEMPL_FIXED = [
    "Based on the interaction history, predict whether the user will like this movie. Answer Yes/No.",
    "Predict if the user likes or dislikes this movie based on past behavior. Yes/No.",
]

TEMPL_GENRE = [
    "Considering only the {g} genre, predict whether the user will like this movie. Answer Yes/No.",
    "In the context of {g}, does the user like this movie? Yes/No.",
]


class QFormerAlignmentDataset(RecBaseDataset):
    def __init__(self, config, filename: str, neg_k: int = 5, hard_k: int = 8, p_fixed: float = 0.8) -> None:
        self.neg_k = neg_k
        self.hard_k = hard_k
        self.p_fixed = p_fixed

        dataset_config = config
        ann_path = dataset_config.build_info.storage / filename
        if (ann_path is None) or (not ann_path.exists()):
            raise ValueError(f"Annotation path {ann_path} does not exist.")

        df = pd.read_pickle(ann_path.with_suffix(".pkl")).reset_index(drop=True)
        pos_df = df[df["label"] == 1]
        if pos_df.empty:
            raise ValueError("No positive samples found for alignment stage.")

        self.annotation = df
        self.user_pos_indices = pos_df.groupby("uid").apply(lambda x: x.index.tolist()).to_dict()
        self.users = list(self.user_pos_indices.keys())
        self.all_items = set(df["iid"].unique().tolist())
        self.user_pos_items = pos_df.groupby("uid")["iid"].apply(set).to_dict()

        # Genre maps (assumed present)
        self.item_genres = {}
        for iid, genres in df[["iid", "genres"]].drop_duplicates().itertuples(index=False):
            gset = {g for g in str(genres).split("|") if g}
            if gset:
                self.item_genres[int(iid)] = gset

        self.user_pos_indices_genre = {}
        for uid, indices in self.user_pos_indices.items():
            with_genre = [idx for idx in indices if int(self.annotation.iloc[idx]["iid"]) in self.item_genres]
            if with_genre:
                self.user_pos_indices_genre[uid] = with_genre

        log_step("data path", str(ann_path))
        log_step("users with positives", str(len(self.users)))
        log_step("item vocab", str(len(self.all_items)))

    def __len__(self) -> int:
        return len(self.users)

    def _sample_fixed(self, user: int, pos_indices):
        pos_row = self.annotation.iloc[random.choice(pos_indices)]
        pos_item = int(pos_row["iid"])
        history_set = set(pos_row["his"]) if "his" in pos_row else set()
        banned_items = history_set | {pos_item}

        candidate_pool = list(self.all_items - banned_items)
        if len(candidate_pool) == 0:
            raise ValueError(f"No negatives available for user {user}.")

        replace_flag = len(candidate_pool) < self.neg_k
        neg_items = np.random.choice(candidate_pool, size=self.neg_k, replace=replace_flag)
        instruction = random.choice(TEMPL_FIXED)
        item_text = f"Title: {pos_row['title']}"
        return pos_item, neg_items, instruction, item_text

    def _sample_genre(self, user: int, pos_indices):
        pos_idx = random.choice(self.user_pos_indices_genre[user])
        seed_row = self.annotation.iloc[pos_idx]
        seed_item = int(seed_row["iid"])
        genre = random.choice(list(self.item_genres[seed_item]))

        user_pos_in_genre = [idx for idx in self.user_pos_indices[user] if genre in self.item_genres.get(int(self.annotation.iloc[idx]["iid"]), set())]
        if not user_pos_in_genre:
            return None

        chosen_idx = random.choice(user_pos_in_genre)
        pos_row = self.annotation.iloc[chosen_idx]
        pos_item = int(pos_row["iid"])

        history_set = set(pos_row["his"]) if "his" in pos_row else set()
        banned_items = history_set | {pos_item}

        # hard negatives: liked items not in genre
        hard_pool = [iid for iid in self.user_pos_items.get(user, set()) if genre not in self.item_genres.get(int(iid), set()) and iid != pos_item]
        hard_sample = []
        if hard_pool:
            take = min(self.hard_k, len(hard_pool))
            hard_sample = random.sample(hard_pool, k=take)

        candidate_pool = list(self.all_items - banned_items)
        need_rand = max(self.neg_k - len(hard_sample), 0)
        replace_flag = len(candidate_pool) < need_rand or need_rand == 0
        rand_sample = []
        if need_rand > 0:
            rand_sample = list(np.random.choice(candidate_pool, size=need_rand, replace=replace_flag))

        neg_items = hard_sample + rand_sample
        if len(neg_items) < self.neg_k and candidate_pool:
            extra = list(np.random.choice(candidate_pool, size=self.neg_k - len(neg_items), replace=True))
            neg_items.extend(extra)

        instruction = random.choice(TEMPL_GENRE).format(g=genre)
        item_text = f"Title: {pos_row['title']}"
        return pos_item, np.array(neg_items), instruction, item_text

    def __getitem__(self, index: int):
        user = random.choice(self.users)
        pos_indices = self.user_pos_indices[user]

        use_fixed = (user not in self.user_pos_indices_genre) or (random.random() < self.p_fixed)

        if use_fixed:
            pos_item, neg_items, instruction, item_text = self._sample_fixed(user, pos_indices)
        else:
            genre_sample = self._sample_genre(user, pos_indices)
            if genre_sample is None:
                pos_item, neg_items, instruction, item_text = self._sample_fixed(user, pos_indices)
            else:
                pos_item, neg_items, instruction, item_text = genre_sample

        sample = {
            "u": torch.tensor(user, dtype=torch.long),
            "i_pos": torch.tensor(pos_item, dtype=torch.long),
            "i_negs": torch.tensor(neg_items, dtype=torch.long),
            "instruction": instruction,
            "item_text": item_text,
        }
        return sample
