"""Builder for Q-Former alignment dataset."""

from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import torch

from sigllm.common.registry import registry
from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


TEMPL_FIXED = [
    "Based on the interaction history, predict whether the user will like this movie. Answer Yes/No.",
    "Predict if the user likes or dislikes this movie based on past behavior. Yes/No.",
]

TEMPL_GENRE = [
    "Considering only the {g} genre, predict whether the user will like this movie. Answer Yes/No.",
    "In the context of {g}, does the user like this movie? Yes/No.",
]

# @registry.register_builder("qformer_alignment")
class QFormerAlignmentBuilder(RecBaseDatasetBuilder):
    """Construct Q-Former alignment splits."""

    train_dataset_cls = QFormerAlignmentDataset

    @staticmethod        
    def build_qformer_alignment_samples(
        input_pkl_path: str,
        output_path: str,
        neg_k: int = 32,
        hard_k: int = 8,
        p_fixed: float = 0.8,
        seed: int = 42,
        samples_per_user: int = 1,
    ):
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        df = pd.read_pickle(input_pkl_path).reset_index(drop=True)
        pos_df = df[df["label"] == 1]
        if pos_df.empty:
            raise ValueError("No positive samples found.")

        user_pos_indices = pos_df.groupby("uid").apply(lambda x: x.index.tolist()).to_dict()
        users = list(user_pos_indices.keys())
        all_items = set(df["iid"].unique().tolist())
        user_pos_items = pos_df.groupby("uid")["iid"].apply(set).to_dict()

        item_genres = {}
        for iid, genres in df[["iid", "genres"]].drop_duplicates().itertuples(index=False):
            gset = {g for g in str(genres).split("|") if g}
            if gset:
                item_genres[int(iid)] = gset

        user_pos_indices_genre = {}
        for uid, indices in user_pos_indices.items():
            with_genre = [idx for idx in indices if int(df.iloc[idx]["iid"]) in item_genres]
            if with_genre:
                user_pos_indices_genre[uid] = with_genre

        def np_choice(pool, size: int, replace: bool):
            if size <= 0:
                return np.array([], dtype=np.int64)
            return np_rng.choice(pool, size=size, replace=replace)

        def sample_fixed(user: int):
            pos_indices = user_pos_indices[user]
            pos_row = df.iloc[py_rng.choice(pos_indices)]
            pos_item = int(pos_row["iid"])
            history_set = set(pos_row["his"]) if "his" in pos_row else set()
            banned_items = history_set | {pos_item}
            candidate_pool = list(all_items - banned_items)
            if len(candidate_pool) == 0:
                return None
            neg_items = np_choice(candidate_pool, size=neg_k, replace=(len(candidate_pool) < neg_k))
            instruction = py_rng.choice(TEMPL_FIXED)
            item_text = f"Title: {pos_row['title']}"
            return pos_item, neg_items, instruction, item_text

        def sample_genre(user: int):
            if user not in user_pos_indices_genre:
                return None
            pos_idx = py_rng.choice(user_pos_indices_genre[user])
            seed_row = df.iloc[pos_idx]
            seed_item = int(seed_row["iid"])
            if seed_item not in item_genres:
                return None
            genre = py_rng.choice(list(item_genres[seed_item]))

            user_pos_in_genre = [
                idx
                for idx in user_pos_indices[user]
                if genre in item_genres.get(int(df.iloc[idx]["iid"]), set())
            ]
            if not user_pos_in_genre:
                return None

            chosen_idx = py_rng.choice(user_pos_in_genre)
            pos_row = df.iloc[chosen_idx]
            pos_item = int(pos_row["iid"])

            history_set = set(pos_row["his"]) if "his" in pos_row else set()
            banned_items = history_set | {pos_item}

            hard_pool = [
                iid
                for iid in user_pos_items.get(user, set())
                if genre not in item_genres.get(int(iid), set()) and iid != pos_item
            ]
            hard_sample = []
            if hard_pool:
                take = min(hard_k, len(hard_pool))
                hard_sample = py_rng.sample(hard_pool, k=take)

            candidate_pool = list(all_items - banned_items)
            need_rand = max(neg_k - len(hard_sample), 0)

            rand_sample = []
            if need_rand > 0 and candidate_pool:
                rand_sample = list(np_choice(candidate_pool, size=need_rand, replace=(len(candidate_pool) < need_rand)))

            neg_items = hard_sample + rand_sample
            if len(neg_items) < neg_k and candidate_pool:
                extra = list(np_choice(candidate_pool, size=neg_k - len(neg_items), replace=True))
                neg_items.extend(extra)

            instruction = py_rng.choice(TEMPL_GENRE).format(g=genre)
            item_text = f"Title: {pos_row['title']}"
            return pos_item, np.array(neg_items, dtype=np.int64), instruction, item_text

        samples = []
        for u in users:
            for _ in range(samples_per_user):
                use_fixed = (u not in user_pos_indices_genre) or (py_rng.random() < p_fixed)
                pack = sample_fixed(u) if use_fixed else sample_genre(u)
                if pack is None:
                    pack = sample_fixed(u)
                if pack is None:
                    continue
                pos_item, neg_items, instruction, item_text = pack
                samples.append(
                    {
                        "u": int(u),
                        "i_pos": int(pos_item),
                        "i_negs": [int(x) for x in neg_items],
                        "instruction": instruction,
                        "item_text": item_text,
                    }
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(
            {
                "seed": seed,
                "neg_k": neg_k,
                "hard_k": hard_k,
                "p_fixed": p_fixed,
                "samples_per_user": samples_per_user,
                "samples": samples,
            },
            output_path,
        )
        return output_path, len(samples)
