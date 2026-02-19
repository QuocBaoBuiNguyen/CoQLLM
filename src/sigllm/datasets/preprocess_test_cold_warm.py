import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from sigllm.common import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.data_prep_warm_cold")


def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""
    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


def process_warm_cold(
    data_dir: str = "/content/SigLLM/datasets/ml-1m/",
    min_user_inter: int = 3,
    min_item_inter: int = 3
) -> pd.DataFrame:
    """
    Process train/test data to identify warm and cold interactions based on history.
    """
    
    # Paths
    train_path = os.path.join(data_dir, "train_ood2.pkl")
    test_path = os.path.join(data_dir, "test_ood2.pkl")
    out_path = os.path.join(data_dir, "test_warm_cold_ood2.pkl")

    log_step("[1/5] Load datasets", f"dir={data_dir}")
    train_ = pd.read_pickle(train_path)
    test_ = pd.read_pickle(test_path)
    log_step("Loaded shapes", f"train={train_.shape}, test={test_.shape}")

    # --- User Analysis ---
    log_step("[2/5] Analyze User Interactions", f"Threshold > {min_user_inter}")
    user_info = train_.groupby('uid').agg({"label":'count'})
    all_train_users = set(user_info.index)
    
    # Filter warm users
    warm_users_df = user_info[user_info['label'] > min_user_inter]
    warm_users_set = set(warm_users_df.index)
    log_step("User stats", f"Total={len(all_train_users)}, Warm={len(warm_users_set)}")

    # --- Item Analysis ---
    log_step("[3/5] Analyze Item Interactions", f"Threshold > {min_item_inter}")
    item_info = train_.groupby('iid').agg({"label":'count'})
    all_train_items = set(item_info.index)

    # Filter warm items
    warm_items_df = item_info[item_info['label'] > min_item_inter]
    warm_items_set = set(warm_items_df.index)
    log_step("Item stats", f"Total={len(all_train_items)}, Warm={len(warm_items_set)}")

    # --- Compute Flags ---
    log_step("[4/5] Compute Warm/Cold Flags for Test Set")
    
    # Warm: User is warm AND Item is warm
    test_['warm'] = test_.apply(
        lambda x: 1 if (x['uid'] in warm_users_set and x['iid'] in warm_items_set) else 0, 
        axis=1
    )

    # Cold: User NOT in train AND Item NOT in train (Strict Cold)
    test_['cold'] = test_.apply(
        lambda x: 1 if (x['uid'] not in all_train_users and x['iid'] not in all_train_items) else 0, 
        axis=1
    )
    
    stats_warm = test_['warm'].value_counts(normalize=True).get(1, 0)
    stats_cold = test_['cold'].value_counts(normalize=True).get(1, 0)
    
    log_step("Flag distribution", f"Warm={stats_warm:.1%}, Cold={stats_cold:.1%}")
    log_step("Test DataFrame Stats", "\n" + str(test_[['not_cold','warm','cold']].describe()))

    # --- Save ---
    log_step("[5/5] Save Result", f"path={out_path}")
    test_.to_pickle(out_path)
    
    return test_


if __name__ == "__main__":
    # Ensure src is importable when running from repo root
    if "src" not in sys.path:
        src_path = "src"
        if not sys.path[0].endswith("src"):
            sys.path.insert(0, src_path)

    process_warm_cold()
