"""Placeholder concrete dataset for MovieDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger
from sigllm.datasets.base.rec_base_dataset import RecBaseDataset

LOGGER = NotebookLogger.rich_logger("sigllm.movie_ood_dataset")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

class MovieOODDataset(RecBaseDataset):

	def __init__(
		self,
		dataset_config,
		filename: str = None,
		subset: Literal["all", "warm", "cold"] = "all"
	) -> None:
		ann_path = Path(dataset_config.build_info.storage) / filename
		
		if (ann_path is None) or (not ann_path.exists()):
			raise ValueError(f"Annotation path {ann_path} does not exist.")
		
		df = pd.read_pickle(ann_path.with_suffix(".pkl")).reset_index(drop=True)
		self.annotation = df.copy()

		# SeLLa-matched warm/cold partition (BUG 2, Mismatch 1). SeLLa
		# (prepare_finetune_data.py:142) partitions the WHOLE test set by a single
		# `not_cold` flag: warm = not_cold==1, cold = not_cold==0. The old code
		# used a separate, stricter `warm` column (user AND item each with >3 train
		# interactions), a smaller/different population that is NOT comparable to
		# SeLLa's Table 3 warm. Use not_cold for both so the split matches SeLLa.
		if subset == "warm":
			self.annotation = df[df['not_cold'].isin([1])].copy()

		if subset == "cold":
			self.annotation = df[df['not_cold'].isin([0])].copy()

		# no-history-drop policy — TOGGLEABLE to match either reference:
		#   * SeLLa (codes/step3_train_sella/prepare_finetune_data.py:47-51) DROPS
		#     rows whose his_title has <2 entries, on EVERY split. Easier population.
		#   * CoLLM (minigpt4/datasets/datasets/rec_datasets.py:49,82-83) does NOT
		#     drop anything — it KEEPS all rows and zero-pads short histories.
		# The two references therefore evaluate on DIFFERENT test populations, so
		# our numbers are only comparable to whichever we mirror. Gate on
		# build_info.match_sella_history_filter (default True = SeLLa; set False in
		# the config to reproduce CoLLM's Table numbers on the full population).
		match_sella_history_filter = bool(
			getattr(dataset_config.build_info, "get", lambda *a: True)(
				"match_sella_history_filter", True
			)
		)
		if match_sella_history_filter and 'his_title' in self.annotation.columns:
			_before = len(self.annotation)
			self.annotation = self.annotation[
				self.annotation['his_title'].map(lambda t: len(t) >= 2)
			].reset_index(drop=True)
			log_step(
				"SeLLa his_title>=2 filter",
				f"subset={subset}: {_before} -> {len(self.annotation)} rows",
			)
		elif 'his_title' in self.annotation.columns:
			log_step(
				"CoLLM-parity: his_title>=2 filter DISABLED",
				f"subset={subset}: keeping all {len(self.annotation)} rows (zero-pad short history)",
			)

		self.use_his = False
		self.prompt_flag = False

		if "sessionItems" in self.annotation.columns or "his" in self.annotation.columns:
			used_columns = ['uid','iid','title','his', 'his_title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']

			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.use_his = True
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
			
			self.annotation['InteractedItemIDs'] = self.annotation['InteractedItemIDs'].map(list)
			self.annotation['InteractedItemTitles'] = self.annotation['InteractedItemTitles'].map(list)
		else:
			used_columns = ['uid','iid','title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
		
		log_step("data path", f"{ann_path} | data size: {self.annotation.shape}")
		self.user_num = self.annotation['UserID'].max() + 1
		self.item_num = self.annotation['TargetItemID'].max() + 1

		if self.use_his:
			max_length = 0
			for his in self.annotation['InteractedItemIDs']:
				max_length = max(max_length, len(his))
			self.max_length = min(max_length, 10)
			log_step("Movie OOD datasets, max history length:", str(self.max_length))
	
	def __getitem__(self, index):
		
		row = self.annotation.iloc[index]

		def _add_prompt_flag(sample: dict) -> dict:
			if self.prompt_flag:
				sample["prompt_flag"] = row["prompt_flag"]
			return sample

		user_id = row["UserID"]
		target_item_id = row["TargetItemID"]
		target_title = row["TargetItemTitle"].strip(" ")
		label = row["label"]
		
		if self.use_his:
			history_item_ids = row["InteractedItemIDs"]
			history_titles = row["InteractedItemTitles"]
			history_len = len(history_item_ids)

			interacted_count = history_len - 1 if (history_len > 0 and history_item_ids[0] == 0) else history_len

			max_history_len = self.max_length  

			if history_len < max_history_len:
				pad_size = max_history_len - history_len
				padded_history_ids = ([0] * pad_size) + list(history_item_ids)
			elif history_len > max_history_len:
				padded_history_ids = list(history_item_ids[-max_history_len:])
				interacted_count = max_history_len
			else:
				padded_history_ids = list(history_item_ids)

			recent_titles = history_titles[-interacted_count:] if interacted_count > 0 else []
			processed_titles = self.convert_title_list(recent_titles)

			sample = {
				"UserID": user_id,
				"InteractedItemIDs_pad": np.array(padded_history_ids),
				"InteractedItemTitles": processed_titles,
				"TargetItemID": target_item_id,
				"TargetItemTitle": f"\"{target_title}\"",
				"InteractedNum": interacted_count,
				"label": row["label"],
			}
			return _add_prompt_flag(sample)
		else:
			sample = {
				"UserID": user_id,
				"TargetItemID": target_item_id,
				"TargetItemTitle": target_title,
				"label": label,
			}
			return _add_prompt_flag(sample)
		
