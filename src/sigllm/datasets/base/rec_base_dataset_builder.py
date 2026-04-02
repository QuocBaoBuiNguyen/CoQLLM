"""Base builder for recommendation datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from abc import abstractmethod
from typing import Any, Dict, Optional

from torch import dist

from sigllm.common.config import Config
from sigllm.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from sigllm.common.logging_utils import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_dataset_builder")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

class RecBaseDatasetBuilder(ABC):
    """Abstract base for dataset builders."""
    train_dataset_cls = None

    def __init__(self, dataset_config) -> None:
        self.dataset_config = dataset_config
    
    def build_datasets(self, evaluate_only=False):
        """Construct dataset instances for training/validation/test."""

        if is_dist_avail_and_initialized():
            dist.barrier()
            
        dataset_cls = self.train_dataset_cls

        build_info = self.dataset_config.build_info
        storage_path = build_info.storage

        if storage_path is None:
            log_step("Warning", f"storage path {storage_path} does not exist.") 

        datasets = dict()

        if not evaluate_only:            
            datasets["train"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="train",
            )

            datasets["valid"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="valid_small",
            )
            datasets["test"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="test",
            )
        else:
            datasets['test_warm'] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="test_warm_cold=warm",
            )
            datasets['test_cold'] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="test_warm_cold=cold",
            )

        return datasets
