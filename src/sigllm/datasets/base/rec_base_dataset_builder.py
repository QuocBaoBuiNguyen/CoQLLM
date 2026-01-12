"""Base builder for recommendation datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from abc import abstractmethod
from typing import Any, Dict, Optional

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_dataset_builder")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

class RecBaseDatasetBuilder(ABC):
    """Abstract base for dataset builders."""
    train_dataset_cls = None

    def __init__(self, config: Config) -> None:
        self.config = config
    
    def _attach_train_meta(self, train_ds, name: str, dataset_config):
        train_ds.name = name
        if 'sample_ratio' in dataset_config:
            train_ds.sample_ratio = dataset_config.sample_ratio

    @abstractmethod
    def build_datasets(self, name: str):
        """Construct dataset instances for training/validation/test."""

        dataset_cls = self.train_dataset_cls

        datasets_config = self.config.datasets_cfg
        dataset_config = datasets_config[name]
        
        build_info = self.config.build_info
        evaluate_only = self.config.run_cfg.evaluate
        storage_path = build_info.storage

        if storage_path is None or not storage_path.exists():
            log_step("Warning", f"storage path {storage_path} does not exist.") 

        datasets = dict()

        if not evaluate_only:            
            datasets["train"] = dataset_cls(
                config=self.config,
                filename="train",
            )
            self._attach_train_meta(datasets["train"], name, dataset_config)

            datasets["valid"] = dataset_cls(
                config=self.config,
                filename="valid_small",
            )
            datasets["test"] = dataset_cls(
                config=self.config,
                filename="test",
            )
        else:
            datasets['test_warm'] = dataset_cls(
                config=self.config,
                filename="test_warm_cold=warm",
            )
            datasets['test_cold'] = dataset_cls(
                config=self.config,
                filename="test_warm_cold=cold",
            )

        return datasets