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

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def build_datasets(self, evaluate_only: bool = False):
        """Construct dataset instances for training/validation/test."""

        dataset_cls = self.train_dataset_cls

        build_info = self.config.build_info
        storage_path = build_info.storage

        if storage_path is None or not storage_path.exists():
            log_step("Warning", f"storage path {storage_path} does not exist.") 

        datasets = dict()

        if evaluate_only:
            datasets["train"] = dataset_cls(
                config=Config(
                    ann_path=storage_path / "train",
                )
            )
            datasets["valid"] = dataset_cls(
                config=Config(
                    ann_path=storage_path / "valid_small",
                )
        )
            datasets["test"] = dataset_cls(
                config=Config(
                    ann_path=storage_path / "test",
                )
            )
        else:
            datasets['test_warm'] = dataset_cls(
                config=Config(
                    ann_path=storage_path / "test_warm_cold=warm",
                )
            )
            datasets['test_cold'] = dataset_cls(
                config=Config(
                    ann_path=storage_path / "test_warm_cold=cold",
                )
            )

        return datasets