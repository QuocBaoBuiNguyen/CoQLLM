"""Builder implementation stub for the MovieOOD dataset."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Dict

from sigllm.common.registry import registry
from sigllm.datasets.movie.movie_ood_dataset import MovieOODDataset
from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder


@registry.register_builder("movie_ood")
class MovieOODBuilder(RecBaseDatasetBuilder):
    """Construct MovieOOD dataset splits, following the requested pattern."""
    train_dataset_cls = MovieOODDataset
