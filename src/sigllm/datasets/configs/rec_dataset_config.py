"""Placeholder dataclasses for dataset configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RecDatasetConfig:
    """Simple placeholder for future dataset configuration values."""

    ann_path: Path
