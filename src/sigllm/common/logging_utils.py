"""Shared logging helpers for notebooks and scripts."""

from __future__ import annotations

import logging


class NotebookLogger:
    """Factory for Colab-friendly loggers with Rich fallback."""

    @staticmethod
    def rich_logger(name: str = "sigllm") -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        handler: logging.Handler
        formatter: logging.Formatter
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(markup=True)
            formatter = logging.Formatter("%(message)s")
        except ModuleNotFoundError:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
