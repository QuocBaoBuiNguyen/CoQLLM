import logging
import sys
from datetime import datetime, timedelta, timezone


VN_TZ = timezone(timedelta(hours=7))


class VietnamTimeFormatter(logging.Formatter):
    """Formatter that renders timestamps in Vietnam time (UTC+7)."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=VN_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

class NotebookLogger:
    @staticmethod
    def rich_logger(name: str = "coqllm") -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            for h in logger.handlers:
                logger.removeHandler(h)

        handler: logging.Handler
        formatter: logging.Formatter
        
        try:
            from rich.logging import RichHandler
            from rich.console import Console
            
            handler = RichHandler(
                console=Console(file=sys.stdout, force_terminal=True),
                markup=True,
                show_time=False,
                show_path=True,
                enable_link_path=False
            )
            formatter = VietnamTimeFormatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        except ModuleNotFoundError:
            handler = logging.StreamHandler(sys.stdout)
            formatter = VietnamTimeFormatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
            
        return logger
