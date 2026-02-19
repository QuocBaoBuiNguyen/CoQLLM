import logging
import sys

class NotebookLogger:
    @staticmethod
    def rich_logger(name: str = "sigllm") -> logging.Logger:
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
                show_path=True,
                enable_link_path=False
            )
            formatter = logging.Formatter("%(message)s")
        except ModuleNotFoundError:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
            
        return logger