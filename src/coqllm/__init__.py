from pathlib import Path

from coqllm.common import registry

__version__ = "0.0.1"

registry.register_path("library_root", str(Path(__file__).resolve().parents[2]))
