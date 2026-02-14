from datetime import datetime
import os
from urllib.parse import urlparse

from sigllm.common import registry

def now():
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)
