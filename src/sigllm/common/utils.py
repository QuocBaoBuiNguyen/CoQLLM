from datetime import datetime
from urllib.parse import urlparse

def now():
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")