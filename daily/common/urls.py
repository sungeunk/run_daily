"""Shared public service URL defaults for daily artifact links."""

import re
from urllib.parse import urlparse

DEFAULT_BACKUP_HOST = "dg2fizz.ikor.intel.com"
DAILY_FILE_BROWSER_PORT = 8081
DEFAULT_DAILY_FILE_BROWSER_ORIGIN = f"http://{DEFAULT_BACKUP_HOST}:{DAILY_FILE_BROWSER_PORT}"
DEFAULT_DAILY_REPORT_BASE_URL = f"{DEFAULT_DAILY_FILE_BROWSER_ORIGIN}/daily"


def normalize_daily_report_url(url: str) -> str:
	"""Replace the retired daily2 route without changing other URL components."""
	parsed = urlparse(url)
	path = re.sub(r"(?<=/)daily2(?=/|$)", "daily", parsed.path)
	return parsed._replace(path=path).geturl()


def daily_artifact_base_url(url: str) -> str:
	"""Return a browser base ending in one daily route."""
	base = normalize_daily_report_url(url).rstrip("/")
	return base if urlparse(base).path.endswith("/daily") else f"{base}/daily"
