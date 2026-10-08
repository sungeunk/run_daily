#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "aiohttp",
#     "beautifulsoup4",
#     "pyyaml",
#     "requests",
#     "tabulate",
#     "tqdm",
# ]
# ///
"""
download-openvino.py — Download OpenVINO prebuilt packages from the Intel storage server.

Usage examples:
    # Download latest nightly for current platform
    python download-openvino.py

    # Download from a direct archive URL
    python download-openvino.py --download-url http://<server>/path/to/archive.zip

    # Download by commit ID
    python download-openvino.py --commit-id fa6b0ec1

    # Show manifest for a local manifest.yml
    python download-openvino.py --manifest ./openvino_nightly/2025.1.0_abcd1234/manifest.yml

    # Install from local directory or archive
    python download-openvino.py --install ./openvino_nightly/2025.1.0_abcd1234/

    # Force re-download even if same version exists
    python download-openvino.py --force

    # Keep old packages after downloading new ones
    python download-openvino.py --keep-old

    # Download the build corresponding to GitHub PR #12345
    python download-openvino.py --pr-number 12345

    # Clean up expired cache entries (persistent cache maintenance)
    python download-openvino.py --cleanup-cache
"""

import sys

try:
    import aiohttp
    import argparse
    import asyncio
    import base64
    import datetime
    from dataclasses import dataclass
    import hashlib
    import json
    import logging as log
    import os
    import platform
    import re
    import requests
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import zipfile

    import tqdm.asyncio as tqdm_asyncio
    import yaml

    from bs4 import BeautifulSoup
    from functools import total_ordering
    from pathlib import Path
    from tabulate import tabulate
    from tqdm import tqdm

except ImportError:
    print(
        "Please install required modules:\n"
        "  pip install aiohttp requests tqdm pyyaml beautifulsoup4 tabulate"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

IS_WINDOWS: bool = platform.system() == "Windows"
PWD: Path = Path.cwd()

# Build server roots
_MASTER_COMMIT_ROOT = (
    "http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci"
    "/private_builds/dldt/master/commit"
)
_MASTER_NIGHTLY_ROOT = (
    "http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci"
    "/private_builds/dldt/master/nightly"
)
_PRE_COMMIT_ROOT = (
    "http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci"
    "/private_builds/dldt/master/pre_commit"
)
_RELEASES_ROOT = (
    "http://ov-share-03.iotg.sclab.intel.com/volatile/openvino_ci"
    "/private_builds/dldt/releases"
)
_IOTG_PACKAGES_ROOT = "https://ov-share-01.iotg.sclab.intel.com/iotg_ovino/packages"
_IOTG_NIGHTLY_ROOT = f"{_IOTG_PACKAGES_ROOT}/nightly"
_IOTG_RELEASES_ROOT = f"{_IOTG_PACKAGES_ROOT}/releases"
_IOTG_RC_ROOT = f"{_IOTG_PACKAGES_ROOT}/release_candidates"

# Builds older than OpenVINO 2025.4 are out of support and never searched.
_MIN_SUPPORTED_VERSION = (2025, 4)

_IOTG_COMPONENTS: tuple[str, ...] = ("openvino", "genai", "tokenizers")
_IOTG_ARCHIVE_PREFIX: dict[str, str] = {
    "openvino": "openvino_toolkit_",
    "genai": "openvino_genai_",
    "tokenizers": "openvino_tokenizers_",
}
_RELEASE_ROOTS_CACHE_FILENAME = "RELEASE_SEARCH_ROOTS_CACHE.json"
_COMMIT_URL_CACHE_FILENAME = "COMMIT_URL_CACHE.json"
_PACKAGE_CHECK_CACHE_FILENAME = "PACKAGE_CHECK_CACHE.json"
_ROOT_HTML_CACHE_FILENAME = "ROOT_HTML_CACHE.json"
_CACHE_VERSION = "2"
_DEFAULT_NO_PROXY_LIST = (
    "localhost,intel.com,192.168.0.0/16,"
    "172.16.0.0/12,127.0.0.0/8,10.0.0.0/8"
)
_COMMIT_URL_POSITIVE_TTL = datetime.timedelta(days=30)
_COMMIT_URL_NEGATIVE_TTL = datetime.timedelta(hours=6)
_PACKAGE_CHECK_POSITIVE_TTL = datetime.timedelta(days=7)
_PACKAGE_CHECK_NEGATIVE_TTL = datetime.timedelta(hours=6)
# Directory listings gain new commits continuously, so keep this TTL short;
# a long TTL hides builds published after the listing was cached.
_ROOT_HTML_POSITIVE_TTL = datetime.timedelta(hours=1)
_ROOT_HTML_NEGATIVE_TTL = datetime.timedelta(hours=1)
_RELEASE_ROOTS_POSITIVE_TTL = datetime.timedelta(days=1)
_CACHE_DIR: Path | None = None
_IGNORE_HTML_CACHE: bool = False
_TEXT_CACHE: dict[str, str] = {}

# Detect Ubuntu version (Linux only)
UBUNTU_VER: str = ""
if platform.system() == "Linux":
    try:
        output = subprocess.check_output(["lsb_release", "-r"], text=True)
        m = re.search(r"Release:[ \t]+(\d+)\.(\d+)", output)
        if m:
            UBUNTU_VER = f"{m.group(1)}_{m.group(2)}"
    except FileNotFoundError:
        log.warning("lsb_release not found; assuming non-Ubuntu Linux.")
    except Exception as exc:
        log.warning("Could not determine Ubuntu version: %s", exc)


# ---------------------------------------------------------------------------
# Required package lists
# ---------------------------------------------------------------------------

def required_openvino_packages_list() -> list[str]:
    """Return the static list of required OpenVINO cpack filenames."""
    return [
        "benchmark_app.zip",
        "core.zip",
        "core_c.zip",
        "core_c_dev.zip",
        "core_dev.zip",
        "cpp_samples.zip",
        "cpu.zip",
        "gpu.zip",
        "ir.zip",
        "onnx.zip",
        "openvino_req_files.zip",
        "ovc.zip",
        "paddle.zip",
        "pyopenvino_python3.11.zip",
        "pyopenvino_python3.12.zip",
        "pytorch.zip",
        "setupvars.zip",
        "tbb.zip",
        "tbb_dev.zip",
        "tensorflow.zip",
        "tensorflow_lite.zip",
    ]


def required_genai_packages_list() -> list[str]:
    """Return the static list of required GenAI cpack filenames."""
    return [
        "openvino_tokenizers.zip",
        "core_c_genai.zip",
        "core_c_genai_dev.zip",
        "core_genai.zip",
        "core_genai_dev.zip",
        "pygenai_3_11.zip",
        "pygenai_3_12.zip",
    ]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

# Module-level flag controlled by the --verify-ssl CLI argument.
# Defaults to False so that Intel internal servers with self-signed
# certificates work out-of-the-box.  Pass --verify-ssl on the CLI to
# enable certificate validation for environments with a trusted CA.
_VERIFY_SSL: bool = False

# Each file is tried once plus two retries on size mismatch or transfer error.
_DOWNLOAD_ATTEMPTS = 3
# A transfer is aborted only when no data arrives for this long (no total limit).
_DOWNLOAD_STALL_TIMEOUT_SEC = 30


def get_text_from_web(url: str) -> str:
    """
    Fetch plain text from *url*; return empty string on failure.

    Caching strategy (hybrid):
    1. In-memory cache (_TEXT_CACHE) for same-session reuse
    2. Persistent cache (ROOT_HTML_CACHE.json) for session-to-session reuse
    """
    # Check in-memory cache first
    cached_text = _TEXT_CACHE.get(url)
    if cached_text is not None:
        return cached_text

    # Check persistent cache (if _CACHE_DIR is set)
    if _CACHE_DIR and not _IGNORE_HTML_CACHE:
        cached_entry = _load_cache_entry(
            _ROOT_HTML_CACHE_FILENAME,
            url,
            _ROOT_HTML_POSITIVE_TTL,
            _ROOT_HTML_NEGATIVE_TTL,
        )
        if cached_entry is not None and bool(cached_entry.get("ok")):
            cached_text = cached_entry.get("value")
            if isinstance(cached_text, str) and cached_text:
                log.debug("Using persistent cached HTML for %s", url)
                _TEXT_CACHE[url] = cached_text
                return cached_text

    # Fetch from network
    try:
        res = requests.get(url, verify=_VERIFY_SSL, timeout=30)
        if res.ok:
            text = res.text
            _TEXT_CACHE[url] = text
            # Store in persistent cache (success)
            if _CACHE_DIR:
                _store_cache_entry(_ROOT_HTML_CACHE_FILENAME, url, True, text)
            return text
        log.error("HTTP %s %s — %s", res.status_code, res.reason, url)
    except requests.exceptions.RequestException as exc:
        log.error("get_text_from_web: %s", exc)

    # Do not cache failures: transient network issues should be retried.
    return ""


def download_file(url: str, out_path: Path) -> Path | None:
    """
    Download a single file from *url* into *out_path* with a tqdm progress bar.

    A transfer whose size does not match ``Content-Length`` (or that stalls for
    ``_DOWNLOAD_STALL_TIMEOUT_SEC``) is discarded and retried.

    Returns the resolved Path to the saved file, or None on failure.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / Path(url).name
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            # (connect, read) timeout: read applies to the gap between received chunks.
            res = requests.get(
                url, stream=True, verify=_VERIFY_SSL,
                timeout=(30, _DOWNLOAD_STALL_TIMEOUT_SEC),
            )
            if not res.ok:
                log.error("HTTP %s %s — %s", res.status_code, res.reason, url)
                return None

            expected = int(res.headers.get("content-length", 0)) or None
            written = 0
            with open(filepath, "wb") as fh, tqdm(
                desc=filepath.name,
                total=expected or 0,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in res.iter_content(chunk_size=1024 * 64):
                    written += fh.write(chunk)
                    bar.update(len(chunk))
            if expected is None or written == expected:
                return filepath.resolve()
            reason = f"size mismatch {written}/{expected} bytes"
        except requests.exceptions.RequestException as exc:
            reason = f"{type(exc).__name__}: {exc}"
        filepath.unlink(missing_ok=True)
        log.warning("Download attempt %d/%d failed (%s): %s", attempt, _DOWNLOAD_ATTEMPTS, reason, url)
    log.error("Failed to download after %d attempts: %s", _DOWNLOAD_ATTEMPTS, url)
    return None


async def _async_download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    out_path: Path,
) -> Path | None:
    """
    Download a single file asynchronously; return saved Path or None.

    A transfer whose size does not match ``Content-Length`` (or that stalls) is
    discarded and retried up to ``_DOWNLOAD_ATTEMPTS`` times in total.
    """
    filepath = out_path / Path(url).name
    # ssl=False when _VERIFY_SSL is False; ssl=None lets aiohttp use
    # the default CA bundle when verification is requested.
    ssl_param = None if _VERIFY_SSL else False
    async with semaphore:
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                async with session.get(url, ssl=ssl_param) as res:
                    if not res.ok:
                        log.error("HTTP %s %s — %s", res.status, res.reason, url)
                        return None
                    expected = res.content_length
                    written = 0
                    with open(filepath, "wb") as fh:
                        async for chunk in res.content.iter_chunked(1024 * 64):
                            written += fh.write(chunk)
                if expected is None or written == expected:
                    return filepath.resolve()
                reason = f"size mismatch {written}/{expected} bytes"
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            filepath.unlink(missing_ok=True)
            log.warning("Download attempt %d/%d failed (%s): %s", attempt, _DOWNLOAD_ATTEMPTS, reason, url)
        log.error("Failed to download after %d attempts: %s", _DOWNLOAD_ATTEMPTS, url)
        return None


async def async_download_files(url_list: list[str], out_path: Path) -> list[Path]:
    """
    Download *url_list* concurrently (max 5 at once) into *out_path*.

    Returns the downloaded file Paths. Raises RuntimeError when any file fails,
    so a partially downloaded package is never treated as complete.
    """
    semaphore = asyncio.Semaphore(5)
    # No total limit (large archives on slow links); abort only on a stalled transfer.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_DOWNLOAD_STALL_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _async_download_one(session, semaphore, url, out_path)
            for url in url_list
        ]
        results = await tqdm_asyncio.tqdm.gather(*tasks, desc="Downloading")

    failed = [url for url, path in zip(url_list, results) if path is None]
    if failed:
        raise RuntimeError(f"{len(failed)}/{len(url_list)} download(s) failed: {failed}")
    return [path for path in results if path is not None]


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

@total_ordering
class OVVersion:
    """
    Parse and compare OpenVINO version strings.

    Format: ``{year}.{major}.{minor}-{commit_number}-{commit_id}``
    Example: ``2025.3.0-19807-44526285f24``
    """

    _REGEX = re.compile(r"^(\d+)\.(\d+)\.(\d+)-(\d+)-([a-zA-Z0-9]+)")

    def __init__(self, version_string: str) -> None:
        self.raw = version_string.strip()
        m = self._REGEX.match(self.raw)
        if not m:
            raise ValueError(f"Invalid OV version string: '{self.raw}'")
        self.year = int(m.group(1))
        self.major = int(m.group(2))
        self.minor = int(m.group(3))
        self.commit_number = int(m.group(4))
        self.commit_id = m.group(5)
        self._tuple = (self.year, self.major, self.minor, self.commit_number)

    def __repr__(self) -> str:
        return f"OVVersion('{self.raw}')"

    def __str__(self) -> str:
        return (
            f"{self.year}.{self.major}.{self.minor}"
            f"-{self.commit_number}-{self.commit_id}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OVVersion):
            return NotImplemented
        return self._tuple == other._tuple and self.commit_id == other.commit_id

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, OVVersion):
            return NotImplemented
        return self._tuple < other._tuple


@dataclass(frozen=True, slots=True)
class LocalPrebuiltVersions:
    """Versions reported by a locally extracted OpenVINO prebuilt."""

    openvino: str
    genai: str
    openvino_module_path: Path
    genai_module_path: Path


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_url_from_cpack(cpack_url: str) -> str:
    idx = cpack_url.rfind("cpack")
    if idx > 0:
        return cpack_url[:idx] + "manifest.yml"
    return ""


def save_manifest(cpack_url: str, dest: Path) -> Path | None:
    """Download the manifest.yml associated with *cpack_url* into *dest*."""
    url = _manifest_url_from_cpack(cpack_url)
    if not url:
        log.warning("Could not derive manifest URL from: %s", cpack_url)
        return None
    return download_file(url, dest)


def generate_manifest_table(manifest_path: Path) -> str:
    """Parse *manifest_path* and return a GitHub-flavoured markdown table."""
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return ""
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh.read())
        rows = []
        for repo in data["components"]["dldt"]["repository"]:
            if repo["name"] in {"openvino", "openvino_tokenizers", "openvino.genai"}:
                rows.append(
                    [repo["name"], repo["url"], repo["branch"], repo["revision"]]
                )
        return tabulate(
            rows,
            headers=["name", "url", "branch", "revision"],
            tablefmt="github",
            stralign="left",
        )
    except (OSError, yaml.YAMLError, KeyError) as exc:
        log.error("Failed to parse manifest %s: %s", manifest_path, exc)
    return ""


# ---------------------------------------------------------------------------
# Build server / version helpers
# ---------------------------------------------------------------------------

def _private_cpack_subpath() -> str:
    """Return the platform-specific cpack sub-directory."""
    if IS_WINDOWS:
        return "private_windows_vs2022_release/cpack"
    return f"private_linux_ubuntu_{UBUNTU_VER}_release/cpack"


def _private_cpack_subpath_candidates() -> list[str]:
    """Return candidate cpack sub-directories in preferred order."""
    if IS_WINDOWS:
        return [
            "private_windows_vs2022_release/cpack",
            "private_windows_vs2019_release/cpack",
        ]
    return [
        f"private_linux_ubuntu_{UBUNTU_VER}_release/cpack",
        "private_linux_ubuntu_22_04_release/cpack",
    ]


def _host_platform_name() -> str:
    """Return a human-readable host platform name used in log messages."""
    if IS_WINDOWS:
        return "Windows"
    return f"Linux (Ubuntu {UBUNTU_VER.replace('_', '.')})" if UBUNTU_VER else "Linux"


def _validate_cpack_url_for_host(cpack_url: str) -> tuple[bool, str]:
    """
    Validate whether *cpack_url* matches the current host platform path.

    Returns (True, "") when compatible, otherwise (False, <reason>).
    """
    expected_subpaths = _private_cpack_subpath_candidates()
    if any(subpath in cpack_url for subpath in expected_subpaths):
        return True, ""

    expected_subpath = expected_subpaths[0]

    if IS_WINDOWS and "private_linux_ubuntu_" in cpack_url:
        return False, (
            "Host is Windows, but URL targets Linux Ubuntu artifacts. "
            f"Expected path containing '{expected_subpath}'."
        )

    if (not IS_WINDOWS) and "private_windows_vs2022_release" in cpack_url:
        return False, (
            "Host is Linux, but URL targets Windows artifacts. "
            f"Expected path containing '{expected_subpath}'."
        )

    if not IS_WINDOWS and "private_linux_ubuntu_" in cpack_url:
        return False, (
            "Host Ubuntu version does not match URL path. "
            f"Expected path containing '{expected_subpath}'."
        )

    return False, (
        "Unsupported or unknown cpack platform path for this host. "
        f"Expected path containing '{expected_subpath}'."
    )


def _get_ov_version_from_url(cpack_url: str) -> OVVersion | None:
    """Fetch the manifest and return the parsed OVVersion, or None."""
    manifest_url = _manifest_url_from_cpack(cpack_url)
    text = get_text_from_web(manifest_url)
    if not text:
        return None
    try:
        data = yaml.safe_load(text)
        return OVVersion(data["components"]["dldt"]["version"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse OV version from manifest: %s", exc)
    return None


def _load_cached_version(output_dir: Path) -> OVVersion | None:
    """Load the last-downloaded version from the cache file, or None."""
    cache = output_dir / "OV_NIGHTLY_VERSION"
    try:
        return OVVersion(cache.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        return None


def _save_cached_version(output_dir: Path, version: OVVersion) -> None:
    """Write *version* to the version cache file."""
    cache = output_dir / "OV_NIGHTLY_VERSION"
    try:
        cache.write_text(str(version), encoding="utf-8")
    except OSError as exc:
        log.error("Failed to write version cache: %s", exc)


def _is_newer(existing: OVVersion | None, cpack_url: str) -> bool:
    """Return True if the build at *cpack_url* is newer than *existing*."""
    if existing is None:
        return True
    remote = _get_ov_version_from_url(cpack_url)
    if remote is None:
        return False
    return existing < remote


def check_required_packages(url: str, strict: bool = False) -> bool:
    """
    Verify that OV and GenAI packages are present at *url*.

    Args:
        url: The cpack directory URL to check
        strict: If True, all packages must be present. If False, warn on missing but continue.

    Returns True when core packages are present (or if strict=False and at least some packages found).
    """
    cached_entry = _load_cache_entry(
        _PACKAGE_CHECK_CACHE_FILENAME,
        url,
        _PACKAGE_CHECK_POSITIVE_TTL,
        _PACKAGE_CHECK_NEGATIVE_TTL,
    )
    if cached_entry is not None and bool(cached_entry.get("ok")):
        log.debug("Using cached package-check result for %s: True", url)
        return True

    text = get_text_from_web(url)
    if not text:
        return False

    if _is_iotg_url(url):
        if _resolve_iotg_component_archive(url, "openvino"):
            _store_cache_entry(_PACKAGE_CHECK_CACHE_FILENAME, url, True)
            return True
        log.warning("Missing host archive package at: %s", url)
        return False

    def package_present(package_name: str) -> bool:
        package_base = package_name.removesuffix(".zip")
        patterns = [
            re.escape(package_name),
            rf"-{re.escape(package_base)}\.zip",
            rf"-{re.escape(package_base)}-",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    required_ov_packages = [
        package_name
        for package_name in required_openvino_packages_list()
        if not package_name.startswith("pyopenvino_python3.")
    ]
    missing_ov_packages = [
        package_name
        for package_name in required_ov_packages
        if not package_present(package_name)
    ]
    if missing_ov_packages:
        log.warning("Missing %d required OpenVINO package(s) at: %s", len(missing_ov_packages), url)
        for package_name in missing_ov_packages[:3]:
            log.warning("  - %s", package_name)
        if len(missing_ov_packages) > 3:
            log.warning("  ... and %d more", len(missing_ov_packages) - 3)
        return False

    if not re.search(r"pyopenvino_python3\.\d+\.zip", text):
        log.warning("Missing any pyopenvino package at: %s", url)
        return False

    genai_packages = required_genai_packages_list()
    present_genai_packages = [
        package_name for package_name in genai_packages if package_present(package_name)
    ]
    if present_genai_packages:
        required_genai_packages = [
            package_name
            for package_name in genai_packages
            if not package_name.startswith("pygenai_")
        ]
        missing_genai_packages = [
            package_name
            for package_name in required_genai_packages
            if not package_present(package_name)
        ]
        has_pygenai_package = re.search(r"pygenai_3_\d+\.zip", text) is not None
        if not has_pygenai_package:
            missing_genai_packages.append("pygenai_3_<python>.zip")
        if missing_genai_packages:
            log.warning("Missing %d required GenAI package(s) at: %s", len(missing_genai_packages), url)
            for package_name in missing_genai_packages[:3]:
                log.warning("  - %s", package_name)
            if len(missing_genai_packages) > 3:
                log.warning("  ... and %d more", len(missing_genai_packages) - 3)
            return False
    elif strict:
        log.warning("Missing GenAI packages at: %s", url)
        return False

    _store_cache_entry(_PACKAGE_CHECK_CACHE_FILENAME, url, True)
    return True


def _is_iotg_url(url: str) -> bool:
    """Return True when *url* points at the iotg package share."""
    return url.startswith(_IOTG_PACKAGES_ROOT)


def _iotg_version_prefix(name: str) -> tuple[int, int] | None:
    """Return the ``(year, major)`` pair of an iotg directory name, or None."""
    m = re.match(r"^(\d{4})\.(\d+)\.\d+", name.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _is_supported_iotg_version(name: str) -> bool:
    """Return True when *name* is 2025.4 or newer."""
    prefix = _iotg_version_prefix(name)
    return prefix is not None and prefix >= _MIN_SUPPORTED_VERSION


def _iotg_build_sort_key(label: str) -> tuple[int, ...]:
    """Return a numeric sort key so 2026.10.0 ranks above 2026.9.0 (and RC10 above RC9)."""
    return tuple(int(part) for part in re.findall(r"\d+", label))


def _parse_components(spec: str | list[str] | None) -> list[str]:
    """Validate and normalize an iotg component selection."""
    if spec is None:
        return list(_IOTG_COMPONENTS)

    raw = spec.split(",") if isinstance(spec, str) else spec
    selected: list[str] = []
    for item in raw:
        name = item.strip()
        if not name or name in selected:
            continue
        if name not in _IOTG_COMPONENTS:
            raise ValueError(f"Unknown iotg component: {name}")
        selected.append(name)

    if "openvino" not in selected:
        raise ValueError("The 'openvino' component is required as the merge base.")
    return selected


def _host_iotg_targets() -> list[str]:
    """Return preferred iotg OS name tokens for the current host, best first."""
    if IS_WINDOWS:
        return ["windows"]
    if platform.system() == "Darwin":
        return ["macos"]

    targets: list[str] = []
    for prefix, name in (("26_", "ubuntu26"), ("24_", "ubuntu24"),
                         ("22_", "ubuntu22"), ("20_", "ubuntu20")):
        if UBUNTU_VER.startswith(prefix):
            targets.append(name)
            break

    # Fallback order for Linux x86_64 hosts.
    for name in ("ubuntu24", "ubuntu22", "ubuntu26", "ubuntu20", "rhel9", "rhel8", "centos8"):
        if name not in targets:
            targets.append(name)
    return targets


def _host_iotg_arch_token() -> str:
    """Return the architecture token used in iotg archive filenames."""
    machine = platform.machine().lower()
    return "arm64" if machine in {"arm64", "aarch64"} else "x86_64"


def _find_iotg_host_archive_url(
    listing_url: str,
    html_text: str,
    prefix: str = "openvino_toolkit_",
) -> str | None:
    """
    Pick the best host-matching iotg archive URL from a directory listing.

    *prefix* selects the component (``openvino_toolkit_``, ``openvino_genai_``
    or ``openvino_tokenizers_``).  Debug-symbol packages are named
    ``pdb_openvino_*`` and are therefore excluded by the anchored pattern.
    """
    candidates = re.findall(rf"href=\"({re.escape(prefix)}[^\"]+)\"", html_text)
    if not candidates:
        return None

    # Keep only package archives and skip checksum sidecars.
    archives = [c for c in candidates if c.endswith((".tgz", ".tar.gz", ".zip"))]
    if not archives:
        return None

    os_tokens = _host_iotg_targets()
    arch_token = _host_iotg_arch_token()
    archive_exts = (".zip",) if IS_WINDOWS else (".tgz", ".tar.gz")

    def rank(name: str) -> tuple[int, int]:
        lname = name.lower()
        if arch_token not in lname:
            return (0, 0)
        if not lname.endswith(archive_exts):
            return (0, 0)
        for idx, token in enumerate(os_tokens):
            if token in lname:
                return (len(os_tokens) - idx, 1)
        return (0, 0)

    best = max(archives, key=rank)
    if rank(best) == (0, 0):
        return None

    return f"{listing_url.rstrip('/')}/{best}"


def _iotg_component_listing_dirs(base_url: str, component: str) -> list[str]:
    """
    Return candidate listing directories for *component* under an iotg build.

    Covers both layouts:
    * nightly  -> ``archives/``, ``genai_archives/``, ``tokenizers_archives/``
    * releases -> ``archives/<os>/``, ``genai/archives/<os>/``, ``tokenizers/archives/``
    """
    base = base_url.rstrip("/")
    if IS_WINDOWS:
        os_dir = "windows"
    elif platform.system() == "Darwin":
        os_dir = "macos"
    else:
        os_dir = "linux"

    if component == "openvino":
        subpaths = ["archives", f"archives/{os_dir}"]
    elif component == "genai":
        subpaths = ["genai_archives", f"genai/archives/{os_dir}", "genai/archives"]
    elif component == "tokenizers":
        subpaths = [
            "tokenizers_archives",
            "tokenizers/archives",
            f"tokenizers/archives/{os_dir}",
        ]
    else:
        raise ValueError(f"Unknown iotg component: {component}")

    # Release/RC builds never use the flat nightly layout, so skip probing it.
    if not base.startswith(_IOTG_NIGHTLY_ROOT):
        subpaths = [sub for sub in subpaths if not sub.endswith("_archives")] or subpaths

    return [f"{base}/{sub}" for sub in subpaths]


def _resolve_iotg_component_archive(base_url: str, component: str) -> str | None:
    """Return the host-matching archive URL for *component*, or None."""
    prefix = _IOTG_ARCHIVE_PREFIX[component]
    for listing_url in _iotg_component_listing_dirs(base_url, component):
        text = get_text_from_web(listing_url)
        if not text:
            continue
        archive_url = _find_iotg_host_archive_url(listing_url, text, prefix)
        if archive_url:
            return archive_url
    return None


def _load_github_token() -> str:
    """
    Load the GitHub personal access token used for API authentication.

    Resolution order:
    1. ``github.token`` in ``.temp/user_env.json`` (relative to the skills
       workspace root, which is 4 directory levels above this script).
    2. ``GITHUB_TOKEN`` environment variable (legacy / CI fallback).

    Returns an empty string when no token is configured.
    """
    # <SKILLS_DIR>/.github/skills/download-openvino/scripts/download-openvino.py
    #  → parent × 4 = <SKILLS_DIR>. Copies outside the skills repo use GITHUB_TOKEN.
    script_parents = Path(__file__).resolve().parents
    skills_root = script_parents[4] if len(script_parents) > 4 else script_parents[-1]
    user_env_path = skills_root / ".temp" / "user_env.json"

    if user_env_path.exists():
        try:
            with open(user_env_path, encoding="utf-8") as fh:
                data = json.load(fh)
            token = data.get("github", {}).get("token", "").strip()
            if token:
                log.debug("GitHub token loaded from %s", user_env_path)
                return token
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            log.warning("Failed to read GitHub token from %s: %s", user_env_path, exc)

    # Fallback: environment variable
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        log.debug("GitHub token loaded from GITHUB_TOKEN environment variable.")
    return token


def _github_api_headers() -> dict[str, str]:
    """Return common GitHub API request headers, including auth when available."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _load_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        log.warning(
            "GitHub token not configured. "
            "Set 'github.token' in .temp/user_env.json or the GITHUB_TOKEN env var "
            "to avoid API rate limits (60 req/hr unauthenticated)."
        )
    return headers


def get_pr_head_commit(pr_number: int) -> str | None:
    """
    Fetch the head commit SHA of a GitHub PR from openvinotoolkit/openvino.

    Returns the full SHA string or None on failure.
    """
    api_url = (
        f"https://api.github.com/repos/openvinotoolkit/openvino/pulls/{pr_number}"
    )
    try:
        res = requests.get(
            api_url,
            headers=_github_api_headers(),
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        sha: str = data["head"]["sha"]
        log.info("PR #%d head commit: %s", pr_number, sha)
        return sha
    except requests.exceptions.HTTPError as exc:
        log.error("GitHub API HTTP error fetching PR #%d: %s", pr_number, exc)
    except (requests.exceptions.RequestException, KeyError, TypeError) as exc:
        log.error("Failed to fetch PR #%d head commit: %s", pr_number, exc)
    return None


def find_url_for_pr(
    pr_number: int,
    release_roots: list[str] | None = None,
    strict_branch: bool = False,
) -> str | None:
    """
    Resolve a GitHub PR number to a cpack URL.

    Fetches the PR head commit SHA from the GitHub API and then delegates to
    :func:`find_url_for_commit` to locate the matching build on the server.

    Returns the cpack URL or None when no matching build is found.
    """
    commit_sha = get_pr_head_commit(pr_number)
    if not commit_sha:
        log.error("Could not retrieve head commit for PR #%d.", pr_number)
        return None
    url = find_url_for_commit(commit_sha, release_roots, strict_branch)
    if url is None:
        log.error(
            "No build found for PR #%d (commit %s).", pr_number, commit_sha[:12]
        )
    return url


def get_latest_commit_list_from_github(per_page: int = 40) -> list[str]:
    """
    Return the most recent *per_page* commit SHAs from openvinotoolkit/openvino master
    via the GitHub REST API.  Returns an empty list on failure.
    """
    api_url = "https://api.github.com/repos/openvinotoolkit/openvino/commits"
    try:
        res = requests.get(
            api_url,
            headers=_github_api_headers(),
            params={"sha": "master", "per_page": per_page},
            timeout=15,
        )
        res.raise_for_status()
        return [c["sha"] for c in res.json()]
    except requests.exceptions.HTTPError as exc:
        log.error("GitHub API HTTP error: %s", exc)
    except requests.exceptions.RequestException as exc:
        log.error("GitHub API request failed: %s", exc)
    return []


def _list_child_dirs(url: str) -> list[str]:
    """Return child directory names from a web listing page."""
    html = get_text_from_web(url)
    if not html:
        return []

    names: list[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href", "")
        m = re.fullmatch(r"([\w.-]+)/", href)
        if not m:
            continue
        name = m.group(1)
        if name in {".", ".."}:
            continue
        names.append(name)

    return sorted(set(names), reverse=True)


def _collect_commit_entries(root_url: str) -> list[tuple[str, datetime.datetime, str]]:
    """
    Collect (commit_id, timestamp, root_url) entries from a commit listing root.
    """
    root_html = get_text_from_web(root_url)
    if not root_html:
        return []

    entries: list[tuple[str, datetime.datetime, str]] = []
    soup = BeautifulSoup(root_html, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if len(href) <= 20 or not href.endswith("/"):
            continue

        commit_id = href[:-1]
        date_text = str(link.next_sibling or "")
        m = re.search(r"(\d{2}-[a-zA-Z]{3}-\d{4} \d{2}:\d{2})", date_text)
        if not m:
            continue

        try:
            dt = datetime.datetime.strptime(m.group(1), "%d-%b-%Y %H:%M")
        except ValueError:
            continue

        entries.append((commit_id, dt, root_url))

    return entries


def _release_search_roots() -> list[str]:
    """Build release search roots from releases/<year>/<milestone> tree since 2025/4."""
    roots: list[str] = []
    years = _list_child_dirs(_RELEASES_ROOT)
    for year in years:
        try:
            year_number = int(year)
        except ValueError:
            log.debug("Skip release directory with non-numeric year: %s", year)
            continue

        year_root = f"{_RELEASES_ROOT}/{year}"
        milestones = _list_child_dirs(year_root)
        for milestone in milestones:
            try:
                milestone_number = int(milestone.split("_", maxsplit=1)[0])
            except ValueError:
                log.debug("Skip release directory with non-numeric milestone: %s/%s", year, milestone)
                continue

            if (year_number, milestone_number) < (2025, 4):
                continue

            release_base = f"{year_root}/{milestone}"
            for leaf in ("commit", "nightly", "pre_commit"):
                roots.append(f"{release_base}/{leaf}")
    return roots


def _cache_contains_unsupported_release_roots(roots: list[str]) -> bool:
    """Return whether cached roots include milestones older than 2025/4."""
    for root in roots:
        match = re.search(r"/releases/(\d+)/([^/]+)/", root)
        if match is None:
            continue

        year_number = int(match.group(1))
        try:
            milestone_number = int(match.group(2).split("_", maxsplit=1)[0])
        except ValueError:
            continue

        if (year_number, milestone_number) < (2025, 4):
            return True

    return False


def _collect_iotg_nightly_archive_urls(limit: int = 20) -> list[str]:
    """Return latest iotg nightly build base URLs in descending version order."""
    versions = _list_child_dirs(_IOTG_NIGHTLY_ROOT)
    if not versions:
        return []

    parsed: list[tuple[OVVersion, str]] = []
    for ver in versions:
        if not _is_supported_iotg_version(ver):
            log.debug("Skip unsupported nightly directory: %s", ver)
            continue
        try:
            parsed.append((OVVersion(ver), ver))
        except ValueError:
            log.debug("Skip non-version nightly directory: %s", ver)

    parsed.sort(key=lambda item: item[0], reverse=True)
    return [f"{_IOTG_NIGHTLY_ROOT}/{ver}" for _, ver in parsed[:limit]]


def list_iotg_builds(channel: str) -> list[tuple[str, str]]:
    """
    Return ``(label, base_url)`` pairs for *channel*, newest first.

    Supported channels: ``nightly``, ``release``, ``rc``.
    Versions older than 2025.4 are never returned.
    """
    if channel == "nightly":
        return [(url.rsplit("/", 1)[1], url) for url in _collect_iotg_nightly_archive_urls(limit=100)]

    if channel == "release":
        root = _IOTG_RELEASES_ROOT
        builds = [
            (ver, f"{root}/{ver}")
            for ver in _list_child_dirs(root)
            if _is_supported_iotg_version(ver)
        ]
        builds.sort(key=lambda item: _iotg_build_sort_key(item[0]), reverse=True)
        return builds

    if channel == "rc":
        root = _IOTG_RC_ROOT
        builds = []
        for ver in _list_child_dirs(root):
            if not _is_supported_iotg_version(ver):
                continue
            for rc_name in _list_child_dirs(f"{root}/{ver}"):
                if re.fullmatch(r"RC\d+", rc_name, re.IGNORECASE):
                    builds.append((f"{ver}/{rc_name}", f"{root}/{ver}/{rc_name}"))
        builds.sort(key=lambda item: _iotg_build_sort_key(item[0]), reverse=True)
        return builds

    raise ValueError(f"Unknown iotg channel: {channel}")


def resolve_iotg_base_url(version: str, channel: str = "auto") -> str | None:
    """
    Resolve an iotg build selector to a base URL.

    Accepted *version* forms:
    * ``latest`` — newest build in *channel* (``nightly`` when channel is auto)
    * ``2026.5.0-22987-f1a195380ab`` — nightly build directory
    * ``2026.3.0`` — release (or newest RC when ``--channel rc``)
    * ``2026.3.0/RC3`` — explicit release candidate
    """
    selector = version.strip().strip("/")

    if selector.lower() == "latest":
        target = "nightly" if channel == "auto" else channel
        builds = list_iotg_builds(target)
        if not builds:
            log.error("No builds found in channel: %s", target)
            return None
        return builds[0][1]

    if not _is_supported_iotg_version(selector):
        log.error(
            "Version '%s' is unsupported; only %d.%d and newer are searched.",
            selector,
            *_MIN_SUPPORTED_VERSION,
        )
        return None

    rc_match = re.fullmatch(r"(\d+\.\d+\.\d+)/(RC\d+)", selector, re.IGNORECASE)
    if rc_match:
        candidates = [(selector, f"{_IOTG_RC_ROOT}/{rc_match.group(1)}/{rc_match.group(2).upper()}")]
    elif re.fullmatch(r"\d+\.\d+\.\d+", selector):
        release_url = f"{_IOTG_RELEASES_ROOT}/{selector}"
        rc_builds = [b for b in list_iotg_builds("rc") if b[0].startswith(f"{selector}/")]
        if channel == "rc":
            candidates = rc_builds
        elif channel == "release":
            candidates = [(selector, release_url)]
        else:
            candidates = [(selector, release_url), *rc_builds]
    else:
        candidates = [(selector, f"{_IOTG_NIGHTLY_ROOT}/{selector}")]

    for label, base_url in candidates:
        if _list_child_dirs(base_url):
            log.info("Resolved iotg build '%s': %s", label, base_url)
            return base_url

    log.error("No iotg build found for version: %s", version)
    return None


def _load_release_roots_cache(cache_file: Path, ttl_minutes: int) -> list[str] | None:
    """Load cached release roots when the cache exists and is not stale."""
    if ttl_minutes <= 0 or not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        generated_at = payload.get("generated_at")
        roots = payload.get("roots")
        if not generated_at or not isinstance(roots, list):
            return None

        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(generated_at)
        if age > datetime.timedelta(minutes=ttl_minutes):
            return None

        return [str(root) for root in roots]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _save_release_roots_cache(cache_file: Path, roots: list[str]) -> None:
    """Persist release roots cache to disk."""
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "roots": roots,
    }
    try:
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("Failed to write release roots cache: %s", exc)


def _cache_file_path(filename: str) -> Path | None:
    """Return cache file path under the active output directory."""
    if _CACHE_DIR is None:
        return None
    return _CACHE_DIR / filename


def _utc_now() -> datetime.datetime:
    """Return timezone-aware current UTC time."""
    return datetime.datetime.now(datetime.timezone.utc)


def _load_kv_cache(filename: str) -> dict[str, dict[str, object]]:
    """Load a small JSON key-value cache from the active cache directory."""
    cache_path = _cache_file_path(filename)
    if cache_path is None or not cache_path.exists():
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError, ValueError):
        return {}

    # Check cache version for compatibility
    version = payload.get("version")
    if version != _CACHE_VERSION:
        log.debug(
            "Cache version mismatch for %s (expected %s, got %s); discarding",
            filename,
            _CACHE_VERSION,
            version,
        )
        return {}

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_kv_cache(filename: str, entries: dict[str, dict[str, object]]) -> None:
    """Persist a small JSON key-value cache under the active cache directory."""
    cache_path = _cache_file_path(filename)
    if cache_path is None:
        return

    payload = {
        "version": _CACHE_VERSION,
        "generated_at": _utc_now().isoformat(),
        "entries": entries,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temporary file first, then rename (atomic operation)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(cache_path)
    except OSError as exc:
        log.debug("Failed to write cache %s: %s", cache_path.name, exc)


def _load_cache_entry(
    filename: str,
    key: str,
    positive_ttl: datetime.timedelta,
    negative_ttl: datetime.timedelta,
) -> dict[str, object] | None:
    """Return a fresh cache entry for *key*, or None when absent/stale."""
    entries = _load_kv_cache(filename)
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return None

    timestamp = entry.get("timestamp")
    ok = bool(entry.get("ok"))
    if not isinstance(timestamp, str):
        return None

    try:
        age = _utc_now() - datetime.datetime.fromisoformat(timestamp)
    except ValueError:
        return None

    ttl = positive_ttl if ok else negative_ttl
    if age > ttl:
        return None
    return entry


def _store_cache_entry(
    filename: str,
    key: str,
    ok: bool,
    value: str | None = None,
) -> None:
    """Store a cache entry for *key* with optional string value."""
    entries = _load_kv_cache(filename)
    entry: dict[str, object] = {
        "ok": ok,
        "timestamp": _utc_now().isoformat(),
    }
    if value is not None:
        entry["value"] = value
    entries[key] = entry
    _save_kv_cache(filename, entries)


def _delete_cache_entry(filename: str, key: str) -> None:
    """Remove a cache entry for *key* without disturbing other cache data."""
    entries = _load_kv_cache(filename)
    if key not in entries:
        return
    del entries[key]
    _save_kv_cache(filename, entries)


def _cleanup_expired_cache_entries(
    filename: str,
    positive_ttl: datetime.timedelta,
    negative_ttl: datetime.timedelta,
) -> int:
    """
    Remove expired cache entries from disk cache.

    Returns the number of entries removed.
    """
    cache_path = _cache_file_path(filename)
    if not cache_path or not cache_path.exists():
        return 0

    try:
        entries = _load_kv_cache(filename)
        original_count = len(entries)
        now = _utc_now()

        # Filter out expired entries
        fresh_entries = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            timestamp_str = entry.get("timestamp")
            if not isinstance(timestamp_str, str):
                continue

            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                age = now - timestamp
                ok = bool(entry.get("ok"))
                ttl = positive_ttl if ok else negative_ttl

                if age <= ttl:
                    fresh_entries[key] = entry
            except (ValueError, TypeError):
                pass

        # Write back cleaned cache
        if len(fresh_entries) < original_count:
            _save_kv_cache(filename, fresh_entries)
            removed = original_count - len(fresh_entries)
            log.info("Cleaned %s cache: removed %d expired entries", filename, removed)
            return removed
    except Exception as exc:
        log.debug("Failed to cleanup cache %s: %s", filename, exc)

    return 0


def get_release_search_roots(
    output_dir: Path,
    *,
    refresh_cache: bool,
    ttl_minutes: int,
    cache_file: Path | None = None,
) -> list[str]:
    if cache_file is None:
        cache_file = output_dir / _RELEASE_ROOTS_CACHE_FILENAME

    if not refresh_cache:
        cached = _load_release_roots_cache(cache_file, ttl_minutes)
        if cached is not None:
            if not _cache_contains_unsupported_release_roots(cached):
                log.info("Using cached release roots (%d).", len(cached))
                return cached
            log.info("Refreshing cached release roots to exclude milestones before 2025/4.")

    roots = _release_search_roots()
    _save_release_roots_cache(cache_file, roots)
    log.info("Refreshed release roots cache (%d).", len(roots))
    return roots


def _commit_search_roots(release_roots: list[str] | None = None) -> list[str]:
    """Return all commit lookup roots in priority order (release first, then master)."""
    if release_roots is None:
        release_roots = _release_search_roots()
    
    # Priority: Release roots first (newer, official builds), then master/nightly
    base_roots = []
    if release_roots:
        base_roots.extend(release_roots)
    
    base_roots.extend([_MASTER_COMMIT_ROOT, _MASTER_NIGHTLY_ROOT, _PRE_COMMIT_ROOT])
    return base_roots


def _branch_label_for_url(url: str) -> str:
    """
    Describe the build server branch a commit/cpack URL came from.

    Returns e.g. ``releases/2026/4`` or ``master/commit``, or ``unknown`` when the
    URL does not match a known root layout.
    """
    if url.startswith(_RELEASES_ROOT):
        match = re.search(r"/releases/([^/]+)/([^/]+)/", url + "/")
        return f"releases/{match.group(1)}/{match.group(2)}" if match else "releases"
    match = re.search(r"/dldt/master/([^/]+)/", url + "/")
    if match:
        return f"master/{match.group(1)}"
    return "unknown"


def _check_resolved_branch(
    commit_prefix: str, url: str, release_roots: list[str] | None, strict_branch: bool
) -> str | None:
    """
    Warn (or refuse) when a commit resolved to a different branch than the release tree.

    The per-commit archives under ``releases/<year>/<milestone>/commit`` are only kept
    for a short rolling window (~2 weeks). Once a commit ages out, the search falls
    through to the master roots, which pin different genai/openvino_tokenizers
    revisions - silently confounding a same-branch bisect.

    Returns *url* when the resolution is acceptable, otherwise None.
    """
    if not release_roots or url.startswith(_RELEASES_ROOT):
        return url

    branch_label = _branch_label_for_url(url)
    message = (
        "Commit %s was not found in the release tree and resolved to '%s' instead. "
        "That build pins different genai/openvino_tokenizers revisions, so mixing it "
        "with release-branch builds can invalidate a same-branch comparison. "
        "Per-commit release archives are retained only ~2 weeks."
    )
    if strict_branch:
        log.error(message, commit_prefix, branch_label)
        log.error("Refusing the cross-branch build because --strict-branch was given.")
        return None

    log.warning(message, commit_prefix, branch_label)
    log.warning("Use --strict-branch to fail instead of falling back across branches.")
    return url


def get_latest_master_urls(
    args: argparse.Namespace,
) -> tuple[list[str], bool]:
    """
    Scrape the private build server for recent master-branch cpack URLs.

    Candidates are ordered by the GitHub master commit history (newest first),
    not by the server upload time: an older commit can be rebuilt and uploaded
    after a newer one, and date ordering would then pick the stale build.
    ``args.skip_latest`` drops that many of the newest valid builds.

    Returns ``(list_of_valid_urls, had_existing_version)``.
    """
    master_roots = [_MASTER_COMMIT_ROOT]
    commit_entries: list[tuple[str, datetime.datetime, str]] = []
    for root in master_roots:
        log.info("Querying master build list: %s", root)
        commit_entries.extend(_collect_commit_entries(root))

    iotg_nightly_urls = _collect_iotg_nightly_archive_urls(limit=20)
    if iotg_nightly_urls:
        log.info("Found %d iotg nightly archive candidate(s).", len(iotg_nightly_urls))

    if not commit_entries and not iotg_nightly_urls:
        log.error("Failed to retrieve master commit/nightly lists.")
        return [], False

    existing_version = _load_cached_version(args.output)
    skip_latest = max(getattr(args, "skip_latest", 0), 0)
    # Skipping recent builds intentionally selects older ones, so the newer-than-cache filter must not apply.
    accept_any_version = args.force or skip_latest > 0

    root_by_commit = {commit_id: root for commit_id, _, root in commit_entries}
    github_commits = get_latest_commit_list_from_github(per_page=100)
    if github_commits:
        ordered = [(c, root_by_commit[c]) for c in github_commits if c in root_by_commit]
        log.info(
            "Ordering %d server build(s) by GitHub master history (%d commits checked).",
            len(ordered), len(github_commits),
        )
    else:
        log.warning(
            "Could not retrieve GitHub commit list; falling back to server upload time, "
            "which may select a stale rebuild."
        )
        commit_entries.sort(key=lambda x: x[1], reverse=True)
        ordered = [(c, root) for c, _, root in commit_entries[:80]]

    valid_urls: list[str] = []
    seen_urls: set[str] = set()
    for commit_id, root in ordered:
        selected_url = None
        for cpack_subpath in _private_cpack_subpath_candidates():
            url = f"{root}/{commit_id}/{cpack_subpath}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if not check_required_packages(url):
                continue
            if skip_latest > 0:
                log.info("Skipping newest build (--skip-latest): %s", url)
                skip_latest -= 1
            elif accept_any_version or _is_newer(existing_version, url):
                selected_url = url
            else:
                log.debug("Skipping non-newer build at %s.", url)
            break

        if selected_url:
            valid_urls.append(selected_url)
        if len(valid_urls) >= 10:
            break

    if not valid_urls:
        log.warning("No new valid master builds found.")
    else:
        log.info("Found %d valid build URL(s).", len(valid_urls))

    # Add iotg nightly candidates as fallback sources.
    for nightly_url in iotg_nightly_urls:
        if nightly_url in seen_urls:
            continue
        if not accept_any_version and existing_version is not None:
            try:
                if not existing_version < OVVersion(nightly_url.rsplit("/", 1)[-1]):
                    continue
            except ValueError:
                continue
        if check_required_packages(nightly_url):
            valid_urls.append(nightly_url)
            seen_urls.add(nightly_url)
        if len(valid_urls) >= 15:
            break

    return valid_urls, existing_version is not None


def find_url_for_commit(
    commit_prefix: str,
    release_roots: list[str] | None = None,
    strict_branch: bool = False,
) -> str | None:
    """
    Search known build server paths for a build whose commit hash starts with
    *commit_prefix*.  Returns the full cpack URL or None.

    When the commit is only found outside the release tree, the resolved branch is
    reported (see :func:`_check_resolved_branch`); *strict_branch* turns that into
    a hard failure instead of a warning.
    """
    cached_entry = _load_cache_entry(
        _COMMIT_URL_CACHE_FILENAME,
        commit_prefix,
        _COMMIT_URL_POSITIVE_TTL,
        _COMMIT_URL_NEGATIVE_TTL,
    )
    if cached_entry is not None and bool(cached_entry.get("ok")):
        cached_url = cached_entry.get("value")
        if isinstance(cached_url, str) and cached_url:
            if check_required_packages(cached_url):
                log.info("Using cached commit URL for %s: %s", commit_prefix, cached_url)
                return _check_resolved_branch(
                    commit_prefix, cached_url, release_roots, strict_branch
                )
            log.warning(
                "Cached commit URL is incomplete for %s; invalidating: %s",
                commit_prefix,
                cached_url,
            )
            _delete_cache_entry(_COMMIT_URL_CACHE_FILENAME, commit_prefix)

    search_roots = _commit_search_roots(release_roots)
    log.info("Searching commit prefix %s in %d root(s).", commit_prefix, len(search_roots))
    for root in search_roots:
        html = get_text_from_web(root)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href", "")
            m = re.search(r"([\w]+)/", href)
            if m:
                full_id = m.group(1)
                if full_id.startswith(commit_prefix):
                    for cpack_subpath in _private_cpack_subpath_candidates():
                        url = f"{root}/{full_id}/{cpack_subpath}"
                        if check_required_packages(url):
                            log.info(
                                "Found commit %s on branch %s at: %s",
                                full_id,
                                _branch_label_for_url(url),
                                url,
                            )
                            _store_cache_entry(_COMMIT_URL_CACHE_FILENAME, commit_prefix, True, url)
                            return _check_resolved_branch(
                                commit_prefix, url, release_roots, strict_branch
                            )
    log.error("No build found for commit prefix: %s", commit_prefix)
    return None


# ---------------------------------------------------------------------------
# Download and install helpers
# ---------------------------------------------------------------------------

def _parse_ov_version_and_commit(url: str, text: str) -> tuple[str, str]:
    """
    Extract the OV package version string and commit ID from *url* and *text*.

    Raises ValueError when either cannot be found.
    """
    commit_m = re.search(r"(custom_build|commit|pre_commit)/([\w]+)", url)
    if not commit_m:
        raise ValueError(f"Cannot parse commit ID from URL: {url}")
    commit_id = commit_m.group(2)

    req0 = required_openvino_packages_list()[0]
    if IS_WINDOWS:
        ver_m = re.search(
            rf"inference-engine_Release-(\d+\.\d+\.\d+\.\d+)-win64-{re.escape(req0)}",
            text,
        )
    else:
        ver_m = re.search(
            rf"inference-engine-(\d+\.\d+\.\d+\.\d+)-Linux-{re.escape(req0)}",
            text,
        )
    if not ver_m:
        raise ValueError(f"Cannot parse OV package version from page at: {url}")

    return ver_m.group(1), commit_id


def _verify_sha256(archive: Path, archive_url: str) -> bool:
    """Compare *archive* against its ``.sha256`` sidecar on the server."""
    sidecar = get_text_from_web(f"{archive_url}.sha256")
    tokens = sidecar.split() if sidecar else []
    if not tokens:
        log.error("No usable .sha256 sidecar for %s", archive.name)
        return False

    expected = tokens[0].strip().lower()
    digest = hashlib.sha256()
    with open(archive, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected:
        log.error("SHA256 mismatch for %s", archive.name)
        return False

    log.info("SHA256 verified: %s", archive.name)
    return True


def _extracted_root(extract_dir: Path) -> Path:
    """Return the package root inside *extract_dir* (single top-level dir or itself)."""
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def _archive_dir_name(archive_url: str) -> str:
    """Return the archive filename without its extension."""
    name = archive_url.rsplit("/", 1)[-1]
    for ext in (".tar.gz", ".tgz", ".zip"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _overlay_tree(src: Path, dest: Path, preserve_root_files: set[str]) -> None:
    """Copy *src* over *dest*, keeping existing root files listed in *preserve_root_files*."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.name in preserve_root_files and target.exists():
            log.debug("Preserving existing %s", target.name)
        else:
            shutil.copy2(item, target)


def _fetch_and_extract_iotg_component(
    archive_url: str,
    component: str,
    dest_dir: Path,
    verify_sha256: bool,
) -> Path:
    """Download *archive_url* for *component* and extract it into *dest_dir*."""
    archive = download_file(archive_url, dest_dir)
    if not archive:
        raise RuntimeError(f"Failed to download {component} package: {archive_url}")

    if verify_sha256 and not _verify_sha256(archive, archive_url):
        raise RuntimeError(f"Checksum verification failed: {archive.name}")

    extract_dir = dest_dir / f"_extract_{component}"
    extract_dir.mkdir(parents=True)

    if not decompress(archive, extract_dir, delete_after=True):
        raise RuntimeError(f"Failed to extract {component} package: {archive.name}")

    return _extracted_root(extract_dir)


def download_iotg_bundle(
    base_url: str,
    out_path: Path,
    components: str | list[str] | None = None,
    verify_sha256: bool = False,
    force: bool = False,
) -> tuple[list[Path], Path]:
    """
    Download OpenVINO plus optional GenAI/tokenizers packages from an iotg build
    and merge them into a single directory driven by one ``setupvars`` script.

    Every selected component must be available for the host; the existing
    installation is replaced only after the merged tree is fully built.

    Returns ``(installed_dirs, versioned_output_dir)``.
    """
    selected = _parse_components(components)

    archive_urls: dict[str, str] = {}
    for component in selected:
        url = _resolve_iotg_component_archive(base_url, component)
        if not url:
            raise ValueError(f"No host-matching '{component}' package found at: {base_url}")
        archive_urls[component] = url

    setup_ext = ".bat" if IS_WINDOWS else ".sh"
    out_path.mkdir(parents=True, exist_ok=True)

    expected_dir = out_path / _archive_dir_name(archive_urls["openvino"])
    if not force and has_existing_prebuilt(expected_dir):
        log.info("Prebuilt already exists at %s. Skipping download.", expected_dir)
        _update_latest_setup_file(expected_dir / f"setupvars{setup_ext}", out_path)
        return [expected_dir], expected_dir

    staging_dir = Path(tempfile.mkdtemp(dir=out_path, prefix=".iotg_"))
    try:
        merged_root = _fetch_and_extract_iotg_component(
            archive_urls["openvino"], "openvino", staging_dir, verify_sha256
        )
        dir_name = expected_dir.name if merged_root.parent == staging_dir else merged_root.name

        # setupvars from the OpenVINO toolkit stays authoritative after overlays.
        preserve = {"setupvars.bat", "setupvars.ps1", "setupvars.sh"}
        for component in (c for c in selected if c != "openvino"):
            overlay_root = _fetch_and_extract_iotg_component(
                archive_urls[component], component, staging_dir, verify_sha256
            )
            _overlay_tree(overlay_root, merged_root, preserve)
            log.info("Merged '%s' into %s", component, dir_name)

        versioned_dir = out_path / dir_name
        previous_dir = versioned_dir.with_name(f"{dir_name}.previous")
        shutil.rmtree(previous_dir, ignore_errors=True)
        if versioned_dir.exists():
            versioned_dir.rename(previous_dir)
        shutil.move(str(merged_root), str(versioned_dir))
        shutil.rmtree(previous_dir, ignore_errors=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    log.info("Installed iotg bundle: %s", versioned_dir)
    setup_script = versioned_dir / f"setupvars{setup_ext}"
    if not setup_script.exists():
        log.warning("setupvars script not found: %s", setup_script)
    _update_latest_setup_file(setup_script, out_path)

    return [versioned_dir], versioned_dir


def download_openvino_packages(url: str, out_path: Path) -> tuple[list[Path], Path]:
    """
    Download all required OpenVINO cpack archives from *url* into a versioned
    sub-directory of *out_path*.

    Returns ``(list_of_downloaded_paths, versioned_output_dir)``.
    """
    text = get_text_from_web(url)
    ov_ver, commit_id = _parse_ov_version_and_commit(url, text)

    versioned_dir = out_path / f"{ov_ver}_{commit_id[:8]}"
    versioned_dir.mkdir(parents=True, exist_ok=True)

    os_tag = "win64" if IS_WINDOWS else "Linux"
    prefix = f"inference-engine{'_Release' if IS_WINDOWS else ''}-{ov_ver}-{os_tag}"
    urls = [
        f"{url}/{prefix}-{pkg}"
        for pkg in required_openvino_packages_list()
        # Python-version archives vary per build; fetch only the published ones.
        if not pkg.startswith("pyopenvino_python3.") or f"{prefix}-{pkg}" in text
    ]

    downloaded = asyncio.run(async_download_files(urls, versioned_dir))
    return downloaded, versioned_dir


def resolve_versioned_dir(url: str, out_path: Path) -> Path | None:
    """
    Resolve the expected extracted install directory for *url*.

    Returns None when version/commit cannot be parsed from the cpack listing.
    """
    text = get_text_from_web(url)
    if not text:
        return None

    try:
        ov_ver, commit_id = _parse_ov_version_and_commit(url, text)
    except ValueError:
        return None

    return out_path / f"{ov_ver}_{commit_id[:8]}"


def missing_device_plugins(versioned_dir: Path) -> list[str]:
    """Return the CPU/GPU plugin libraries missing from an extracted prebuilt."""
    if IS_WINDOWS:
        lib_dir = versioned_dir / "runtime" / "bin" / "intel64" / "Release"
        names = [f"openvino_intel_{dev}_plugin.dll" for dev in ("cpu", "gpu")]
    else:
        lib_dir = versioned_dir / "runtime" / "lib" / "intel64"
        names = [f"libopenvino_intel_{dev}_plugin.so" for dev in ("cpu", "gpu")]
    return [name for name in names if not (lib_dir / name).is_file()]


def has_existing_prebuilt(versioned_dir: Path) -> bool:
    """Return True if a complete extracted prebuilt (setupvars + CPU/GPU plugins) exists."""
    setup_ext = ".bat" if IS_WINDOWS else ".sh"
    setup_script = versioned_dir / f"setupvars{setup_ext}"
    if not (versioned_dir.is_dir() and setup_script.is_file()):
        return False
    missing = missing_device_plugins(versioned_dir)
    if missing:
        log.warning("Existing prebuilt %s is incomplete (missing %s).", versioned_dir, ", ".join(missing))
        return False
    return True


def query_local_prebuilt_versions(versioned_dir: Path) -> LocalPrebuiltVersions | None:
    """Query OpenVINO and GenAI versions through a prebuilt's configured Python environment."""
    setup_ext = ".bat" if IS_WINDOWS else ".sh"
    setup_script = versioned_dir / f"setupvars{setup_ext}"
    if not setup_script.is_file():
        return None

    query = (
        "import json, openvino, openvino_genai; "
        "print(json.dumps({'openvino': openvino.__version__, "
        "'genai': openvino_genai.__version__, "
        "'openvino_module_path': openvino.__file__, "
        "'genai_module_path': openvino_genai.__file__}))"
    )
    encoded_query = base64.b64encode(query.encode("utf-8")).decode("ascii")
    python_command = f"python -c \"import base64; exec(base64.b64decode('{encoded_query}'))\""
    if IS_WINDOWS:
        command = ["cmd.exe", "/d", "/s", "/c", f'call "{setup_script}" >nul && {python_command}']
    else:
        command = ["bash", "-c", f'source "{setup_script}" >/dev/null && {python_command}']

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        versions = json.loads(result.stdout.strip())
        openvino_version = versions["openvino"]
        genai_version = versions["genai"]
        openvino_module_path = versions["openvino_module_path"]
        genai_module_path = versions["genai_module_path"]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        log.debug("Could not query local prebuilt versions in %s: %s", versioned_dir, exc)
        return None

    if not all(
        isinstance(value, str)
        for value in (openvino_version, genai_version, openvino_module_path, genai_module_path)
    ):
        log.debug("Local prebuilt returned invalid version data in %s", versioned_dir)
        return None

    versioned_dir_resolved = versioned_dir.resolve()
    openvino_path = Path(openvino_module_path).resolve()
    genai_path = Path(genai_module_path).resolve()
    if not all(
        module_path.is_relative_to(versioned_dir_resolved)
        for module_path in (openvino_path, genai_path)
    ):
        log.debug("Local prebuilt imports modules outside its directory: %s", versioned_dir)
        return None

    return LocalPrebuiltVersions(
        openvino=openvino_version,
        genai=genai_version,
        openvino_module_path=openvino_path,
        genai_module_path=genai_path,
    )


def find_latest_local_prebuilt(output_dir: Path) -> tuple[Path, LocalPrebuiltVersions] | None:
    """Return the newest local prebuilt whose OpenVINO and GenAI versions Python can query."""
    candidates: list[tuple[OVVersion, Path, LocalPrebuiltVersions]] = []
    for versioned_dir in output_dir.iterdir():
        if not versioned_dir.is_dir() or not has_existing_prebuilt(versioned_dir):
            continue

        versions = query_local_prebuilt_versions(versioned_dir)
        if versions is None:
            continue

        try:
            candidates.append((OVVersion(versions.openvino), versioned_dir, versions))
        except ValueError:
            log.debug("Skip local prebuilt with unrecognized OpenVINO version: %s", versions.openvino)

    if not candidates:
        return None

    _, versioned_dir, versions = max(candidates, key=lambda candidate: candidate[0])
    return versioned_dir, versions


def download_genai_packages(url: str, dest: Path) -> list[Path]:
    """
    Download all required GenAI cpack archives from *url* into *dest*.

    Returns a list of downloaded file Paths.
    """
    text = get_text_from_web(url)
    if not text:
        return []

    os_tag = "win64" if IS_WINDOWS else "Linux"
    req0 = required_genai_packages_list()[0]
    m = re.search(
        rf"OpenVINOGenAI-([\d.]+)-{os_tag}-{re.escape(req0)}", text
    )
    if m:
        genai_ver = m.group(1)
        urls = [
            f"{url}/OpenVINOGenAI-{genai_ver}-{os_tag}-{pkg}"
            for pkg in required_genai_packages_list()
            if not pkg.startswith("pygenai_")
            or f"OpenVINOGenAI-{genai_ver}-{os_tag}-{pkg}" in text
        ]
        return asyncio.run(async_download_files(urls, dest))

    legacy_match = re.search(
        rf"(openvino_tokenizers-[\d.]+-{os_tag}\.zip)",
        text,
    )
    if legacy_match:
        return asyncio.run(async_download_files([f"{url}/{legacy_match.group(1)}"], dest))

    log.warning(
        "No GenAI package layout recognized at %s. Continuing without GenAI packages.",
        url,
    )
    return []


def decompress(archive: Path, dest: Path, delete_after: bool = False) -> Path | None:
    """
    Extract *archive* (.zip or .tgz/.gz) into *dest*.

    Returns the expected root directory inside *dest*, or None on failure.
    """
    suffix = archive.suffix.lower()
    stem = archive.stem

    try:
        if suffix == ".zip":
            if IS_WINDOWS:
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(dest)
            else:
                subprocess.run(
                    ["unzip", "-o", "-q", str(archive), "-d", str(dest)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        elif suffix in {".tgz", ".gz"}:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(dest, filter="tar", numeric_owner=True)
        else:
            log.warning("Unknown archive format: %s", archive.name)
            return None
    except (zipfile.BadZipFile, tarfile.TarError, subprocess.CalledProcessError) as exc:
        log.error("Decompression failed for %s: %s", archive.name, exc)
        return None

    if delete_after:
        try:
            archive.unlink()
        except OSError as exc:
            log.warning("Could not delete archive %s: %s", archive.name, exc)

    return dest / stem


def _update_latest_setup_file(setup_script: Path, output_dir: Path) -> None:
    """Write the path to the latest setupvars script into output_dir."""
    latest = output_dir / "latest_ov_setup_file.txt"
    try:
        latest.write_text(str(setup_script), encoding="utf-8")
        log.info("Latest setup script recorded: %s", setup_script)
    except OSError as exc:
        log.error("Failed to write latest setup file: %s", exc)


def install_openvino(archive: Path, output_dir: Path) -> None:
    """Extract a single OV archive and update the latest-setup pointer."""
    if not archive.exists():
        log.warning("Archive not found: %s", archive)
        return
    unpacked = decompress(archive, output_dir)
    if unpacked:
        setup_ext = ".bat" if IS_WINDOWS else ".sh"
        setup_script = unpacked / f"setupvars{setup_ext}"
        _update_latest_setup_file(setup_script, output_dir)


def cleanup_old_artifacts(output_dir: Path, keep_dirs: int = 20) -> None:
    """Remove stale archives and keep only the *keep_dirs* most recent dirs."""
    for archive in list(output_dir.glob("*.zip")) + list(output_dir.glob("*.tgz")):
        log.info("Removing old archive: %s", archive.name)
        try:
            archive.unlink()
        except OSError as exc:
            log.warning("Could not remove %s: %s", archive.name, exc)

    dirs = sorted(
        (d for d in output_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for old_dir in dirs[keep_dirs:]:
        log.info("Removing old directory: %s", old_dir.name)
        try:
            shutil.rmtree(old_dir)
        except OSError as exc:
            log.warning("Could not remove directory %s: %s", old_dir.name, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and install OpenVINO prebuilt packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output",
        metavar="DIR",
        type=Path,
        default=PWD / "openvino_nightly",
        help="Directory where packages will be stored.",
    )
    parser.add_argument(
        "-d", "--download-url", "--download_url",
        dest="download_url",
        metavar="URL",
        type=str,
        default=None,
        help=(
            "Direct archive URL ending in .zip/.tgz for single-file download, "
            "or a cpack directory URL to download that build."
        ),
    )
    parser.add_argument(
        "-c", "--commit-id", "--commit_id",
        dest="commit_id",
        metavar="HASH",
        type=str,
        default=None,
        help="Short or full commit hash to locate the corresponding build.",
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        type=Path,
        default=None,
        help="Print a markdown table from a local manifest.yml file and exit.",
    )
    parser.add_argument(
        "-i", "--install",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Install from a local archive (.zip/.tgz) or directory containing "
            "archives, then exit."
        ),
    )
    parser.add_argument(
        "--latest-commit", "--latest_commit",
        dest="latest_commit",
        action="store_true",
        help="Print the latest valid master-branch commit URL and exit.",
    )
    parser.add_argument(
        "--skip-latest", "--skip_latest",
        dest="skip_latest",
        metavar="N",
        type=int,
        default=0,
        help=(
            "Skip the N newest valid master builds (auto-latest mode only). "
            "Implies accepting builds older than the cached version."
        ),
    )
    parser.add_argument(
        "--prefer-local",
        action="store_true",
        help=(
            "In auto-latest mode, reuse the newest complete local prebuilt "
            "instead of querying the build server."
        ),
    )
    parser.add_argument(
        "--no-proxy", "--no_proxy",
        dest="no_proxy",
        action="store_true",
        help="Accepted for compatibility; internal servers always bypass the proxy.",
    )
    parser.add_argument(
        "--strict-branch",
        action="store_true",
        help=(
            "Fail instead of falling back to a master build when --commit-id / "
            "--pr-number is not found in the release tree. Use this for same-branch "
            "bisects: per-commit release archives are kept only ~2 weeks, and master "
            "builds pin different genai/openvino_tokenizers revisions."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the same version is already present.",
    )
    parser.add_argument(
        "--keep-old", "--keep_old",
        dest="keep_old",
        action="store_true",
        help="Do not remove old packages/directories after a successful download.",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        default=False,
        help=(
            "Enable SSL certificate verification. Disabled by default "
            "because Intel's internal build servers use self-signed certs."
        ),
    )
    parser.add_argument(
        "--check-packages",
        metavar="URL",
        type=str,
        default=None,
        help="Verify that all required packages exist at a cpack URL and exit.",
    )
    parser.add_argument(
        "--refresh-release-roots-cache",
        action="store_true",
        default=False,
        help="Force refresh of releases/<year>/<milestone> search roots cache.",
    )
    parser.add_argument(
        "--release-roots-cache-ttl-minutes",
        metavar="MIN",
        type=int,
        default=120,
        help="Cache TTL in minutes for release search roots (<=0 disables cache reads).",
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        default=False,
        help="Remove expired entries from all cache files and exit.",
    )
    parser.add_argument(
        "--iotg-version",
        metavar="VERSION",
        type=str,
        default=None,
        help=(
            "iotg package share build selector: 'latest', a nightly directory "
            "(2026.5.0-22987-f1a195380ab), a release (2026.3.0) or a release "
            "candidate (2026.3.0/RC3). Builds older than 2025.4 are rejected."
        ),
    )
    parser.add_argument(
        "--channel",
        choices=("auto", "nightly", "release", "rc"),
        default="auto",
        help="iotg channel used to resolve --iotg-version (default: auto).",
    )
    parser.add_argument(
        "--components",
        metavar="LIST",
        type=str,
        default="openvino,genai,tokenizers",
        help=(
            "Comma-separated iotg components to install and merge "
            "(openvino,genai,tokenizers). 'openvino' is always required."
        ),
    )
    parser.add_argument(
        "--list-iotg-versions",
        metavar="CHANNEL",
        nargs="?",
        const="nightly",
        choices=("nightly", "release", "rc"),
        default=None,
        help="List available iotg builds for a channel and exit.",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        default=False,
        help="Verify downloaded iotg archives against their .sha256 sidecars.",
    )
    parser.add_argument(
        "--pr-number",
        metavar="NUMBER",
        type=int,
        default=None,
        help=(
            "GitHub PR number for openvinotoolkit/openvino. "
            "The script fetches the PR's head commit from the GitHub API "
            "and searches the build server for the matching prebuilt packages. "
            "Set GITHUB_TOKEN to avoid API rate limits."
        ),
    )
    return parser


def main() -> None:
    log.basicConfig(
        level=log.INFO,
        format="[%(filename)s:%(lineno)4d:%(funcName)20s] %(levelname)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Apply SSL and proxy settings before any network operations.
    global _VERIFY_SSL  # noqa: PLW0603
    _VERIFY_SSL = args.verify_ssl
    if not _VERIFY_SSL:
        import urllib3  # suppress InsecureRequestWarning for internal servers
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Always bypass proxy for internal build servers.
    os.environ["no_proxy"] = _DEFAULT_NO_PROXY_LIST
    os.environ["NO_PROXY"] = _DEFAULT_NO_PROXY_LIST

    # ------------------------------------------------------------------ #
    # Mode 0: Show manifest                                                #
    # ------------------------------------------------------------------ #
    if args.manifest:
        table = generate_manifest_table(args.manifest)
        if table:
            log.info("\n%s", table)
        else:
            log.error("Failed to generate manifest table from: %s", args.manifest)
            sys.exit(1)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Mode 1: Check package completeness at a URL                         #
    # ------------------------------------------------------------------ #
    if args.check_packages:
        ok = check_required_packages(args.check_packages)
        if ok:
            log.info("All required packages present at: %s", args.check_packages)
            sys.exit(0)
        else:
            log.error("Missing packages at: %s", args.check_packages)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Common setup                                                         #
    # ------------------------------------------------------------------ #
    args.output.mkdir(parents=True, exist_ok=True)
    global _CACHE_DIR, _IGNORE_HTML_CACHE  # noqa: PLW0603
    _CACHE_DIR = args.output
    _IGNORE_HTML_CACHE = bool(args.force)

    # ------------------------------------------------------------------ #
    # Mode 1b: List iotg builds                                            #
    # ------------------------------------------------------------------ #
    if args.list_iotg_versions:
        builds = list_iotg_builds(args.list_iotg_versions)
        if not builds:
            log.error("No builds found in channel: %s", args.list_iotg_versions)
            sys.exit(1)
        log.info(
            "\n%s",
            tabulate(
                [[label, url] for label, url in builds],
                headers=["build", "url"],
                tablefmt="github",
            ),
        )
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Mode 0b: Cleanup expired cache                                       #
    # ------------------------------------------------------------------ #
    if args.cleanup_cache:
        log.info("Cleaning up expired cache entries...")
        removed_commit = _cleanup_expired_cache_entries(
            _COMMIT_URL_CACHE_FILENAME,
            _COMMIT_URL_POSITIVE_TTL,
            _COMMIT_URL_NEGATIVE_TTL,
        )
        removed_packages = _cleanup_expired_cache_entries(
            _PACKAGE_CHECK_CACHE_FILENAME,
            _PACKAGE_CHECK_POSITIVE_TTL,
            _PACKAGE_CHECK_NEGATIVE_TTL,
        )
        removed_html = _cleanup_expired_cache_entries(
            _ROOT_HTML_CACHE_FILENAME,
            _ROOT_HTML_POSITIVE_TTL,
            _ROOT_HTML_NEGATIVE_TTL,
        )
        total_removed = removed_commit + removed_packages + removed_html
        log.info("Cleanup complete: %d total entries removed", total_removed)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Mode 2: Local install                                                #
    # ------------------------------------------------------------------ #
    if args.install:
        target = args.install
        if target.is_dir():
            archives = list(target.glob("*.zip")) + list(target.glob("*.tgz"))
            if not archives:
                log.error("No .zip/.tgz archives found in: %s", target)
                sys.exit(1)
            for arc in archives:
                install_openvino(arc, args.output)
        elif target.is_file():
            install_openvino(target, args.output)
        else:
            log.error("Install path not found: %s", target)
            sys.exit(1)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Mode 3: Download (single file, cpack dir, commit, or auto-latest)   #
    # ------------------------------------------------------------------ #

    if args.iotg_version:
        base_url = resolve_iotg_base_url(args.iotg_version, args.channel)
        if not base_url:
            sys.exit(1)
        try:
            _, versioned_dir = download_iotg_bundle(
                base_url, args.output, args.components, args.verify_sha256, args.force
            )
        except (ValueError, RuntimeError) as exc:
            log.error("iotg download failed: %s", exc)
            sys.exit(1)
        log.info("Successfully installed iotg build to: %s", versioned_dir)
        if not args.keep_old:
            cleanup_old_artifacts(args.output)
        sys.exit(0)

    if args.prefer_local and not any(
        (args.force, args.download_url, args.commit_id, args.pr_number, args.latest_commit)
    ):
        local_prebuilt = find_latest_local_prebuilt(args.output)
        if local_prebuilt:
            versioned_dir, versions = local_prebuilt
            setup_ext = ".bat" if IS_WINDOWS else ".sh"
            _update_latest_setup_file(versioned_dir / f"setupvars{setup_ext}", args.output)
            _save_cached_version(args.output, OVVersion(versions.openvino))
            log.info(
                "Using local prebuilt at %s (OpenVINO: %s, GenAI: %s).",
                versioned_dir,
                versions.openvino,
                versions.genai,
            )
            sys.exit(0)

    # Case A: single archive URL
    if args.download_url and args.download_url.endswith((".zip", ".tgz")):
        path = download_file(args.download_url, args.output)
        if path:
            install_openvino(path, args.output)
            sys.exit(0)
        log.error("Failed to download: %s", args.download_url)
        sys.exit(1)

    # Resolve target URL list
    if args.download_url:
        target_urls = [args.download_url.rstrip("/")]
    elif args.commit_id:
        release_roots = get_release_search_roots(
            args.output,
            refresh_cache=args.refresh_release_roots_cache,
            ttl_minutes=args.release_roots_cache_ttl_minutes,
        )
        url = find_url_for_commit(args.commit_id, release_roots, args.strict_branch)
        target_urls = [url] if url else []
    elif args.pr_number:
        release_roots = get_release_search_roots(
            args.output,
            refresh_cache=args.refresh_release_roots_cache,
            ttl_minutes=args.release_roots_cache_ttl_minutes,
        )
        url = find_url_for_pr(args.pr_number, release_roots, args.strict_branch)
        target_urls = [url] if url else []
    else:
        target_urls, have_existing = get_latest_master_urls(args)
        if have_existing and not target_urls:
            log.info("No newer version found. Nothing to do.")
            sys.exit(0)

    if not target_urls:
        log.error("No target URLs specified or found.")
        sys.exit(1)

    # Case B: cpack directory URL(s) — try each in order
    processed_any = False
    for cpack_url in target_urls:
        log.info("Attempting: %s", cpack_url)
        try:
            if not check_required_packages(cpack_url):
                log.warning("Incomplete packages at %s, trying next.", cpack_url)
                continue

            if args.latest_commit:
                log.info("Valid latest commit: %s", cpack_url)
                sys.exit(0)

            if _is_iotg_url(cpack_url):
                log.info("Downloading iotg nightly bundle …")
                _, versioned_dir = download_iotg_bundle(
                    cpack_url, args.output, args.components, args.verify_sha256, args.force
                )
                log.info("Successfully installed iotg nightly package to: %s", versioned_dir)
                processed_any = True
                break

            if not args.force:
                existing_dir = resolve_versioned_dir(cpack_url, args.output)
                if existing_dir and has_existing_prebuilt(existing_dir):
                    log.info(
                        "Prebuilt already exists at %s. Skipping download.",
                        existing_dir,
                    )
                    setup_ext = ".bat" if IS_WINDOWS else ".sh"
                    setup_script = existing_dir / f"setupvars{setup_ext}"
                    _update_latest_setup_file(setup_script, args.output)
                    new_version = _get_ov_version_from_url(cpack_url)
                    if new_version:
                        _save_cached_version(args.output, new_version)
                    processed_any = True
                    break

            log.info("Downloading OpenVINO packages …")
            ov_files, versioned_dir = download_openvino_packages(cpack_url, args.output)

            log.info("Downloading GenAI packages …")
            genai_files = download_genai_packages(cpack_url, versioned_dir)

            all_archives = ov_files + genai_files
            if not all_archives:
                raise RuntimeError("No files were downloaded successfully.")

            log.info("Decompressing %d archive(s) …", len(all_archives))
            for arc in tqdm(all_archives, desc="Extracting"):
                if decompress(arc, arc.parent, delete_after=True) is None:
                    raise RuntimeError(f"Failed to extract {arc.name}")

            missing = missing_device_plugins(versioned_dir)
            if missing:
                raise RuntimeError(f"Installed package is missing {', '.join(missing)}")

            setup_ext = ".bat" if IS_WINDOWS else ".sh"
            setup_script = versioned_dir / f"setupvars{setup_ext}"
            _update_latest_setup_file(setup_script, args.output)

            new_version = _get_ov_version_from_url(cpack_url)
            if new_version:
                _save_cached_version(args.output, new_version)

            manifest_path = save_manifest(cpack_url, versioned_dir)
            if manifest_path:
                table = generate_manifest_table(manifest_path)
                if table:
                    log.info("--- Manifest ---\n%s\n----------------", table)

            log.info("Successfully installed packages to: %s", versioned_dir)
            processed_any = True
            break

        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to process %s: %s", cpack_url, exc)
            if cpack_url == target_urls[-1]:
                log.error("All attempts failed.")
                sys.exit(1)

    if not processed_any:
        log.error("No valid prebuilt was processed from the requested target URL(s).")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #
    if not args.keep_old:
        log.info("Cleaning up old artifacts in %s …", args.output)
        cleanup_old_artifacts(args.output)

    sys.exit(0)


if __name__ == "__main__":
    main()
