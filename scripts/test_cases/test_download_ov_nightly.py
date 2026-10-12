"""Regression tests for automatic master package selection."""

import argparse
import datetime
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "download_ov_nightly.py"
SPEC = importlib.util.spec_from_file_location("download_ov_nightly", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


@pytest.mark.parametrize("source", ["cpack", "iotg"])
@pytest.mark.parametrize("cached_on_master", [False, True])
@pytest.mark.parametrize("github_available", [False, True])
def test_latest_ignores_non_master_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    cached_on_master: bool,
    github_available: bool,
) -> None:
    cached_commit = "abcdef1234567890"
    latest_commit = "1234567890abcdef"
    (tmp_path / "OV_NIGHTLY_VERSION").write_text(
        f"2026.5.0-99999-{cached_commit[:11]}", encoding="utf-8"
    )
    args = argparse.Namespace(output=tmp_path, force=False, skip_latest=0)
    root = downloader._MASTER_COMMIT_ROOT
    cpack_url = f"{root}/{latest_commit}/test/cpack"
    nightly_url = f"{downloader._IOTG_NIGHTLY_ROOT}/2026.5.0-23000-{latest_commit}"
    entries = (
        [(latest_commit, datetime.datetime(2026, 10, 10), root)]
        if source == "cpack"
        else []
    )
    history = [latest_commit, cached_commit] if cached_on_master else [latest_commit]
    if not github_available:
        history = []
        if cached_on_master:
            entries.append((cached_commit, datetime.datetime(2026, 10, 9), root))
    monkeypatch.setattr(downloader, "_collect_commit_entries", lambda root: entries)
    monkeypatch.setattr(
        downloader, "_collect_iotg_nightly_archive_urls",
        lambda limit: [nightly_url] if source == "iotg" else [],
    )
    monkeypatch.setattr(downloader, "get_latest_commit_list_from_github", lambda per_page: history)
    monkeypatch.setattr(downloader, "_private_cpack_subpath_candidates", lambda: ["test/cpack"])
    monkeypatch.setattr(
        downloader, "check_required_packages", lambda url: url in {cpack_url, nightly_url}
    )
    monkeypatch.setattr(
        downloader, "_get_ov_version_from_url",
        lambda url: downloader.OVVersion(f"2026.5.0-23000-{latest_commit}"),
    )

    urls, have_existing = downloader.get_latest_master_urls(args)

    expected_url = cpack_url if source == "cpack" else nightly_url
    assert urls == ([] if cached_on_master else [expected_url])
    assert have_existing is cached_on_master