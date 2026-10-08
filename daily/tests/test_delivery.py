from __future__ import annotations

import pytest

from common.delivery import (
    DEFAULT_BACKUP_HOST, REMOTE_BASE_DIR, backup_server_url, stage_report_images,
)

pytestmark = pytest.mark.dev_only


def test_default_backup_targets_dg2fizz_file_browser(monkeypatch) -> None:
    monkeypatch.delenv("MAIL_RELAY_SERVER", raising=False)

    assert DEFAULT_BACKUP_HOST == "dg2fizz.ikor.intel.com"
    assert REMOTE_BASE_DIR == "/mnt/hdd/daily/data"
    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://dg2fizz.ikor.intel.com:8081/daily2/"
    )


def test_mail_relay_override_does_not_change_public_url(monkeypatch) -> None:
    monkeypatch.setenv("MAIL_RELAY_SERVER", "reports.example.com")

    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://dg2fizz.ikor.intel.com:8081/daily2/"
    )


def test_backup_url_honors_public_base_override(monkeypatch) -> None:
    monkeypatch.setenv("DAILY_REPORT_BASE_URL", "https://reports.example.com/results/daily2")

    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "https://reports.example.com/results/daily2/"
    )


def test_relay_rejects_ssh_destination(monkeypatch) -> None:
    from common.delivery import _resolve_host

    monkeypatch.setenv("MAIL_RELAY_SERVER", "user@reports.example.com")

    with pytest.raises(ValueError, match="bare hostname"):
        _resolve_host(None)


def test_stage_report_images_ignores_skipped_tests_and_malformed_records(tmp_path) -> None:
    image = tmp_path / "model_p1_pid42_output.png"
    image.write_bytes(b"image")
    output = tmp_path / "staged"
    summary = {"tests": [
        {"outcome": "skipped", "metrics": {
            "test_type": "image_generation", "model": "model", "precision": "FP16",
            "data": [{"image_path": str(image)}],
        }},
        {"outcome": "passed", "metrics": {
            "test_type": "image_generation", "model": "model", "precision": "FP16",
            "data": ["malformed", {"image_path": str(image)}],
        }},
    ]}

    staged = stage_report_images(summary, output, "20261005_2349")

    assert len(staged) == 1
    assert next(iter(staged.values())).name == "daily.20261005_2349.image.model_FP16_1.png"
