from __future__ import annotations

import pytest

from common.delivery import DEFAULT_BACKUP_HOST, REMOTE_BASE_DIR, backup_server_url

pytestmark = pytest.mark.dev_only


def test_default_backup_targets_dg2fizz_file_browser(monkeypatch) -> None:
    monkeypatch.delenv("MAIL_RELAY_SERVER", raising=False)

    assert DEFAULT_BACKUP_HOST == "dg2fizz.ikor.intel.com"
    assert REMOTE_BASE_DIR == "/mnt/hdd/daily/data"
    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://dg2fizz.ikor.intel.com:8081/daily2/"
    )


def test_backup_url_honors_relay_override(monkeypatch) -> None:
    monkeypatch.setenv("MAIL_RELAY_SERVER", "reports.example.com")

    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://reports.example.com:8081/daily2/"
    )


def test_backup_url_honors_public_origin_override(monkeypatch) -> None:
    monkeypatch.setenv("DAILY_REPORT_BASE_URL", "https://reports.example.com/results")

    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "https://reports.example.com/results/daily2/"
    )


def test_relay_rejects_ssh_destination(monkeypatch) -> None:
    monkeypatch.setenv("MAIL_RELAY_SERVER", "user@reports.example.com")

    with pytest.raises(ValueError, match="bare hostname"):
        backup_server_url(filename="daily.20260901_1143.raw")
