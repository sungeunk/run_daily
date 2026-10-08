from __future__ import annotations

from pathlib import Path

import pytest

from common.delivery import (
    DEFAULT_BACKUP_HOST, REMOTE_BASE_DIR, backup_server_url, stage_report_images,
)

pytestmark = pytest.mark.dev_only


@pytest.mark.parametrize("fail_publish", [False, True])
def test_backup_publishes_atomically(tmp_path: Path, monkeypatch, fail_publish: bool) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys

    from common import delivery

    path = tmp_path / "daily.20261008_1200.summary.json"
    path.write_text("{}")
    events = []

    def put(local, remote):
        events.append(("put", local, remote))

    def rename(source, destination):
        events.append(("rename", source, destination))
        if fail_publish:
            raise OSError("publish failed")

    sftp = SimpleNamespace(put=put, posix_rename=rename,
                           remove=lambda remote: events.append(("remove", remote)))
    client = SimpleNamespace(open_sftp=lambda: nullcontext(sftp))
    monkeypatch.setitem(sys.modules, "paramiko", SimpleNamespace(SFTPError=OSError, SSHException=OSError))
    monkeypatch.setattr(delivery, "_open_ssh_client", lambda *args: nullcontext(client))
    monkeypatch.setattr(delivery, "_ensure_remote_directory", lambda *args: None)

    uploaded = delivery.scp_backup([path], relay_server="reports.example.com")

    assert events[0][0] == "put"
    assert events[0][2].endswith(".upload")
    assert events[1] == ("rename", events[0][2], events[0][2].rsplit(".", 2)[0])
    assert uploaded == ([] if fail_publish else [path])
    if fail_publish:
        assert events[2] == ("remove", events[0][2])


def test_default_backup_targets_dg2fizz_file_browser(monkeypatch) -> None:
    monkeypatch.delenv("MAIL_RELAY_SERVER", raising=False)

    assert DEFAULT_BACKUP_HOST == "dg2fizz.ikor.intel.com"
    assert REMOTE_BASE_DIR == "/mnt/hdd/daily/data"
    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://dg2fizz.ikor.intel.com:8081/daily/"
    )


def test_mail_relay_override_does_not_change_public_url(monkeypatch) -> None:
    monkeypatch.setenv("MAIL_RELAY_SERVER", "reports.example.com")

    assert backup_server_url(filename="daily.20260901_1143.raw").startswith(
        "http://dg2fizz.ikor.intel.com:8081/daily/"
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


def test_email_preserves_visible_quality_previews(tmp_path: Path) -> None:
    from common.delivery import _html_report_body

    report = tmp_path / "daily.20261008_1310.html"
    original = (
        '<!doctype html>\n<html><body><h2>Output Quality Checks</h2>\n'
        '<div class="quality-preview"><b>Current</b><pre>generated &lt;output&gt;</pre></div>'
        '<div class="quality-preview"><b>Baseline</b><img src="data:image/jpeg;base64,cHJldmlldw==" /></div>'
        '</body></html>'
    )
    report.write_text(original, encoding="utf-8")

    body = _html_report_body(report)

    assert body == original
    assert report.read_text(encoding="utf-8") == original


def test_email_embeds_thumbnail_as_related_mime_part() -> None:
    from email import policy
    from email.parser import BytesParser
    from common.delivery import _build_html_email_message

    body = '<html><body><pre>Text excerpt</pre><img src="data:image/jpeg;base64,cHJldmlldw==" width="128" /><img src="data:image/jpeg;base64,cHJldmlldw==" /></body></html>'
    message = BytesParser(policy=policy.default).parsebytes(
        _build_html_email_message("recipient@example.com", "test", body),
    )
    html_body = message.get_body(preferencelist=("html",)).get_content()
    images = [part for part in message.walk() if part.get_content_type() == "image/jpeg"]

    assert len(images) == 1
    assert images[0].get_payload(decode=True) == b"preview"
    assert images[0].get_content_disposition() == "inline"
    assert html_body.count(f'cid:{images[0]["Content-ID"][1:-1]}') == 2
    assert "data:image" not in html_body
    assert '<pre>Text excerpt</pre>' in html_body


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_sendmail_sends_inline_previews_as_mime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    from types import SimpleNamespace
    from common.delivery import send_mail

    path = tmp_path / "report.html"
    path.write_text('<img src="data:image/jpeg;base64,cHJldmlldw==" />', encoding="utf-8")
    calls = []
    monkeypatch.setattr("common.delivery.platform.system", lambda: system)
    monkeypatch.setattr("common.delivery.shutil.which", lambda _: "/usr/sbin/sendmail")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("common.delivery.subprocess.run", capture)

    assert send_mail(path, "recipient@example.com", "test")
    if system == "Windows":
        assert calls[0][0][0] == "ssh"
        assert calls[0][0][-1] == "/usr/sbin/sendmail -t -oi"
    else:
        assert calls[0][0] == ["/usr/sbin/sendmail", "-t", "-oi"]
    assert b"multipart/related" in calls[0][1]["input"]
    assert b"Content-Type: image/jpeg" in calls[0][1]["input"]


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_inline_image_delivery_failure_does_not_send_broken_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    from common.delivery import send_mail

    path = tmp_path / "report.html"
    path.write_text('<img src="data:image/jpeg;base64,cHJldmlldw==" />', encoding="utf-8")
    calls = []
    monkeypatch.setattr("common.delivery.platform.system", lambda: system)
    monkeypatch.setattr("common.delivery.shutil.which", lambda _: None)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def fail(command, **kwargs):
        calls.append(command)
        raise OSError("sendmail unavailable")

    monkeypatch.setattr("common.delivery.subprocess.run", fail)

    assert not send_mail(path, "recipient@example.com", "test")
    assert len(calls) == 1
