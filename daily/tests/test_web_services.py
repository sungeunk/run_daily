"""Check that service registration is independent of the checkout location."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("checkout_name", ["run_daily", 'relocated repo%"name'])
def test_service_install_uses_checkout_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkout_name: str
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo_root = tmp_path / checkout_name
    web_root = repo_root / "web_server"
    web_root.mkdir(parents=True)
    shutil.copy2(source_root / "web_server/manage-web-services.sh", web_root)
    shutil.copytree(source_root / "web_server/caddy/systemd", web_root / "caddy/systemd")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    systemctl = bin_dir / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o755)
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    subprocess.run(
        ["bash", str(web_root / "manage-web-services.sh"), "install"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    escaped_root = str(repo_root).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    for service in ("caddy.service", "daily-viewer.service", "daily-results-mcp.service"):
        unit = config_home / "systemd/user" / service
        assert unit.resolve() == web_root / "caddy/systemd" / service
        drop_in = unit.parent / f"{service}.d/10-run-daily-root.conf"
        assert drop_in.read_text(encoding="utf-8") == (
            f'[Service]\nWorkingDirectory="{escaped_root}"\n'
        )


def test_ingest_uses_relocated_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo_root = tmp_path / "relocated checkout"
    scripts = repo_root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(source_root / "scripts/ingest_db.sh", scripts)
    requirements = repo_root / "daily/requirements.txt"
    requirements.parent.mkdir()
    requirements.touch()
    launcher = tmp_path / "uv"
    launcher.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$PYTHONPATH" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("UV_BIN", str(launcher))
    monkeypatch.setenv("PYTHONPATH", "existing/modules")
    monkeypatch.delenv("REQUIREMENTS_FILE", raising=False)

    result = subprocess.run(
        ["bash", str(scripts / "ingest_db.sh"), "--force"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert lines[:6] == [
        f"existing/modules:{repo_root / 'daily'}",
        "run", "--with-requirements", str(requirements), "python", "-u",
    ]
    assert lines[-1] == "--force"