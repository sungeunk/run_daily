#!/usr/bin/env python3
"""Post-run delivery: scp backup + mail send.

Ported from ``scripts/common_utils.backup_files`` / ``send_mail``.

Two upstream bugs are corrected during the port:

* The old ``backup_files`` only ran scp on Windows (``if is_windows():`` with
  no else branch), which silently skipped backup on every Linux rig.
* ``send_mail`` shelled out via ``shell=True`` with an unquoted subject —
  fine for fixed strings, but we now quote with shlex to be safe.
"""

from __future__ import annotations

import html
import json
import logging
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


log = logging.getLogger(__name__)


# Default target server. The legacy scripts used the same host for both
# publishing (``http://...``) and scp'ing (bare hostname), so we keep one
# source of truth here and derive both from it.
DEFAULT_BACKUP_HOST = 'dg2raptorlake.ikor.intel.com'

# Remote directory under which every machine's artefacts live. Kept distinct
# from the legacy ``/var/www/html/daily`` path so the new pytest-based pipeline
# can coexist with the old one without mixing files.
REMOTE_BASE_DIR = '/var/www/html/daily2'


def _html_report_body(report_path: Path) -> str:
    """Return an HTML body that preserves the text report formatting."""
    report_text = report_path.read_text(encoding='utf-8')
    escaped = html.escape(report_text)
    return (
        '<html><body>'
        '<pre style="font-family:Consolas,Monaco,monospace;'
        'white-space:pre-wrap;line-height:1.35;margin:0">'
        f'{escaped}'
        '</pre>'
        '</body></html>'
    )


def _html_analysis_summary_block(summary_json: Path) -> str:
    """Return a small HTML summary from ``summary.json`` analysis block."""
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    try:
        payload = json.loads(summary_json.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ''

    if not isinstance(payload, dict):
        return ''

    analysis = payload.get('analysis')
    if not isinstance(analysis, dict):
        return ''

    overall = str(analysis.get('overall_status', 'unknown'))
    baseline = analysis.get('baseline') if isinstance(analysis.get('baseline'), dict) else {}
    last_known_good = (
        analysis.get('last_known_good') if isinstance(analysis.get('last_known_good'), dict) else {}
    )
    functional = analysis.get('functional') if isinstance(analysis.get('functional'), dict) else {}
    performance = analysis.get('performance') if isinstance(analysis.get('performance'), dict) else {}

    baseline_text = 'not found'
    if baseline.get('status') == 'found':
        baseline_text = f"{baseline.get('stamp', '')} / {baseline.get('ov_version', 'unknown')}"

    lkg_text = None
    if last_known_good:
        lkg_text = 'not found'
    if last_known_good.get('status') == 'found':
        lkg_text = (
            f"{last_known_good.get('stamp', '')} / "
            f"{last_known_good.get('ov_version', 'unknown')}"
        )

    lkg_line = ''
    if lkg_text is not None:
        lkg_line = f'<li>last known good: {html.escape(str(lkg_text))}</li>'

    # Keep this intentionally compact so the full report still remains the source of detail.
    return (
        '<div style="margin-bottom:12px">'
        '<strong>Analysis summary</strong>'
        '<ul style="margin:6px 0 0 18px;padding:0">'
        f'<li>overall: {html.escape(overall)}</li>'
        f'<li>baseline: {html.escape(str(baseline_text))}</li>'
        f'{lkg_line}'
        f'<li>functional: failed={_safe_int(functional.get("failed", 0))} '
        f'error={_safe_int(functional.get("error", 0))}</li>'
        f'<li>performance: compared={_safe_int(performance.get("compared", 0))} '
        f'regressed={_safe_int(performance.get("regressed", 0))}</li>'
        '</ul>'
        '</div>'
    )


def _resolve_host(relay_server: str | None) -> str:
    """Pick the scp target host.

    Precedence: explicit arg → ``MAIL_RELAY_SERVER`` env → ``DEFAULT_BACKUP_HOST``.
    """
    return relay_server or os.environ.get('MAIL_RELAY_SERVER') or DEFAULT_BACKUP_HOST


def backup_server_url(base_url: str | None = None, filename: str = '') -> str:
    """Return the public URL for a backed-up artefact.

    The scp target is ``<host>:/var/www/html/daily2/<node>/`` and the relay
    exposes it at ``http://<host>/daily2/<node>/<file>``.
    """
    if base_url is None:
        base_url = f'http://{_resolve_host(None)}'
    return f'{base_url.rstrip("/")}/daily2/{platform.node()}/{filename}'


def scp_backup(files: Iterable[Path], *, relay_server: str | None = None
               ) -> list[Path]:
    """Copy ``files`` to the backup server via scp.

    Returns the list of files that were successfully uploaded.

    Host precedence: ``relay_server`` arg → ``MAIL_RELAY_SERVER`` env
    → ``DEFAULT_BACKUP_HOST``. The remote directory is created on demand so
    new hosts don't need manual setup.
    """
    relay = _resolve_host(relay_server)
    remote_dir = f'{REMOTE_BASE_DIR}/{platform.node()}'
    remote = f'{relay}:{remote_dir}/'

    is_windows = platform.system() == 'Windows'
    scp_bin = 'scp.exe' if is_windows else 'scp'
    ssh_bin = 'ssh.exe' if is_windows else 'ssh'

    # The legacy ``daily2/`` is owned ``root:www-data`` with 775, so mkdir
    # only works if the ssh user is in ``www-data``. Probe first; if the dir
    # is missing *and* we can't create it, surface an actionable error so
    # the admin knows what to fix rather than watching scp fail mysteriously.
    probe = subprocess.run(
        [ssh_bin, relay, f'test -d {remote_dir} || mkdir -p {remote_dir}'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        log.error(
            'backup: remote dir %s missing and mkdir failed (rc=%d). '
            'Add the ssh user to the www-data group on %s, or have an admin '
            'pre-create %s. ssh stderr: %s',
            remote_dir, probe.returncode, relay, remote_dir,
            (probe.stderr or '').strip(),
        )
        return []

    uploaded: list[Path] = []
    for f in files:
        f = Path(f)
        if not f.exists():
            log.error('backup: missing %s', f)
            continue
        cmd = [scp_bin, str(f), remote]
        log.info('backup: %s → %s', f.name, remote)
        rc = subprocess.call(cmd)
        if rc == 0:
            uploaded.append(f)
        else:
            log.error('backup: scp failed for %s (rc=%d)', f, rc)
    return uploaded


def send_mail(report_path: Path, recipients: str, title: str, *,
              suffix_title: str = '', now_stamp: str = '',
              summary_json: Path | None = None,
              relay_server: str | None = None) -> bool:
    """Send ``report_path`` as an HTML email to ``recipients``.

    ``recipients`` is the same comma-separated string the old ``--mail``
    flag accepted. Returns True on success.
    """
    if not recipients:
        return False

    full_title = f'[{platform.node()}/{now_stamp}] {title} {suffix_title}'.strip()
    analysis_block = _html_analysis_summary_block(summary_json) if summary_json else ''
    body = _html_report_body(report_path)
    if analysis_block:
        body = body.replace('<html><body>', f'<html><body>{analysis_block}', 1)

    if platform.system() == 'Windows':
        user_profile = os.environ.get('USERPROFILE')
        if not user_profile:
            log.error('send_mail: USERPROFILE env not set')
            return False
        id_rsa = Path(user_profile) / '.ssh' / 'id_rsa'
        relay = _resolve_host(relay_server)
        # Remote mail via ssh — the Windows build doesn't have ``mail(1)``.
        quoted_title = shlex.quote(full_title)
        quoted_to = shlex.quote(recipients)
        remote_cmd = (f'mail --content-type=text/html -s {quoted_title} '
                      f'{quoted_to}')
        with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                         suffix='.html', delete=False) as tmp:
            tmp.write(body)
            body_file = Path(tmp.name)
        cmd = (f'ssh -i "{id_rsa}" {relay} "{remote_cmd}" '
               f'< "{body_file}"')
        log.info('send_mail: %s → %s', full_title, recipients)
        try:
            return subprocess.call(cmd, shell=True) == 0
        finally:
            body_file.unlink(missing_ok=True)
    else:
        cmd = [
            'mail',
            '--content-type=text/html',
            '-s',
            full_title,
            recipients,
        ]

    log.info('send_mail: %s → %s', full_title, recipients)
    result = subprocess.run(cmd, input=body, text=True)
    return result.returncode == 0


def write_pip_freeze(output_path: Path) -> None:
    """Capture installed packages for build reproducibility."""
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    output_path.write_text(result.stdout, encoding='utf-8')


def mail_title_suffix(summary: dict) -> str:
    """Produce the ``(geomean/passed)`` suffix used in mail subjects.

    Consumes the report builder's summary dict so we avoid reparsing JSON.
    """
    from statistics import geometric_mean

    values: list[float] = []
    for t in summary.get('tests', []):
        if t.get('outcome') != 'passed':
            continue
        m = t.get('metrics', {})
        if m.get('test_type') == 'llm_benchmark':
            for d in m.get('data', []):
                perf = d.get('perf') or []
                # 1st-inference latency — mirrors the old geomean input.
                if perf and isinstance(perf[0], (int, float)):
                    values.append(float(perf[0]))

    geomean = geometric_mean(values) if values else 0.0
    passed = summary.get('totals', {}).get('passed', 0)
    return f'({geomean:.2f}/{passed})'
