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

import logging
import os
import platform
import shlex
import subprocess
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
              relay_server: str | None = None) -> bool:
    """Send ``report_path`` as an HTML email to ``recipients``.

    ``recipients`` is the same comma-separated string the old ``--mail``
    flag accepted. Returns True on success.
    """
    if not recipients:
        return False

    full_title = f'[{platform.node()}/{now_stamp}] {title} {suffix_title}'.strip()

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
        cmd = (f'ssh -i "{id_rsa}" {relay} "{remote_cmd}" '
               f'< "{report_path}"')
    else:
        cmd = (f'cat {shlex.quote(str(report_path))} | '
               f'mail --content-type=text/html '
               f'-s {shlex.quote(full_title)} {shlex.quote(recipients)}')

    log.info('send_mail: %s → %s', full_title, recipients)
    return subprocess.call(cmd, shell=True) == 0


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
