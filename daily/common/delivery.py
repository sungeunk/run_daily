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
import re
import shlex
import shutil
import subprocess
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from typing import Any, Iterable

from common.paths import machine_name, relative_dir


log = logging.getLogger(__name__)


# Default target server. The legacy scripts used the same host for both
# publishing (``http://...``) and scp'ing (bare hostname), so we keep one
# source of truth here and derive both from it.
DEFAULT_BACKUP_HOST = 'dg2raptorlake.ikor.intel.com'
DEFAULT_BACKUP_USER = 'sungeunk'

# Sender shown in the mail client. Overridable with ``DAILY_MAIL_FROM``.
DEFAULT_MAIL_FROM = f'jenkins <jenkins@{DEFAULT_BACKUP_HOST}>'

# Remote directory under which every machine's artefacts live. Kept distinct
# from the legacy ``/var/www/html/daily`` path so the new pytest-based pipeline
# can coexist with the old one without mixing files.
REMOTE_BASE_DIR = '/var/www/html/daily2'

_UNSAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._-]+')

_STAGED_IMAGE_RE = re.compile(r'^daily\.\d{8}_\d{3,4}\.image\.(?P<slot>.+)$')

# Scratch files the genai benchmark writes next to the artefacts, e.g.
# ``flux.1-schnell_p0_iter1_pid8196_output.png``.
_GENAI_SCRATCH_RE = re.compile(
    r'_p\d+_(?:iter\d+_)?pid\d+_(?:input\.txt|output\.png)$')


def backup_relative_dir(name: str = '') -> str:
    """Path under the backup root that ``name`` belongs in."""
    return relative_dir(name)


def staged_image_slot(model: str, precision: str, idx: int, suffix: str) -> str:
    """Slot name ``stage_report_images`` gives the ``idx``-th image of a test.

    Deterministic from the summary alone, so a report regenerated later can
    find the staged copy without the original staging run.
    """
    label = _UNSAFE_NAME_RE.sub('_', f'{model}_{precision}').strip('_') or 'image'
    return f'{label}_{idx}{suffix.lower()}'


def stage_report_images(summary: dict, output_dir: Path, stamp: str
                        ) -> dict[str, Path]:
    """Copy generated images to publishable filenames.

    Returns ``{original image_path: staged Path}``. The staged name carries
    the run stamp so it sorts and publishes with the run's other artefacts.
    """
    staged: dict[str, Path] = {}
    for test in summary.get('tests', []):
        metrics = test.get('metrics') or {}
        if metrics.get('test_type') != 'image_generation':
            continue
        for idx, entry in enumerate(metrics.get('data') or []):
            source = entry.get('image_path')
            if not source or source in staged:
                continue
            source_path = Path(source)
            if not source_path.is_file():
                continue
            slot = staged_image_slot(metrics.get('model', ''),
                                     metrics.get('precision', ''),
                                     idx, source_path.suffix)
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f'daily.{stamp}.image.{slot}'
            try:
                shutil.copyfile(source_path, target)
            except OSError as exc:
                log.error('stage image: copy failed for %s: %s', source_path, exc)
                continue
            staged[source] = target
    return staged


def cleanup_genai_scratch(output_dir: Path) -> int:
    """Delete the genai benchmark's raw image/prompt dumps.

    Only safe once the report has embedded its thumbnails and the staged
    copies have been published; callers run this at the end of a run.
    """
    removed = 0
    for path in Path(output_dir).iterdir():
        if not path.is_file() or not _GENAI_SCRATCH_RE.search(path.name):
            continue
        try:
            path.unlink()
        except OSError as exc:
            log.warning('cleanup: could not remove %s: %s', path, exc)
            continue
        removed += 1
    return removed


def image_slot_name(staged_name: str) -> str:
    """The ``<model>_<precision>_<idx>.<ext>`` part of a staged image name.

    Stable across runs, so it pairs a run's image with the baseline's.
    """
    match = _STAGED_IMAGE_RE.match(staged_name)
    return match.group('slot') if match else staged_name


def staged_images_for(root: Path, stamp: str) -> dict[str, Path]:
    """Staged images of the run identified by ``stamp``, keyed by slot name.

    Searches the whole output tree so a baseline from an earlier month is
    still found.
    """
    return {
        image_slot_name(p.name): p
        for p in sorted(Path(root).rglob(f'daily.{stamp}.image.*'))
    }


def _html_report_body(report_path: Path) -> str:
    """Return an HTML body for either a raw HTML report or plain text report."""
    report_text = report_path.read_text(encoding='utf-8')
    if report_path.suffix.lower() == '.html':
        return report_text

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
    bisect_delta = (
        analysis.get('bisect_delta') if isinstance(analysis.get('bisect_delta'), dict) else {}
    )
    functional = analysis.get('functional') if isinstance(analysis.get('functional'), dict) else {}
    performance = analysis.get('performance') if isinstance(analysis.get('performance'), dict) else {}

    baseline_text = 'not found'
    if baseline.get('status') == 'found':
        baseline_text = f"{baseline.get('stamp', '')} / {baseline.get('ov_version', 'unknown')}"

    failed = _safe_int(functional.get('failed', 0))
    error = _safe_int(functional.get('error', 0))
    issue_count = _safe_int(functional.get('issue_count', failed + error))

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

    bisect_lines = ''
    if bisect_delta:
        status = str(bisect_delta.get('status', 'unavailable'))
        if status == 'available':
            issue_ref = (
                f"{bisect_delta.get('issue_stamp', '')} / "
                f"{bisect_delta.get('issue_ov_version', 'unknown')}"
            )
            good_ref = (
                f"{bisect_delta.get('last_good_stamp', '')} / "
                f"{bisect_delta.get('last_good_ov_version', 'unknown')}"
            )
            bisect_lines = (
                f'<li>bisect delta: issue={html.escape(issue_ref)} '
                f'vs last-good={html.escape(good_ref)}</li>'
                f'<li>bisect counts: compared={_safe_int(bisect_delta.get("compared_count", 0))} '
                f'regressed={_safe_int(bisect_delta.get("regressed_count", 0))} '
                f'functional_issues={_safe_int(bisect_delta.get("functional_issue_count", 0))}</li>'
                f'<li>build/sha changed: '
                f'build={html.escape(str(bisect_delta.get("build_changed")))} '
                f'sha={html.escape(str(bisect_delta.get("sha_changed")))}</li>'
            )
        else:
            bisect_lines = '<li>bisect delta: unavailable</li>'

    # Keep this intentionally compact so the full report still remains the source of detail.
    return (
        '<div style="margin-bottom:12px">'
        '<strong>Analysis summary</strong>'
        '<ul style="margin:6px 0 0 18px;padding:0">'
        f'<li>overall: {html.escape(overall)}</li>'
        f'<li>baseline: {html.escape(str(baseline_text))}</li>'
        f'{lkg_line}'
        f'{bisect_lines}'
        f'<li>functional: issues={issue_count} failed={failed} error={error}</li>'
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


def _build_html_email_message(recipients: str, subject: str, html_body: str) -> bytes:
    """Build a UTF-8 multipart email (plain + html) as RFC-compliant bytes."""
    msg = EmailMessage()
    msg['To'] = recipients
    msg['Subject'] = subject
    msg['Date'] = formatdate(localtime=True)

    # Without an explicit From, the relay stamps the SSH account and Outlook
    # resolves it to that person's GAL display name instead of the job's.
    sender = os.environ.get('DAILY_MAIL_FROM', '').strip() or DEFAULT_MAIL_FROM
    msg['From'] = sender

    msg.set_content(
        'This message contains an HTML report. '
        'Please use an HTML-capable email client to view the formatted content.',
        charset='utf-8',
    )
    msg.add_alternative(html_body, subtype='html', charset='utf-8')
    return msg.as_bytes()


def backup_server_url(base_url: str | None = None, filename: str = '') -> str:
    """Return the public URL for a backed-up artefact.

    The scp target is ``<host>:/var/www/html/daily2/<node>/<YYYY.MM>/`` and the
    relay exposes it at ``http://<host>/daily2/<node>/<YYYY.MM>/<file>``.
    """
    if base_url is None:
        base_url = f'http://{_resolve_host(None)}'
    return (f'{base_url.rstrip("/")}/daily2/'
            f'{backup_relative_dir(filename)}/{filename}')


def jenkins_console_url() -> str:
    """This build's Jenkins console, or '' when not running under Jenkins.

    The console is not copied to the relay: it is only complete after the job
    finishes, which is after this process has already published everything.
    Jenkins log rotation eventually removes it.
    """
    build_url = os.environ.get('BUILD_URL', '').strip()
    return f'{build_url.rstrip("/")}/consoleText' if build_url else ''


def artefact_links(files: Iterable[Path], *, base_url: str | None = None
                   ) -> list[tuple[str, str]]:
    """Return ``(label, url)`` pairs for artefacts published on the relay.

    Labels are derived from the filename suffix so the report header reads
    ``html`` / ``raw log`` rather than the full stamped filename.
    Ordered most-useful-first; unknown suffixes keep their extension as the
    label and sort last.
    """
    label_order = {
        'html': (1, 'html'),
        'console.log': (2, 'console'),
        'raw': (3, 'raw log'),
        'summary.json': (4, 'summary json'),
        'pytest.json': (5, 'pytest json'),
        'requirements.txt': (6, 'requirements'),
        'monitor.parquet': (7, 'monitor data'),
    }

    def _classify(name: str) -> tuple[int, str]:
        # Match the longest compound suffix first ('summary.json' before 'json').
        for suffix, entry in sorted(label_order.items(), key=lambda kv: -len(kv[0])):
            if name.endswith(f'.{suffix}'):
                return entry
        return (99, Path(name).suffix.lstrip('.') or 'file')

    entries = []
    for f in files:
        name = Path(f).name
        rank, label = _classify(name)
        entries.append((rank, label, backup_server_url(base_url, name)))

    if console_url := jenkins_console_url():
        entries.append((*label_order['console.log'], console_url))

    return [(label, url) for _, label, url in sorted(entries, key=lambda e: (e[0], e[1]))]


def render_links_block(files: Iterable[Path], *, base_url: str | None = None
                       ) -> str:
    """Return the ``[ Links ]`` text block for the top of the daily report.

    Empty string when there is nothing to link, so callers can prepend
    unconditionally.
    """
    links = artefact_links(files, base_url=base_url)
    if not links:
        return ''

    width = max(len(label) for label, _ in links)
    lines = ['[ Links ]']
    lines.extend(f'- {label.ljust(width)}  {url}' for label, url in links)
    return '\n'.join(lines) + '\n'


def _open_ssh_client(relay: str, username: str = DEFAULT_BACKUP_USER) -> Any:
    """Open an SSH client using the user's agent or default key files."""
    import paramiko

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        relay,
        username=username,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        allow_agent=True,
        look_for_keys=True,
    )
    return client


def _ensure_remote_directory(sftp: Any, remote_dir: str) -> None:
    """Create the remote directory when it does not already exist.

    Walks the path so a machine's first run of the month creates both the
    machine and the ``YYYY.MM`` level.
    """
    parts = remote_dir.strip('/').split('/')
    for depth in range(1, len(parts) + 1):
        path = '/' + '/'.join(parts[:depth])
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def prepend_links_html(html_path: Path, links_block: str) -> None:
    """Insert the same links into the HTML report as a small link list.

    Parses the ``- label  url`` lines back out of *links_block* so the text
    and HTML reports can't drift apart.
    """
    items = []
    for line in links_block.splitlines():
        line = line.strip()
        if not line.startswith('- '):
            continue
        label, _, url = line[2:].rpartition('  ')
        label, url = label.strip(), url.strip()
        if not url:
            continue
        items.append(
            f'<li><a href="{html.escape(url, quote=True)}">'
            f'{html.escape(label or url)}</a></li>'
        )
    if not items:
        return

    block = (
        '<div style="margin:0 0 14px;font-family:Consolas,Monaco,monospace;'
        'font-size:13px">'
        '<strong>Links</strong>'
        f'<ul style="margin:6px 0 0 18px;padding:0">{"".join(items)}</ul>'
        '</div>'
    )

    current = html_path.read_text(encoding='utf-8')
    marker = '<body>'
    idx = current.find(marker)
    if idx == -1:
        html_path.write_text(block + current, encoding='utf-8')
        return
    insert_at = idx + len(marker)
    html_path.write_text(
        current[:insert_at] + block + current[insert_at:], encoding='utf-8'
    )


def scp_backup(files: Iterable[Path], *, relay_server: str | None = None
               ) -> list[Path]:
    """Copy ``files`` to the backup server via Paramiko SFTP.

    Returns the list of files that were successfully uploaded.

    Host precedence: ``relay_server`` arg → ``MAIL_RELAY_SERVER`` env
    → ``DEFAULT_BACKUP_HOST``. The remote directory is created on demand so
    new hosts don't need manual setup.
    """
    relay = _resolve_host(relay_server)

    try:
        import paramiko
    except ImportError:
        log.error('backup: paramiko is required for Python-based SSH backup')
        return []

    try:
        with _open_ssh_client(relay, DEFAULT_BACKUP_USER) as client:
            with client.open_sftp() as sftp:
                uploaded: list[Path] = []
                ensured: set[str] = set()
                for f in files:
                    f = Path(f)
                    if not f.exists():
                        log.error('backup: missing %s', f)
                        continue

                    remote_dir = f'{REMOTE_BASE_DIR}/{backup_relative_dir(f.name)}'
                    if remote_dir not in ensured:
                        try:
                            _ensure_remote_directory(sftp, remote_dir)
                        except (OSError, paramiko.SFTPError) as exc:
                            log.error(
                                'backup: remote directory %s is unavailable on %s: %s',
                                remote_dir, relay, exc,
                            )
                            continue
                        ensured.add(remote_dir)

                    remote_path = f'{remote_dir}/{f.name}'
                    log.info('backup: %s -> %s:%s/', f.name, relay, remote_dir)
                    try:
                        sftp.put(str(f), remote_path)
                    except (OSError, paramiko.SFTPError) as exc:
                        log.error('backup: upload failed for %s: %s', f, exc)
                        continue
                    uploaded.append(f)
                return uploaded
    except (OSError, paramiko.SSHException) as exc:
        log.error('backup: SSH connection to %s failed: %s', relay, exc)
        return []


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

    full_title = f'[{machine_name()}/{now_stamp}] {title} {suffix_title}'.strip()
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
        # Remote send via ssh + sendmail so MIME headers/charset are preserved.
        raw_message = _build_html_email_message(recipients, full_title, body)
        cmd = [
            'ssh',
            '-i',
            str(id_rsa),
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=20',
            f'{DEFAULT_BACKUP_USER}@{relay}',
            '/usr/sbin/sendmail -t -oi',
        ]
        log.info('send_mail: %s → %s', full_title, recipients)
        try:
            result = subprocess.run(cmd, input=raw_message, timeout=60)
            if result.returncode == 0:
                return True
            log.warning('send_mail: sendmail path failed (rc=%d), falling back to mail(1)', result.returncode)
        except subprocess.TimeoutExpired:
            log.warning('send_mail: sendmail path timed out on %s, falling back to mail(1)', relay)

        # Fallback: legacy remote mail command. List-based (no shell=True) so
        # the title/recipients never pass through a second, local shell's
        # metacharacter parsing on top of the remote shlex.quote()-ing below.
        remote_cmd = (
            f'mail --content-type=text/html -s {shlex.quote(full_title)} '
            f'{shlex.quote(recipients)}'
        )
        cmd = [
            'ssh',
            '-i',
            str(id_rsa),
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=20',
            f'{DEFAULT_BACKUP_USER}@{relay}',
            remote_cmd,
        ]
        try:
            result = subprocess.run(cmd, input=body.encode('utf-8'), timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.error('send_mail: fallback mail(1) path timed out on %s', relay)
            return False
    else:
        cmd = [
            'mail',
            '--content-type=text/html; charset=UTF-8',
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
    """Produce the ``(geomean/success series/failed series)`` suffix used in
    mail subjects.

    Consumes the report builder's summary dict so we avoid reparsing JSON.
    Series counts come from each test's ``expected_series`` — the same field
    ``analysis.report._series_counts`` uses — rather than pytest's
    test-function counts, since one test function can stand for several
    benchmark cases (e.g. one LLM test covers N prompts x 1st/2nd, so "2
    passed" would hide that only 4 of the expected series actually landed).
    """
    from statistics import geometric_mean

    values: list[float] = []
    success_series = 0
    failed_series = 0
    for t in summary.get('tests', []):
        m = t.get('metrics', {})
        expected = int(m.get('expected_series') or 0)
        outcome = t.get('outcome')
        if outcome == 'passed':
            success_series += expected
            if m.get('test_type') == 'llm_benchmark':
                for d in m.get('data', []):
                    perf = d.get('perf') or []
                    # 1st-inference latency — mirrors the old geomean input.
                    if perf and isinstance(perf[0], (int, float)):
                        values.append(float(perf[0]))
        elif outcome in ('failed', 'error'):
            failed_series += expected

    geomean = geometric_mean(values) if values else 0.0
    return f'({geomean:.2f}/{success_series}/{failed_series})'
