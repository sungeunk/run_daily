#!/usr/bin/env python3
"""Subprocess runner that streams stdout to a shared raw-log sink.

Design notes
------------
* Streaming: output is read line-by-line so a shared log file can interleave
  live progress from a long-running test. pytest captures stdout per-test, but
  the raw log is a separate, always-open file that survives across tests.
* Encoding: force UTF-8 with errors='ignore' — daily runs often include model
  output that is not ASCII-clean, and we don't want a decode error to mask
  real test failures.
* Timeout: uses Popen.wait(timeout=...); on timeout we kill the process and
  return whatever was collected so far with returncode=-1.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from .fs_utils import is_windows


CmdSpec = Union[str, list[str]]


@dataclass
class CmdResult:
    cmd: str
    returncode: int
    output: str
    duration_sec: float


def _prepare_cmd(cmd: CmdSpec):
    """Normalise `cmd` for ``subprocess.Popen``.

    * A list is passed through unchanged — use this when arguments contain
      spaces (e.g. ``python -c "import foo; foo.bar()"``).
    * A string is split with shlex on POSIX (handles quoted substrings),
      left as a string on Windows where it can be handed to cmd.exe directly.
    """
    if isinstance(cmd, list):
        return cmd
    if is_windows():
        return cmd
    return shlex.split(cmd)


def _render_cmd(cmd: CmdSpec) -> str:
    return cmd if isinstance(cmd, str) else ' '.join(shlex.quote(str(p)) for p in cmd)


def run_cmd(
    cmd: CmdSpec,
    *,
    timeout_sec: int,
    log_sink: Optional[Callable[[str], None]] = None,
) -> CmdResult:
    """Run `cmd`, tee output to `log_sink` line-by-line, return CmdResult."""
    start = time.time()
    lines: list[str] = []

    def _emit(line: str) -> None:
        lines.append(line)
        if log_sink is not None:
            log_sink(line)

    cmd_str = _render_cmd(cmd)
    try:
        proc = subprocess.Popen(
            _prepare_cmd(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            text=True,
            encoding='UTF-8',
            errors='ignore',
            bufsize=1,  # line-buffered
        )
    except OSError as e:
        _emit(f'[cmd_runner] spawn failed: {e}\n')
        return CmdResult(cmd=cmd_str, returncode=-1, output=''.join(lines),
                         duration_sec=time.time() - start)

    reader_done = threading.Event()

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                _emit(line)
        finally:
            reader_done.set()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        _emit(f'[cmd_runner] timeout after {timeout_sec}s, killing\n')
        proc.kill()
        proc.wait()

    reader_done.wait(timeout=5)
    returncode = proc.returncode if proc.returncode is not None else -1

    return CmdResult(
        cmd=cmd_str,
        returncode=returncode,
        output=''.join(lines),
        duration_sec=time.time() - start,
    )
