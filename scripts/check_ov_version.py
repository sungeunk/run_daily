#!/usr/bin/env python3
"""Verify that the installed OpenVINO / GenAI packages report a usable version.

Must run with setupvars already sourced: openvino_genai resolves its native
libraries through `import openvino` at import time.
"""

from __future__ import annotations

import sys


def report(module_name: str) -> str | None:
    """Returns the version string, or None if the module is unusable."""
    try:
        module = __import__(module_name)
    except Exception as e:
        print(f'{module_name}: import failed: {e}')
        return None

    try:
        version = module.get_version()
    except Exception as e:
        print(f'{module_name}: get_version() failed: {e}')
        return None

    if not version or not str(version).strip():
        print(f'{module_name}: empty version')
        return None

    print(f'{module_name}: {version}')
    return str(version).strip()


def main() -> int:
    versions = [report(name) for name in ('openvino', 'openvino_genai')]
    if any(v is None for v in versions):
        print('ERROR: OpenVINO / GenAI version check failed')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
