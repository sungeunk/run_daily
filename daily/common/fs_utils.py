#!/usr/bin/env python3
"""Filesystem helpers, platform-aware path handling."""

import os
import platform
import stat


def is_windows() -> bool:
    return platform.system() == 'Windows'


def convert_path(path) -> str:
    path = str(path)
    if is_windows():
        return path.replace('/', '\\')
    return path.replace('\\', '/')


def exists_path(path) -> bool:
    try:
        return os.path.exists(path)
    except OSError:
        return False


def force_delete_file(file_path) -> bool:
    """Delete a file, clearing the read-only bit on Windows if needed."""
    if not os.path.exists(file_path):
        return True

    try:
        os.remove(file_path)
        return True
    except PermissionError:
        try:
            os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IWRITE)
            os.remove(file_path)
            return True
        except OSError:
            return False
    except OSError:
        return False
