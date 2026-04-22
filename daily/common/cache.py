#!/usr/bin/env python3
"""Cache cleanup used before/after each test to prevent cross-test leakage."""

from __future__ import annotations

from glob import glob
from pathlib import Path

from .fs_utils import convert_path, exists_path, force_delete_file


CACHE_PATTERNS = ('*.cl_cache', '*.blob', '*.png')


def clear_caches(cache_dir: Path, model_dir: Path) -> int:
    """Remove generated cache files. Returns number of files deleted."""
    targets = [
        convert_path(str(cache_dir)),
        convert_path(f'{model_dir}/stable-diffusion-v3.0/transformer/model_cache'),
        convert_path(f'{model_dir}/stable-diffusion-xl/unet/model_cache'),
    ]

    deleted = 0
    for directory in targets:
        if not exists_path(directory):
            continue
        for pattern in CACHE_PATTERNS:
            for file in glob(convert_path(f'{directory}/{pattern}')):
                if force_delete_file(file):
                    deleted += 1
    return deleted
