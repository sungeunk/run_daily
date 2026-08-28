"""Shared helpers for the daily_results GNAI toolkit scripts.

Each tool script is invoked as a standalone process by `gnai toolkits serve`,
so this module wires up sys.path to reach `daily/viewer/queries.py` (the
same query layer the Streamlit viewer uses) and provides JSON-safe output.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from pathlib import Path

DAILY_DIR = Path("/home/sungeunk/repo/run_daily/daily")
DB_PATH = Path("/var/www/html/daily2/daily_llm_benchmark.duckdb")

if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

from viewer import queries  # noqa: E402


def _sanitize(value):
    """Recursively replace NaN/Inf and non-JSON-native scalars so the
    output is strict JSON (json.dumps silently emits the invalid NaN/
    Infinity tokens otherwise, which not every MCP client can parse)."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def emit(obj) -> None:
    print(json.dumps(_sanitize(obj), default=str, indent=2))


def input_str(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"GNAI_INPUT_{name}", default) or default


def input_int(name: str, default: int) -> int:
    raw = os.environ.get(f"GNAI_INPUT_{name}")
    return int(raw) if raw not in (None, "") else default


def df_records(df) -> list[dict]:
    return df.to_dict(orient="records")
