"""Newest runs for a machine (metadata: ts, ov_version, purpose, pass/fail)."""
from _common import DB_PATH, emit, input_str, input_int, df_records
from viewer import queries

machine = input_str("MACHINE")
limit = input_int("LIMIT", 10)

df = queries.recent_runs(DB_PATH, machine, limit=limit)
emit(df_records(df))
