"""Per-series regression signal: median(last recent_days) vs
median(baseline_days before that), for every series on one machine."""
from _common import DB_PATH, emit, input_str, input_int, df_records
from viewer import queries

machine = input_str("MACHINE")
recent_days = input_int("RECENT_DAYS", 7)
baseline_days = input_int("BASELINE_DAYS", 21)

df = queries.trend_regressions(
    DB_PATH, machine, recent_days=recent_days, baseline_days=baseline_days
)
emit(df_records(df))
