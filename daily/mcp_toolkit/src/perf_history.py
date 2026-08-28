"""Time series for one exact perf point (model/precision/tokens/exec_mode on
one machine), with rolling median/MAD baseline, z-score and pct-diff."""
from _common import DB_PATH, emit, input_str, input_int, df_records
from viewer import queries

machine = input_str("MACHINE")
model = input_str("MODEL")
precision = input_str("PRECISION")
in_token = input_int("IN_TOKEN", 0)
out_token = input_int("OUT_TOKEN", 0)
exec_mode = input_str("EXEC_MODE")
days = input_int("DAYS", 60)

df = queries.series_history(
    DB_PATH, machine, model, precision, in_token, out_token, exec_mode, days=days
)
emit(df_records(df))
