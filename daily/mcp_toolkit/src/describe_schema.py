"""List tables and columns in the daily benchmark DuckDB, for building
follow-up queries (e.g. with daily_run_sql)."""
from _common import DB_PATH, emit, df_records
from viewer.queries import _read_only

with _read_only(DB_PATH) as con:
    df = con.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchdf()

emit(df_records(df))
