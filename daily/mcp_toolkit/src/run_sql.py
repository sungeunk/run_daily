"""Ad-hoc read-only SQL escape hatch for the daily benchmark DuckDB.

Safety:
- The connection itself is opened read_only=True, so DuckDB physically
  refuses any write regardless of the SQL text.
- On top of that, as defense in depth, we reject anything that isn't a
  single SELECT/WITH statement, and reject statement/session-control
  keywords that read-only mode doesn't already block on their own.
"""
import re
import sys

from _common import DB_PATH, emit, input_str, df_records
from viewer.queries import _read_only

BLOCKED_KEYWORDS = (
    "ATTACH", "DETACH", "COPY", "PRAGMA", "INSTALL", "LOAD",
    "CALL", "EXPORT", "IMPORT", "SET", "CREATE", "INSERT",
    "UPDATE", "DELETE", "DROP", "ALTER",
)


def validate(sql: str) -> str:
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT/WITH (read-only) statements are allowed.")
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", stripped, re.IGNORECASE):
            raise ValueError(f"Statement contains a disallowed keyword: {kw}")
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped += " LIMIT 500"
    return stripped


sql = input_str("SQL")
try:
    safe_sql = validate(sql)
except ValueError as exc:
    print(f"Rejected: {exc}", file=sys.stderr)
    sys.exit(1)

with _read_only(DB_PATH) as con:
    df = con.execute(safe_sql).fetchdf()

emit(df_records(df))
