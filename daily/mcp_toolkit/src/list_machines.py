"""List all machines that have daily benchmark results, with their most
recent run's timestamp, OpenVINO version and pass/fail counts."""
from _common import DB_PATH, emit
from viewer.queries import _read_only

with _read_only(DB_PATH) as con:
    rows = con.execute(
        """
        SELECT r.machine,
               count(*) AS total_runs,
               max(r.ts) AS latest_run_ts,
               arg_max(r.ov_version, r.ts) AS latest_ov_version,
               arg_max(r.passed_tests, r.ts) AS latest_passed_tests,
               arg_max(r.failed_tests, r.ts) AS latest_failed_tests,
               arg_max(r.total_tests, r.ts) AS latest_total_tests
        FROM runs r
        GROUP BY r.machine
        ORDER BY r.machine
        """
    ).fetchdf()

emit(rows.to_dict(orient="records"))
