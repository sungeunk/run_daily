# Daily LLM Viewer

Pipeline: `/var/www/html/daily/<MACHINE>/daily.*.{summary.json|pickle,.report}`
→ `ingest/cli.py` → `bench.duckdb` → `streamlit run app.py`.

## Layout

```
viewer/
├── schema.sql                  # DuckDB tables + perf_with_buckets / perf_stats views
├── profiles/default.yaml       # Row order for the Excel-paste tab
├── ingest/
│   ├── record.py               # RunRecord dataclass (format-neutral)
│   ├── writer.py               # DuckDB upsert + display profile loader
│   ├── loader_new.py           # summary.json (new format)
│   ├── loader_old.py           # .report + .pickle (legacy)
│   └── cli.py                  # `python -m viewer.ingest.cli`
├── queries.py                  # Streamlit-side DuckDB helpers
└── app.py                      # Streamlit entry
```

## Ingest

```bash
# Auto-detect under a tree (handles both formats)
python -m viewer.ingest.cli --root /var/www/html/daily --db viewer/bench.duckdb

# Single file
python -m viewer.ingest.cli --input /var/www/html/daily/LNL-02/daily.20250722_0136.2025.3.0-19553-f705706fbce.pickle
python -m viewer.ingest.cli --input output/daily.20260422_0903.summary.json

# Force re-ingest (ignores file_hash dedup)
python -m viewer.ingest.cli --root /var/www/html/daily --force

# Only new / only old
python -m viewer.ingest.cli --root /var/www/html/daily --format new
```

Each invocation also refreshes `display_rows` from `profiles/default.yaml`.
Pass `--profile other.yaml` to load an alternate one, or `--skip-profile` to
leave the table alone.

## Viewer

```bash
streamlit run viewer/app.py -- --db viewer/bench.duckdb
# or set DAILY_DB=.../bench.duckdb in the env
```

Tabs:

| Tab         | What it does                                                         |
|-------------|----------------------------------------------------------------------|
| Excel       | Wide matrix (rows from `display_rows`, columns from selected runs)   |
| Trend       | Per-series line chart with rolling median ± 2σ band                  |
| Regressions | MAD-based z-score + pct-diff flags for the latest run                |
| Geomean     | Geometric-mean trend across a bucket (machine-wide health)           |
| Noise       | CV (σ/median) table — useful for iGPU fluctuation diagnostics        |

## Schema notes

* `perf` holds raw numbers (raw `in_token` / `out_token`). Bucketing into
  `'short'` / `'long'` / `'0'` happens in the `perf_with_buckets` view.
* `perf_stats` computes rolling median + MAD over a 30-day back-window, per
  `(machine, model, precision, in_token, out_token, exec_mode)` series.
* Regression direction: `unit IN ('ms', 's', '%')` → lower is better;
  otherwise higher is better (FPS, tps).

## Extending profiles

Copy `profiles/default.yaml` to e.g. `profiles/iGPU.yaml`, tweak the rows,
then `python -m viewer.ingest.cli --profile profiles/iGPU.yaml --skip-profile`
(the `--skip-profile` suppresses the default one); or run twice to have
both available in the sidebar.

`in_spec` / `out_spec` accept:
* `'short'` / `'long'` / `'0'` — match against `in_bucket` / `out_bucket`
* `'*'` — any (used for SD pipelines where tokens aren't meaningful)
* `'<int>'` — exact numeric match (e.g. `'802'` for vision prompts)
