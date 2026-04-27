# daily-llm-viewer — Design

> Streamlit-based analysis tool for the LLM daily test results.
> Supersedes `_old_viewer.py` + `_old_ingest.py` + `_old_schema.sql`.

- **Working dir:** `/home/sungeunk/repo/run_daily2/daily`
- **Results root:** `/var/www/html/daily/<MACHINE>/`
- **Language policy:** `ko-KR` for conversation; English for code / comments / docs.

---

## Pipeline

| Track | Flow |
|---|---|
| Old | `.pickle` (run_daily2/scripts pipeline) + `.report` (text) → ingest → DuckDB |
| New | `.summary.json` (`daily/run.py` → `report/builder.py`) → ingest → DuckDB |

- Both tracks converge on the `RunRecord` dataclass, then `writer.upsert_run`.
- Viewer consumes DuckDB read-only via `queries.py`; Streamlit app in `app.py`.

---

## Files

### `daily/viewer/schema.sql`
DuckDB schema. Source of truth. `ensure_schema()` runs this on every ingest.

- **Tables:** `runs`, `system_devices`, `perf`, `display_rows`, `viewer_settings`
- **Views:** `perf_with_buckets`, `perf_flat`, `perf_stats`, `latest_run_per_machine`

Notes:
- `perf` stores RAW `in_token` / `out_token`. No bucketing at write time (per user requirement B).
- `perf_with_buckets` derives `in_bucket` / `out_bucket` (`'short'` / `'long'` / `'0'`) with `threshold=100`.
- `perf_stats` uses correlated subqueries for rolling median + MAD over a 30-day back-window. Kept for legacy point-vs-band helpers; Regression tab now recomputes the selected-series rolling band from purpose-filtered `perf_flat` rows so personal PR results do not pollute the chart.
- `runs.rawlog_path` stores a FILE PATH, not the log body (old schema stored the whole log in a TEXT column; removed to keep DB small).
- `runs.source_format` is `'old' | 'new'` — keep this for debugging data provenance.

### `daily/viewer/profiles/default.yaml`
Replacement for `FIXED_ROW_ORDER`. Controls row order in Excel-paste tab.

Match spec semantics:

| Spec | Meaning |
|---|---|
| `short` | `token > 0 AND < 100` |
| `long`  | `token >= 100` |
| `0`     | `token == 0` (exact) |
| `*`     | any value (token-agnostic; used for SD/LCM pipelines) |
| `<int>` | exact numeric (e.g. `'802'` for phi-3.5-vision-instruct) |

**Extension:** add e.g. `profiles/iGPU.yaml` for machine-specific rows. Load via `python -m viewer.ingest.cli --profile <yaml>`. Sidebar picks between loaded profiles.

### `daily/viewer/ingest/record.py`
`RunRecord` + `DeviceRecord` + `PerfRow` dataclasses. Format-neutral — loaders fill these, writer consumes them.

**Do not** add format-specific fields here. Put them on the loader.

### `daily/viewer/ingest/_common.py`
Shared helpers for loaders.

Exports: `parse_stamp_from_name`, `workweek_of`, `split_ov_version`, `file_hash` (sha256 content hash of source file), `run_id_of` (sha1 of `machine|iso_ts|report_file`, 20 chars).

### `daily/viewer/ingest/loader_new.py`
`summary.json` → `RunRecord`. Raw token counts preserved.

- **Test-type handlers:** `llm_benchmark`, `benchmark_app`, `sd_genai`, `sd_dgfx`.
- **Dropped handlers:** `qwen_usage`, `whisper_base` — see [Dropped qwen_usage and Whisper base](#dropped-qwen_usage-and-whisper-base).
- **Meta handling:** `summary.meta` is optional — falls back to filename stamp + parent-dir machine + `generated_at` ts. Early summary files (before meta landed) still work.

### `daily/viewer/ingest/loader_old.py`
Legacy `.pickle` + `.report` → `RunRecord`.

- **Pickle trick:** `_TolerantUnpickler.find_class` replaces missing classes (`test_cases.TestBenchmark` etc.) with `_UnknownClass` stubs. Key tuple is `(model, precision, TestClass)` — we read `TestClass.__name__` to dispatch to the correct extractor.
- **Class handlers:**

| Class | Handler |
|---|---|
| `TestBenchmark` | `_benchmark_perf` |
| `TestBenchmarkapp` | `_benchmark_app_perf` |
| `TestStableDiffusion` | `_sd_perf_ms` (legacy C++ binary — ms in pickle, divided by 1000 at ingest) |
| `TestStableDiffusionGenai` | `_sd_perf_sec` (pickle already seconds) |
| `TestStableDiffusionDGfxE2eAi` | `_sd_perf_sec` (pickle already seconds) |

- **Dropped handlers:** `TestMeasuredUsageCpp`, `TestWhisperBase` — see [Dropped qwen_usage and Whisper base](#dropped-qwen_usage-and-whisper-base).
- **Version parsing:** OV version pulled from filename `daily.<stamp>.<ov_version>.pickle` AND from `.report` text (`Purpose` + `OpenVINO` line). Report text is preferred when present.
- **SD unit normalization:** All SD pipelines land in the DB as `unit='s'` regardless of source class. Legacy ms pickles are divided by 1000 at load time. This removes the 1000× ambiguity that made SD-XL pipeline look like milliseconds in the viewer.

### `daily/viewer/ingest/writer.py`
DuckDB upsert (transactional per `RunRecord`) + display-profile loader.

- **Dedup:** `perf` is replaced wholesale per run (DELETE + INSERT). Duplicate PKs within a run are deduplicated to last-wins.
- **File-hash skip:** `already_ingested()` checks `runs.file_hash`. Skip unless `--force`.

### `daily/viewer/ingest/cli.py`
Single CLI; `--root` scans, `--input` handles one file.

- **Auto-detect (`_classify`):** `*.summary.json` → new; `*.pickle` (with sibling `.report`) → old; `*.report` alone → old if sibling pickle exists, else skip.
- **Progress:** Inline progress bar. Failures collected and reported at end; not fatal.
- **Profile load:** Each run (re)loads `profiles/default.yaml` into `display_rows` unless `--skip-profile`.

### `daily/viewer/queries.py`
DuckDB helpers used by `app.py`. All return pandas DataFrames.

| Function | Description |
|---|---|
| `list_machines` | `SELECT DISTINCT machine` |
| `list_runs` | Runs metadata for one machine, newest first |
| `list_profiles` | `SELECT DISTINCT profile FROM display_rows` |
| `build_excel_matrix` | Cross-join `display_rows` × selected runs, LEFT JOIN `perf_with_buckets`, then manual pivot (pandas `pivot_table` drops NaN index and explodes on `dropna=False`) |
| `extra_rows` | Perf rows in selected runs NOT covered by the display profile — QA aid |
| `series_history` | Time series of one `(machine, model, precision, in_token, out_token, exec_mode)`, with rolling median / MAD recomputed from purpose-filtered `perf_flat` rows. Used for the trend plot in the Regression tab. |
| `trend_regressions` | **NEW** — per-series comparison: median of recent window vs median of baseline window. `worsening_pct` is sign-normalised (positive = worse regardless of ms vs FPS). This is what the Regression tab drives off. |
| `regressions_for_run` | **OLD** point-vs-band flags. Kept for possible mail-alert reuse but not wired into the UI any more. |
| `noise_summary` | Per-series CV over a window |
| `geomean_trend` | `exp(avg(ln(value)))` per run, filtered by `exec_mode` / `in_bucket` / `out_bucket` / `excluded_models` |

### `daily/viewer/app.py`
Streamlit entry. 5 tabs.

| Tab | Purpose |
|---|---|
| Dashboard | First tab. Selects the latest run whose purpose/description matches `DEFAULT_RUN_FILTER` (fallback: `daily_CB`), then reviews summary / `.report` / `.pytest.json` / `.raw` artifacts to show whether tests actually ran, pass/fail totals, grouped failure causes, and per-test raw-log messages. Artifact siblings are derived from `source_path` stem (`.summary.json` / `.pickle` suffix stripped) — `report_file` is source-identity only, never fed to text/JSON readers. |
| Excel | Select runs → wide matrix (profile rows × run stamps) + tab-separated paste block + "extra rows" expander |
| Regression | MERGED tab (was separate Trend + Regressions). Shows a ranked table of series by worsening %, plus a one-series-at-a-time trend plot for the selected row. Baseline median + recent median rendered as Scatter traces with legend entries; rolling ±2σ band drawn with a visible fill (rgba 0.28) so it reads on bright monitors. |
| Geomean | `exec_mode × in_bucket × out_bucket` geomean trend + band + latest-point alert (user-requested) |
| Noise | Per-series CV table sorted desc (iGPU diagnostics) |

Sidebar:
- **Machine filter:** "Daily machines only" checkbox (default `True`). When `True`, dropdown is filtered by `DAILY_MACHINES` constant (`dg2alderlake`, `MTL-01`, `ARLH-01`, `BMG-02`, `LNL-03`, `LNL-04`, `DUT4580PTLH`, `DUT6047BMGFRD`). Falls back to full list if none intersect.
- **Default filter:** Excel tab purpose filter defaults to `daily_CB timer` via `DEFAULT_RUN_FILTER` constant.
- **Days slider:** 7–60 day range for trend history. All tab-local "days" inputs (Geomean, Noise) capped at 60 for consistency.

Other:
- **Unit display:** every user-visible numeric carries its unit — table median columns render as `8.060 s`, trend heading shows `[s]`, caption prints `Recent median = 8.060 s`. Prevents SD pipeline seconds from being mis-read as ms.
- **Caching:** all queries wrapped with `@st.cache_data` keyed on DB mtime to auto-invalidate on re-ingest.
- **Config source:** `DAILY_DB` env var or `-- --db <path>` trailing arg (streamlit swallows its own flags).

### `daily/run.py`
Daily suite entry — runs pytest, builds reports, ships mail/xlsx.

- **Exit policy:** always returns `0` if the run completed end-to-end (report built). Test failures are reflected in JSON summary / mail, not exit code. Only infra failures (no pytest JSON produced) propagate a non-zero rc. Changed per user request; relied on by Jenkins to distinguish "test failed" from "run itself broke".

### Legacy / independent modules
| Path | Status |
|---|---|
| `daily/viewer/_old_viewer.py` | Legacy reference. Do not import. |
| `daily/viewer/_old_ingest.py` | Legacy reference. Do not import. |
| `daily/viewer/_old_schema.sql` | Legacy reference. Do not use. |
| `daily/viewer/perf_rows.py` | Used by `xlsx_update.py` path (flatten summary → lookup dict). Kept as-is. **NOTE:** this one DOES bucket (`'short'` / `'long'`) for the xlsx template — do not reuse for DB ingest (which must keep raw tokens). |
| `daily/viewer/xlsx_update.py` | Separate xlsx append path invoked from `run.py`. Independent of the DuckDB pipeline. |

---

## Design decisions

### FIXED_ROW_ORDER replacement
- **Choice:** YAML profile + DB `display_rows` table.
- **Rejected:** Python list in code (no version control per-row, harder to diff); DB-only (no git history).
- **Why:** user adds/removes models frequently. YAML is reviewable in git; DB copy is for fast JOINs in queries. Match spec (`short` / `long` / `*` / `<int>`) is expressive enough without code changes.

### Raw tokens in DB vs bucketed
- **Choice:** raw only.
- **User request:** B — ingest is data-as-is; viewer does the bucketing.
- **Implementation:** `perf_with_buckets` view derives `in_bucket` / `out_bucket`. Threshold lives in SQL — change the view to retune, no re-ingest needed.

### Regression detection method (current approach)
- **Choice:** two-window median comparison — recent-window median vs baseline-window median.
- **Rejected:**
  - Single-point robust z-score — user feedback: they want to see whether the recent BLOCK is drifting, not whether today is an outlier. Single-point tests flip on any blip and are useless when data is noisy or occasionally corrupted.
  - mean/std z-score — too sensitive to iGPU outliers.
- **Why:** iGPU runs are noisy and individual data points are sometimes contaminated. Comparing medians of two time windows (default: last 7d vs 21d prior) washes both out while still catching slow drift. Matches common benchmark tools (ASV / pytest-benchmark) which emphasize median-like robust summaries and percentage thresholds rather than reacting to a single point.
- **Defaults:** `recent_days=7`, `baseline_days=21`, `min_recent_points=5`, `min_baseline_points=7`, `pct_threshold_from_sidebar=0.05`, `z_threshold_from_sidebar=3.0`, `noisy_cv_threshold=0.10`.
  > Values mirror [`queries.trend_regressions`](queries.py) signature. Single source of truth.
- **Direction normalisation:** `worsening_pct` is signed so positive always means "worse" — ms/s/%: `+pct` when recent>baseline; FPS/tps: `+pct` when recent<baseline. `worsening_z` uses baseline MAD (`sigma ≈ 1.4826 * MAD`) so recent noise does not hide or inflate the comparison. UI sorts by threshold-normalised severity = `max(worsening_pct / pct_threshold, worsening_z / z_threshold)`.
- **Purpose filter:** regression summary and selected-series history are filtered to purpose containing `daily_CB timer` so personal PR runs do not pollute baseline/recent windows. The chart rolling band is recomputed from filtered `perf_flat` rows, not from `perf_stats`, because `perf_stats` is an all-purpose view.
- **UI:** single merged "Regression" tab. Ranked table (worst first by threshold-normalised severity) + one trend chart per selected row. See `app.py:_tab_regression`.
- **Supersedes:** earlier rolling z-score approach (`queries.regressions_for_run`) is still defined but unused by the UI.

### One series per trend chart
- **Choice:** enforce single-series plot in the Regression tab.
- **Why:** mixing models with wildly different value ranges on one chart makes it unreadable. Table-plus-single-chart is the pattern that scales.
- **Implementation:** row selection in the summary table drives which series is plotted. Default = row 0 (worst drift). The y-axis range has a minimum relative span so tiny stable differences (e.g. `31.6 ms` vs `31.95 ms`) are not over-zoomed into visually noisy fluctuations.

### SD pipeline unit normalization
- **Problem observed:** legacy `TestStableDiffusion` pickle stored ms; newer `TestStableDiffusionGenai` / `TestStableDiffusionDGfxE2eAi` pickles stored seconds. The original `_sd_perf` lumped them all into `unit='ms'`, so SD-XL pipeline (actually 8 s) showed up as 8 ms in comparisons.
- **Fix:** split into `_sd_perf_ms` (legacy, `/1000`) and `_sd_perf_sec` (new, as-is). All SD pipelines now land as `unit='s'`.
- **Remediation:** re-ingested with `--force` after the fix. Confirmed: 0 remaining ms-labeled SD rows in DB.
- **Commit:** `708a853`.

### Units visible in every UI surface
- **Choice:** attach unit suffix to table cells, plot headings, and captions.
- **Why:** raw numbers without units caused the SD-XL confusion above. User should never have to remember which test type is in which unit.
- **Implementation:** regression table builds display-only `recent` / `baseline` columns like `8.060 s`; raw numeric columns hidden via `column_config` but retained for plot/caption code paths.

### Dropped qwen_usage and Whisper base
- **User request:** remove both entirely.
- **Action:** removed handlers from `loader_new.py` + `loader_old.py`, deleted existing rows from DB.
- **Note:** `whisper-large-v3` (NOT Whisper base) is a different model in the GenAI pipeline and is kept.

### Geomean alerting
- **User request:** C — user explicitly wants geomean-level trend alerts (not just per-series).
- **Implementation:** `geomean_trend()` groups by `(machine, exec_mode, in_bucket, out_bucket)`. Tab computes median + MAD of the geomean series itself and shows a banner if latest point falls outside `±z·σ` AND `±pct%`.

### Daily machines filter
- **Choice:** hardcoded `DAILY_MACHINES` tuple in `app.py` with sidebar checkbox (default `True`).
- **Rejected:** DB-backed list (operational set doesn't need runtime mutation and git is the right source of truth).
- **Machines:** `dg2alderlake`, `MTL-01`, `ARLH-01`, `BMG-02`, `LNL-03`, `LNL-04`, `DUT4580PTLH`, `DUT6047BMGFRD`.
- **Fallback:** if the checkbox filter leaves the dropdown empty (fresh DB), falls back to the full machine list.

### `run.py` exit-code policy
- **User request:** always return `0` when the run completes, not when tests pass. Infra failure vs test failure must be distinguishable.
- **Choice:** return `0` after `build_reports` succeeds, regardless of pytest rc. Still propagates non-zero if pytest never produced the JSON (real infra break).
- **Rationale:** Jenkins treats non-zero as "build broke". The team wants "test failed" signalled via mail/report, not Jenkins.

### `run_id`
- **Choice:** `sha1(machine|iso_ts|report_file)[:20]`.
- **Why:** including `iso_ts` makes re-runs minutes apart distinct even if the file stem somehow collides.

### `file_hash` for dedup
- **Choice:** sha256 content hash of the source file (pickle or summary.json).
- **Why:** stable across path changes / rsyncs / backups. Replaces the old `(path|size|mtime)` scheme which broke on copy.
- **Cost:** hashing is I/O-bound but amortised — 6006 files in 57s on `/var/www/html/daily`.

### rawlog storage
- **Choice:** store PATH only (`rawlog_path`); never the body.
- **Old behavior:** `_old_schema.sql` stored the full text in `runs.rawlog` — this exploded the DB.
- **Mitigation:** if log access is needed from the UI, open the file from the path. Viewer doesn't do this yet.

### Pickle unpickle without legacy modules
- **Choice:** `_TolerantUnpickler.find_class` returns `_UnknownClass` stub on `ModuleNotFoundError`.
- **Why:** legacy pickles reference `test_cases.test_benchmark.TestBenchmark` etc. We don't want to import that package into the viewer. We only need the class NAME to dispatch extractors.

### Transaction granularity
- **Choice:** one transaction per `RunRecord` (`runs` + `system_devices` + `perf` together).
- **Why:** partial-state bugs in the old ingest (separate inserts) were painful. Run is the natural unit.

### Streamlit caching
- **Choice:** `@st.cache_data` with `_v=DB.stat().st_mtime` as a tiebreaker argument.
- **Why:** Streamlit cache invalidates when any cache key changes. Passing mtime means re-ingest auto-refreshes all cached queries without manual invalidation.

### Trend plot axis orientation
- **Choice:** `xaxis autorange reversed` — newest on the left.
- **Why:** user request — matches the team's convention of reading newest-first.

---

## SQL match rules

- **Location:** `queries.build_excel_matrix` (also mirrored in `extra_rows`).
- **Rule:** `display_rows.<spec>` matches `perf_with_buckets` as follows:

```
spec = '*'                         -> always matches
spec = 'short'/'long'/'0'          -> matches the corresponding <side>_bucket column
TRY_CAST(spec AS INTEGER) = token  -> exact numeric match
else                               -> no match
```

All three are OR'd. `TRY_CAST` returns `NULL` for non-integer strings, which cleanly fails the comparison.

---

## How to run

```bash
# Ingest all
cd daily && python -m viewer.ingest.cli --root /var/www/html/daily --db viewer/bench.duckdb

# Ingest one (new format)
cd daily && python -m viewer.ingest.cli --input output/daily.<stamp>.summary.json

# Ingest one (old format)
cd daily && python -m viewer.ingest.cli --input /var/www/html/daily/LNL-02/daily.<stamp>.<ver>.pickle

# Force re-ingest
# (append --force to any of the above)

# Reload profile only
python -m viewer.ingest.cli --root /dev/null --profile viewer/profiles/default.yaml

# Launch viewer
cd daily && streamlit run viewer/app.py -- --db viewer/bench.duckdb

# Alternative DB via env var
DAILY_DB=/path/to/bench.duckdb streamlit run viewer/app.py
```

**Python env:** use `/home/sungeunk/miniforge3/envs/daily/bin/python` (conda `daily` env). System `python3` doesn't have duckdb/streamlit.

---

## Conventions

### Units

| Unit | Meaning | Direction |
|---|---|---|
| `ms` | latency | lower is better |
| `s` | seconds (SD / LCM / whisper / flux pipelines) | lower is better |
| `FPS` | throughput (benchmark_app) | higher is better |
| `tps` | tokens per second | higher is better |
| `%` | percent | treated as lower-is-better for regression direction |

### Other
- **`exec_mode_values`:** `"1st"`, `"2nd"`, `"pipeline"`, `"batch:<n>"`, `"tps"`.
- **`token_bucket_threshold`:** `100`.
- **`regression_window_defaults`:** see [Regression detection method](#regression-detection-method-current-approach).

---

## Known gaps & future work

- **No unit tests.** All modules were smoke-tested against real data. If the codebase grows, pytest on `_common.py` (stamp/hash parsers), `loader_new` / `loader_old` (fixture files), and `queries` (synthetic DB) would be worth it.
- **`perf_stats` still uses correlated subqueries (O(n·m) conceptually).** ~0.1s at current scale (27k rows, then 6006 runs on full tree). If query time exceeds 1s, rewrite as a Python-side precomputation that writes into a cached `perf_stats_cached` table on ingest.
- **Email regression alerts from `run.py`.** Out of scope for this change. Hook would be: after `run.py`'s `send_mail`, run `queries.trend_regressions` for the latest machine and append a section to the mail body.
- **`regressions_for_run` is defined but unused.** Might still be useful for mail alerts or a per-run QA view. If it hasn't been touched by 2026-Q3, delete it.
- **Profile CLI currently always loads `default.yaml` on every ingest unless `--skip-profile`.** Make `--profile` additive by default — skip-if-exists semantics, or always upsert only the named profile without touching others.
- **Machine name for pickles ingested from non-canonical locations.** `loader_old.py` derives machine from parent-dir name. If a pickle sits in a dir not named after the machine, the value is wrong (seen with `res/` fixture → `machine='res'`). Acceptable for real `/var/www/html/daily/<MACHINE>/` layout. If we need to rescue fixtures, add a `--machine` override to `cli.py`.
- **Display profile dropdown shows only `default` today.** Only one YAML exists. Consider hiding the dropdown until a second profile lands, or pre-ship an iGPU-focused profile to motivate the machinery.
