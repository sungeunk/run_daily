-- daily/viewer/schema.sql
--
-- Schema for the daily benchmark DuckDB.
--
-- Design notes:
-- * `runs` holds per-execution metadata. `source_format` distinguishes the
--   legacy pickle/.report ingest path from the new summary.json path so the
--   viewer can fall back on raw-log paths when a field is missing.
-- * `perf` stores raw numbers exactly as the loaders emit them. Token
--   bucketing ('short'/'long'/'0') lives in a view so the threshold can be
--   changed without a re-ingest.
-- * `perf_stats` is the rolling-window view used by the regression / trend
--   tabs. It uses median + MAD (robust to iGPU fluctuation outliers) and
--   annotates each point with z-score, pct-diff vs. baseline median, and an
--   `is_noisy` flag when the series itself has high CV.
-- * `display_rows` captures the old FIXED_ROW_ORDER as data, per profile, so
--   the Excel-paste tab stays deterministic without hard-coding in Python.

-- ---------------------------------------------------------------------------
-- Core tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    source_format  TEXT NOT NULL,          -- 'old' | 'new'
    report_file    TEXT NOT NULL,
    machine        TEXT NOT NULL,
    device         TEXT,                   -- 'GPU', 'GPU.1', ...
    purpose        TEXT,
    description    TEXT,
    ts             TIMESTAMP NOT NULL,
    ww             TEXT,
    ov_version     TEXT,
    ov_build       TEXT,
    ov_sha         TEXT,
    genai_version  TEXT,
    genai_commit   TEXT,
    tok_commit     TEXT,
    short_run      BOOLEAN DEFAULT FALSE,
    source_path    TEXT,                   -- ingested source file (pickle or summary.json)
    rawlog_path    TEXT,                   -- path to the raw log if available
    file_hash      TEXT,
    ingested_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_devices (
    run_id             TEXT NOT NULL,
    device_index       INTEGER NOT NULL,
    device             TEXT,
    driver             TEXT,
    eu                 INTEGER,
    clock_freq_mhz     DOUBLE,
    global_mem_size_gb DOUBLE,
    PRIMARY KEY (run_id, device_index)
);

CREATE TABLE IF NOT EXISTS perf (
    run_id     TEXT NOT NULL,
    model      TEXT NOT NULL,
    precision  TEXT NOT NULL,
    in_token   INTEGER NOT NULL DEFAULT 0,
    out_token  INTEGER NOT NULL DEFAULT 0,
    exec_mode  TEXT NOT NULL,
    value      DOUBLE,
    unit       TEXT,
    PRIMARY KEY (run_id, model, precision, in_token, out_token, exec_mode)
);

CREATE TABLE IF NOT EXISTS display_rows (
    profile    TEXT NOT NULL,
    seq        INTEGER NOT NULL,        -- 0-based row order
    model      TEXT NOT NULL,
    precision  TEXT NOT NULL,
    in_spec    TEXT NOT NULL,           -- 'short' | 'long' | '0' | '*' | '<int>'
    out_spec   TEXT NOT NULL,
    exec_mode  TEXT NOT NULL,
    label      TEXT,                    -- optional human label
    PRIMARY KEY (profile, seq)
);

-- Viewer-tunable knobs stored in-DB so a single UI restart picks them up.
-- Kept separate from schema defaults so a user can override without editing
-- SQL files.
CREATE TABLE IF NOT EXISTS viewer_settings (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

INSERT INTO viewer_settings (key, value) VALUES
    ('token_bucket_threshold', '100'),
    ('regression_window', '14'),
    ('regression_z_threshold', '3.0'),
    ('regression_pct_threshold', '0.05'),
    ('noisy_cv_threshold', '0.10')
ON CONFLICT (key) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_runs_ts_machine   ON runs(ts, machine);
CREATE INDEX IF NOT EXISTS idx_runs_machine_ts   ON runs(machine, ts);
CREATE INDEX IF NOT EXISTS idx_perf_series       ON perf(model, precision, in_token, out_token, exec_mode);
CREATE INDEX IF NOT EXISTS idx_sys_run           ON system_devices(run_id);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Adds 'short' / 'long' / '0' buckets as derived columns. Threshold is
-- hard-coded (100) to match the historical viewer; change in-place to retune.
CREATE OR REPLACE VIEW perf_with_buckets AS
SELECT
    p.*,
    CASE
        WHEN in_token  = 0 THEN '0'
        WHEN in_token  < 100 THEN 'short'
        ELSE 'long'
    END AS in_bucket,
    CASE
        WHEN out_token = 0 THEN '0'
        WHEN out_token < 100 THEN 'short'
        ELSE 'long'
    END AS out_bucket
FROM perf p;

-- Flat join for the Streamlit tabs: one row per (run, perf point) with the
-- machine/ts denormalised. All downstream analyses use this.
CREATE OR REPLACE VIEW perf_flat AS
SELECT
    r.machine,
    r.device,
    r.ts,
    r.ts::DATE AS date,
    r.ww,
    r.ov_version,
    r.ov_build,
    r.ov_sha,
    r.purpose,
    r.short_run,
    r.source_format,
    p.run_id,
    p.model,
    p.precision,
    p.in_token,
    p.out_token,
    p.in_bucket,
    p.out_bucket,
    p.exec_mode,
    p.value,
    p.unit
FROM runs r
JOIN perf_with_buckets p USING (run_id);

-- Rolling statistics per series. Baseline = median of previous N points
-- (exclusive of current row) within the same (machine, series). MAD is
-- computed with a correlated subquery because DuckDB window functions can't
-- express median-of-abs-deviations directly.
--
-- The subquery cost is fine at daily cadence (a few thousand runs * ~60
-- series), but if it ever hurts, materialise it in a CTE or a Python helper.
CREATE OR REPLACE VIEW perf_stats AS
WITH base AS (
    SELECT
        machine, device, ts, date, ww,
        ov_version, ov_build, ov_sha, purpose, short_run,
        run_id, model, precision,
        in_token, out_token, in_bucket, out_bucket,
        exec_mode, value, unit
    FROM perf_flat
),
with_baseline AS (
    SELECT
        b.*,
        (
            SELECT median(b2.value)
            FROM base b2
            WHERE b2.machine = b.machine
              AND b2.model = b.model
              AND b2.precision = b.precision
              AND b2.in_token = b.in_token
              AND b2.out_token = b.out_token
              AND b2.exec_mode = b.exec_mode
              AND b2.ts < b.ts
              AND b2.ts >= b.ts - INTERVAL '30 DAY'
        ) AS win_median,
        (
            SELECT count(*)
            FROM base b2
            WHERE b2.machine = b.machine
              AND b2.model = b.model
              AND b2.precision = b.precision
              AND b2.in_token = b.in_token
              AND b2.out_token = b.out_token
              AND b2.exec_mode = b.exec_mode
              AND b2.ts < b.ts
              AND b2.ts >= b.ts - INTERVAL '30 DAY'
        ) AS win_n
    FROM base b
),
with_mad AS (
    SELECT
        w.*,
        (
            SELECT median(abs(b2.value - w.win_median))
            FROM base b2
            WHERE b2.machine = w.machine
              AND b2.model = w.model
              AND b2.precision = w.precision
              AND b2.in_token = w.in_token
              AND b2.out_token = w.out_token
              AND b2.exec_mode = w.exec_mode
              AND b2.ts < w.ts
              AND b2.ts >= w.ts - INTERVAL '30 DAY'
        ) AS win_mad
    FROM with_baseline w
)
SELECT
    *,
    1.4826 * win_mad AS win_sigma,
    CASE
        WHEN win_mad IS NULL OR win_median IS NULL OR win_mad = 0 THEN NULL
        ELSE (value - win_median) / (1.4826 * win_mad)
    END AS z_score,
    CASE
        WHEN win_median IS NULL OR win_median = 0 THEN NULL
        ELSE (value - win_median) / win_median
    END AS pct_diff,
    CASE
        WHEN win_median IS NULL OR win_median = 0 OR win_mad IS NULL THEN NULL
        ELSE win_mad / win_median
    END AS cv
FROM with_mad;

-- Latest run per machine, useful for the Regressions tab's default selection.
CREATE OR REPLACE VIEW latest_run_per_machine AS
SELECT machine, arg_max(run_id, ts) AS run_id, max(ts) AS ts
FROM runs
GROUP BY machine;
