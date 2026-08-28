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
    -- Normalised at ingest so the viewer can exclude CI/PR runs without
    -- pattern-matching free-form purpose text at query time.
    run_kind       TEXT DEFAULT 'daily',   -- 'daily' | 'pr' | 'test' | 'manual'
    description    TEXT,
    ts             TIMESTAMP NOT NULL,
    ww             TEXT,
    ov_version     TEXT,
    ov_build       TEXT,
    ov_sha         TEXT,
    host_info      TEXT,
    host_memory_size_gb DOUBLE,
    host_memory_speed_mhz DOUBLE,
    genai_version  TEXT,
    genai_commit   TEXT,
    tok_commit     TEXT,
    short_run      BOOLEAN DEFAULT FALSE,
    -- pytest outcome counts and wall time for the whole run, taken from
    -- summary.json so the fleet view does not have to re-read every file.
    total_tests    INTEGER,
    passed_tests   INTEGER,
    failed_tests   INTEGER,
    error_tests    INTEGER,
    skipped_tests  INTEGER,
    -- Benchmark cases (perf series) lost to skipped tests, from each test's
    -- ``expected_series``. A pytest test can carry several cases.
    skipped_cases  INTEGER,
    duration_sec   DOUBLE,
    source_path    TEXT,                   -- ingested source file (pickle or summary.json)
    build_url      TEXT,                   -- Jenkins BUILD_URL of the run, when any
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
    prompt_idx INTEGER NOT NULL DEFAULT 0,  -- source prompt index (0,1,2,...); more stable join key than token buckets
    PRIMARY KEY (run_id, model, precision, in_token, out_token, exec_mode)
);

CREATE TABLE IF NOT EXISTS display_rows (
    profile    TEXT NOT NULL,
    seq        INTEGER NOT NULL,        -- 0-based row order
    model      TEXT NOT NULL,
    precision  TEXT NOT NULL,
    in_spec    TEXT NOT NULL,           -- '<prompt_idx>' | '*'
    out_spec   TEXT NOT NULL,
    exec_mode  TEXT NOT NULL,
    label      TEXT,                    -- optional human label
    PRIMARY KEY (profile, seq)
);


CREATE TABLE IF NOT EXISTS analysis_results (
    run_id                TEXT PRIMARY KEY,
    baseline_run_id       TEXT,
    overall_status        TEXT NOT NULL,
    compared_count        INTEGER NOT NULL DEFAULT 0,
    improved_count        INTEGER NOT NULL DEFAULT 0,
    same_count            INTEGER NOT NULL DEFAULT 0,
    regressed_count       INTEGER NOT NULL DEFAULT 0,
    functional_fail_count INTEGER NOT NULL DEFAULT 0,
    updated_at            TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analysis_comparisons (
    run_id          TEXT NOT NULL,
    baseline_run_id TEXT,
    model           TEXT NOT NULL,
    precision       TEXT NOT NULL,
    in_token        INTEGER NOT NULL,
    out_token       INTEGER NOT NULL,
    exec_mode       TEXT NOT NULL,
    unit            TEXT,
    current_value   DOUBLE,
    baseline_value  DOUBLE,
    improvement_pct DOUBLE,
    verdict         TEXT NOT NULL,
    threshold_pct   DOUBLE,
    -- History context computed by daily/analysis/engine.py. Persisted so the
    -- viewer can show "is this outside normal fluctuation" without recomputing.
    history_count      INTEGER,
    history_median     DOUBLE,
    history_mad        DOUBLE,
    history_sigma      DOUBLE,
    history_cv         DOUBLE,
    worsening_z        DOUBLE,
    reference_source   TEXT,
    within_fluctuation BOOLEAN,
    PRIMARY KEY (run_id, model, precision, in_token, out_token, exec_mode)
);

-- Per-test machine telemetry, reduced to min/max/mean by
-- daily/common/machine_monitor.py. Raw JSONL samples stay on disk; only the
-- summary is ingested so regression review can tell a slow run apart from a
-- throttled or otherwise disturbed machine.
CREATE TABLE IF NOT EXISTS machine_monitor_stats (
    run_id                    TEXT NOT NULL,
    nodeid                    TEXT NOT NULL,
    model                     TEXT,
    precision                 TEXT,
    samples                   INTEGER,
    duration_sec              DOUBLE,
    gpu_clock_mhz_mean        DOUBLE,
    gpu_clock_mhz_min         DOUBLE,
    gpu_clock_mhz_max         DOUBLE,
    gpu_clock_max_mhz         DOUBLE,
    gpu_utilization_mean      DOUBLE,
    gpu_power_watts_mean      DOUBLE,
    gpu_power_watts_max       DOUBLE,
    gpu_temp_c_mean           DOUBLE,
    gpu_temp_c_max            DOUBLE,
    cpu_clock_mhz_mean        DOUBLE,
    cpu_usage_percent_mean    DOUBLE,
    cpu_temp_c_max            DOUBLE,
    host_memory_usage_mean    DOUBLE,
    page_faults_per_sec_mean  DOUBLE,
    throttled_sample_ratio    DOUBLE,
    throttle_reasons          TEXT,
    sample_duration_ms_max    DOUBLE,
    monitor_file              TEXT,
    PRIMARY KEY (run_id, nodeid)
);

CREATE TABLE IF NOT EXISTS functional_issues (
    run_id    TEXT NOT NULL,
    nodeid    TEXT NOT NULL,
    outcome   TEXT NOT NULL,
    message   TEXT,
    model     TEXT,
    precision TEXT,
    PRIMARY KEY (run_id, nodeid, outcome)
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

-- Runs manually excluded from every cohort-based analysis (Geomean,
-- Regression, Noise, Functional, Dashboard latest-run) via the viewer's
-- Exclusions tab — e.g. a build that only measured a handful of models,
-- which would otherwise collapse the common-series intersection for the
-- whole cohort. The Excel tab's manual run picker deliberately ignores this
-- table since it lets the user pick any run on purpose.
CREATE TABLE IF NOT EXISTS run_exclusions (
    run_id      TEXT PRIMARY KEY,
    machine     TEXT NOT NULL,
    stamp       TEXT NOT NULL,      -- denormalized for display without a join
    reason      TEXT,
    excluded_at TIMESTAMP DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_runs_ts_machine   ON runs(ts, machine);
CREATE INDEX IF NOT EXISTS idx_runs_machine_ts   ON runs(machine, ts);
CREATE INDEX IF NOT EXISTS idx_perf_series       ON perf(model, precision, in_token, out_token, exec_mode);
CREATE INDEX IF NOT EXISTS idx_sys_run           ON system_devices(run_id);
CREATE INDEX IF NOT EXISTS idx_analysis_status   ON analysis_results(overall_status, run_id);
CREATE INDEX IF NOT EXISTS idx_monitor_run       ON machine_monitor_stats(run_id);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Adds 'short' / 'long' / '0' buckets as derived columns. Threshold is
-- hard-coded (100) to match the historical viewer; change in-place to retune.
-- These image-generation models emit seconds, but the viewer displays their
-- latency in milliseconds to match the Excel/report convention.
CREATE OR REPLACE VIEW perf_with_buckets AS
SELECT
    p.*,
    CASE
        WHEN p.model IN (
            'flux.1-schnell',
            'stable-diffusion-v1-5',
            'stable-diffusion-3.5-large-turbo'
        ) AND p.unit = 's' THEN p.value * 1000
        ELSE p.value
    END AS viewer_value,
    CASE
        WHEN p.model IN (
            'flux.1-schnell',
            'stable-diffusion-v1-5',
            'stable-diffusion-3.5-large-turbo'
        ) AND p.unit = 's' THEN 'ms'
        ELSE p.unit
    END AS viewer_unit,
    CASE
        WHEN p.in_token  = 0 THEN '0'
        WHEN p.in_token  < 100 THEN 'short'
        ELSE 'long'
    END AS in_bucket,
    CASE
        WHEN p.out_token = 0 THEN '0'
        WHEN p.out_token < 100 THEN 'short'
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
    r.run_kind,
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
    p.viewer_value AS value,
    p.viewer_unit AS unit
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

-- One machine-health row per run. `gpu_clock_ratio` is the headline signal:
-- a run that spent its time well below the card's own max clock is a
-- fluctuation suspect rather than a code regression.
CREATE OR REPLACE VIEW run_machine_health AS
SELECT
    run_id,
    count(*)                          AS monitored_tests,
    sum(samples)                      AS total_samples,
    max(throttled_sample_ratio)       AS max_throttle_ratio,
    avg(throttled_sample_ratio)       AS avg_throttle_ratio,
    max(gpu_temp_c_max)               AS max_gpu_temp_c,
    avg(gpu_clock_mhz_mean)           AS avg_gpu_clock_mhz,
    avg(gpu_utilization_mean)         AS avg_gpu_utilization,
    avg(gpu_power_watts_mean)         AS avg_gpu_power_watts,
    avg(cpu_usage_percent_mean)       AS avg_cpu_usage,
    max(cpu_temp_c_max)               AS max_cpu_temp_c,
    avg(host_memory_usage_mean)       AS avg_host_memory_usage,
    max(sample_duration_ms_max)       AS max_sample_duration_ms,
    CASE
        WHEN max(gpu_clock_max_mhz) IS NULL OR max(gpu_clock_max_mhz) = 0 THEN NULL
        ELSE avg(gpu_clock_mhz_mean) / max(gpu_clock_max_mhz)
    END                               AS gpu_clock_ratio
FROM machine_monitor_stats
GROUP BY run_id;
