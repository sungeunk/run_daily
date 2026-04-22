
-- ddl/bench_schema.sql
PRAGMA enable_verification;

-- 1) 리포트 헤더(런)
CREATE TABLE IF NOT EXISTS runs (
  run_id           TEXT PRIMARY KEY,  -- 해시(머신+파일명) 등 고유키
  report_file      TEXT NOT NULL,     -- "daily.20260114_0104....report"
  machine          TEXT NOT NULL,
  purpose          TEXT,
  rawlog           TEXT,
  ts               TIMESTAMP,         -- "datetime": "20260114_1559" 파싱
  ww               TEXT,              -- "2026.WW2.3"
  ov_version       TEXT,              -- version.openvino
  genai_version    TEXT,              -- version.openvino.genai
  ov_commit        TEXT,              -- commit_id.openvino
  genai_commit     TEXT,              -- commit_id.openvino.genai
  tok_commit       TEXT,              -- commit_id.openvino_tokenizers
  source_path      TEXT,              -- 적재한 파일의 경로
  file_hash        TEXT UNIQUE,       -- 파일 해시(중복 적재 방지)
  ingested_at      TIMESTAMP DEFAULT now()
);

-- 2) 시스템 디바이스(리스트)
CREATE TABLE IF NOT EXISTS system_devices (
  run_id           TEXT,
  device_index     INTEGER,           -- 배열 인덱스
  device           TEXT,
  driver           TEXT,
  eu               INTEGER,
  clock_freq_mhz   DOUBLE,            -- "2400"→2400.0
  global_mem_size_gb DOUBLE,
  PRIMARY KEY (run_id, device_index),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- 3) 성능(perf)
CREATE TABLE IF NOT EXISTS perf (
  run_id     TEXT,
  model      TEXT,
  precision  TEXT,
  in_token   INTEGER,
  out_token  INTEGER,
  exec_mode  TEXT,    -- "1st/2nd/pipeline/fps" 문자열
  value      DOUBLE,  -- perf
  unit       TEXT,    -- perf_unit
  PRIMARY KEY (run_id, model, precision, in_token, out_token, exec_mode),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- 조회 최적화를 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_runs_ts_machine ON runs(ts, machine);
CREATE INDEX IF NOT EXISTS idx_perf_model ON perf(model, precision, in_token, out_token);
CREATE INDEX IF NOT EXISTS idx_sys_run ON system_devices(run_id);

-- 최근 10일(또는 N일) 집계를 위한 뷰 예시
CREATE OR REPLACE VIEW perf_last10 AS
SELECT
  r.ts::DATE AS date,
  r.machine,
  sd.device, sd.driver, sd.eu,
  p.model, p.precision, p.in_token, p.out_token, p.exec_mode,
  p.value, p.unit
FROM runs r
LEFT JOIN system_devices sd USING(run_id)
LEFT JOIN perf p USING(run_id)
WHERE r.ts >= current_date - INTERVAL 10 DAY;

-- 최근일 vs 기준선(직전 1~9일 중앙값) 비교 뷰 예시
CREATE OR REPLACE VIEW perf_today_vs_baseline AS
WITH daily AS (
  SELECT
    r.ts::DATE AS date,
    r.machine, p.model, p.precision, p.in_token, p.out_token, p.exec_mode, p.unit,
    median(p.value) AS value
  FROM runs r
  JOIN perf p USING(run_id)
  WHERE r.ts >= current_date - INTERVAL 10 DAY
  GROUP BY ALL
),
with_baseline AS (
  SELECT
    d.*,
    (
      SELECT median(value) FROM daily d2
      WHERE d2.machine=d.machine AND d2.model=d.model AND d2.precision=d.precision
        AND d2.in_token=d.in_token AND d2.out_token=d.out_token AND d2.exec_mode=d.exec_mode
        AND d2.date < d.date
    ) AS baseline
  FROM daily d
)
SELECT
  *,
  CASE WHEN baseline IS NULL THEN NULL
       ELSE (value - baseline)/NULLIF(baseline, 0) END AS pct_diff
FROM with_baseline
WHERE date = (SELECT max(date) FROM daily);
