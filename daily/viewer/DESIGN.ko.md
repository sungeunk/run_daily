# daily-llm-viewer — 설계

> LLM daily test 결과를 분석하기 위한 Streamlit 기반 도구입니다.
> `_old_viewer.py` + `_old_ingest.py` + `_old_schema.sql`을 대체합니다.
>
> **주의:** 이 문서는 한국어 번역본입니다. Source of truth는 [DESIGN.md](DESIGN.md)입니다.

- **작업 디렉터리:** `/home/sungeunk/repo/run_daily2/daily`
- **결과 루트:** `/var/www/html/daily/<MACHINE>/`
- **언어 정책:** 대화는 `ko-KR`, 코드 / 주석 / 문서는 영어.

---

## 파이프라인

| 트랙 | 흐름 |
|---|---|
| Old | `.pickle` (run_daily2/scripts pipeline) + `.report` (text) -> ingest -> DuckDB |
| New | `.summary.json` (`daily/run.py` -> `report/builder.py`) -> ingest -> DuckDB |

- 두 트랙 모두 `RunRecord` dataclass로 수렴한 뒤 `writer.upsert_run`으로 저장됩니다.
- Viewer는 `queries.py`를 통해 DuckDB를 read-only로 읽고, Streamlit 앱은 `app.py`에 있습니다.

---

## 파일

### `daily/viewer/schema.sql`
DuckDB 스키마입니다. Source of truth이며, ingest 때마다 `ensure_schema()`가 실행합니다.

- **Tables:** `runs`, `system_devices`, `perf`, `display_rows`, `viewer_settings`
- **Views:** `perf_with_buckets`, `perf_flat`, `perf_stats`, `latest_run_per_machine`

Notes:
- `perf`는 RAW `in_token` / `out_token`을 저장합니다. write 시점에는 bucketing하지 않습니다. (사용자 요구사항 B)
- `perf_with_buckets`는 `threshold=100` 기준으로 `in_bucket` / `out_bucket` (`'short'` / `'long'` / `'0'`)을 파생합니다.
- `perf_stats`는 30일 back-window에 대해 correlated subquery로 rolling median + MAD를 계산합니다. legacy point-vs-band helper용으로 유지합니다. Regression 탭은 개인 PR 결과가 chart를 오염시키지 않도록 purpose-filtered `perf_flat` rows에서 selected-series rolling band를 다시 계산합니다.
- `runs.rawlog_path`는 로그 본문이 아니라 FILE PATH를 저장합니다. old schema는 전체 로그를 TEXT column에 저장해서 DB가 커졌기 때문에 제거했습니다.
- `runs.source_format`은 `'old' | 'new'`입니다. 데이터 출처 디버깅을 위해 유지합니다.

### `daily/viewer/profiles/default.yaml`
`FIXED_ROW_ORDER`를 대체합니다. Excel-paste 탭의 row order를 제어합니다.

Match spec 의미:

| Spec | 의미 |
|---|---|
| `short` | `token > 0 AND < 100` |
| `long` | `token >= 100` |
| `0` | `token == 0` (exact) |
| `*` | any value (token-agnostic; SD/LCM pipelines에 사용) |
| `<int>` | exact numeric (예: phi-3.5-vision-instruct의 `'802'`) |

**확장:** machine-specific row가 필요하면 예를 들어 `profiles/iGPU.yaml`을 추가합니다. `python -m viewer.ingest.cli --profile <yaml>`로 로드합니다. Sidebar에서 로드된 profile 중 선택합니다.

### `daily/viewer/ingest/record.py`
`RunRecord` + `DeviceRecord` + `PerfRow` dataclass입니다. Format-neutral이며, loader가 채우고 writer가 소비합니다.

**하지 말 것:** format-specific field를 여기에 추가하지 마세요. 그런 정보는 loader에 둡니다.

### `daily/viewer/ingest/_common.py`
loader 공용 helper입니다.

Exports: `parse_stamp_from_name`, `workweek_of`, `split_ov_version`, `file_hash` (source file의 sha256 content hash), `run_id_of` (`machine|iso_ts|report_file`의 sha1, 20 chars).

### `daily/viewer/ingest/loader_new.py`
`summary.json` -> `RunRecord`. Raw token count를 보존합니다.

- **Test-type handlers:** `llm_benchmark`, `benchmark_app`, `sd_genai`, `sd_dgfx`.
- **Dropped handlers:** `qwen_usage`, `whisper_base` — [Dropped qwen_usage and Whisper base](#dropped-qwen_usage-and-whisper-base)를 참고하세요.
- **Meta handling:** `summary.meta`는 optional입니다. 없으면 filename stamp + parent-dir machine + `generated_at` ts로 fallback합니다. meta가 추가되기 전의 초기 summary 파일도 동작합니다.

### `daily/viewer/ingest/loader_old.py`
legacy `.pickle` + `.report` -> `RunRecord`.

- **Pickle trick:** `_TolerantUnpickler.find_class`는 누락된 class (`test_cases.TestBenchmark` 등)를 `_UnknownClass` stub으로 대체합니다. Key tuple은 `(model, precision, TestClass)`이고, `TestClass.__name__`을 읽어 올바른 extractor로 dispatch합니다.
- **Class handlers:**

| Class | Handler |
|---|---|
| `TestBenchmark` | `_benchmark_perf` |
| `TestBenchmarkapp` | `_benchmark_app_perf` |
| `TestStableDiffusion` | `_sd_perf_ms` (legacy C++ binary — pickle에는 ms, ingest 때 1000으로 나눔) |
| `TestStableDiffusionGenai` | `_sd_perf_sec` (pickle 값이 이미 seconds) |
| `TestStableDiffusionDGfxE2eAi` | `_sd_perf_sec` (pickle 값이 이미 seconds) |

- **Dropped handlers:** `TestMeasuredUsageCpp`, `TestWhisperBase` — [Dropped qwen_usage and Whisper base](#dropped-qwen_usage-and-whisper-base)를 참고하세요.
- **Version parsing:** OV version은 filename `daily.<stamp>.<ov_version>.pickle`과 `.report` text (`Purpose` + `OpenVINO` line)에서 가져옵니다. Report text가 있으면 그것을 우선합니다.
- **SD unit normalization:** 모든 SD pipeline은 source class와 관계없이 DB에 `unit='s'`로 저장됩니다. Legacy ms pickle은 load 시점에 1000으로 나눕니다. 이로써 SD-XL pipeline이 viewer에서 8초가 아니라 8ms처럼 보이던 1000배 ambiguity를 제거했습니다.

### `daily/viewer/ingest/writer.py`
DuckDB upsert (각 `RunRecord` 단위 transaction) + display-profile loader입니다.

- **Dedup:** run 단위로 `perf`를 통째로 교체합니다. (DELETE + INSERT) 한 run 안의 duplicate PK는 last-wins로 deduplicate합니다.
- **File-hash skip:** `already_ingested()`는 `runs.file_hash`를 확인합니다. `--force`가 없으면 skip합니다.

### `daily/viewer/ingest/cli.py`
단일 CLI입니다. `--root`는 scan, `--input`은 단일 file을 처리합니다.

- **Auto-detect (`_classify`):** `*.summary.json` -> new; sibling `.report`가 있는 `*.pickle` -> old; `.report` 단독 입력은 sibling pickle이 있으면 old, 아니면 skip.
- **Progress:** inline progress bar. 실패는 모아서 마지막에 보고하며 fatal은 아닙니다.
- **Profile load:** default profile은 additive입니다. ingest CLI invocation마다 profile이 없을 때만 `profiles/default.yaml`을 로드합니다. 명시적인 `--profile` 또는 `--force`는 `--skip-profile`이 없는 한 지정 profile을 refresh합니다.

### `daily/viewer/queries.py`
`app.py`에서 사용하는 DuckDB helper입니다. 모든 함수는 pandas DataFrame을 반환합니다.

| Function | 설명 |
|---|---|
| `list_machines` | `SELECT DISTINCT machine` |
| `list_runs` | machine 하나에 대한 run metadata, newest first |
| `list_profiles` | `SELECT DISTINCT profile FROM display_rows` |
| `build_excel_matrix` | `display_rows` x selected runs를 cross-join하고 `perf_with_buckets`에 LEFT JOIN한 뒤 manual pivot. pandas `pivot_table`은 NaN index를 drop하고 `dropna=False`에서 폭발하므로 직접 처리합니다. |
| `extra_rows` | selected runs 안에서 display profile에 포함되지 않은 perf rows — QA 보조용 |
| `series_history` | `(machine, model, precision, in_token, out_token, exec_mode)` 하나의 time series. purpose-filtered `perf_flat` rows에서 rolling median / MAD를 다시 계산합니다. Regression 탭의 trend plot에서 사용합니다. |
| `trend_regressions` | **NEW** — recent window median과 baseline window median을 비교하는 per-series regression signal. `worsening_pct`는 ms와 FPS 등 단위 차이에 관계없이 positive = worse가 되도록 sign-normalised됩니다. Regression 탭의 기반입니다. |
| `noise_summary` | window 기준 per-series CV |
| `geomean_trend` | run별 `exp(avg(ln(value)))`, `exec_mode` / `in_bucket` / `out_bucket` / `excluded_models`로 filter |

### `daily/viewer/app.py`
Streamlit entry입니다. 5개 tab이 있습니다.

| Tab | 목적 |
|---|---|
| Dashboard | 첫 번째 tab입니다. `DEFAULT_RUN_FILTER`와 purpose/description이 match되는 최신 run을 선택합니다. fallback은 `daily_CB`입니다. summary / `.report` / `.pytest.json` / `.raw` artifact를 검토해서 test가 실제로 실행되었는지, pass/fail total, grouped failure cause, per-test pytest-json/summary message, raw pytest log text를 보여줍니다. Artifact sibling은 `source_path` stem에서 파생합니다. (`.summary.json` / `.pickle` suffix 제거) `report_file`은 source identity일 뿐이며 text/JSON reader에 직접 넣지 않습니다. |
| Excel | run 선택 -> wide matrix (profile rows x run stamps) + tab-separated paste block + "extra rows" expander |
| Regression | MERGED tab입니다. 이전의 Trend + Regressions를 합쳤습니다. worsening % 기준으로 ranked table을 보여주고, 선택된 row에 대해 one-series-at-a-time trend plot을 표시합니다. Baseline median + recent median은 legend entry가 보이도록 Scatter trace로 렌더링합니다. rolling ±2σ band는 밝은 모니터에서도 보이도록 visible fill (rgba 0.28)로 그립니다. |
| Geomean | `exec_mode x in_bucket x out_bucket` geomean trend + band + latest-point alert (사용자 요청) |
| Noise | Per-series CV table sorted desc (iGPU diagnostics) |

Sidebar:
- **Machine filter:** "Daily machines only" checkbox (default `True`). `True`이면 dropdown은 `DAILY_MACHINES` constant (`dg2alderlake`, `MTL-01`, `ARLH-01`, `BMG-02`, `LNL-03`, `LNL-04`, `DUT4580PTLH`, `DUT6047BMGFRD`)로 filter됩니다. 교집합이 없으면 full list로 fallback합니다.
- **Default filter:** Excel tab의 purpose filter는 `DEFAULT_RUN_FILTER` constant를 통해 `daily_CB timer`가 기본값입니다.
- **Days slider:** trend history는 7-60일 범위입니다. Geomean, Noise의 tab-local "days" input도 일관성을 위해 60으로 cap합니다.

Other:
- **Unit display:** 사용자에게 보이는 모든 numeric에는 unit을 붙입니다. table median column은 `8.060 s`, trend heading은 `[s]`, caption은 `Recent median = 8.060 s`처럼 표시합니다. SD pipeline seconds가 ms로 오해되는 것을 막습니다.
- **Caching:** 모든 query는 DB mtime을 key로 포함하는 `@st.cache_data`로 감쌉니다. re-ingest 후 cache가 자동 invalidation됩니다.
- **Config source:** `DAILY_DB` env var 또는 `-- --db <path>` trailing arg를 사용합니다. streamlit이 자체 flag를 먼저 소비하기 때문입니다.

### `daily/run.py`
Daily suite entry입니다. pytest를 실행하고 report를 만들고 mail/xlsx를 전달합니다.

- **Exit policy:** run이 end-to-end로 완료되어 report가 생성되면 test failure와 관계없이 항상 `0`을 반환합니다. Test failure는 JSON summary / mail에 반영하고 exit code로 표현하지 않습니다. pytest JSON이 생성되지 않는 진짜 infra failure만 non-zero rc를 propagate합니다. Jenkins가 "test failed"와 "run itself broke"를 구분할 수 있도록 사용자 요청에 따라 변경되었습니다.

### Legacy / independent modules
| Path | 상태 |
|---|---|
| `daily/viewer/_old_viewer.py` | Legacy reference. import하지 마세요. |
| `daily/viewer/_old_ingest.py` | Legacy reference. import하지 마세요. |
| `daily/viewer/_old_schema.sql` | Legacy reference. 사용하지 마세요. |
| `daily/viewer/perf_rows.py` | `xlsx_update.py` path에서 사용합니다. summary를 flatten해서 lookup dict로 만듭니다. 그대로 유지합니다. **NOTE:** 이 경로는 xlsx template용으로 bucketing (`'short'` / `'long'`)을 수행합니다. RAW token을 유지해야 하는 DB ingest에는 재사용하지 마세요. |
| `daily/viewer/xlsx_update.py` | `run.py`에서 호출하는 별도 xlsx append path입니다. DuckDB pipeline과 독립적입니다. |

---

## 설계 결정

### FIXED_ROW_ORDER replacement
- **Choice:** YAML profile + DB `display_rows` table.
- **Rejected:** Python list in code (row별 git diff가 어렵고 version control이 약함); DB-only (git history 없음).
- **Why:** 사용자가 model을 자주 추가/삭제합니다. YAML은 git에서 review하기 좋고, DB copy는 queries에서 빠른 JOIN을 위해 사용합니다. Match spec (`short` / `long` / `*` / `<int>`)은 code change 없이도 충분히 표현력이 있습니다.

### Raw tokens in DB vs bucketed
- **Choice:** raw only.
- **User request:** B — ingest는 data-as-is이고, viewer가 bucketing합니다.
- **Implementation:** `perf_with_buckets` view가 `in_bucket` / `out_bucket`을 파생합니다. Threshold는 SQL에 있으므로 view만 바꾸면 되고 re-ingest는 필요 없습니다.

### Regression detection method (current approach)
- **Choice:** two-window median comparison — recent-window median vs baseline-window median.
- **Rejected:**
  - Single-point robust z-score — 사용자 피드백: 오늘의 outlier가 아니라 최근 block이 drift 중인지 보고 싶습니다. Single-point test는 작은 blip에도 뒤집히고, data가 noisy하거나 가끔 corrupt될 때 쓸모가 떨어집니다.
  - mean/std z-score — iGPU outlier에 너무 민감합니다.
- **Why:** iGPU run은 noisy하고 개별 data point가 가끔 오염됩니다. 두 time window의 median을 비교하면 outlier와 ingest noise가 완화되면서 slow drift는 잡을 수 있습니다. ASV / pytest-benchmark 같은 benchmark tool도 single point 반응보다 median-like robust summary와 percentage threshold를 중시합니다.
- **Window defaults:** `recent_days=7`, `baseline_days=21`, `min_recent_points=5`, `min_baseline_points=7`.
  > 이 값들은 [`queries.trend_regressions`](queries.py) signature와 mirror됩니다. window size와 minimum sample count의 single source of truth는 function signature로 유지합니다.
- **UI threshold defaults:** `pct_threshold_from_sidebar=0.05`, `z_threshold_from_sidebar=3.0`, `noisy_cv_threshold=0.10`.
  > 이 값들은 `trend_regressions`가 아니라 Streamlit sidebar에서 제어합니다.
- **Direction normalisation:** `worsening_pct`는 positive가 항상 "worse"가 되도록 sign 처리합니다. ms/s/%는 recent>baseline이면 `+pct`, FPS/tps는 recent<baseline이면 `+pct`입니다. `worsening_z`는 baseline MAD (`sigma ≈ 1.4826 * MAD`)를 사용해서 recent noise가 비교를 숨기거나 부풀리지 않도록 합니다. UI는 threshold-normalised severity = `max(worsening_pct / pct_threshold, worsening_z / z_threshold)` 기준으로 정렬합니다.
- **Purpose filter:** regression summary와 selected-series history는 purpose에 `daily_CB timer`가 포함된 run만 사용합니다. 개인 PR run이 baseline/recent window를 오염시키지 않도록 하기 위함입니다. Chart rolling band는 `perf_stats`가 아니라 filtered `perf_flat` rows에서 재계산합니다. `perf_stats`는 all-purpose view이기 때문입니다.
- **UI:** single merged "Regression" tab. Ranked table (worst first by threshold-normalised severity) + selected row별 one trend chart. `app.py:_tab_regression`을 참고하세요.
- **Supersedes:** 이전 rolling z-score point-vs-band helper는 제거했습니다. `trend_regressions`가 UI와 mail alert에서 사용하는 단일 regression signal입니다.

### One series per trend chart
- **Choice:** Regression tab에서 single-series plot을 강제합니다.
- **Why:** 값 범위가 크게 다른 model을 하나의 chart에 섞으면 읽기 어렵습니다. Table-plus-single-chart가 scale되는 pattern입니다.
- **Implementation:** summary table의 row selection이 plot할 series를 결정합니다. 기본값은 row 0 (worst drift)입니다. y-axis range에는 minimum relative span이 있어 `31.6 ms` vs `31.95 ms` 같은 작은 안정적 차이를 과도하게 zoom해서 visually noisy하게 만들지 않습니다.

### SD pipeline unit normalization
- **Problem observed:** legacy `TestStableDiffusion` pickle은 ms를 저장했고, newer `TestStableDiffusionGenai` / `TestStableDiffusionDGfxE2eAi` pickle은 seconds를 저장했습니다. 기존 `_sd_perf`는 모두 `unit='ms'`로 묶어 SD-XL pipeline (실제로는 8 s)이 comparison에서 8 ms처럼 보였습니다.
- **Fix:** `_sd_perf_ms` (legacy, `/1000`)와 `_sd_perf_sec` (new, as-is)로 분리했습니다. 이제 모든 SD pipeline은 `unit='s'`로 저장됩니다.
- **Remediation:** fix 후 `--force`로 re-ingest했습니다. 확인 결과 DB에 남은 ms-labeled SD row는 0개입니다.
- **Commit:** `708a853`.

### Units visible in every UI surface
- **Choice:** table cell, plot heading, caption에 unit suffix를 붙입니다.
- **Why:** unit 없는 raw number 때문에 SD-XL confusion이 생겼습니다. 사용자가 어떤 test type이 어떤 unit인지 외울 필요가 없어야 합니다.
- **Implementation:** regression table은 `8.060 s` 같은 display-only `recent` / `baseline` column을 만듭니다. raw numeric column은 `column_config`로 숨기지만 plot/caption code path를 위해 유지합니다.

### Dropped qwen_usage and Whisper base
- **User request:** 둘 다 완전히 제거합니다.
- **Action:** `loader_new.py` + `loader_old.py`에서 handler를 제거했고, DB의 기존 row도 삭제했습니다.
- **Note:** `whisper-large-v3`는 Whisper base가 아니라 GenAI pipeline의 다른 model이므로 유지합니다.

### Geomean alerting
- **User request:** C — per-series뿐 아니라 geomean-level trend alert도 명시적으로 원했습니다.
- **Implementation:** `geomean_trend()`는 `(machine, exec_mode, in_bucket, out_bucket)` 기준으로 group합니다. Tab은 geomean series 자체의 median + MAD를 계산하고 latest point가 `±z·σ` AND `±pct%` 밖이면 banner를 표시합니다.

### Daily machines filter
- **Choice:** `app.py`의 hardcoded `DAILY_MACHINES` tuple + sidebar checkbox (default `True`).
- **Rejected:** DB-backed list. 운영 대상 machine set은 runtime mutation이 필요 없고 git이 source of truth로 적합합니다.
- **Machines:** `dg2alderlake`, `MTL-01`, `ARLH-01`, `BMG-02`, `LNL-03`, `LNL-04`, `DUT4580PTLH`, `DUT6047BMGFRD`.
- **Fallback:** checkbox filter 후 dropdown이 비면 (fresh DB) full machine list로 fallback합니다.

### `run.py` exit-code policy
- **User request:** test pass 여부가 아니라 run이 완료되었으면 항상 `0`을 반환해야 합니다. Infra failure와 test failure는 구분되어야 합니다.
- **Choice:** pytest rc와 관계없이 `build_reports`가 성공하면 `0` 반환. pytest JSON이 생성되지 않는 진짜 infra break만 non-zero를 propagate합니다.
- **Rationale:** Jenkins는 non-zero를 "build broke"로 처리합니다. 팀은 "test failed"를 Jenkins status가 아니라 mail/report로 표현하기를 원합니다.

### `run_id`
- **Choice:** `sha1(machine|iso_ts|report_file)[:20]`.
- **Why:** `iso_ts`를 포함하면 file stem이 우연히 충돌해도 몇 분 간격의 re-run을 구분할 수 있습니다.

### `file_hash` for dedup
- **Choice:** source file (pickle 또는 summary.json)의 sha256 content hash.
- **Why:** path change / rsync / backup 후에도 안정적입니다. copy에서 깨지던 old `(path|size|mtime)` scheme을 대체합니다.
- **Cost:** hashing은 I/O-bound지만 amortised됩니다. `/var/www/html/daily` 기준 6006 files에 57초였습니다.

### rawlog storage
- **Choice:** body가 아니라 PATH만 저장합니다. (`rawlog_path`)
- **Old behavior:** `_old_schema.sql`은 full text를 `runs.rawlog`에 저장했고, 이 때문에 DB가 커졌습니다.
- **Mitigation:** UI에서 log access가 필요하면 path에서 파일을 엽니다. Dashboard는 `.raw`를 raw pytest log expander로 표시합니다.

### Pickle unpickle without legacy modules
- **Choice:** `_TolerantUnpickler.find_class`는 `ModuleNotFoundError`에서 `_UnknownClass` stub을 반환합니다.
- **Why:** legacy pickle은 `test_cases.test_benchmark.TestBenchmark` 등을 참조합니다. Viewer에 해당 package를 import하고 싶지 않습니다. Extractor dispatch에는 class NAME만 필요합니다.

### Transaction granularity
- **Choice:** `RunRecord` 하나당 transaction 하나 (`runs` + `system_devices` + `perf` together).
- **Why:** old ingest의 partial-state bug (separate insert)는 고통스러웠습니다. Run이 자연스러운 단위입니다.

### Streamlit caching
- **Choice:** `_v=DB.stat().st_mtime`을 tiebreaker argument로 넣은 `@st.cache_data`.
- **Why:** Streamlit cache는 cache key가 바뀌면 invalidation됩니다. mtime을 넘기면 re-ingest 후 cached query가 수동 invalidation 없이 자동 refresh됩니다.

### Trend plot axis orientation
- **Choice:** `xaxis autorange reversed` — newest on the left.
- **Why:** 사용자 요청입니다. 팀의 newest-first 읽기 convention과 맞습니다.

---

## SQL match rules

- **Location:** `queries.build_excel_matrix` (`extra_rows`에도 mirror됨).
- **Rule:** `display_rows.<spec>`는 다음 방식으로 `perf_with_buckets`와 match합니다.

```text
spec = '*'                         -> always matches
spec = 'short'/'long'/'0'          -> matches the corresponding <side>_bucket column
TRY_CAST(spec AS INTEGER) = token  -> exact numeric match
else                               -> no match
```

세 조건은 OR로 묶입니다. `TRY_CAST`는 non-integer string에 대해 `NULL`을 반환하므로 비교가 깔끔하게 실패합니다.

---

## 실행 방법

```bash
# 전체 ingest
cd daily && conda run -n daily python -m viewer.ingest.cli --root /var/www/html/daily --db viewer/bench.duckdb

# 단일 파일 ingest (new format)
cd daily && conda run -n daily python -m viewer.ingest.cli --input output/daily.<stamp>.summary.json

# 단일 파일 ingest (old format)
cd daily && conda run -n daily python -m viewer.ingest.cli --input /var/www/html/daily/LNL-02/daily.<stamp>.<ver>.pickle

# 강제 re-ingest
# 위 명령에 --force 추가

# Profile만 다시 로드
conda run -n daily python -m viewer.ingest.cli --root /dev/null --profile viewer/profiles/default.yaml

# Viewer 실행
cd daily && conda run -n daily streamlit run viewer/app.py -- --db viewer/bench.duckdb

# env var로 다른 DB 지정
DAILY_DB=/path/to/bench.duckdb conda run -n daily streamlit run viewer/app.py
```

**Python env:** conda env `daily`를 사용합니다. (`/home/sungeunk/miniforge3/envs/daily/bin/python`) System `python3`에는 duckdb/streamlit이 없습니다.

---

## Conventions

### Units

| Unit | 의미 | Direction |
|---|---|---|
| `ms` | latency | lower is better |
| `s` | seconds (SD / LCM / whisper / flux pipelines) | lower is better |
| `FPS` | throughput (benchmark_app) | higher is better |
| `tps` | tokens per second | higher is better |
| `%` | percent | regression direction에서는 lower-is-better로 처리 |

### Other
- **`exec_mode_values`:** `"1st"`, `"2nd"`, `"pipeline"`, `"batch:<n>"`, `"tps"`.
- **`token_bucket_threshold`:** `100`.
- **`regression_window_defaults`:** [Regression detection method](#regression-detection-method-current-approach)를 참고하세요.

---

## Known gaps & future work

- **Unit-test coverage 확장.** viewer helper/loader/query/mail-alert formatting에 대한 focused pytest coverage를 추가했습니다. Future work: old pickle variant edge case와 더 큰 synthetic DuckDB history fixture를 보강합니다.
- **`perf_stats`는 여전히 correlated subquery를 사용합니다. (개념적으로 O(n·m))** 2026-04-27 기준 322k `perf_stats` rows에서 약 0.57초로 측정되어 아직 cached table은 필요 없습니다. query time이 1초를 넘으면 ingest 시점에 Python-side precomputation으로 `perf_stats_cached` table에 쓰는 방식으로 바꾸는 것을 고려합니다.
- **`run.py`의 email regression alert.** mail delivery 전에 best-effort report section으로 구현했습니다. Future work: 수신자가 더 풍부한 형식을 원하면 threshold tuning이나 전용 HTML table 추가를 고려합니다.
- **Non-canonical location에서 ingest한 pickle의 machine name.** `loader_old.py`는 기본적으로 parent-dir name에서 machine을 파생합니다. `/var/www/html/daily/<MACHINE>/` 밖에 있는 fixture/archive 파일을 rescue할 때는 `viewer.ingest.cli --machine <name>`을 사용합니다.
- **추가 display profile.** profile이 하나뿐이면 sidebar는 profile dropdown을 숨깁니다. 실제로 선택할 두 번째 display layout이 생기면 iGPU-focused profile을 추가합니다.
