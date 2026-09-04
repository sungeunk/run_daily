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
| New | `.summary.json` (`daily/run.py` -> `report/builder.py`) -> ingest -> DuckDB |

- ingest는 `RunRecord` dataclass를 만든 뒤 `writer.upsert_run`으로 저장합니다.
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
- `runs.source_format`은 `'old' | 'new'`입니다. legacy loader 제거 이전에 ingest된 row의 출처 디버깅을 위해 유지합니다.

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
- **Meta handling:** `summary.meta`는 optional입니다. 없으면 filename stamp + parent-dir machine + `generated_at` ts로 fallback합니다. meta가 추가되기 전의 초기 summary 파일도 동작합니다.

### `daily/viewer/ingest/writer.py`
DuckDB upsert (각 `RunRecord` 단위 transaction) + display-profile loader입니다.

- **Dedup:** run 단위로 `perf`를 통째로 교체합니다. (DELETE + INSERT) 한 run 안의 duplicate PK는 last-wins로 deduplicate합니다.
- **File-hash skip:** `already_ingested()`는 `runs.file_hash`를 확인합니다. `--force`가 없으면 skip합니다.

### `daily/viewer/ingest/cli.py`
단일 CLI입니다. `--root`는 scan, `--input`은 단일 file을 처리합니다.

- **Detection (`_classify`):** `*.summary.json` -> new. 그 외는 skip합니다.
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
Streamlit entry입니다. 4개 tab이 있습니다.

| Tab | 목적 |
|---|---|
| Dashboard | machine 이름 부분 문자열 filter로 좁힌 뒤, machine마다 bordered card 하나를 보여줍니다. card 구성 순서는 latest run의 failing models -> rig 변경 note -> newest clean run과 newest failed run을 좌우 두 run card로 -> metric별 geomean trend입니다. |
| Excel | run 선택 -> wide matrix (profile rows x run stamps) + tab-separated paste block + "extra rows" expander |
| Compare | run A와 run B를 series 단위로 비교합니다. 모든 A/B delta는 각 run 이전 history와도 비교해서 단순 scatter인지 실제 변화인지 구분합니다. |
| Exclusions | 특정 machine+run을 모든 cohort 기반 view에서 수동으로 제외합니다. |

Sidebar에는 global 설정만 둡니다: DB path, "Refresh database", display profile,
chart y-axis range.

Analysis scope는 tab별입니다 (`_scope_controls`): history depth (run 단위, 3-20),
`Purpose` filter, run당 최소 successful series 수. sidebar state를 공유하지 않고
각 view가 자기 window를 소유합니다. machine 선택도 tab별입니다 (`_machine_picker`).

Other:
- **Unit display:** 사용자에게 보이는 모든 numeric에는 unit을 붙입니다. trend heading은 `[s]`, caption은 `Recent median = 8.060 s`처럼 표시합니다. SD pipeline seconds가 ms로 오해되는 것을 막습니다.
- **Caching:** 모든 query는 `_cache_version()`을 key로 포함하는 `@st.cache_data`로 감쌉니다. (아래 참조)
- **Config source:** `DAILY_DB` env var 또는 `-- --db <path>` trailing arg를 사용합니다. streamlit이 자체 flag를 먼저 소비하기 때문입니다.

### `daily/run.py`
Daily suite entry입니다. pytest를 실행하고 report를 만들고 mail/xlsx를 전달합니다.

- **Exit policy:** run이 end-to-end로 완료되어 report가 생성되면 test failure와 관계없이 항상 `0`을 반환합니다. Test failure는 JSON summary / mail에 반영하고 exit code로 표현하지 않습니다. pytest JSON이 생성되지 않는 진짜 infra failure만 non-zero rc를 propagate합니다. Jenkins가 "test failed"와 "run itself broke"를 구분할 수 있도록 사용자 요청에 따라 변경되었습니다.

### Legacy / independent modules
모두 제거되었습니다. 삭제 결정을 다시 되풀지 않도록 항목은 남겨둡니다.

| Path | Status |
|---|---|
| `daily/viewer/_old_viewer.py` | 삭제됨. `app.py`로 대체, import하는 곳 없음. |
| `daily/viewer/_old_ingest.py` | 삭제됨. `ingest/`로 대체. |
| `daily/viewer/_old_schema.sql` | 삭제됨. `schema.sql`로 대체. |
| `daily/viewer/perf_rows.py` | xlsx 경로와 함께 삭제됨. `summary.json`을 master xlsx lookup key 형태로 평탄화하고 구 `FIXED_ROW_ORDER` template을 위해 token을 `'short'` / `'long'`으로 bucketing했습니다. DuckDB 경로는 이를 사용한 적이 없으며 `ingest/loader_new.py`가 자체 추출하고 raw token을 유지합니다. |
| `daily/viewer/xlsx_update.py` | 삭제됨. master xlsx workflow(`run.py --xlsx-update`)는 더 이상 존재하지 않으며 DuckDB viewer가 그 역할을 대체했습니다. |

---

## 설계 결정

### FIXED_ROW_ORDER replacement
- **Choice:** YAML profile + DB `display_rows` table.
- **Rejected:** Python list in code (row별 git diff가 어렵고 version control이 약함); DB-only (git history 없음).
- **Why:** 사용자가 model을 자주 추가/삭제합니다. YAML은 git에서 review하기 좋고, DB copy는 queries에서 빠른 JOIN을 위해 사용합니다. Match spec (`<prompt_idx>` / `*` / `<int>`)은 code change 없이도 충분히 표현력이 있습니다.

### Excel-paste join key: prompt_idx over token buckets
- **Problem observed:** `in_spec`이 `'short'`/`'long'` token-count bucket(threshold 100)으로 매칭되던 방식은, 일부 model이 2개보다 많은 prompt를 사용(예: `phi-4-multimodal-instruct`는 4개)하면서 서로 다른 prompt가 같은 `'long'` bucket으로 뭉개져 하나의 `display_rows` row만 매칭될 수 있는 문제가 있었습니다.
- **Fix:** 원본 `.jsonl`의 prompt index를 `perf.prompt_idx`로 저장하고, `display_rows.in_spec`에 그 index 값을 직접 넣습니다(`'0'`, `'1'`, `'2'`, ...). `build_excel_matrix` / `extra_rows`는 bucket 대신 `prompt_idx`로 매칭합니다.
- **Migration:** 기존 DB는 `ALTER TABLE ... ADD COLUMN ... DEFAULT 0`으로 `prompt_idx`를 얻습니다. 이 변경 전에 ingest된 기존 row는 `--force`로 재적재하기 전까지 `prompt_idx=0`으로 조회됩니다.

### Raw tokens in DB vs bucketed
- **Choice:** raw only.
- **User request:** B — ingest는 data-as-is이고, viewer가 bucketing합니다.
- **Implementation:** `perf_with_buckets` view가 `in_bucket` / `out_bucket`을 파생합니다. Threshold는 SQL에 있으므로 view만 바꾸면 되고 re-ingest는 필요 없습니다.

### Regression detection method (query layer)
- **Scope:** 이 항목은 `queries.trend_regressions` / `compare_runs_with_trend`를 설명합니다. 전용 "Regression" tab은 제거되었고, 해당 방식은 Compare tab(각 A/B delta를 그 run 이전 history와 비교)과 mail alert에서 사용됩니다.
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
- **Purpose filter:** regression summary와 selected-series history는 tab의 `Purpose` scope control로 filter합니다. 개인 PR run이 baseline/recent window를 오염시키지 않도록 하기 위함입니다. Chart rolling band는 `perf_stats`가 아니라 filtered `perf_flat` rows에서 재계산합니다. `perf_stats`는 all-purpose view이기 때문입니다.
- **Supersedes:** 이전 rolling z-score point-vs-band helper는 제거했습니다. `trend_regressions`가 mail alert에서 사용하는 단일 regression signal입니다.

### One series per trend chart
- **Choice:** single-series plot을 강제합니다 (Compare tab).
- **Why:** 값 범위가 크게 다른 model을 하나의 chart에 섞으면 읽기 어렵습니다. Table-plus-single-chart가 scale되는 pattern입니다.
- **Implementation:** summary table의 row selection이 plot할 series를 결정합니다. y-axis range에는 minimum relative span이 있어 `31.6 ms` vs `31.95 ms` 같은 작은 안정적 차이를 과도하게 zoom해서 visually noisy하게 만들지 않습니다.

### SD pipeline unit normalization
- **Problem observed:** legacy `TestStableDiffusion` pickle은 ms를 저장했고, newer `TestStableDiffusionGenai` / `TestStableDiffusionDGfxE2eAi` pickle은 seconds를 저장했습니다. 기존 `_sd_perf`는 모두 `unit='ms'`로 묶어 SD-XL pipeline (실제로는 8 s)이 comparison에서 8 ms처럼 보였습니다.
- **Fix:** `_sd_perf_ms` (legacy, `/1000`)와 `_sd_perf_sec` (new, as-is)로 분리했습니다. 이제 모든 SD pipeline은 `unit='s'`로 저장됩니다.
- **Remediation:** fix 후 `--force`로 re-ingest했습니다. 확인 결과 DB에 남은 ms-labeled SD row는 0개입니다.
- **Commit:** `708a853`.

### Units visible in every UI surface
- **Choice:** table cell, plot heading, caption에 unit suffix를 붙입니다.
- **Why:** unit 없는 raw number 때문에 SD-XL confusion이 생겼습니다. 사용자가 어떤 test type이 어떤 unit인지 외울 필요가 없어야 합니다.
- **Implementation:** regression table은 `8.060 s` 같은 display-only `recent` / `baseline` column을 만듭니다. raw numeric column은 `column_config`로 숨기지만 plot/caption code path를 위해 유지합니다.

### Geomean on the dashboard
- **Choice:** machine card마다 metric별 geomean trend를 둘고, 해당 machine의 모든 run이 측정한 series로만 제한합니다 (`geomean_matrix`).
- **Why:** failure로 model을 잃은 run은 geomean이 아니라 success count가 움직여야 합니다. 그렇지 않으면 failure가 performance 변화처럼 읽힙니다.
- **History:** 단독 "Geomean" tab(bucket geomean + ±2σ band + latest-point banner, `geomean_trend` 기반)은 제거되었습니다. `geomean_trend`는 현재 UI에서 사용되지 않습니다.

### Daily machines filter
- **Choice:** `app.py`의 hardcoded `DAILY_MACHINES` tuple을 `_machines_in_scope()`에서 무조건 적용합니다.
- **Rejected:** DB-backed list. 운영 대상 machine set은 runtime mutation이 필요 없고 git이 source of truth로 적합합니다. sidebar toggle도 제거했습니다. report root에는 일회성 folder가 남아 있어 이를 보여주는 것은 의미가 없었습니다.
- **Fallback:** DB의 machine과 교집합이 없으면 (fresh DB) full machine list로 fallback합니다.

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
- **Choice:** `_v=_cache_version()`을 tiebreaker argument로 넣은 `@st.cache_data`. `_cache_version() = max(DB mtime, queries.py mtime)`입니다.
- **Why:** Streamlit cache는 cache key가 바뀌면 invalidation됩니다. DB mtime만 넣으면 re-ingest는 커버되지만, 서버가 오래 떠 있는 상태에서 `queries.py`를 배포하면 이전 SQL로 만들어진 frame이 계속 반환되어 column schema가 코드와 어긍나는 문제가 있었습니다. query module mtime을 함께 보면 코드 변경도 cache를 invalidation합니다.
- **Note:** `_db_version()`은 DB 전용으로 남겨둡니다. DB 존재 확인과, refresh 중 다른 session이 DB를 재생성했는지 감지하는 데 쓰입니다.

### Trend plot axis orientation
- **Choice:** default `xaxis autorange` — oldest on the left, newest on the right.
- **Why:** 사용자 요청입니다. chart는 시간순으로 읽고, table은 newest-first를 유지합니다.

---

## UI design guidelines

viewer는 운영 도구입니다. 목표는 화려함이 아니라 데이터가 명확히 구분되는 것입니다.
이 섹션을 house style로 간주하고, tab마다 새 styling을 만들지 마십시오.

### 참고 자료

| 주제 | 출처 |
|---|---|
| Table 정렬/밀도/구분선 | [Pencil & Paper — Data Table Design UX Patterns](https://www.pencilandpaper.io/articles/ux-pattern-analysis-enterprise-data-tables) |
| 숫자 표기 | [Datawrapper — data tables](https://www.datawrapper.de/blog/data-tables/) |
| Chart palette, dashboard 구성 | [IBM Carbon — Data visualization](https://carbondesignsystem.com/data-visualization/getting-started/) |
| Dashboard 목적, 인지 부하, 의미 있는 색 사용 | [Grafana — Dashboard best practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/) |
| 어떤 signal을 보여줄 것인가 | [Google SRE Book — Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) |
| App 전체 theming | [Streamlit — Theming](https://docs.streamlit.io/develop/concepts/configuration/theming) |

### 규칙

1. **한 화면은 한 질문에만 답합니다.** 각 card가 답하는 질문은 "이 machine은 오늘
   정상인가, 아니라면 언제부터인가"입니다. 이에 도움이 안 되는 것은 card에 넣지
   않습니다.
2. **나쁜 소식을 먼저.** card 순서는 failing models -> rig 변경 note -> run card
   2개 -> trend입니다. 무언가 깨졌다는 사실을 알려고 scroll하게 하지 않습니다.
3. **색은 의미만 전달합니다.** 장식용 색은 쓰지 않으며, 아래 status palette만
   허용합니다. 중립적인 값은 색을 입히지 않습니다.
4. **숫자는 우측 정렬 + 고정 소수점**이며 항상 unit을 함께 표시합니다
   (`8.06 s`, `41.2 min`, `85`). 가운데 정렬은 쓰지 않습니다.
5. **텍스트는 좌측 정렬**하고, column 이름을 모든 cell에 반복하지 않습니다.
6. **Zebra stripe를 쓰지 않습니다.** Streamlit dataframe에는 이미 hover/selection
   state가 있어 교차 배경이 세 번째 회색으로 충돌합니다. 그룹핑은
   `st.container(border=True)` card로 합니다.
7. **Chart는 왼쪽에서 오른쪽으로 시간순**으로 읽히며, y축에 최소 span을 두어
   안정적인 series가 톱니 모양으로 보이지 않게 합니다. 최신 run에는 점선 원을
   씨우고, 모든 점의 hover에 OV version과 purpose를 담습니다.
8. **빈 상태는 글로 쓴다** (`never`, `none.`, `—`). 공백이나 `None`으로 두지
   않습니다.
9. **Theme은 한 곳에.** App 수준의 색/폰트/radius는 `.streamlit/config.toml`에,
   코드 수준의 색은 module 상수 하나에 둡니다. 호출부에 inline하지 않습니다.

### Status / case 색상

| 의미 | 색 | 사용처 |
|---|---|---|
| success / 정상 | `#1f77b4` | case bar, success count |
| skipped / 중립 | `#9e9e9e` | case bar, skip count |
| failed / 문제 | `#d62728` | case bar, failed count |

Run status는 emoji prefix(`🟢 success`, `🟡 stale`, `🔴 failed`, `⚪ unknown`)로도
표시해서 흑백이나 색각 이상 환경에서도 식별되도록 합니다. palette를 다시 잡을 경우
색각 안전한 Okabe-Ito set(`#0072B2` 파랑, `#999999` 회색, `#D55E00` 주황)을
우선합니다. 적록색약에서도 파랑/빨강 구분이 유지됩니다.

### 알려진 예외

- `_case_bar()`는 raw HTML을 쓴다. `st.progress`는 단색이라 success/skip/failed
  분할을 막대 하나에 보여줄 수 없습니다.
- 색상이 아직 `app.py`에 literal로 남아 있습니다. `config.toml`의
  `theme.chartCategoricalColors`로 옮기는 것은 미완료 작업입니다.

---

## SQL match rules

- **Location:** `queries.build_excel_matrix` (`extra_rows`에도 mirror됨).
- **Rule:** `display_rows.<spec>`는 다음 방식으로 `perf_with_buckets`와 match합니다.

```text
spec = '*'                         -> always matches
TRY_CAST(spec AS INTEGER) = prompt_idx (in_spec) / out_token (out_spec)
                                    -> exact match
else                                -> no match
```

세 조건은 OR로 묶입니다. `TRY_CAST`는 non-integer string에 대해 `NULL`을 반환하므로 비교가 깔끔하게 실패합니다.

---

## 실행 방법

```bash
# 전체 ingest
cd daily && conda run -n daily python -m viewer.ingest.cli --root /var/www/html/daily --db ../daily_output/<machine>/bench.duckdb

# 단일 파일 ingest (new format)
cd daily && conda run -n daily python -m viewer.ingest.cli --input output/daily.<stamp>.summary.json

# 단일 파일 ingest
cd daily && conda run -n daily python -m viewer.ingest.cli --input /var/www/html/daily/LNL-02/daily.<stamp>.summary.json

# 강제 re-ingest
# 위 명령에 --force 추가

# Profile만 다시 로드
conda run -n daily python -m viewer.ingest.cli --root /dev/null --profile viewer/profiles/default.yaml

# Viewer 실행
cd daily && conda run -n daily streamlit run viewer/app.py -- --db ../daily_output/<machine>/bench.duckdb

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

- **Unit-test coverage 확장.** viewer helper/loader/query/mail-alert formatting에 대한 focused pytest coverage를 추가했습니다. Future work: 더 큰 synthetic DuckDB history fixture를 보강합니다.
- **`perf_stats`는 여전히 correlated subquery를 사용합니다. (개념적으로 O(n·m))** 2026-04-27 기준 322k `perf_stats` rows에서 약 0.57초로 측정되어 아직 cached table은 필요 없습니다. query time이 1초를 넘으면 ingest 시점에 Python-side precomputation으로 `perf_stats_cached` table에 쓰는 방식으로 바꾸는 것을 고려합니다.
- **`run.py`의 email regression alert.** mail delivery 전에 best-effort report section으로 구현했습니다. Future work: 수신자가 더 풍부한 형식을 원하면 threshold tuning이나 전용 HTML table 추가를 고려합니다.
- **추가 display profile.** profile이 하나뿐이면 sidebar는 profile dropdown을 숨깁니다. 실제로 선택할 두 번째 display layout이 생기면 iGPU-focused profile을 추가합니다.
- **Dead code 제거 완료 (2026-09).** `_old_viewer.py`, `_old_ingest.py`, `_old_schema.sql`, `perf_rows.py`, `xlsx_update.py`를 삭제했고, Regression/Geomean/Noise/Functional tab 제거로 호출자를 잃은 query 함수도 함께 제거했습니다: `geomean_trend`, `noise_summary`, `list_run_kinds`, `machine_stats_for_run`, `monitor_samples_for_run`, `functional_summary_for_runs`, `fetch_functional_history`, `fetch_functional_summary`, `fetch_analysis_overview`. `trend_regressions`와 `fetch_run_comparison`은 mail alert와 `compare_runs_with_trend`가 여전히 사용해서 유지했습니다.
- **Colour token을 `config.toml`로.** card와 chart 색이 아직 `app.py`에 literal로 남아 있습니다. `.streamlit/config.toml`로 옮겨 theme을 한 곳에서 지정합니다.
