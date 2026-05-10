# LLM Daily 리팩토링/재구현 설계

## 목적

현재 Jenkins에서 동작 중인 LLM daily 파이프라인을 유지보수하기 쉬운 구조로 정리하고, 실행 결과를 누적/분석/알림까지 연결 가능한 형태로 재구현한다.

핵심 목표는 다음과 같다.

1. LLM daily 결과를 DuckDB에 누적 저장한다.
2. Streamlit 대시보드에서 regression, 버전별 성능 변화, functional 이슈를 빠르게 리뷰한다.
3. `daily/run.py` 실행 후 생성되는 report 최상단에 이번 실행의 분석 summary를 자동으로 추가한다.
4. regression 또는 functional 이슈가 발생하면 마지막 성공 build와 비교한 bisect 보조 정보를 메일로 발송한다.

## 현재 구현 상태 요약

현재 코드베이스는 이미 수집/누적/조회 기반이 갖춰져 있다. 신규 개발의 중심은 새로운 저장소를 만드는 것이 아니라, 기존 데이터를 활용하는 비교/판정/자동화 계층을 분리하고 안정화하는 것이다.

### 구현 완료

- pytest 기반 daily 실행 래퍼
  - `daily/run.py`에서 테스트 실행, `summary.json`, text report 생성, 후처리 메일/백업/xlsx 연결을 수행한다.
- DuckDB ingest 인프라
  - `daily/viewer/ingest/cli.py`, `daily/viewer/ingest/writer.py`, `daily/viewer/schema.sql` 기반으로 new `summary.json` 포맷과 legacy `pickle/report` 포맷을 모두 적재한다.
- Streamlit 조회 UI
  - `daily/viewer/app.py`에서 Dashboard, Excel, Regression, Geomean, Noise, Functional, Compare 탭을 제공한다.
  - `daily/viewer/queries.py`에서 trend/regression/geomean/noise 조회 함수를 제공한다.
- 통계 기반 regression 신호 계산
  - recent window와 baseline window의 median, MAD, z-score, pct 변화량 기반 계산이 구현되어 있다.
  - run 완료 후 regression alert를 report에 append하는 경로가 있다.
- 기본 delivery 기능
  - `daily/common/delivery.py`를 통해 메일 발송과 scp 백업을 수행한다.
- 기본 파이프라인 테스트
  - `daily/tests/test_viewer_pipeline.py`에서 ingest, query, regression 계산 경로를 검증한다.
- 독립 분석 엔진 분리 (P0 완료)
  - `daily/analysis` 패키지(`types/baseline/verdict/functional/engine/report/persistence`)가 추가되었다.
  - `AnalysisResult` contract 기반으로 판정/요약/저장을 공통화했다.
- run-to-analysis 오케스트레이션 정리
  - `daily/run.py`는 내부 분석 헬퍼를 제거하고 분석 엔진 호출 경로로 통합되었다.
  - DB 파일이 없는 첫 실행에서도 분석이 부트스트랩되도록 수정되었다.
- summary/report/DB 동시 반영
  - `summary.json`에 `analysis` block을 기록한다.
  - report 최상단에 `[ Analysis summary ]`를 prepend한다.
  - `analysis_results`, `analysis_comparisons`, `functional_issues` 집계 테이블이 추가되었다.
- 분석 저장 안정화
  - 분석 DB 저장은 트랜잭션 단위로 처리되며 실패 시 rollback한다.
  - rerun 시 stale row 정리, `threshold_pct` 저장, legacy DB 컬럼 마이그레이션(`ADD COLUMN IF NOT EXISTS`)이 반영되었다.
- 분석 테스트 확대
  - `daily/tests/test_analysis_engine.py`에서 verdict 경계값, baseline 우선순위, invalid/NaN, green-only baseline, rollback, schema migration 등을 검증한다.

### 부분 구현

- functional 이슈 리뷰
  - functional 히스토리/집계 뷰는 추가되었지만, stderr 기반 category 분류는 아직 placeholder 상태다.
- OpenVINO 버전별 성능 변화 분석
  - `ov_version`, `ov_build`, `ov_sha` 저장과 run-to-run 직접 비교는 가능하지만, last-good 비교 패널과 model-level summary 전용 테이블은 남아 있다.
- report/summary/DB 재사용 체계
  - report, summary, DB, mail, dashboard는 analysis block을 재사용한다. bisect delta/template은 아직 별도 구현이 필요하다.
- last known good 기반 bisect 연동
  - `baseline.find_last_known_good`와 issue run summary/mail hint는 구현되었다. issue run과 last good run의 상세 delta 생성은 남아 있다.

### 미구현

- issue run vs last good run delta 생성
- bisect 보조 메일 템플릿 고도화와 자동 발송
- noisy/insufficient dual gate 고도화 판정

## 핵심 설계 방향

`daily/run.py`에 섞여 있는 비교/판정/report summary 로직을 독립 분석 계층으로 분리한다.

목표 인터페이스는 다음과 같다.

```text
summary.json + DuckDB 누적 데이터
    -> analysis engine
    -> structured analysis result
    -> summary.json analysis block
    -> report [ Analysis summary ] prepend
    -> mail summary
    -> dashboard / bisect query 재사용
```

이 구조를 통해 daily run, manual run, dashboard, mail, bisect가 동일한 판정 결과를 사용하도록 만든다.

## 분석 엔진 설계

### 패키지 구조

권장 모듈 배치는 다음과 같다.

```text
daily/
  analysis/
    __init__.py
    types.py          # AnalysisConfig, SeriesKey, ComparisonRow, AnalysisResult
    baseline.py       # baseline run 선택, last known good 탐색
    verdict.py        # improved/same/regressed/noisy/unavailable 판정
    functional.py     # summary.json totals/tests 기반 functional issue 집계
    engine.py         # analyze_run(summary_json, db_path, config)
    report.py         # [ Analysis summary ] text 렌더링
    persistence.py    # summary.json analysis block 및 DB 집계 저장
```

### 입력

1. 현재 run의 `summary.json`
2. 누적 DuckDB 경로, 기본값 `daily/viewer/bench.duckdb`
3. 분석 설정
   - pct threshold
   - z threshold
   - noisy CV threshold
   - baseline selection policy
   - purpose/short_run 필터 정책
   - top regression 표시 개수

### 출력

분석 엔진은 report 문자열을 직접 만드는 대신 구조화된 `AnalysisResult`를 먼저 반환한다.

예상 JSON 형태는 다음과 같다.

```json
{
  "overall_status": "yellow",
  "baseline": {
    "status": "found",
    "run_id": "LNL-03:20260508_1200:summary",
    "stamp": "20260508_1200",
    "ov_version": "2026.2.0-21664-ad5d8e0f99b",
    "selection_reason": "same machine, short_run, purpose"
  },
  "functional": {
    "total": 120,
    "passed": 118,
    "failed": 2,
    "error": 0,
    "skipped": 0,
    "issues": [
      {
        "nodeid": "daily/tests/test_llm_benchmark.py::test_xxx",
        "outcome": "failed",
        "message": "short normalized failure message"
      }
    ]
  },
  "performance": {
    "compared": 84,
    "improved": 10,
    "same": 70,
    "regressed": 4,
    "unavailable": 0
  },
  "models": [
    {
      "model": "llama",
      "avg_improvement_pct": -0.042,
      "improved": 1,
      "same": 3,
      "regressed": 2
    }
  ],
  "top_regressions": [
    {
      "model": "llama",
      "precision": "FP16",
      "in_token": 32,
      "out_token": 128,
      "exec_mode": "2nd",
      "unit": "ms",
      "current_value": 12.0,
      "baseline_value": 10.0,
      "improvement_pct": -0.2,
      "verdict": "regressed"
    }
  ]
}
```

### Baseline 선택

MVP baseline은 동일 머신의 가장 최근 비교 가능한 run을 사용한다.

우선순위는 다음과 같다.

1. same machine + same short_run + same purpose + older timestamp
2. same machine + same short_run + older timestamp
3. same machine + older timestamp

비교 가능한 성능 항목은 아래 key가 모두 같은 경우로 제한한다.

```text
model, precision, in_token, out_token, exec_mode
```

bisect 모드에서는 baseline 대신 `overall_status = green`인 마지막 성공 run을 탐색한다.

### 성능 비교 방향

metric unit에 따라 개선 방향을 정규화한다.

```text
ms, s, %      -> lower is better
tps, FPS 등  -> higher is better
```

엔진 내부에서는 `improvement_pct`를 항상 같은 의미로 사용한다.

```text
positive improvement_pct -> 개선
negative improvement_pct -> 악화
```

### Verdict 판정

MVP는 단순 threshold 기반 판정을 사용한다.

```text
improvement_pct >= +T      -> improved
-T < improvement_pct < +T  -> same
improvement_pct <= -T      -> regressed
```

초기 기본값은 운영 노이즈를 고려해 `T = 5%`로 둔다.

고도화 단계에서는 historical points가 충분한 series에 대해 dual gate를 적용한다.

```text
regressed = worsening_pct >= pct_threshold AND worsening_z >= z_threshold
noisy = recent_cv >= noisy_cv_threshold
insufficient = recent_n 또는 baseline_n 부족
```

### Functional 판정

MVP는 `summary.json`의 `totals`와 `tests`를 기준으로 한다.

```text
functional_fail_count = totals.failed + totals.error
functional issue = failed 또는 error 또는 timeout outcome
```

추후에는 stderr 텍스트 기반 분류를 다음 category로 정규화한다.

```text
model_error, openvino_error, timeout, infra_error, unknown
```

### Overall status

overall status는 downstream에서 공통으로 사용하는 build health 신호다.

```text
red    = functional fail/error 있음
yellow = functional 이슈는 없지만 performance regression 있음
green  = baseline 비교 가능하고 functional/regression 없음
gray   = baseline 없음 또는 비교 불가
```

functional 이슈가 있으면 performance 결과보다 우선한다.

### Report 렌더링

`daily/analysis/report.py`는 구조화된 `AnalysisResult`를 받아 text block만 렌더링한다.

report 최상단에는 항상 다음 블록을 prepend한다.

```text
[ Analysis summary ]
- Functional: total=... passed=... failed=... error=... skipped=...
- Baseline: stamp=... ov=...
- Performance: compared=... improved=... same=... regressed=...
- Model deltas:
- Top regressions:
- Overall verdict: ...
```

baseline이 없으면 비교 불가 사유를 명시한다.

```text
- Baseline comparison: no older run found for this machine.
```

기존 `[ Summary ]`와 상세 결과 섹션은 그대로 유지한다.

### Summary JSON 기록

분석 결과는 report에만 쓰지 않고 `summary.json`에도 `analysis` block으로 저장한다.

이 block은 mail, dashboard, bisect, backfill이 동일하게 재사용하는 contract가 된다.

### DB 집계 저장

MVP에서는 기존 `runs`, `perf`, `perf_flat`을 유지한다. 이후 아래 집계 테이블을 추가한다.

```text
analysis_results
  run_id, baseline_run_id, overall_status, compared_count,
  improved_count, same_count, regressed_count,
  functional_fail_count, created_at

analysis_comparisons
  run_id, baseline_run_id, model, precision, in_token, out_token,
  exec_mode, unit, current_value, baseline_value,
  improvement_pct, verdict, threshold_pct

functional_issues
  run_id, nodeid, outcome, message
```

`functional_issues.category`는 stderr 기반 분류 체계가 확정된 뒤 추가한다.

이 테이블은 dashboard의 버전-버전 비교, functional 히스토리, last known good 탐색을 빠르게 만들기 위한 캐시 역할을 한다.

## 실행 오케스트레이션

`daily/run.py`는 분석 세부 구현을 갖지 않고 orchestration만 수행하도록 정리한다.

```text
1. pytest 실행
2. pytest JSON -> summary.json 생성
3. text report 생성
4. summary.json ingest/upsert
5. analysis engine 실행
6. summary.json analysis block 기록
7. report 최상단에 [ Analysis summary ] prepend
8. mail/xlsx/backup 후처리
```

manual run과 daily run은 동일 분석 경로를 사용한다.

## 메일 요약 설계

메일은 기존 report 본문을 그대로 첨부/포함하되, 본문 상단에 구조화 summary를 먼저 둔다.

메일 본문 필수 항목은 다음과 같다.

- overall status
- machine, device, stamp, ov_version, ov_build, ov_sha
- baseline stamp/version
- functional fail/error count와 대표 실패 목록
- performance compared/improved/same/regressed count
- 모델별 평균 변화율 top N
- 상위 regression top N
- bisect 후보 정보, 가능한 경우 마지막 green build

## Bisect 보조 설계

이슈 run의 기준은 다음 중 하나다.

- `overall_status = red`
- `overall_status = yellow`
- functional issue count > 0
- regression count > 0

last known good 탐색은 같은 머신과 같은 run profile 안에서 `overall_status = green`인 가장 최근 과거 run을 찾는다.

비교 메일에는 다음 정보를 포함한다.

- issue run: stamp, ov_version, ov_build, ov_sha
- last good run: stamp, ov_version, ov_build, ov_sha
- build/sha delta
- 새로 발생한 functional failures
- regressed model/precision/token/exec_mode 목록
- 공통 모델 비교 수와 regression 비율

## Streamlit 보강 계획

기존 Streamlit viewer는 조회 기반이 충분하므로, 분석 엔진 결과를 재사용하는 패널을 추가한다.

우선순위는 다음과 같다.

1. Dashboard에 latest run build health 카드 추가
2. functional issue 히스토리/집계 뷰 추가
3. run-to-run 직접 비교 뷰 추가
4. model-level improved/same/regressed summary 테이블 추가
5. last known good 후보와 issue run 비교 패널 추가

## 마일스톤

### M1. 분석 엔진 분리

- `daily/run.py` 내부의 baseline 선택, verdict, model delta, analysis summary 로직을 `daily/analysis`로 이동한다.
- `analyze_run(summary_json, db_path, config)` API를 만든다.
- `AnalysisResult`를 report 렌더링과 JSON 기록 양쪽에서 사용한다.

### M2. Summary JSON + Report 통합

- `summary.json`에 `analysis` block을 기록한다.
- report 최상단에 `[ Analysis summary ]`를 prepend한다.
- baseline 없음, 비교 항목 없음, functional 실패 케이스를 명시적으로 표현한다.

### M3. DB 집계와 Dashboard 확장

- `analysis_results`, `analysis_comparisons`, `functional_issues` 테이블을 추가한다.
- functional 히스토리와 run-to-run 비교 dashboard를 추가한다.

### M4. 메일 템플릿 고도화

- 모델별 변화, 상위 regression, functional 상태를 메일 본문 상단에 구조화한다.
- report 전문은 하단 또는 첨부로 유지한다.

### M5. Bisect 보조 알림

- last known good 탐색 API를 구현한다.
- issue run과 last good run의 delta를 메일에 포함한다.

## TODO 체크리스트

### 기획/정책

- [x] Jenkins LLM daily 산출물 포맷 1차 확보
- [~] 성능 메트릭 단위 표준화 고도화
- [~] regression threshold 운영 정책 확정
- [ ] baseline 대상 필터 정책 확정: machine, purpose, short_run, success-only 여부
- [ ] functional 이슈 분류 체계 확정: fail, timeout, infra, model, OpenVINO

### 데이터/DB

- [x] DuckDB schema와 ingest 경로 구현
- [x] run/file_hash 기반 중복 처리 구현
- [x] new/old 포맷 ingestion CLI 구현
- [x] `analysis_results` 테이블 추가
- [x] `analysis_comparisons` 테이블 추가
- [x] `functional_issues` 테이블 추가
- [ ] backfill 운영 스크립트 추가

### 분석 엔진

- [x] trend 기반 median/MAD regression query 구현
- [x] run-to-run improved/same/regressed 1차 로직 구현
- [x] `daily/analysis` 패키지로 분석 로직 분리
- [x] `AnalysisResult` 구조 정의
- [x] `summary.json` analysis block 생성기 추가
- [x] functional issue list와 overall status 계산 로직 정규화 (MVP)
- [ ] noisy/insufficient verdict 고도화

### 비교/요약 자동화

- [x] report 상단 comparison summary 1차 구현
- [x] report 렌더러를 `daily/analysis/report.py`로 분리
- [x] mail summary 템플릿 고도화
- [x] manual/daily run 공통 orchestration 정리

### Dashboard

- [x] 기본 필터 UI 구현
- [x] regression table과 trend chart 구현
- [x] geomean/noise tab 구현
- [x] functional 히스토리/집계 뷰 추가
- [x] run-to-run 직접 비교 뷰 추가
- [x] latest run build health 카드 추가

### Bisect 지원

- [x] last known good 탐색 API 구현 (`baseline.find_last_known_good` 구현 + issue run 경로 연동)
- [ ] issue run vs last good run delta 생성
- [ ] bisect 보조 메일 템플릿 작성
- [ ] bisect 경로 테스트 추가

### 운영/품질

- [~] CLI/env 혼합 설정을 중앙 설정으로 정리
- [x] 기본 로깅/에러 핸들링/계속 진행 정책 일부 구현
- [~] ingest/query/viewer 테스트 존재
- [x] analysis engine unit test 추가
- [ ] 샘플 데이터 기반 E2E rehearsal 추가
- [ ] 실제 daily 머신 1대에서 shadow mode 운영

## 우선순위

### P0

- `daily/analysis` 패키지 생성
- `AnalysisResult` contract 정의
- functional 집계 + overall status 계산 정규화
- `summary.json` analysis block 기록
- report 상단 `[ Analysis summary ]` 유지

### P1

- 메일 본문 템플릿 고도화
- DB 집계 테이블 추가
- Streamlit functional 히스토리/집계 뷰 추가
- run-to-run 직접 비교 뷰 추가

### P2

- last known good 탐색
- bisect 보조 메일 자동화
- noisy series와 dual gate 판정 고도화
- backfill/E2E rehearsal 자동화

## 다음 작업 (착수 순서)

P0(분석 엔진 분리)는 완료되었고, 현재는 P1/P2 항목을 순차 진행한다.

### P0 — 분석 엔진 분리 (M1·M2) 완료

1. **`daily/analysis` 패키지 골격 생성** (완료)
2. **`run.py` 내부 로직 이동** (완료)
3. **`engine.analyze_run()` API 확정** (완료)
4. **`summary.json` analysis block 기록** (완료)
5. **`run.py` orchestration 정리** (완료)
6. **analysis engine unit test 추가** (완료)

### P1 — DB 집계·메일·Dashboard (M3·M4)

현재 착수 우선순위.

7. **DB 집계 테이블 추가**
   `schema.sql`에 `analysis_results`, `analysis_comparisons`, `functional_issues`를 추가하고 backfill 스크립트를 작성한다.

8. **메일 본문 템플릿 고도화** (부분 완료)
  `delivery.py`에 summary.json `analysis` block 기반의 compact summary를 추가했다. 모델별 변화/top regression/delta 전용 HTML은 bisect 템플릿 단계에서 보강한다.

9. **Streamlit functional 히스토리 뷰 추가** (완료)
  `queries.py`에 functional summary/history 조회를 추가하고, `app.py` Functional 탭을 연결했다.

10. **run-to-run 직접 비교 뷰 추가** (완료)
   `analysis_comparisons` 우선 + `perf` fallback 기반으로 두 run 비교 Compare 탭을 추가했다.

### P2 — Bisect·고도화 (M5)

11. **last known good 탐색 API**
  `baseline.py`에 `overall_status = green` 기반 탐색 함수를 추가하고,
  issue run(`red`/`yellow`) summary에 bisect hint로 연동했다. (완료)

12. **bisect 보조 메일 템플릿**
    issue run과 last good run의 delta를 구조화해서 메일에 포함한다.

13. **noisy series dual gate 고도화**
    `verdict.py`에 `worsening_pct AND worsening_z` dual gate와 `noisy`/`insufficient` 레이블을 추가한다.

## 완료 기준

- 동일 입력 재실행 시 DB 중복 없이 갱신된다.
- `daily/run.py` 실행 후 report 최상단에 `[ Analysis summary ]`가 항상 생성된다.
- baseline이 있으면 비교 run 정보와 compared/improved/same/regressed 수가 표시된다.
- baseline이 없으면 비교 불가 사유가 표시된다.
- functional 실패가 있으면 overall status에서 performance regression보다 우선 표시된다.
- `summary.json`에 downstream 재사용 가능한 `analysis` block이 기록된다.
- 메일 본문에 모델별 변화, 상위 regression, functional 상태가 포함된다.
- dashboard에서 버전/모델별 성능 변화와 functional 이슈 히스토리를 1분 내 확인할 수 있다.
- regression 또는 functional 이슈 발생 시 마지막 성공 build 비교 정보가 메일에 포함된다.

## 검증 계획

### 단위 테스트

- verdict 경계값 테스트: improved/same/regressed
- unit 방향성 테스트: latency unit과 throughput unit
- baseline 선택 우선순위 테스트
- no baseline/no comparable rows 테스트
- functional fail 우선순위와 overall status 테스트
- model-level average/count aggregation 테스트

### 통합 테스트

- baseline/current summary 2개 ingest 후 analysis result 생성 검증
- report 최상단 `[ Analysis summary ]` prepend 검증
- `summary.json` analysis block 기록 검증
- mail body 렌더링 검증
- last known good 탐색 검증

### 운영 검증

- 실제 daily 머신 1대에서 1주 shadow mode 운영
- 기존 report와 신규 analysis summary 차이 수집
- 오탐/미탐 케이스 기준으로 threshold 조정
- noisy series 목록을 별도 수집해 threshold 정책에 반영

## 분석 방법론

### 단일 측정값 비교의 한계

머신 노이즈, CPU frequency scaling, memory interference, OS scheduling 등으로 동일 코드의 성능 측정값도 실행마다 수 % 편차가 발생한다. 단일 값 비교는 fluctuation을 regression으로 오인하거나 실제 regression을 놓칠 수 있다.

### 핵심 원칙

1. 반복 측정과 median 사용
   - mean은 outlier에 민감하므로 median을 기준 통계로 사용한다.
   - pyperf도 이 이유로 mean 대신 median과 MAD를 활용한다.
2. dual gate 판정
   - pct 변화 단독으로는 노이즈 환경에서 오탐이 많다.
   - `pct_change > threshold AND z_score > z_threshold`를 동시에 만족할 때 regression으로 판정한다.
3. CV 기반 noisy series 분리
   - CV가 일정 기준 이상인 series는 신뢰 판정이 어렵기 때문에 `noisy`로 분리한다.
4. MVP와 고도화 단계 분리
   - MVP는 latest comparable baseline과 단순 threshold를 사용한다.
   - 데이터가 충분히 쌓이면 median/MAD, z-score, CV, Mann-Whitney U test, EWMA/CUSUM을 단계적으로 적용한다.

### Median + MAD

$$\hat{\sigma} = 1.4826 \times \text{MAD}, \quad \text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$

MAD는 정규 분포를 강하게 가정하지 않는 robust 산포 추정량이다. 이상값 일부가 mean/stddev를 크게 왜곡하는 상황에서도 median/MAD는 안정적으로 동작한다.

### z-score

$$z = \frac{x - \text{median}}{\hat{\sigma}}$$

`z > 3`은 3σ 이상 이탈을 의미한다. Shewhart control chart의 3σ 기준과 같은 논리이며, 정규 분포 기준 약 0.27% 확률의 이상 이탈에 해당한다.

### Mann-Whitney U test

baseline window와 recent window의 분포가 같은지 비모수적으로 검정한다. 정규 분포를 가정하지 않으며 작은 샘플에서도 사용할 수 있다. p-value와 effect size를 함께 보아 regression 판정의 보조 신호로 사용한다.

### EWMA / CUSUM

단발 이상값보다 지속적인 소폭 하락 추세를 감지하는 데 적합하다. Shewhart 3σ는 큰 급변을 잘 잡지만 느린 drift에는 둔감하므로, 장기 운영 단계에서 EWMA/CUSUM을 보조 지표로 추가한다.

### 권장 초기 파라미터

| 파라미터 | 초기값 | 근거 |
|---|---:|---|
| `REGRESSION_PCT_THRESHOLD` | 5% | 노이즈 환경에서 실질 변화로 보기 위한 최솟값 |
| `REGRESSION_Z_THRESHOLD` | 3.0 | Shewhart 3σ 기준 |
| `REGRESSION_NOISY_CV_THRESHOLD` | 10% | CV > 10% series는 신뢰 판정에서 분리 |
| 최소 recent points | 5 | window 기반 통계의 실용적 최솟값 |
| 최소 baseline points | 7 | baseline median/MAD 안정성 확보 |
| 동적 임계 공식 | `max(3%, 2 * CV)` | series별 노이즈 수준 반영 |

## 참고 자료

1. pyperf - Analyze benchmark results: <https://pyperf.readthedocs.io/en/latest/analyze.html>
2. pyperf - Run a benchmark: <https://pyperf.readthedocs.io/en/latest/run_benchmark.html>
3. Google Benchmark - User Guide: <https://google.github.io/benchmark/user_guide.html>
4. Control chart: <https://en.wikipedia.org/wiki/Control_chart>
5. Median absolute deviation: <https://en.wikipedia.org/wiki/Median_absolute_deviation>
6. Mann-Whitney U test: <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>