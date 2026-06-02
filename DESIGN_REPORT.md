# LLM Daily Analysis/Report 구현 요약

## 1. 요청 배경 및 목표

사용자 요청의 핵심은 다음과 같았다.

- report를 **HTML 형태로 제공**
- 최근 약 10개 daily 결과를 기준으로, 단순 비교가 아니라 **통계 분포 기반**으로 current build를 리뷰
- 단순 변화율(예: 10ms -> 9ms = 10% 개선)만으로 improved/regression 판단하지 않고,
  기존 변동 폭(fluctuation) 안의 변화인지 구분
- 결과 나열보다 **regressed / improved 판단의 신뢰도**를 높이는 것

## 2. 반영한 설계 방향

기존 `daily/analysis` 엔진 구조를 유지하면서, 판정 기준을 아래처럼 확장했다.

1. 시리즈별 히스토리 수집
- 동일 머신 + 동일 run profile(short_run/purpose) 기준
- 최근 `N=10`개(`history_window`) 시리즈 값 수집

2. 비교 기준(reference) 변경
- baseline 1회 값 대신, 히스토리가 충분하면
  단위 방향성을 반영해 **Top-K 평균**(`reference_top_k=5`)을 참조값으로 사용
- latency 계열(ms/s/%)은 낮을수록 좋음, throughput 계열은 높을수록 좋음

3. 분포 기반 fluctuation guard 추가
- median/MAD 기반 sigma 추정
- current와 reference 차이가 `fluctuation_sigma_scale * sigma` 이내면
  improved/regressed가 아닌 **same으로 강등**
- 즉, 퍼센트 임계치를 넘어도 통계적으로 의미가 약하면 보수적으로 same 처리

4. HTML 보고서 추가
- text report와 별도로 분석 중심 HTML 리포트를 생성
- Top regressions / Top improvements와 분포 정보(CV/sigma/history count)를 함께 노출

5. Baseline 선택 정책 명확화
- HTML의 `Baseline`은 현재 run과 비교 가능한 과거 run 1건을 의미한다.
- 현재 정책은 범위를 넓게 잡지 않고, 우선적으로 `purpose=daily_CB timer` 인 과거 run만 baseline 후보로 사용한다.
- 후보는 같은 machine 이고 현재보다 오래된 run 이어야 하며, 실제 perf series overlap 이 있는 경우만 인정한다.
- baseline 자체는 Run Summary 에 표시되는 대표 과거 run 이고, 각 row의 실제 성능 비교값은 히스토리가 충분하면 baseline 1점 대신 `top-k mean` 을 우선 사용한다.

6. Current Run 메타데이터 가시성 강화
- Run Summary 에 current run의 환경 정보를 함께 노출해, 성능 변화와 실행 환경 변화를 한 화면에서 해석할 수 있도록 했다.
- 표시 항목:
  - OpenVINO version
  - purpose
  - machine name
  - GPU driver / GPU info
  - host info
  - memory size / memory speed

7. 메타데이터 ingest/persistence 보강
- 기존에는 일부 HTML 필드가 `n/a` 로 보일 수 있었기 때문에, 렌더링만이 아니라 ingest → schema → persistence → render 전체 경로를 함께 확장했다.
- legacy `.report` 와 new `summary.json` 경로 모두에서 host/GPU runtime metadata 를 회수하도록 보강했다.

8. 운영 편의 옵션 추가
- `generate_html_report.sh` 에 `--rebuild-db` 옵션을 추가해, 머신별 DuckDB를 삭제 후 전체 ingest + report 재생성을 한 번에 수행할 수 있게 했다.

## 3. 코드 변경 요약

### 3.1 타입/설정 확장
- 파일: `daily/analysis/types.py`
- 추가 설정:
  - `history_window` (default 10)
  - `reference_top_k` (default 5)
  - `fluctuation_sigma_scale` (default 1.5)
- `ComparisonRow` 확장 필드:
  - `history_count`, `reference_source`
  - `history_median`, `history_mad`, `history_sigma`, `history_cv`
  - `worsening_z`, `within_fluctuation`
- `CurrentRunInfo` 추가:
  - `ov_version`, `purpose`, `machine_name`
  - `gpu_driver_version`, `gpu_info`
  - `host_info`, `memory_size`, `memory_speed`
- `AnalysisResult.current_run` 추가

### 3.2 엔진 로직 확장
- 파일: `daily/analysis/engine.py`
- `_fetch_comparison_rows`에서 아래를 수행:
  - 최근 히스토리 로딩
  - top-k mean 계산 및 reference 적용
  - `verdict_from_signal` 기반 판정
  - fluctuation guard 적용 후 same 강등
- baseline은 별도 selector를 통해 선택되며,
  실제 row 비교의 reference는 `topk_mean` 우선, 부족할 때만 `baseline` fallback 사용
- 보조 함수 추가:
  - `_load_series_history`
  - `_history_stats`
  - `_is_within_fluctuation`
  - `_fetch_run_context`
- 기존 테스트 호환을 위해 `_fetch_comparison_rows`가 `run_id(str)` 입력도 처리

### 3.3 baseline 선택 정책 정리
- 파일: `daily/analysis/baseline.py`
- baseline 선택 우선순위:
  1. 같은 machine + 같은 short_run + `purpose=daily_CB timer` + older run
  2. 같은 machine + `purpose=daily_CB timer` + older run
- 공통 후보 조건:
  - 현재 run 제외
  - 현재보다 과거 run
  - perf row overlap 존재
- 찾지 못하면 `BaselineInfo(status="not_found")`
- yellow/red 결과에 대해서는 별도로 `last known good` 탐색 지원

### 3.4 report 렌더러 확장
- 파일: `daily/analysis/report.py`
- 텍스트 summary에 반영:
  - fluctuation guard로 same 처리된 건수 표시
  - top regressions에 reference source/history count 포함
- HTML 렌더링/저장 추가:
  - `render_analysis_html(result)`
  - `write_analysis_html(report_path, result)`
- Run Summary 확장:
  - Current OV / purpose / machine
  - GPU driver / GPU info
  - Host info / memory size / memory speed
  - Baseline / Selection reason
- Analysis Methodology 영역에 `Reference` 와 `baseline fallback` 의미를 텍스트로 명시

### 3.5 HTML 생성 orchestration 확장
- 파일: `scripts/generate_analysis_report.py`
- 역할:
  - `--root` 기준 artefact ingest
  - DB에서 current run 선택
  - current run 메타데이터 조회 및 `CurrentRunInfo` 구성
  - HTML output 파일 생성
- current run 메타데이터는 `runs` + `system_devices` 테이블을 합쳐 구성한다.
- memory size는 host RAM 기준을 우선 표시하고, GPU VRAM 은 GPU info 문자열에 포함한다.

### 3.6 ingest/schema/persistence 확장
- 파일: `daily/viewer/ingest/record.py`
  - `RunRecord`에 `host_info`, `host_memory_size_gb`, `host_memory_speed_mhz` 추가
- 파일: `daily/viewer/schema.sql`
  - `runs` 테이블에 host metadata 컬럼 추가
- 파일: `daily/viewer/ingest/writer.py`
  - 신규 컬럼 migration 및 upsert 반영
- 파일: `daily/viewer/ingest/loader_old.py`
  - legacy `.report` 의 `System Info`/`CPU Info`/`GPU Info` 블록에서 host/GPU runtime metadata 파싱
- 파일: `daily/viewer/ingest/loader_new.py`
  - sibling report가 있으면 동일 파서를 재사용해 current runtime metadata 확보

### 3.7 persistence 확장
- 파일: `daily/analysis/persistence.py`
- `summary.json`의 `analysis` block에
  히스토리/분포 관련 row 메타데이터까지 직렬화 저장

### 3.8 스크립트 UX 개선
- 파일: `daily/generate_html_report.sh`
- 추가 옵션:
  - `--machine`, `-m`
  - `--rebuild-db`
  - `--help`
- `--rebuild-db` 사용 시 머신별 DB 파일(`.duckdb`, `.wal`, `.tmp`)을 삭제 후 재생성한다.

### 3.9 테스트 추가
- 파일: `daily/tests/test_analysis_engine.py`
- 추가 검증:
  - top-k history reference 우선 사용
  - fluctuation guard 동작
  - HTML 렌더/파일 생성 스모크

## 4. 생성되는 산출물

run 완료 후 기존 산출물 외에 아래가 추가된다.

- `daily.<stamp>.report` (기존)
- `daily.<stamp>.summary.json` (기존)
- `analysis.current_<stamp>.generated_<timestamp>.html` (**신규**)  
  분석 중심 뷰(Overall/Top regression/improvement/분포 지표/Run Summary)

또한 머신별 DB는 다음 패턴으로 사용된다.

- `daily/viewer/bench.<machine>.duckdb`

## 5. 현재 검증 상태

- 정적/구문 체크:
  - Python 수정 파일 기준 **에러 없음**
  - `daily/generate_html_report.sh` 구문 검사 통과
  - help 출력 및 `--rebuild-db` 옵션 노출 확인 완료
- 기능 검증:
  - MTL-01 기준 DB를 삭제 후 전체 ingest 재실행
  - 약 1,091개 candidate 재적재 완료
  - 생성된 HTML에서 아래 값들이 실데이터로 표시됨을 확인
    - Current OV
    - Current purpose
    - Machine
    - GPU driver / GPU info
    - Host info
    - Memory size / Memory speed
    - Baseline
- Baseline/Reference 동작 확인:
  - Run Summary 의 `Baseline` 은 대표 과거 run 정보로 표시
  - 각 series 비교는 `topk_mean` 우선, 필요 시 `baseline` fallback 사용
- 단위 테스트 전체 실행:
  - 환경 의존성으로 중단 가능성 있음
  - 대표 이슈: `ModuleNotFoundError: psutil` (`daily/conftest.py` import 시)

즉, 코드 변경 자체의 정합성은 확인했지만,
전체 pytest 기반 실행 검증은 실행 환경 패키지 준비 후 재확인이 필요하다.

## 6. 기대 효과

- noise가 큰 머신에서의 과잉 regression/improved 판정 감소
- 단일 baseline point 의존도 감소
- daily 리뷰 시 "변화량" + "분포 내 의미"를 동시에 해석 가능
- report 소비자(메일/대시보드/수동 리뷰) 기준으로 설명력이 높은 HTML 결과 제공
- 현재 run 환경 메타데이터를 함께 노출해, 성능 변화와 환경 변화를 같은 화면에서 해석 가능
- legacy/new artefact 혼재 환경에서도 host/GPU 정보 누락 없이 재구성 가능
- 필요 시 DB 재생성을 포함한 전체 리포트 갱신을 운영자가 한 번에 실행 가능

## 7. 후속 권장 작업

1. 실행 환경에 `psutil` 포함 후 테스트 재실행
2. 운영 데이터로 임계치 튜닝
   - `history_window`, `reference_top_k`, `fluctuation_sigma_scale`
3. 필요 시 Run Summary 의 `Selection reason` 문구를 사용자 친화적으로 정리
4. 필요 시 HTML를 메일 본문과 연계(링크 또는 inline 요약 강화)
5. 장기적으로는 dual gate(noisy/insufficient/z-score) 파라미터를 머신별 프로파일로 분리
