# 여러 머신 Daily 결과 통합 HTML 메일 구현 플랜

## 1. 목표

여러 머신에서 매일 실행되는 테스트 결과를 `daily_results` MCP 서버를 통해 수집하고, 한눈에 상태를 확인할 수 있는 통합 HTML 메일 한 통으로 발송한다.

통합 리포트 생성기는 중앙 DuckDB를 직접 읽지 않는다. 머신별 결과 선택, 이슈 집계, 성능 분석 데이터 조회는 모두 MCP 도구를 통해 수행한다.

## 2. 핵심 원칙

1. 같은 날짜에 실행된 결과를 하나의 daily cycle로 묶는다.
2. OpenVINO 버전은 머신마다 달라도 허용한다.
3. 각 머신에서는 조건에 맞는 마지막 run 하나를 선택한다.
4. run 선택 시 날짜, `purpose`, 실행자를 핵심 조건으로 사용한다.
5. 메일 본문은 성공/실패 상태를 짧게 보여주고, 상세 분석은 viewer 링크에서 확인한다.
6. 명령행 인자는 최소화하고 운영 설정은 JSON 파일로 관리한다.

## 3. 전체 구조

```mermaid
flowchart LR
  A[각 머신의 Daily test] --> B[산출물 업로드와 중앙 ingest]
  B --> C[중앙 DuckDB]
  C --> D[daily_results MCP 서버]
  D --> E[통합 리포트 생성기]
  E --> F[통합 HTML 리포트]
  F --> G[메일 발송]
```

각 머신의 테스트와 중앙 ingest가 끝날 시간을 고려해 중앙 머신 한 곳에서 통합 리포트 생성기를 실행한다.

## 4. 머신별 run 선택 정책

### 4.1 선택 기준

리포트 대상 날짜를 먼저 결정한 뒤, 각 머신에서 다음 조건을 모두 만족하는 run 중 마지막 run을 선택한다.

1. `machine`이 설정된 예상 머신과 일치한다.
2. run의 기준 날짜가 리포트 대상 날짜와 일치한다.
3. `purpose`가 설정값과 일치한다.
4. 실행자(`triggered_by`)가 설정값과 일치한다.
5. partial run이나 명시적으로 제외된 run이 아니다.
6. 조건을 만족하는 run이 여러 개면 timestamp가 가장 늦은 run을 선택한다.

즉, OpenVINO version/build가 아니라 다음 key로 후보군을 제한한다.

```text
(report_date, machine, purpose, triggered_by)
```

후보군 안에서는 다음 순서로 하나를 선택한다.

```text
ORDER BY ts DESC, ingested_at DESC
LIMIT 1
```

`ts`가 같을 때는 나중에 ingest된 결과를 선택한다. 동일 조건의 중복 run이 있으면 선택된 run 외의 개수도 warning으로 남긴다.

### 4.2 날짜 처리

머신 시간이 조금씩 다르고 테스트가 자정을 넘길 수 있으므로 날짜의 의미를 명확히 고정한다.

- 기본 기준은 run 시작 시각인 `runs.ts`의 날짜이다.
- 모든 timestamp는 중앙 DB에서 사용하는 timezone 기준으로 비교한다.
- 운영 timezone은 JSON 설정의 `timezone`으로 명시한다.
- 필요하면 `day_start_hour`를 두어 오전 0시 이후 끝난 run을 전날 cycle로 묶을 수 있게 한다.
- 이전 날짜의 run을 누락 머신의 최신 결과처럼 자동 대체하지 않는다.

권장 초기값은 다음과 같다.

```json
{
  "timezone": "Asia/Seoul",
  "day_start_hour": 0
}
```

실제 DB의 `ts`가 timezone 정보가 없는 local time이면 서버와 각 머신의 timezone을 먼저 통일한다.

### 4.3 OpenVINO 버전 처리

머신별로 선택된 run의 OpenVINO version/build가 달라도 모두 리포트에 포함한다.

- 버전이 다르다는 이유로 run을 제외하지 않는다.
- 각 머신 행에 실제 OpenVINO version을 표시한다.
- 여러 버전이 섞여 있으면 상단에 간단한 안내를 표시하되 실패로 판정하지 않는다.
- 성능 비교는 각 머신의 해당 run과 그 머신에 적합한 과거 reference 사이에서 수행한다.

### 4.4 실행자 메타데이터 추가

현재 `runs` schema에는 실행자 전용 필드가 없다. `purpose`나 `description`에서 실행자를 추측하지 않고 다음 경로를 추가한다.

1. daily 실행 시 `triggered_by`를 수집한다.
2. `summary.json`의 `meta.triggered_by`에 저장한다.
3. ingest record와 `runs` 테이블에 `triggered_by` 컬럼을 추가한다.
4. MCP 응답에 `triggered_by`를 포함한다.
5. 과거 데이터처럼 값이 없는 run은 `unknown`으로 취급한다.

실행자 값의 우선순위는 구현 시 환경에 맞춰 확정하되, 예시는 다음과 같다.

1. 명시적인 환경변수 `DAILY_TRIGGERED_BY`
2. Jenkins가 제공하는 build user 정보
3. OS 사용자명
4. `unknown`

자동 cron 실행은 사람 이름 대신 `scheduler`처럼 고정된 값을 사용하는 것이 좋다.

## 5. 예상 머신과 운영 설정

예상 머신 목록은 DB에서 자동 추론하지 않고 JSON 설정으로 관리한다. 그래야 결과를 올리지 못한 머신도 `missing`으로 표시할 수 있다.

예상 설정 파일:

```json
{
  "schema_version": 1,
  "mcp_url": "http://dg2raptorlake.ikor.intel.com:8090/mcp",
  "viewer_base_url": "http://dg2raptorlake.ikor.intel.com:8501",
  "report_base_url": "http://dg2raptorlake.ikor.intel.com/daily2",
  "timezone": "Asia/Seoul",
  "day_start_hour": 6,
  "purpose": "daily_pipeline timer",
  "triggered_by": "scheduler",
  "expected_machines": [
    "ARLH-01",
    "BMG-02",
    "LNL-03",
    "LNL-04",
    "MTL-01",
    "PTLH-01",
    "PTLH-02",
    "RAPTOR-ELLY",
    "dg2alderlake"
  ],
  "mail": {
    "recipients": ["sungeun.kim@intel.com"],
    "subject_prefix": "Daily GPU",
    "relay_server": "dg2raptorlake.ikor.intel.com"
  },
  "schedule": {
    "max_wait_minutes": 60,
    "poll_interval_seconds": 300
  },
  "report": {
    "max_functional_issues": 20,
    "top_regressions": 10,
    "top_improvements": 5
  }
}
```

저장소에는 실제 수신자나 환경별 값을 넣지 않은 example 파일만 commit한다. 운영 설정 파일은 필요하면 git ignore 대상 위치에 둔다.

## 6. MCP 서버 확장

`daily/mcp_server/server.py`에 통합 리포트 전용 도구를 추가한다.

```text
daily_results_daily_digest(
    report_date: string,
    purpose: string,
    triggered_by: string,
    expected_machines: list[string],
    max_functional_issues: int = 20,
    top_regressions: int = 10,
    top_improvements: int = 5
+) -> JSON
```

실제 구현 시 MCP SDK가 지원하는 인자 타입에 맞게 최종 시그니처를 확정한다.

### 6.1 MCP 응답 형식

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-10T02:00:00+09:00",
  "selection": {
    "report_date": "2026-09-09",
    "purpose": "daily_pipeline timer",
    "triggered_by": "scheduler"
  },
  "summary": {
    "status": "ready",
    "expected_machines": 9,
    "completed_machines": 9,
    "successful_machines": 8,
    "failed_machines": 1
  },
  "machines": [],
  "functional_issues": [],
  "top_regressions": [],
  "top_improvements": [],
  "warnings": []
}
```

머신별 항목에는 다음 정보를 포함한다.

- machine, run ID, timestamp
- `purpose`, `triggered_by`
- OpenVINO version/build/SHA
- total, passed, failed, error, skipped
- 전체 상태와 performance verdict 개수
- viewer 상세 URL
- 개별 HTML report URL과 Jenkins build URL이 있으면 함께 제공

### 6.2 전용 MCP 도구가 필요한 이유

`daily_results_run_sql`은 최대 500행만 반환한다. 전체 머신의 상세 결과는 이 제한을 넘을 수 있고 클라이언트에서 여러 결과를 join해야 한다.

전용 도구에서 run 선택과 집계를 끝내고 메일에 필요한 크기로 제한된 payload만 반환한다. Regression 계산은 기존 query 및 통계 로직을 재사용하며, 최신 run에 `analysis_comparisons`가 없을 때도 history로 계산할 수 있어야 한다.

## 7. 통합 HTML 리포트

메일의 목적은 전체 상태를 빠르게 판단하는 것이다. 상세 로그와 모든 성능 row를 메일에 넣지 않는다.

### 7.1 상단 요약

- 전체 상태: `GREEN`, `YELLOW`, `RED`, `INCOMPLETE`
- 대상 날짜
- 완료 머신 수 / 예상 머신 수
- 성공 머신 수 / 실패 머신 수
- 사용된 `purpose`와 실행자
- 머신별 OpenVINO version이 여러 개인 경우 버전 개수

### 7.2 머신 요약 테이블

예상 머신마다 한 행을 표시한다.

| 항목 | 내용 |
|---|---|
| Machine | 머신 이름 |
| Status | success, failed, missing, stale |
| Time | 선택된 마지막 run 시각 |
| OpenVINO | 해당 머신의 실제 version |
| Tests | passed / failed / error / skipped |
| Perf | regression 개수 |
| Details | viewer 링크 |

정상 머신은 한 줄로 간단히 표시한다. 실패, error, regression, missing 상태는 색상과 짧은 문구로 강조한다.

### 7.3 이슈 요약

이슈가 있는 머신만 별도 영역에 표시한다.

- functional failure: 실패 test 이름과 핵심 메시지 한 줄
- error: 실행 또는 infrastructure 오류 요약
- performance regression: 가장 큰 항목 몇 개와 변화율
- missing/stale: 기대한 날짜의 적합한 run이 없다는 설명

failure message는 길이를 제한하고 HTML escape한다. 전체 stack trace나 모든 performance row는 메일에 넣지 않는다.

### 7.4 Viewer 상세 링크

각 머신 행과 이슈 항목에 viewer 상세 링크를 넣는다. 사용자가 링크를 누르면 선택된 `run_id` 또는 machine/date 조건이 적용된 화면을 바로 열 수 있어야 한다.

권장 방식:

```text
<viewer_base_url>/?run_id=<run_id>
```

현재 viewer가 `run_id` deep link를 지원하지 않으면 다음 기능을 추가한다.

1. URL query parameter에서 `run_id`를 읽는다.
2. 해당 run의 machine과 날짜를 자동 선택한다.
3. 관련 summary, failure, performance 탭으로 이동할 수 있게 한다.

링크 우선순위는 다음과 같다.

1. viewer의 선택된 run 상세 화면
2. 개별 HTML report
3. Jenkins console log

메일에는 주로 viewer 링크 하나만 노출하고, 상세 화면에서 report와 Jenkins 링크를 제공한다.

## 8. 통합 리포트 생성기와 최소 인자

다음 entry point를 추가한다.

```text
daily/generate_fleet_report.py
```

CLI 인자는 최소한으로 유지한다.

```text
--config PATH
--date YYYY-MM-DD
--dry-run
--force
```

- `--config`: JSON 설정 파일 경로. 기본 경로를 제공한다.
- `--date`: 재생성이나 테스트가 필요할 때만 지정한다. 기본값은 자동 계산한 이전 daily 날짜이다.
- `--dry-run`: HTML만 생성하고 메일은 보내지 않는다.
- `--force`: 이미 발송한 날짜를 다시 발송한다.

MCP URL, 머신 목록, purpose, 실행자, 수신자, polling, 표시 개수 같은 운영 값은 모두 JSON 설정에서 읽는다.

생성기의 역할은 다음과 같다.

1. JSON 설정을 읽고 검증한다.
2. `daily/common/mcp_client.py`를 통해 `daily_results_daily_digest`를 호출한다.
3. 응답 schema와 선택 조건을 검증한다.
4. 간결한 HTML을 생성하고 파일로 저장한다.
5. `daily/common/delivery.py::send_mail`을 사용해 발송한다.
6. 수집 또는 발송 실패 시 0이 아닌 종료 코드를 반환한다.

## 9. 준비 상태 확인과 스케줄링

모든 머신이 같은 시각에 끝난다고 가정하지 않는다.

1. 통합 리포트 job을 01:30 또는 02:00 무렵 시작한다.
2. 같은 `report_date`, `purpose`, `triggered_by` 조건으로 MCP를 주기적으로 호출한다.
3. 모든 예상 머신의 적합한 run이 준비되면 즉시 발송한다.
4. JSON 설정의 최대 대기시간이 지나면 polling을 중단한다.
5. deadline까지 결과가 없는 머신을 포함한 `INCOMPLETE` 리포트를 한 번 발송한다.
6. 일시적인 MCP 및 메일 오류는 제한된 횟수와 backoff로 재시도한다.
7. MCP 수집 자체가 실패하면 내용이 빈 정상 메일은 발송하지 않는다.

## 10. 중복 발송 방지

중복 방지 key는 OpenVINO build를 포함하지 않는다. 머신마다 버전이 다를 수 있기 때문이다.

```text
<report-date>_<purpose>_<triggered-by>
```

상태 파일에는 수신자 목록, 생성된 리포트 경로, 발송 시각과 결과를 기록한다. 이미 성공적으로 발송한 key는 `--force`가 없으면 다시 보내지 않는다.

상태 파일은 atomic file replacement로 갱신한다.

## 11. 실패 및 상태 정책

- MCP 연결 실패: 재시도 후 0이 아닌 코드로 종료하고 정상 메일은 발송하지 않는다.
- deadline에 일부 머신 누락: `INCOMPLETE` 메일을 발송한다.
- functional failure 또는 error: 전체 상태를 `RED`로 표시한다.
- 유의미한 performance regression만 있음: `YELLOW`로 표시한다.
- 머신별 OpenVINO version 차이: 정보로 표시하고 실패로 판정하지 않는다.
- `purpose` 또는 실행자가 다른 run: 선택하지 않고 기대 결과 누락으로 처리한다.
- 메일 발송 실패: 생성된 HTML을 유지하고 0이 아닌 코드로 종료한다.

## 12. 구현 단계

### Phase 1: 실행자 메타데이터

- run metadata에 `triggered_by`를 추가한다.
- `summary.json`, ingest record, DuckDB schema/writer에 반영한다.
- 기존 데이터의 값은 `unknown`으로 처리한다.

### Phase 2: MCP contract와 query layer

- 날짜, 머신, purpose, 실행자로 후보 run을 필터링한다.
- 머신별 마지막 run을 선택한다.
- 머신 상태와 functional issue를 집계한다.
- 제한된 regression/improvement를 계산한다.
- viewer deep link를 포함한 digest 도구를 등록한다.

### Phase 3: HTML 생성

- JSON 설정 loader와 validation을 추가한다.
- 요약 중심의 HTML renderer를 구현한다.
- 이슈가 있는 항목에 viewer 링크를 제공한다.
- 긴 문자열과 동적 내용을 안전하게 처리한다.

### Phase 4: 발송과 운영 적용

- 기존 메일 발송 코드를 재사용한다.
- readiness polling, deadline, retry, 중복 방지를 구현한다.
- viewer의 `run_id` deep link를 지원한다.
- cron wrapper와 운영 문서를 추가한다.
- 제한된 수신자로 검증 후 운영 발송을 활성화한다.

## 13. 테스트 계획

### Run 선택 테스트

- 같은 날짜에 버전이 다른 머신 결과가 모두 선택됨
- 같은 머신과 날짜에서 마지막 run만 선택됨
- machine, purpose, 실행자 필터가 모두 적용됨
- partial/excluded run이 제외됨
- 자정 경계와 timezone 설정이 올바르게 적용됨
- 이전 날짜 run이 누락 머신을 대신하지 않음

### MCP 테스트

- expected machine의 missing 상태 유지
- functional issue와 performance 결과 개수 제한
- 500행 SQL 제한과 무관한 digest 생성
- 저장된 comparison이 없어도 history에서 계산
- viewer URL에 선택한 `run_id` 포함

### HTML 테스트

- 정상 머신은 한 줄로 간결히 표시
- 실패 머신의 핵심 이슈와 viewer 링크 표시
- failure message HTML escaping 및 길이 제한
- complete, incomplete, MCP error 상태 표시
- 여러 OpenVINO version을 오류로 판정하지 않음

### 설정과 발송 테스트

- JSON 필수 필드 및 잘못된 타입 검증
- 최소 CLI 인자의 기본값과 override 확인
- `--dry-run`에서 메일 미발송
- 같은 날짜/purpose/실행자의 중복 발송 차단
- `--force` 재발송
- mail command 실패 시 0이 아닌 종료 코드

## 14. 예상 변경 파일

- `daily/run.py`
- `daily/report/builder.py`
- `daily/viewer/ingest/record.py`
- `daily/viewer/ingest/loader_new.py`
- `daily/viewer/ingest/writer.py`
- `daily/viewer/schema.sql`
- `daily/viewer/queries.py`
- `daily/viewer/app.py`
- `daily/mcp_server/server.py`
- `daily/common/mcp_client.py`
- 신규 `daily/generate_fleet_report.py`
- 신규 JSON example 설정 파일
- `daily/tests/`의 관련 테스트
- `daily/README.md`

## 15. 완료 조건

- 각 머신에서 같은 날짜, 지정 purpose, 지정 실행자의 마지막 run이 선택된다.
- 머신별 OpenVINO version이 달라도 결과에 포함된다.
- 모든 cross-machine 데이터는 MCP를 통해서만 수집된다.
- 설정된 모든 머신이 success, failed, missing, stale 중 하나로 표시된다.
- 메일은 성공/실패와 핵심 이슈를 간결하게 보여준다.
- 이슈가 있는 모든 머신에서 viewer 상세 화면으로 이동할 수 있다.
- 운영 옵션은 JSON에 모이고 CLI는 최소한으로 유지된다.
- 중복 발송과 주요 실패 상황을 종료 코드와 로그로 확인할 수 있다.
- run 선택, MCP 집계, HTML 생성, viewer link, 발송에 대한 테스트가 존재한다.
