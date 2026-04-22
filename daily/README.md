# daily — pytest 기반 일일 테스트 수트 (WIP)

`scripts/run_llm_daily.py` 의 대체 구현. 마이그레이션 기간 동안 기존
`scripts/` 는 참조용으로 남아있으며, 모든 테스트가 여기로 포팅되면 제거 예정.

## 목표

1. **pytest** (+ `pytest-json-report`) 기반 — 표준 프레임워크/선택 문법/플러그인 생태계
2. 개별 테스트 선택: `pytest -k "llama"`, `--collect-only` 로 목록 확인
3. 전체 런에 대한 단일 raw log: 모든 subprocess stdout 이 세션 스코프 파일
   (`daily.<timestamp>.raw`) 하나로 tee
4. 테스트 간 독립성: `isolate_test` autouse fixture 가 매 테스트 전/후 모델 캐시 정리
5. 한 번의 런에 통합 리포트 하나: pytest JSON 에서 생성된 text (`.report`)
   + 정규화된 JSON (`.summary.json`)
6. 기계 판독성: `.summary.json` 이 downstream 앱과의 계약

## 디렉토리 구조

```
daily/
  conftest.py             공통 fixture + CLI 옵션
  pytest.ini              pytest 설정
  run.py                  entry point (pytest + 리포트 빌더 래핑)
  common/                 config, cmd runner, fs 헬퍼, 캐시 cleaner
  parsers/                출력 파서 (llm_benchmark 등)
  tests/                  실제 테스트 파일
  report/                 리포트 빌더 (pytest JSON → text + summary JSON)
  viewer/                 perf_rows 평탄화 + xlsx 업데이트
```

## 머신별 설정

타겟 OpenVINO 디바이스는 장비마다 다릅니다 (`GPU`, `GPU.0`, `GPU.1` …).
쉘 rc 파일에 한 번만 설정:

```bash
echo 'export DAILY_DEVICE=GPU.1' >> ~/.bashrc
```

`run.py` 와 `pytest` 가 자동으로 읽어갑니다. 특정 실행에서만 덮어쓰려면
`--device ...` 를 CLI 로 전달.

## 실행

```bash
# 전체 실행
python daily/run.py

# 스모크 런 (llama 만, 짧은 토큰)
python daily/run.py --short-run -k llama

# 실행될 테스트 목록만 확인
python daily/run.py -- --collect-only -q

# pytest 직접 실행 (run.py 의 파일 네이밍 + 리포트 단계 건너뜀)
cd daily && pytest tests/test_llm_benchmark.py -v --short-run
```

### 백업 & 메일 (cron 용)

```bash
export MAIL_RELAY_SERVER=dg2raptorlake.ikor.intel.com
python daily/run.py \
    --backup \
    --mail sungeun.kim@intel.com \
    --description "LLM nightly"
```

| 플래그 | 동작 |
|------|--------|
| `--backup` | report + summary.json + raw log + pip-freeze 를 `$MAIL_RELAY_SERVER:/var/www/html/daily2/<hostname>/` 로 scp (기본 호스트: `dg2raptorlake.ikor.intel.com`) |
| `--mail <addrs>` | 텍스트 리포트 (HTML) 를 콤마 구분 수신자에게 메일 |
| `--description` | 메일 제목에 쓰일 자유 텍스트 태그 |
| `--pip-freeze` | `daily.<ts>.requirements.txt` 도 함께 작성 (`--backup`/`--mail` 사용 시 자동 활성) |

`MAIL_RELAY_SERVER` 환경변수가 없으면 기본값 `dg2raptorlake.ikor.intel.com`
을 사용. 다른 서버로 보내려면 환경변수로 오버라이드. Linux 에서는 메일
자체는 로컬 `mail(1)` 로 발송.

백업은 `/var/www/html/daily2/<hostname>/` 아래에 저장 — 기존
`/var/www/html/daily/` (구 스크립트) 와 분리되어 파일이 섞이지 않음.
`<hostname>` 디렉토리는 첫 백업 시 자동 생성 시도.

**서버 측 사전 설정 (한 번만):**

```bash
# 릴레이 서버에서 root 권한으로 — <ssh_user> 는 각 클라이언트가
# scp 할 때 사용하는 ssh 계정 이름 (예: sungeunk)
sudo mkdir -p /var/www/html/daily2
sudo chown <ssh_user>:www-data /var/www/html/daily2
sudo chmod 775 /var/www/html/daily2
```

- owner 를 ssh 사용자로 두면 클라이언트가 `<hostname>` 서브디렉토리를
  스스로 생성 (첫 백업 시 자동 `mkdir -p`).
- group 을 `www-data` 로 유지하면 nginx/apache 가 파일을 읽어서 HTTP 로
  서빙 가능 (백업 URL 접근용).
- 여러 ssh 사용자가 쓸 필요가 있다면 대신 `www-data` 그룹에 각 사용자를
  추가 (`sudo usermod -aG www-data <ssh_user>` 후 재로그인).

ssh 사용자에게 권한이 없으면 `<hostname>` 디렉토리 자동 생성이 실패하고,
어떤 조치가 필요한지 에러 메시지로 안내.

### 공유 master xlsx (OneDrive / SharePoint)

팀이 SharePoint 에 xlsx 하나를 유지 — 각 컬럼이 하나의 일일 런.
`--xlsx-update` 에 로컬 싱크된 경로를 주면 오늘의 결과로 새 컬럼을 추가하고
in-place 저장. OneDrive 가 이후 sync 를 담당.

```bash
python daily/run.py \
    --xlsx-update "/home/me/OneDrive/daily-perf.xlsx" \
    --xlsx-sheet Daily
```

시트 해석 방식 (모두 구성 가능):

| knob | 기본값 | 의미 |
|------|---------|---------|
| `--xlsx-sheet` | 첫 번째 시트 | 쓸 시트 |
| `--xlsx-key-cols` | `1,2,3,4,5` (A..E) | `(model, precision, in, out, exec)` 가 있는 컬럼 |
| `--xlsx-header-rows` | `3` | commit / workweek / datetime 용 헤더 행 수 |

writer 동작:
1. `header_rows + 1` 행부터 키 컬럼들을 읽어 행 순서를 파악
   — `FIXED_ROW_ORDER` 상수 유지 불필요.
2. 가장 오른쪽에 컬럼 하나 추가.
3. 헤더 3셀 (`ov_version`, `workweek`, `datetime`) 과, 이번 런 결과에 있는
   키에 해당하는 값 하나씩 채움. 매칭 안 되는 행은 빈 셀로 남김.

빈 셀로 남는 행은 xlsx 가 기대한 메트릭을 이번 런이 만들지 않았다는 뜻 —
테스트가 skip/fail 됐거나, xlsx 의 키 튜플이 `perf_rows.py` 가 내놓는 값과
다른 경우. 조건문으로 우회하지 말고 xlsx 나 파서를 맞춰야 함.

### 수동 xlsx 업데이트 (cron 외 상황)

```bash
python -m daily.viewer.xlsx_update \
    --summary daily/output/daily.20260422_0104.summary.json \
    --xlsx "/path/to/master.xlsx"
```

## 전체 cron 예시 (Linux)

야간 잡이 실제로 하는 일 전부 — OpenVINO 소싱, 머신별 디바이스 설정,
전체 수트 실행, share / mail / master xlsx 로 결과 전달.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/sungeunk/repo/run_daily2

# 1. OpenVINO 환경 — download-openvino 가 기록한 최신 포인터 사용.
source "$(cat ov_pkg/latest_ov_setup_file.txt)"

# 2. 머신별 설정 (보통 ~/.bashrc 에 있지만, cron 잡이 login shell 에
#    의존하지 않도록 여기서도 선언.)
export DAILY_DEVICE=GPU.1
export MAIL_RELAY_SERVER=dg2raptorlake.ikor.intel.com

# 3. 수트 실행 + 결과 전송.
python daily/run.py \
    --backup \
    --mail your.email@intel.com \
    --description "LLM nightly" \
    --xlsx-update "/home/sungeunk/OneDrive/daily-perf.xlsx" \
    --xlsx-sheet Daily
```

crontab 엔트리 (매일 01:00 실행):

```cron
0 1 * * * /home/sungeunk/repo/run_daily2/daily/cron.sh \
    >> /home/sungeunk/repo/run_daily2/output/cron.log 2>&1
```

## Windows (PowerShell)

Linux 스크립트와 동일한 흐름 — PowerShell + `setupvars.bat` 로 변환.

```powershell
# daily\run-daily.ps1
$ErrorActionPreference = 'Stop'
Set-Location C:\dev\run_daily2

# 1. OpenVINO 환경 — setupvars.bat 을 현재 PowerShell 프로세스에 흡수.
#    setupvars.bat 은 `set` 으로 env 를 내보내므로 그 출력을 파싱해
#    $env: 로 재적용 → 이후 python 호출이 상속받음.
$setupBat = Get-Content ov_pkg\latest_ov_setup_file.txt
cmd /c "`"$setupBat`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}

# 2. 머신별 설정 (한 번만 [System.Environment]::SetEnvironmentVariable 로
#    영구 설정하거나, 이 스크립트를 자족적으로 쓰려면 여기서 반복 선언.)
$env:DAILY_DEVICE = 'GPU.1'
$env:MAIL_RELAY_SERVER = 'dg2raptorlake.ikor.intel.com'

# 3. 수트 실행 + 결과 전송.
python daily\run.py `
    --backup `
    --mail your.email@intel.com `
    --description "LLM nightly" `
    --xlsx-update "C:\Users\me\OneDrive\daily-perf.xlsx" `
    --xlsx-sheet Daily
```

수동 실행:

```powershell
powershell -ExecutionPolicy Bypass -File C:\dev\run_daily2\daily\run-daily.ps1
```

메모:

* `cmd /c "setupvars.bat && set" | ForEach-Object ...` 는 배치 파일의
  환경을 PowerShell 로 가져오는 정석 방법 — PowerShell 은 .bat 을 직접
  `source` 할 수 없음.
* 백업은 `scp.exe` 사용 (최신 Windows 10+ OpenSSH 클라이언트에 기본 포함).
  없으면 OpenSSH 클라이언트 optional feature 설치 필요.
* Windows 메일은 릴레이를 통해 터널링 (`ssh relay "mail ..."`) 이라
  `MAIL_RELAY_SERVER` 설정과 SSH 키 (`%USERPROFILE%\.ssh\id_rsa`) 가
  릴레이에 등록되어 있어야 함.

## 공통 주의사항 (Linux / Windows)

* 스크립트는 준-멱등(idempotent-ish): 재실행 시 새 타임스탬프 + 새 xlsx
  컬럼이 생성되며, 기존 마스터 xlsx 데이터는 덮어쓰지 않음.
* `--backup` 은 `MAIL_RELAY_SERVER` 가 설정되어 있어야 동작. 없으면 경고만
  찍고 스킵. `--mail` 은 Linux 에서 `mail(1)` PATH 에, Windows 에서는
  릴레이 SSH 접근이 필요.
* `--xlsx-update` 는 OneDrive 싱크 중 파일이 잠긴 상태면 non-zero 로 실패.
  재시도하거나, OneDrive 가 건드리지 않는 로컬 경로로 가리키는 방법 검토.

## 산출물

모두 `--output-dir` 아래 생성 (기본: `<repo>/output`):

| 파일                                | 용도                                           |
| ----------------------------------- | ---------------------------------------------- |
| `daily.<ts>.raw`                    | 모든 subprocess 출력 통합 raw log              |
| `daily.<ts>.pytest.json`            | pytest-json-report 원본                        |
| `daily.<ts>.summary.json`           | downstream consumer 용 정규화 스키마           |
| `daily.<ts>.report`                 | 사람이 읽는 텍스트 리포트                      |

### `summary.json` 스키마

```json
{
  "generated_at": 1713499200.0,
  "duration_sec": 1234.5,
  "totals": {"passed": 14, "failed": 0, "error": 0, "skipped": 0, "total": 14},
  "tests": [
    {
      "nodeid": "tests/test_llm_benchmark.py::test_llm_benchmark[llama-2-7b-chat-hf-OV_FP16-4BIT_DEFAULT]",
      "outcome": "passed",
      "duration_sec": 97.3,
      "failure": null,
      "metrics": {
        "test_type": "llm_benchmark",
        "model": "llama-2-7b-chat-hf",
        "precision": "OV_FP16-4BIT_DEFAULT",
        "cmd": "python ... benchmark.py ...",
        "returncode": 0,
        "duration_sec": 97.3,
        "data": [
          {"in_token": 32, "out_token": 256, "perf": [123.4, 45.6], "generated_text": "..."}
        ]
      }
    }
  ]
}
```

## 진행 상태

모든 테스트 포팅 + 백업/메일 + xlsx 업데이트 완료.

- [x] llm_benchmark           (14 케이스)
- [x] benchmark_app           (2 케이스)
- [x] chat_sample             (1 케이스)
- [x] stable_diffusion_genai  (5 케이스, whisper + flux 포함)
- [x] stable_diffusion_dgfx   (2 케이스)
- [x] measured_usage_cpp      (8 케이스, 현재 장비에선 skip; `hw_tracker` 사용)
- [x] whisper_base            (1 케이스, 현재 장비에선 skip; HF 캐시 필요)
- [x] 백업 / 메일 (`--backup`, `--mail`)
- [x] 공유 xlsx 업데이트 (`--xlsx-update`)
- [ ] end-to-end 검증 후 `scripts/` 제거
