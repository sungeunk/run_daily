# daily — pytest-based daily test suite (WIP)

Replacement for `scripts/run_llm_daily.py`. The old orchestration code in
`scripts/` is kept for reference during the migration and will be removed
once every test has been ported here.

## Goals

1. Built on **pytest** (+ `pytest-json-report`) — standard framework, standard
   selection syntax, standard plugin ecosystem.
2. Individual test selection: `pytest -k "llama"`, `--collect-only` to list.
3. Single raw log for the whole run: every subprocess tees stdout into one
   session-scoped file (`daily.<timestamp>.raw`).
4. Tests don't affect each other: `isolate_test` is an `autouse` fixture that
   clears model caches before and after every test.
5. One consolidated report per run: text (`.report`) + normalised JSON
   (`.summary.json`) built from pytest's JSON output.
6. Machine readable: the `.summary.json` is the contract for downstream apps.

## Layout

```
daily/
  conftest.py             shared fixtures + CLI options
  pytest.ini              pytest config
  run.py                  entry point (wraps pytest + report builder)
  common/                 config, cmd runner, fs helpers, cache cleaner
  parsers/                output parsers (currently: llm_benchmark)
  tests/                  the actual tests
  report/                 report builder (pytest JSON → text + summary JSON)
```

## Per-machine setup

The target OpenVINO device differs per rig (`GPU`, `GPU.0`, `GPU.1` …). Set
it once in the machine's shell rc file:

```bash
echo 'export DAILY_DEVICE=GPU.1' >> ~/.bashrc
```

Both `run.py` and `pytest` pick it up automatically. Pass `--device ...`
on the CLI to override for a single run.

## Running

```bash
# full run
python daily/run.py

# smoke run, only llama cases
python daily/run.py --short-run -k llama

# list what would run
python daily/run.py -- --collect-only -q

# pytest directly (skips run.py's output-file naming + report step)
cd daily && pytest tests/test_llm_benchmark.py -v --short-run
```

### Backup & mail (for cron)

```bash
export MAIL_RELAY_SERVER=dg2raptorlake.ikor.intel.com
python daily/run.py \
    --backup \
    --mail sungeun.kim@intel.com \
    --description "LLM nightly"
```

| flag | effect |
|------|--------|
| `--backup` | scp report + summary.json + raw log + pip-freeze to `$MAIL_RELAY_SERVER:/var/www/html/daily/<hostname>/` |
| `--mail <addrs>` | mail the text report (HTML) to comma-separated recipients |
| `--description` | free-text tag used in mail subject |
| `--pip-freeze` | also write `daily.<ts>.requirements.txt` (auto-on with `--backup`/`--mail`) |

`MAIL_RELAY_SERVER` environment variable is required for `--backup` and for
Windows mail delivery. On Linux, mail is sent via the local `mail(1)` binary.

### Shared master xlsx (OneDrive / SharePoint)

The team keeps one xlsx on SharePoint where each column is a daily run.
Point `--xlsx-update` at the locally-synced copy; the run will append a new
column with this day's numbers and save in place. OneDrive handles the sync.

```bash
python daily/run.py \
    --xlsx-update "/home/me/OneDrive/daily-perf.xlsx" \
    --xlsx-sheet Daily
```

How the sheet is interpreted (all configurable):

| knob | default | meaning |
|------|---------|---------|
| `--xlsx-sheet` | first sheet | which sheet to write to |
| `--xlsx-key-cols` | `1,2,3,4,5` (A..E) | columns holding `(model, precision, in, out, exec)` |
| `--xlsx-header-rows` | `3` | rows above data used for commit / workweek / datetime |

The writer:
1. Reads the key columns starting at row `header_rows + 1` to learn row order
   — no `FIXED_ROW_ORDER` constant to maintain.
2. Appends one column at the rightmost position.
3. Fills three header cells (`ov_version`, `workweek`, `datetime`) and one
   value per key that exists in this run's results. Missing rows stay blank.

Rows that stay blank mean the master xlsx expects a metric this run didn't
produce — either the test was skipped/failed, or the key tuple in the xlsx
doesn't match what `perf_rows.py` emits. Adjust the xlsx or the parser
rather than adding conditionals.

### Manual xlsx append (outside the cron pipeline)

```bash
python -m daily.viewer.xlsx_update \
    --summary daily/output/daily.20260422_0104.summary.json \
    --xlsx "/path/to/master.xlsx"
```

## Full cron example

Everything the nightly job actually does — sourcing OpenVINO, setting the
per-machine device, running the full suite, then shipping artefacts to the
share, mail, and the master xlsx.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/sungeunk/repo/run_daily2

# 1. OpenVINO env — always use the latest pointer written by download-openvino.
source "$(cat ov_pkg/latest_ov_setup_file.txt)"

# 2. Per-machine knobs (these typically live in ~/.bashrc; re-declared here so
#    the cron job doesn't rely on a login shell.)
export DAILY_DEVICE=GPU.1
export MAIL_RELAY_SERVER=dg2raptorlake.ikor.intel.com

# 3. Run the suite and ship results.
python daily/run.py \
    --backup \
    --mail your.email@intel.com \
    --description "LLM nightly" \
    --xlsx-update "/home/sungeunk/OneDrive/daily-perf.xlsx" \
    --xlsx-sheet Daily
```

crontab entry (runs every day at 01:00):

```cron
0 1 * * * /home/sungeunk/repo/run_daily2/daily/cron.sh \
    >> /home/sungeunk/repo/run_daily2/output/cron.log 2>&1
```

## Windows (PowerShell)

Same idea as the Linux script, adapted to PowerShell + `setupvars.bat`.

```powershell
# daily\run-daily.ps1
$ErrorActionPreference = 'Stop'
Set-Location C:\dev\run_daily2

# 1. OpenVINO env — source setupvars.bat into the current PowerShell process.
#    setupvars.bat exports env vars via `set`; we parse its output and
#    re-apply them with $env: so the subsequent python call inherits them.
$setupBat = Get-Content ov_pkg\latest_ov_setup_file.txt
cmd /c "`"$setupBat`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}

# 2. Per-machine knobs (set once via [System.Environment]::SetEnvironmentVariable,
#    or repeated here for a self-contained script).
$env:DAILY_DEVICE = 'GPU.1'
$env:MAIL_RELAY_SERVER = 'dg2raptorlake.ikor.intel.com'

# 3. Run the suite and ship results.
python daily\run.py `
    --backup `
    --mail your.email@intel.com `
    --description "LLM nightly" `
    --xlsx-update "C:\Users\me\OneDrive\daily-perf.xlsx" `
    --xlsx-sheet Daily
```

Run it manually:

```powershell
powershell -ExecutionPolicy Bypass -File C:\dev\run_daily2\daily\run-daily.ps1
```

Notes:

* The `cmd /c "setupvars.bat && set" | ForEach-Object ...` dance is the
  canonical way to import a batch file's environment into PowerShell —
  PowerShell can't `source` a .bat directly.
* Backup uses `scp.exe` (ships with modern Windows 10+ OpenSSH client). If
  it's missing, install the OpenSSH client optional feature.
* Mail delivery on Windows tunnels through the relay via
  `ssh relay "mail ..."`, so `MAIL_RELAY_SERVER` must be set and the current
  user's SSH key (`%USERPROFILE%\.ssh\id_rsa`) must be authorised on the
  relay.

## Notes (both platforms)

* The script is idempotent-ish: re-running it creates a fresh run with a new
  timestamp and a new xlsx column. Nothing in the master xlsx is overwritten.
* `--backup` needs `MAIL_RELAY_SERVER` set; otherwise it logs a warning and
  skips. `--mail` needs `mail(1)` on the PATH (Linux) or SSH access to the
  relay (Windows).
* `--xlsx-update` fails the run with a non-zero exit if the xlsx is locked
  by OneDrive mid-sync — retry, or point to a locally-synced path that
  OneDrive doesn't touch during the write window.

## Output artefacts

All written under `--output-dir` (default `<repo>/output`):

| File                                | Purpose                                        |
| ----------------------------------- | ---------------------------------------------- |
| `daily.<ts>.raw`                    | Unified raw log of all subprocess output       |
| `daily.<ts>.pytest.json`            | pytest-json-report raw output                  |
| `daily.<ts>.summary.json`           | Normalised schema for downstream consumers     |
| `daily.<ts>.report`                 | Human readable text report                     |

### `summary.json` schema

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

## Status

All tests ported. Still pending a smoke-run on the target machine.

- [x] llm_benchmark           (14 cases)
- [x] benchmark_app           (2 cases)
- [x] chat_sample             (1 case)
- [x] stable_diffusion_genai  (5 cases, incl. whisper + flux)
- [x] stable_diffusion_dgfx   (2 cases)
- [x] measured_usage_cpp      (8 cases, uses `hw_tracker`)
- [x] whisper_base            (1 case)
- [ ] backup / mail delivery (follow-up in `run.py`)
- [ ] drop `scripts/` once the new suite passes end-to-end
