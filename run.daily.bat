@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM LLM benchmark runner (openvino.genai llm_bench, invoked directly).
REM
REM Runs every BACKENDS x MODEL_DATES combination and prints a per-case summary:
REM   PA + {WW24, WW34, WW35}, then SDPA + {WW24, WW34, WW35}
REM
REM Usage:
REM   run.daily.bat                       download latest nightly OpenVINO and run
REM   run.daily.bat <COMMIT_SHA>          download that build and run
REM   run.daily.bat <path\setupvars.bat>  use an already installed package
REM
REM Optional env:
REM   set DAILY_DEVICE=GPU.1               target device (default GPU.1)
REM   set BENCH_EXTRA=--num_beams 1        extra args appended to benchmark.py
REM   set BACKENDS=PA                      narrow the matrix (e.g. for bisect)
REM   set MODEL_DATES=WW35_...             narrow the matrix (e.g. for bisect)
REM ============================================================================

REM ---- User Config -----------------------------------------------------------
set "CONDA_ENV=daily.py312"
set "SKILL_ROOT=c:\dev\sungeunk\repo\openvino-gpu-plugin-skills"
set "MODEL_DIR=c:\dev\models\daily"
REM One run per BACKENDS x MODEL_DATES pair; models live at %MODEL_DIR%\<date>\<model>\pytorch\ov\<precision>.
if not defined BACKENDS set "BACKENDS=PA"
if not defined MODEL_DATES set "MODEL_DATES=WW35_llm-optimum_2026.4.0-22930-RC1"
if not defined MODEL_NAME set "MODEL_NAME=qwen3.6-35b-a3b"
if not defined PRECISION set "PRECISION=OV_FP16-4BIT_DEFAULT"
set "BENCH_ARGS=-mc 1 -ic 128 -n 1 --apply_chat_template -t visual_text_gen --disable_prompt_permutation"
if not defined DAILY_DEVICE set "DAILY_DEVICE=GPU"
REM ----------------------------------------------------------------------------

set "DOWNLOAD_OV=%SKILL_ROOT%\.github\skills\download-openvino\scripts\download-openvino.py"
set "DOWNLOAD_OUTPUT=%~dp0openvino_nightly"
set "LATEST_SETUP_FILE=%DOWNLOAD_OUTPUT%\latest_ov_setup_file.txt"
set "LLM_BENCH=%~dp0openvino.genai\tools\llm_bench\benchmark.py"
set "PROMPT_FILE=%~dp0prompts\32_1024\%MODEL_NAME%.jsonl"
set "OUTPUT_DIR=%~dp0daily_output\RAPTOR-ELLY"

cd /d "%~dp0"

if not exist "%DOWNLOAD_OV%" (echo [ERROR] download-openvino.py not found: %DOWNLOAD_OV%& exit /b 1)
if not exist "%LLM_BENCH%" (echo [ERROR] benchmark.py not found: %LLM_BENCH%& exit /b 1)
if not exist "%PROMPT_FILE%" (echo [ERROR] prompt file not found: %PROMPT_FILE%& exit /b 1)

call conda activate %CONDA_ENV%
if errorlevel 1 (echo [ERROR] conda activate %CONDA_ENV% failed& exit /b 1)

REM ---- 1. Resolve OpenVINO package -------------------------------------------
if "%~1"=="" (
    echo [1/3] No argument provided. Downloading latest OpenVINO nightly ...
    uv run --script "%DOWNLOAD_OV%" --output "%DOWNLOAD_OUTPUT%"
    if errorlevel 1 (echo [ERROR] OpenVINO download failed& exit /b 1)
    goto read_latest
)

REM An existing path or an explicit .bat argument is treated as setupvars.
if exist "%~1" (
    echo [1/3] Using provided setupvars: %~1
    set "SETUPVARS=%~1"
    goto found
)
if /I "%~x1"==".bat" (
    echo [1/3] Using provided setupvars: %~1
    set "SETUPVARS=%~1"
    goto found
)

echo [1/3] Downloading OpenVINO for commit %~1 ...
uv run --script "%DOWNLOAD_OV%" --commit-id "%~1" --output "%DOWNLOAD_OUTPUT%"
if errorlevel 1 (echo [ERROR] OpenVINO download failed& exit /b 1)

:read_latest
if not exist "%LATEST_SETUP_FILE%" (
    echo [ERROR] %LATEST_SETUP_FILE% not created by download script
    exit /b 1
)
for /f "usebackq delims=" %%f in ("%LATEST_SETUP_FILE%") do set "SETUPVARS=%%f"

:found
if not exist "!SETUPVARS!" (echo [ERROR] Setup script not found: !SETUPVARS!& exit /b 1)

echo [2/3] Executing: !SETUPVARS!
call "!SETUPVARS!"
if errorlevel 1 (echo [ERROR] setupvars failed: !SETUPVARS!& exit /b 1)
call conda activate %CONDA_ENV%

set "VERSION="
for /f "usebackq tokens=*" %%a in (`python -c "from openvino import get_version; print(get_version())"`) do set "VERSION=%%a"
if not defined VERSION (echo [ERROR] Cannot import openvino from env %CONDA_ENV%& exit /b 1)
echo       OpenVINO: !VERSION!

REM ---- 2. Benchmark every backend x model date pair -----------------------------
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"`) do set "TS=%%t"
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "Get-Date -Format yyyy.MM"`) do set "OUTPUT_MONTH=%%m"
set "OUTPUT_DIR=%OUTPUT_DIR%\%OUTPUT_MONTH%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

set "OVERALL_EXIT=0"
set "STEP=0"
for %%b in (%BACKENDS%) do for %%d in (%MODEL_DATES%) do call :run_one "%%b" "%%d"

echo.
echo ----------------------------------------------------------------------------
echo OpenVINO : !VERSION!
echo Device   : %DAILY_DEVICE%
echo Model    : %MODEL_NAME% (%PRECISION%)
for /l %%i in (1,1,!STEP!) do echo   %%i. !LABEL_%%i! : !RESULT_%%i!
if "!OVERALL_EXIT!"=="0" (echo Overall  : PASS) else (echo Overall  : FAIL)
echo Reports  : %OUTPUT_DIR%
echo ----------------------------------------------------------------------------

exit /b !OVERALL_EXIT!

:run_one
set /a STEP+=1
set "BACKEND=%~1"
set "MDATE=%~2"
set "LABEL_!STEP!=%BACKEND% + %MDATE%"
set "MODEL_PATH=%MODEL_DIR%\%MDATE%\%MODEL_NAME%\pytorch\ov\%PRECISION%"
set "REPORT_JSON=%OUTPUT_DIR%\llm_bench_%MODEL_NAME%-%PRECISION%_%BACKEND%_%MDATE%_%TS%.json"
echo.
echo [3/3] Case !STEP!: %BACKEND% + %MDATE% on %DAILY_DEVICE% ...
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model directory not found: %MODEL_PATH%
    set "RESULT_!STEP!=SKIP - model directory missing"
    set "OVERALL_EXIT=1"
    goto :eof
)
python "%LLM_BENCH%" -m "%MODEL_PATH%" -d %DAILY_DEVICE% %BENCH_ARGS% --cb_config "{\"enable_prefix_caching\": false, \"max_num_batched_tokens\": 99999999}" --load_config "{\"ATTENTION_BACKEND\": \"%BACKEND%\"}" -pf "%PROMPT_FILE%" -rj "%REPORT_JSON%" %BENCH_EXTRA%
set "RC=!ERRORLEVEL!"
if !RC! EQU 0 (
    echo       PASS: %BACKEND% + %MDATE%
    set "RESULT_!STEP!=PASS"
) else (
    echo       FAIL: %BACKEND% + %MDATE% [exit !RC!]
    set "RESULT_!STEP!=FAIL - exit !RC!"
    set "OVERALL_EXIT=1"
)
goto :eof
