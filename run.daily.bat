@echo off
setlocal enabledelayedexpansion

call conda activate daily.py312

set "REPO_ROOT=C:\dev\sungeunk\repo"
set "DAILY_DIR=%~dp0"
set "SKILLS_DIR=%REPO_ROOT%\openvino-gpu-plugin-skills"
set "DOWNLOAD_SCRIPT=%SKILLS_DIR%\.github\skills\download-openvino\scripts\download-openvino.py"
set "DOWNLOAD_OUTPUT=%DAILY_DIR%\openvino_nightly"
set "LATEST_SETUP_FILE=%DOWNLOAD_OUTPUT%\latest_ov_setup_file.txt"
set "MODEL_DIR=c:\dev\models\daily"

if not exist "%DOWNLOAD_SCRIPT%" (
    echo Error: download-openvino.py not found: %DOWNLOAD_SCRIPT%
    exit /b 1
)

REM No argument: download latest OpenVINO.
if "%~1"=="" (
    echo No SHA or setupvars path provided. Downloading latest OpenVINO ...
    uv run --script "%DOWNLOAD_SCRIPT%" --output "%DOWNLOAD_OUTPUT%"

    if errorlevel 1 (
        echo Download failed!
        exit /b 1
    )

    if not exist "%LATEST_SETUP_FILE%" (
        echo Error: %LATEST_SETUP_FILE% not created by download script!
        exit /b 1
    )

    for /f "usebackq delims=" %%f in ("%LATEST_SETUP_FILE%") do (
        set "SETUPVARS=%%f"
    )
    goto found
)

REM Existing file path or .bat path: use it as setupvars.
if exist "%~1" (
    set "SETUPVARS=%~1"
) else if /I "%~x1"==".bat" (
    set "SETUPVARS=%~1"
) else (
    REM Otherwise treat the argument as an OpenVINO commit SHA.
    set "SHA=%~1"
    echo Downloading OpenVINO for commit !SHA! ...
    uv run --script "%DOWNLOAD_SCRIPT%" --commit-id "!SHA!" --output "%DOWNLOAD_OUTPUT%"

    if errorlevel 1 (
        echo Download failed!
        exit /b 1
    )

    if not exist "%LATEST_SETUP_FILE%" (
        echo Error: %LATEST_SETUP_FILE% not created by download script!
        exit /b 1
    )

    for /f "usebackq delims=" %%f in ("%LATEST_SETUP_FILE%") do (
        set "SETUPVARS=%%f"
    )
)

:found
if not exist "!SETUPVARS!" (
    echo Error: Setup script not found: !SETUPVARS!
    exit /b 1
)

echo Executing: !SETUPVARS!
call "!SETUPVARS!"

for /f "tokens=*" %%a in ('python -c "from openvino import get_version; print(get_version())"') do (
    set "VERSION=%%a"
)

echo OpenVINO: !VERSION!

if not exist "%MODEL_DIR%" (
    echo Error: model directory not found: %MODEL_DIR%
    exit /b 1
)

pushd "%DAILY_DIR%"
python scripts\run_llm_daily.py --device GPU -m "%MODEL_DIR%" --description "daily_CB"
set "RUN_EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %RUN_EXIT_CODE%