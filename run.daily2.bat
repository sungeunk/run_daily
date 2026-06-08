@echo off
setlocal enabledelayedexpansion

call conda activate daily
call %1

:: get OpenVINO Version
for /f "tokens=*" %%a in ('python -c "from openvino import get_version; print(get_version())"') do (
    set "VERSION=%%a"
)

python daily\run.py -k "whisper and large and v3"
