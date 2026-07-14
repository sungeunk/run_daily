@echo off
setlocal enabledelayedexpansion

call conda activate daily
call %1

:: get OpenVINO Version
for /f "tokens=*" %%a in ('python -c "from openvino import get_version; print(get_version())"') do (
    set "VERSION=%%a"
)

:: TimedOUT: gemma-4-26b-a4b-it
python daily\run.py -k "gpt-oss-20b|qwen3.6-35b-a3b" --device GPU.1
