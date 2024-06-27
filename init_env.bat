@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

:: for python packages
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    pip install -q virtualenv
    virtualenv venv
)

if exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    call %DAILY_ROOT%\venv\Scripts\activate.bat
) else (
    goto end_script
)

pip install -r requirements.txt
pip install -r openvino.genai\llm_bench\python\requirements.txt
pip install -r %GPU_TOOLS%\whisper\optimum_notebook\non_stateful\requirements.txt

:end_script
