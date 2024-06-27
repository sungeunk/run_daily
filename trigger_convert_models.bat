@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

call conda activate daily
IF %ERRORLEVEL% NEQ 0 (
    echo could not call the miniconda. Please install or check it.
    goto end_script
)

call %OV_SETUP_SCRIPT%
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: %OV_SETUP_SCRIPT%
    goto end_script
)

:: convert models
@echo on
python %GPU_TOOLS%\run_llm_daily.py ^
    -m %MODEL_ROOT% ^
    -d %DEVICE% ^
    --convert_models

:end_script
call conda deactivate
