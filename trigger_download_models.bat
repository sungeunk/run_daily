@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

call conda activate daily
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: conda activate daily_token
    echo Please try: conda create -n daily_token python=3.11 -y
    goto end_script
)

:: download models
python %DAILY_ROOT%\download_models.py --output %MODEL_ROOT%

:end_script
call conda deactivate
