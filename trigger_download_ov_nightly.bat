@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

:: set environments
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    echo "Please call init_env.bat to set dev environments."
    exit /b 0
)
call %DAILY_ROOT%\venv\Scripts\activate.bat

:: call download
python %GPU_TOOLS%\download_ov_nightly.py --delete_zip
