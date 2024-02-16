@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: set environments
call %DAILY_ROOT%\venv\Scripts\activate.bat

:: call download
python %GPU_TOOLS%\download_ov_nightly.py
