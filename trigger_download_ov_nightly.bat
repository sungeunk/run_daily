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

:: call download
python %GPU_TOOLS%\download_ov_nightly.py --clean_up

:end_script
call conda deactivate
