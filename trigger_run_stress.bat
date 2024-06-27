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

:: parsing arguments
goto GETOPTS

:Help
echo -h this help messages
echo -ov {ov setup file path}
exit /b 0

:GETOPTS
if /I "%1" == "-h" call :Help
if /I "%1" == "-ov" set OV_SETUP_SCRIPT=%2 & shift
if /I "%1" == "-test" set NO_MAIL=--no_mail
shift
if not "%1" == "" goto GETOPTS


call %OV_SETUP_SCRIPT%
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: %OV_SETUP_SCRIPT%
    goto end_script
)


:: run daily
@echo on
python %GPU_TOOLS%\run_daily.py ^
    -w %DAILY_ROOT% ^
    -m %MODEL_ROOT% ^
    -o %DW_ROOT% ^
    -d %DEVICE% ^
    --gpu_tools_dir %GPU_TOOLS% ^
    --mode stress  %NO_MAIL%

:end_script
call conda deactivate
