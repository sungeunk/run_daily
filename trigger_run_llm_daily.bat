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

set SEND_MAIL=
set TEST=
set REPORT_DESCRIPTION=

:: parsing arguments
goto GETOPTS

:Help
echo -h this help messages
echo -ov {ov setup file path}
exit /b 0

:GETOPTS
if /I "%1" == "-h" call :Help
if /I "%1" == "-ov" set OV_SETUP_SCRIPT=%2 & shift
if /I "%1" == "-desc" set REPORT_DESCRIPTION=--description %2 & shift
if /I "%1" == "-mail" set SEND_MAIL=--mail
if /I "%1" == "-test" set TEST=--test
if /I "%1" == "-d" set DEVICE=%2
shift
if not "%1" == "" goto GETOPTS


call %OV_SETUP_SCRIPT%
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: %OV_SETUP_SCRIPT%
    goto end_script
)


if defined REF_REPORT (
    if exist "%REF_REPORT%" (
        set SET_REF_REPORT=--ref_report %REF_REPORT%
    )
)


:: run daily
@echo on
python %GPU_TOOLS%\run_llm_daily.py ^
    -w %DAILY_ROOT% ^
    -m %MODEL_ROOT% ^
    -o %DW_ROOT% ^
    -d %DEVICE% ^
    --repeat 1 ^
    --ov_dev_data %SEND_MAIL% %TEST% %SET_REF_REPORT% %REPORT_DESCRIPTION% ^
    --benchmark_app %BIN_DIR%\benchmark_app\benchmark_app.exe

:end_script
call conda deactivate
