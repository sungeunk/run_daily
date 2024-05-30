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

set NO_MAIL=

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

if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
call %OV_SETUP_SCRIPT%

:: run daily
@echo on
python %GPU_TOOLS%\run_llm_daily.py ^
    -w %DAILY_ROOT% ^
    -m %MODEL_ROOT% ^
    -o %DW_ROOT% ^
    -d %DEVICE% ^
