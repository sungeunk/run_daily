@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: set environments
:: virtualenv
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    echo "Please call init_env.bat to set dev environments."
    exit /b 0
)
call %DAILY_ROOT%\venv\Scripts\activate.bat

:: openvino
if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
call %OV_SETUP_SCRIPT%

python -c "import importlib;test = importlib.import_module('gpu-tools.run_daily');test.get_now();" > temp.txt
SET /p DATE= < temp.txt
del temp.txt
SET MAIL_FILE=%DW_ROOT%\check_perf.%DATE%.mail
SET RAW_FILE=%DW_ROOT%\check_perf.%DATE%.raw
SET APP=%INTEL_OPENVINO_DIR%\samples\cpp\build\intel64\benchmark_app.exe

:: Begin tag for mail
echo. ^<pre^> >> %MAIL_FILE%
echo. Device: %DEVICE% >> %MAIL_FILE%

:: write header for check_perf v1
(
echo. ^<pre^>
echo. ===============
echo. MODEL V1 result
echo. ===============
) >> %MAIL_FILE%

:: run check_perf v1
python gpu-tools/check_performance.py -d %DEVICE% -a %APP% -c 30 ^
    -m Z:\models ^
    -r result.txt --report_load_time load_time.txt ^
    --cldnn gpu-tools/ref/cldnn.report --ref gpu-tools/ref/onednn.report >> %RAW_FILE%

:: write results for check_perf v1
(
echo.
type result.txt
echo.
type load_time.txt
) >> %MAIL_FILE%

:: write header for check_perf v2
(
echo.
echo. ===============
echo. MODEL V2 result
echo. ===============
) >> %MAIL_FILE%

:: run check_perf v2
python gpu-tools/check_performance.py -d %DEVICE% -a %APP% --version 2 --nstreams 4 --use_device_mem -c 30 ^
    -m Z:\models ^
    -r result.txt --report_load_time load_time.txt ^
    --cldnn gpu-tools/ref/cldnn.v2report --ref gpu-tools/ref/onednn.v2report >> %RAW_FILE%

:: write results for check_perf v2
(
echo.
type result.txt
echo.
type load_time.txt
) >> %MAIL_FILE%

:: End tag for mail
echo. ^</pre^> >> %MAIL_FILE%

:: remove temp files.
del result.txt
del load_time.txt

:: send mail
set MAIL_FILE=%MAIL_FILE:\=/% 
python -c "import importlib;test = importlib.import_module('gpu-tools.run_daily');test.send_mail('%MAIL_FILE%', 'check_perf');"
