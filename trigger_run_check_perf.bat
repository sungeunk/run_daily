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
echo.
echo. ===============
echo. MODEL V1 result
echo. ===============
) >> %MAIL_FILE%

:: run check_perf v1
python gpu-tools/check_performance.py -d %DEVICE% -a %APP% ^
    -m Z:\models ^
    -r result.txt --report_load_time load_time.txt ^
    --cldnn gpu-tools/ref/cldnn.report --ref gpu-tools/ref/onednn.report ^
    --pickle version1.pickle

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
python gpu-tools/check_performance.py -d %DEVICE% -a %APP% --version 2 --nstreams 4 --use_device_mem ^
    -m Z:\models ^
    -r result.txt --report_load_time load_time.txt ^
    --cldnn gpu-tools/ref/cldnn.v2report --ref gpu-tools/ref/onednn.v2report ^
    --pickle version2.pickle

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
@REM set MAIL_FILE=%MAIL_FILE:\=/%
@REM python -c "import importlib;test = importlib.import_module('gpu-tools.run_daily');test.send_mail('%MAIL_FILE%', 'check_perf');"


:end_script
call conda deactivate
