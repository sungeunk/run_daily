@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: set environments
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    echo "Please call init_env.bat to set dev environments."
    exit /b 0
)
call %DAILY_ROOT%\venv\Scripts\activate.bat

if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
call %OV_SETUP_SCRIPT%

:: run daily
python %GPU_TOOLS%\run_daily.py ^
    -w %DAILY_ROOT% ^
    -m %MODEL_ROOT% ^
    -o %DW_ROOT% ^
    --ref_pickle %DW_ROOT%\20240206_1435.pickle ^
    --benchmark_app %INTEL_OPENVINO_DIR%\samples\cpp\build\intel64\benchmark_app.exe ^
    --gpu_tools_dir %GPU_TOOLS%
