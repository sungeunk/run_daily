@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: set environments
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    virtualenv venv
)
call %DAILY_ROOT%\venv\Scripts\activate.bat
call %OV_SETUP_SCRIPT%

:: run daily
python %GPU_TOOLS%\run_daily.py ^
    -w %DAILY_ROOT% ^
    -m %MODEL_ROOT% ^
    -o %DW_ROOT% ^
    --ref_pickle %DW_ROOT%\20240206_1435.pickle ^
    --benchmark_app %INTEL_OPENVINO_DIR%\samples\cpp\build\intel64\benchmark_app.exe ^
    --gpu_tools_dir %GPU_TOOLS%
