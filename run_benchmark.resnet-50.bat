@echo off && setlocal

set OV_SETUP_SCRIPT=%1
set OV_INSTALL_DIR=%~d1%~p1
set MODEL=models\resnet_v1.5_50\resnet_v1.5_50_i8.xml
set DEVICE=GPU

if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)

call %OV_SETUP_SCRIPT%

echo Mode: latency
FOR %%x IN (1 2 3) DO %OV_INSTALL_DIR%\samples\cpp\build\intel64\benchmark_app.exe -m %MODEL% -d %DEVICE% --hint latency -t 10 | findstr Throughput

echo Mode: tput
FOR %%x IN (1 2 3) DO %OV_INSTALL_DIR%\samples\cpp\build\intel64\benchmark_app.exe -m %MODEL% -d %DEVICE% --hint tput -t 10 | findstr Throughput

echo Mode: none
FOR %%x IN (1 2 3) DO %OV_INSTALL_DIR%\samples\cpp\build\intel64\benchmark_app.exe -m %MODEL% -d %DEVICE% --hint none --nstreams 2 --nireq 4 -t 10 | findstr Throughput
