@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)


:: python environment for daily
call conda activate daily
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: conda activate daily
    echo Please try: conda create -n daily python=3.11 -y
    goto end_script
)

pip install -r requirements.txt
pip install -r openvino.genai\llm_bench\python\requirements.txt
pip install -r gpu-tools\whisper\optimum_notebook\non_stateful\requirements.txt

call conda deactivate


:: vcpkg for opencl
cd vcpkg
bootstrap-vcpkg.bat
vcpkg.exe install opencl
cd ..

:end_script
