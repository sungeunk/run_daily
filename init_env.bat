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
pip install -r %GPU_TOOLS%\whisper\optimum_notebook\non_stateful\requirements.txt

call conda deactivate


:: python environment for generate token
call conda activate daily_token
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: conda activate daily_token
    echo Please try: conda create -n daily_token python=3.11 -y
    goto end_script
)

if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
echo call %OV_SETUP_SCRIPT%
call %OV_SETUP_SCRIPT%

pip install -r requirements.txt
pip install -r openvino.genai.lcm\image_generation\lcm_dreamshaper_v7\cpp\requirements.txt
pip install -r openvino.genai.lcm\thirdparty\openvino_tokenizers\requirements-build.txt
pip install openvino.genai.lcm\thirdparty\openvino_tokenizers\[transformers]

call conda deactivate


:: vcpkg for opencl
cd vcpkg
bootstrap-vcpkg.bat
vcpkg.exe install opencl
cd ..

:end_script
call conda deactivate
