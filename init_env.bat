@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

:: for python packages
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    pip install -q virtualenv
    virtualenv venv
)

call %DAILY_ROOT%\venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: %DAILY_ROOT%\venv\Scripts\activate.bat
    goto end_script
)

pip install -r %DAILY_ROOT%\requirements.txt
pip install -r openvino.genai\llm_bench\python\requirements.txt
pip install -r %GPU_TOOLS%\whisper\optimum_notebook\non_stateful\requirements.txt

call deactivate


:: python environment for generate token
if not exist %DAILY_ROOT%\venv_token\Scripts\activate.bat (
    pip install -q virtualenv
    virtualenv venv_token
)

call %DAILY_ROOT%\venv_token\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: %DAILY_ROOT%\venv_token\Scripts\activate.bat
    goto end_script
)

pip install -r %DAILY_ROOT%\requirements.txt

cd %DAILY_ROOT%\openvino.genai.token
pip install -r image_generation\stable_diffusion_1_5\cpp\requirements.txt
pip install openvino openvino-dev
pip install thirdparty\openvino_tokenizers\[transformers]
cd ..

call deactivate

:: vcpkg for opencl
cd vcpkg
bootstrap-vcpkg.bat
vcpkg.exe install opencl
cd ..

:end_script
