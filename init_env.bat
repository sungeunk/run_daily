@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

:: update submodules
cd openvino.genai
git submodule update --init --recursive
cd ..

cd openvino.genai.token
git submodule update --init --recursive
cd ..

cd openvino.genai.chatglm3
git submodule update --init --recursive
cd ..

cd openvino.genai.qwen
git submodule update --init --recursive
cd ..

:: mount model directory
@REM if not exist Z:\models\resnet_v1.5_50\resnet_v1.5_50_i8.xml (
@REM     net use Z: \\sshfs.kr\sungeunk@dg2ubuntu.ikor.intel.com\mnt\nvme\shared
@REM )


:: for python packages
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    pip install -q virtualenv
    virtualenv venv
)

if exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    call %DAILY_ROOT%\venv\Scripts\activate.bat
) else (
    goto end_script
)

pip install -r requirements.txt
pip install -r openvino.genai\llm_bench\python\requirements.txt
pip uninstall openvino-nightly

:end_script
