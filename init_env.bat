@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: update submodules
cd openvino.genai.chatglm3
git submodule update --init --recursive
cd ..

cd openvino.genai.qwen
git submodule update --init --recursive
cd ..

:: mount model directory
if not exist Z:\models\resnet_v1.5_50\resnet_v1.5_50_i8.xml (
    net use Z: \\sshfs.kr\sungeunk@dg2ubuntu.ikor.intel.com\mnt\nvme\shared
)

:: for python packages
if not exist %DAILY_ROOT%\venv\Scripts\activate.bat (
    virtualenv venv
)

call %DAILY_ROOT%\venv\Scripts\activate.bat

pip install -r requirements.txt


