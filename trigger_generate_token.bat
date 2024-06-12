@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

:: parsing arguments
goto GETOPTS

:Help
echo -h this help messages
exit /b 0

:GETOPTS
if /I "%1" == "-h" call :Help
shift
if not "%1" == "" goto GETOPTS

:: set environments
call %VC_ENV_FILE%
if "%VisualStudioVersion%"=="" (
    echo "could not find VC tools"
    exit /b 0
)

:: for python packages
if not exist %DAILY_ROOT%\venv_token\Scripts\activate.bat (
    pip install -q virtualenv
    virtualenv venv_token
)

if exist %DAILY_ROOT%\venv_token\Scripts\activate.bat (
    call %DAILY_ROOT%\venv_token\Scripts\activate.bat
) else (
    goto end_script
)

pip install -r requirements.txt
python gpu-tools\download_ov_nightly.py --download_url https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/windows/w_openvino_toolkit_windows_2024.0.0.14509.34caeefd078_x86_64.zip

set OV_SETUP_SCRIPT=%DAILY_ROOT%\openvino_nightly\w_openvino_toolkit_windows_2024.0.0.14509.34caeefd078_x86_64\setupvars.bat
if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
echo call %OV_SETUP_SCRIPT%
call %OV_SETUP_SCRIPT%


cd openvino.genai.token
git checkout -f 534963a79037ec682f295e2e60e4fba79e973fa5
git submodule update --init --recursive

pip install -r image_generation\stable_diffusion_1_5\cpp\requirements.txt
pip install thirdparty\openvino_tokenizers\[transformers]
cd ..

set LINK_DST="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\INT8"
set LINK_SRC="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
if exist %LINK_SRC% (
    if not exist %LINK_DST% (
        mklink /d %LINK_DST% %LINK_SRC%
    )
)

set LINK_DST="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\INT8"
set LINK_SRC="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
if exist %LINK_SRC% (
    if not exist %LINK_DST% (
        mklink /d %LINK_DST% %LINK_SRC%
    )
)

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ --tokenizer-output-type i32 -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ --tokenizer-output-type i32 -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ --tokenizer-output-type i32 -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ --tokenizer-output-type i32 -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\lcm-dreamshaper-v7\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\
