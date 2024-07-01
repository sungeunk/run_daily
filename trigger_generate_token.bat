@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)


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
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\lcm-dreamshaper-v7\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\
