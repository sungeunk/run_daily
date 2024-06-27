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

call conda activate daily_token
IF %ERRORLEVEL% NEQ 0 (
    echo could not call: conda activate daily_token
    echo Please try: conda create -n daily_token python=3.11 -y
    goto end_script
)


set LINK_DST="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\INT8"
set LINK_SRC="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
if exist %LINK_SRC% (
    if not exist %LINK_DST% (
        mklink /d %LINK_DST% %LINK_SRC%
    )
)

set LINK_DST="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\INT8"
set LINK_SRC="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
if exist %LINK_SRC% (
    if not exist %LINK_DST% (
        mklink /d %LINK_DST% %LINK_SRC%
    )
)

set MODEL_PATH="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v1-5\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v1-5\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v2-1\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\%MODEL_DATE%\stable-diffusion-v2-1\pytorch\dldt\compressed_weights\OV_FP16-INT8_ASYM"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

set MODEL_PATH="%MODEL_ROOT%\%MODEL_DATE%\lcm-dreamshaper-v7\pytorch\dldt\FP16"
convert_tokenizer %MODEL_PATH%\tokenizer\ -o %MODEL_PATH%\tokenizer\

:end_script
call conda deactivate
