::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: set path (User definitions)
SET DAILY_ROOT=%~dp0
SET DW_ROOT="%DAILY_ROOT%\directory_browsing\run_daily"
SET GPU_TOOLS="%DAILY_ROOT%\gpu-tools"
SET MODEL_ROOT="C:\dev\models"
SET OCL_ROOT=%DAILY_ROOT%\thirdparty\OpenCL-SDK-v2023.12.14-Win-x64
SET VC_ENV_FILE="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
:: End (User definitions)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: set environments
if exist %DAILY_ROOT%\openvino_nightly\latest_ov_setup_file.txt (
    SET /p OV_SETUP_SCRIPT= < %DAILY_ROOT%\openvino_nightly\latest_ov_setup_file.txt
)

:: don't set proxy for TOE
SET http_proxy=
SET https_proxy=proxy-dmz.intel.com:912

SET /a "NPROC=%NUMBER_OF_PROCESSORS%*3/4"
