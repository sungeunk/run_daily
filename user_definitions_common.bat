::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: set path (User definitions)
SET DAILY_ROOT=%~dp0
SET DW_ROOT=%DAILY_ROOT%\directory_browsing\run_daily
SET BIN_DIR=%DAILY_ROOT%\bin
SET GPU_TOOLS=%DAILY_ROOT%\gpu-tools
SET MODEL_ROOT=C:\dev\models
SET VC_ENV_FILE_BUILDTOOLS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
SET VC_ENV_FILE_COMMUNITY="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
SET VC_ENV_FILE_PRO="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
SET VCPKG_PRIVATE_ROOT=%DAILY_ROOT%\vcpkg

SET PYTHONIOENCODING=utf-8
SET DEVICE=GPU
SET MAIL_TO=nex.nswe.odt.runtime.kor@intel.com
SET MAIL_RELAY_SERVER=sungeunk@dg2raptorlake.ikor.intel.com
:: End (User definitions)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: set environments
if exist %DAILY_ROOT%\openvino_nightly\latest_ov_setup_file.txt (
    SET /p OV_SETUP_SCRIPT= < %DAILY_ROOT%\openvino_nightly\latest_ov_setup_file.txt
)

:: don't set proxy for TOE
SET http_proxy=http://proxy-dmz.intel.com:912
SET https_proxy=http://proxy-dmz.intel.com:912

SET /a "NPROC=%NUMBER_OF_PROCESSORS%*3/4"
