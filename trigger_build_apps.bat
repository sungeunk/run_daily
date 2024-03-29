@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

SET BUILD_BENCHMARK=1
SET BUILD_CHATGLM=1
SET BUILD_QWEN=1

:: parsing arguments
goto GETOPTS

:Help
echo -h this help messages
echo -ov {ov setup file path}
exit /b 0

:GETOPTS
if /I "%1" == "-h" call :Help
if /I "%1" == "-ov" set OV_SETUP_SCRIPT=%2 & shift
if /I "%1" == "-bb" set BUILD_BENCHMARK=%2 & shift
if /I "%1" == "-bc" set BUILD_CHATGLM=%2 & shift
if /I "%1" == "-bq" set BUILD_QWEN=%2 & shift
shift
if not "%1" == "" goto GETOPTS

:: set environments
call %VC_ENV_FILE%
if "%VisualStudioVersion%"=="" (
    echo "could not find VC tools"
    exit /b 0
)

if not exist %OV_SETUP_SCRIPT% (
    echo "could not find %OV_SETUP_SCRIPT%"
    exit /b 0
)
echo call %OV_SETUP_SCRIPT%
call %OV_SETUP_SCRIPT%

if not exist %OCL_ROOT% (
    echo "could not find %OCL_ROOT%"
    exit /b 0
)
SET Path=%OCL_ROOT%\lib;%OCL_ROOT%\bin;%Path%

:: build benchmark_app
if %BUILD_BENCHMARK% == 1 (
    pushd %INTEL_OPENVINO_DIR%\samples\cpp
    if exist build\ (
        rmdir /S /Q build
    )
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build benchmark_app
    popd
)


:: build chatglm
if %BUILD_CHATGLM% == 1 (
    pushd openvino.genai.chatglm3
    if exist build\ (
        rmdir /S /Q build
    )
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build chatglm
    copy build\thirdparty\openvino_contrib\modules\custom_operations\user_ie_extensions\user_ov_extensions.dll build\llm\chatglm_cpp\
    copy build\_deps\fast_tokenizer-src\third_party\lib\icudt70.dll build\llm\chatglm_cpp\
    copy build\_deps\fast_tokenizer-src\third_party\lib\icuuc70.dll build\llm\chatglm_cpp\
    copy build\_deps\fast_tokenizer-src\lib\core_tokenizers.dll build\llm\chatglm_cpp\
    popd
)

:: build qwen
if %BUILD_QWEN% == 1 (
    pushd openvino.genai.qwen\llm\qwen_cpp
    if exist build\ (
        rmdir /S /Q build
    )
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build main
    popd
)
