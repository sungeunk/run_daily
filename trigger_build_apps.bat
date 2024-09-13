@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

SET BUILD_BENCHMARK=1
SET BUILD_CHATGLM=1
SET BUILD_QWEN=1
SET BUILD_SD=1
SET BUILD_LCM=1
SET BUILD_CHAT_SAMPLE=1

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
if /I "%1" == "-sd" set BUILD_SD=%2 & shift
if /I "%1" == "-lcm" set BUILD_LCM=%2 & shift
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


:: build benchmark_app
if %BUILD_BENCHMARK% == 1 (
    pushd %INTEL_OPENVINO_DIR%\samples\cpp
    if exist build\ (
        rmdir /S /Q build
    )
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%VCPKG_PRIVATE_ROOT%\scripts\buildsystems\vcpkg.cmake
    ninja -j %NPROC% -C build benchmark_app

    mkdir %BIN_DIR%\benchmark_app
    copy /Y build\intel64\benchmark_app.exe %BIN_DIR%\benchmark_app\
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

    mkdir %BIN_DIR%\chatglm
    copy /Y build\llm\chatglm_cpp\chatglm.exe %BIN_DIR%\chatglm\
    copy /Y build\thirdparty\openvino_contrib\modules\custom_operations\user_ie_extensions\user_ov_extensions.dll %BIN_DIR%\chatglm\
    copy /Y build\_deps\fast_tokenizer-src\third_party\lib\icudt70.dll %BIN_DIR%\chatglm\
    copy /Y build\_deps\fast_tokenizer-src\third_party\lib\icuuc70.dll %BIN_DIR%\chatglm\
    copy /Y build\_deps\fast_tokenizer-src\lib\core_tokenizers.dll %BIN_DIR%\chatglm\
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

    mkdir %BIN_DIR%\qwen
    copy /Y build\bin\main.exe %BIN_DIR%\qwen\
    popd
)

:: build stable_diffusion_1_5
if %BUILD_SD% == 1 (
    pushd openvino.genai.token\image_generation\stable_diffusion_1_5\cpp
    if exist build\ (
        rmdir /S /Q build
    )

    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build stable_diffusion

    mkdir %BIN_DIR%\sd
    copy /Y build\stable_diffusion.exe %BIN_DIR%\sd\
    popd
)

:: build lcm_dreamshaper_v7
if %BUILD_LCM% == 1 (
    pushd openvino.genai.lcm\image_generation\lcm_dreamshaper_v7\cpp
    if exist build\ (
        rmdir /S /Q build
    )

    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build lcm_dreamshaper

    mkdir %BIN_DIR%\lcm
    copy /Y build\lcm_dreamshaper.exe %BIN_DIR%\lcm\
    popd
)

:: build chat_sample
if %BUILD_CHAT_SAMPLE% == 1 (
    pushd openvino.genai
    if exist build\ (
        rmdir /S /Q build
    )

    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -j %NPROC% -C build chat_sample openvino_tokenizers

    mkdir %BIN_DIR%\chat_sample
    copy build\samples\cpp\chat_sample\chat_sample.exe %BIN_DIR%\chat_sample\
    copy build\openvino_genai\*dll %BIN_DIR%\chat_sample\
    copy /Y build\_deps\fast_tokenizer-src\lib\*dll %BIN_DIR%\chat_sample\
    copy /Y build\_deps\fast_tokenizer-src\third_party\lib\*dll %BIN_DIR%\chat_sample\
    popd
)
