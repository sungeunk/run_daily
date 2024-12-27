@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

SET BUILD_BENCHMARK=1
SET BUILD_QWEN=1
SET BUILD_SD=1
SET BUILD_LCM=1

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
if /I "%1" == "-bq" set BUILD_QWEN=%2 & shift
if /I "%1" == "-sd" set BUILD_SD=%2 & shift
if /I "%1" == "-lcm" set BUILD_LCM=%2 & shift
shift
if not "%1" == "" goto GETOPTS

:: set environments
if exist %VC_ENV_FILE_BUILDTOOLS% (
    call %VC_ENV_FILE_BUILDTOOLS%
) else (
    if exist %VC_ENV_FILE_COMMUNITY% (
        call %VC_ENV_FILE_COMMUNITY%
    ) else (
        if exist %VC_ENV_FILE_PRO% (
            call %VC_ENV_FILE_PRO%
        ) else (
            echo No Visual Studio compiler.
            exit /b 0
        )
    )
)

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

:: build qwen
if %BUILD_QWEN% == 1 (
    pushd cpp_sample\qwen_cpp
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
    pushd cpp_sample\image_generation\stable_diffusion_1_5\cpp
    if exist build\ (
        rmdir /S /Q build
    )

    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build stable_diffusion

    mkdir %BIN_DIR%\sd
    copy /Y build\stable_diffusion.exe %BIN_DIR%\sd\
    copy /Y build\_deps\tokenizers\src\*dll %BIN_DIR%\sd\
    popd
)

:: build lcm_dreamshaper_v7
if %BUILD_LCM% == 1 (
    pushd cpp_sample\image_generation\lcm_dreamshaper_v7\cpp
    if exist build\ (
        rmdir /S /Q build
    )

    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja -j %NPROC% -C build lcm_dreamshaper

    mkdir %BIN_DIR%\lcm
    copy /Y build\lcm_dreamshaper.exe %BIN_DIR%\lcm\
    copy /Y build\_deps\tokenizers\src\*dll %BIN_DIR%\lcm\
    popd
)
