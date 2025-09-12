@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)

git submodule update --init --recursive

:: python environment for daily
call conda create -n daily python=3.11 -y
call conda activate daily

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

pip install -r requirements.windows.txt

call conda deactivate


:: vcpkg for opencl
cd vcpkg
bootstrap-vcpkg.bat
vcpkg.exe install opencl
cd ..

:end_script
