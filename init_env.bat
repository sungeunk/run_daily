@echo off && setlocal

:: include user definitions
if exist user_definitions_%COMPUTERNAME%.bat (
    call user_definitions_%COMPUTERNAME%.bat
) else (
    call user_definitions_default.bat
)


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

pip install -U --pre -r requirements.txt
pip install -U --pre --no-deps --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino-tokenizers

call conda deactivate


:: vcpkg for opencl
cd vcpkg
bootstrap-vcpkg.bat
vcpkg.exe install opencl
cd ..

:end_script
