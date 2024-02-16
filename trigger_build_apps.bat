@echo off && setlocal

:: include user definitions
call user_definitions_%COMPUTERNAME%.bat

:: set environments
call %VC_ENV_FILE%
if "%VisualStudioVersion%"=="" (
    echo "could not find VC tools"
    exit /b 0
)

call %OV_SETUP_SCRIPT%

:: build benchmark_app
pushd %INTEL_OPENVINO_DIR%\samples\cpp
if exist build\ (
    rmdir /S /Q build
)
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -j %NPROC% -C build benchmark_app
popd

:: build chatglm
pushd openvino.genai.chatglm3
if exist build\ (
    rmdir /S /Q build
)
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -j %NPROC% -C build
copy build\thirdparty\openvino_contrib\modules\custom_operations\user_ie_extensions\user_ov_extensions.dll build\llm\chatglm_cpp\
copy build\_deps\fast_tokenizer-src\third_party\lib\icudt70.dll build\llm\chatglm_cpp\
copy build\_deps\fast_tokenizer-src\third_party\lib\icuuc70.dll build\llm\chatglm_cpp\
copy build\_deps\fast_tokenizer-src\lib\core_tokenizers.dll build\llm\chatglm_cpp\
popd

:: build qwen
pushd openvino.genai.qwen\llm\qwen_cpp
if exist build\ (
    rmdir /S /Q build
)
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -j %NPROC% -C build
popd
