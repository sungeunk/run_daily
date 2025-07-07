@echo off
SETLOCAL enableDelayedExpansion
SETLOCAL ENABLEEXTENSIONS

set POSITIONAL_ARGS=%*

:: Define valid options
set "validOptions=openvino_benchmark_app openvino_model_zoo ov_llm_bench dml_llm_bench llm_bench ipex_llm_bmg asr computer_vision bert base_sd trt_llm_bench msft clpeak quantize golden microbench"

:: Check if -h argument is provided
if "%1"=="-h" (
    goto showHelp
)

:: Check if argument is provided
if "%1"== [] (
    echo No argument provided. Please provide argument to execute AI ML tests.
    echo.
    goto showHelp
)

:: Check if argument is within the valid option selection
set "isValid=0"
for %%o in (%validOptions%) do (
    if /i "%1"=="%%o" (
        set "isValid=1"
        goto :break
    )
)
:break

if "%isValid%"=="0" (
    echo Argument "%1" is not valid.
    echo.
    goto showHelp
)

:: ############### SETUP ENV VAR ###############
echo ############### SETUP ENV VAR ###############

:: Call the Python script and capture the output
for /f "delims=" %%i in ('python scripts/get_dns_suffix.py') do set "MATCHING_FILTER=%%i"

:: Check if the Python script returned a matching filter
if "%MATCHING_FILTER%"=="" (
    echo No matching DNS Suffix Search List entry found with ipconfig /all, system not executed within Intel internal network. Fallback to non-artifactory assets still WIP. 
    exit /b 1
)
echo Intel Geo: %MATCHING_FILTER%

:: Check if .env file exists
if not exist "%cd%\.env" (
    echo Please provide .env file containing artifactory credentials. Fallback to non-artifactory assets still WIP.
    exit /b 1
)

:: Set the appropriate password variable based on the matching filter
set PASSWORD_VAR=ARTIFACTORY_PASSWORD_%MATCHING_FILTER%

:: Read the password from the .env file
for /f "tokens=1,2 delims==" %%a in (.env) do (
	if %%a==ARTIFACTORY_USERNAME set %%a=%%b
    if %%a==%PASSWORD_VAR% set ARTIFACTORY_PASSWORD=%%b
	if %%a==HF_TOKEN set %%a=%%b
    if %%a==EXECUTE_MODE set %%a=%%b
)

:: If the specific password is not found, search for ARTIFACTORY_PASSWORD entry
if "%ARTIFACTORY_PASSWORD%"=="" (
    echo Specific password not found for %PASSWORD_VAR%. Falling back to search for ARTIFACTORY_PASSWORD entry. 
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if %%a==ARTIFACTORY_PASSWORD set ARTIFACTORY_PASSWORD=%%b
    )
)

:: Check if the ARTIFACTORY_USERNAME was set
if "%ARTIFACTORY_USERNAME%"=="" (
    echo No ARTIFACTORY_USERNAME entry found in .env file. Please provide .env file containing artifactory credentials. Fallback to non-artifactory assets still WIP.
    exit /b 1
) else (
	echo ARTIFACTORY_USERNAME configured
)

:: Check if the ARTIFACTORY_PASSWORD was set
if "%ARTIFACTORY_PASSWORD%"=="" (
    echo No ARTIFACTORY_PASSWORD entry found in .env file. Please provide .env file containing artifactory credentials. Fallback to non-artifactory assets still WIP.
    exit /b 1
) else (
	echo ARTIFACTORY_PASSWORD configured
)

:: Check if the HF_TOKEN was set
if "%HF_TOKEN%"=="" (
    echo No HF_TOKEN entry found in .env file. Please provide .env file containing artifactory credentials. Fallback to non-artifactory assets still WIP.
    exit /b 1
) else (
	echo HF_TOKEN configured
)

:: Check if Git is installed
git --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Git is not installed or not set up in the PATH.
    exit /b 1
) else (
    echo Git is installed.
    git --version
)

:: Check if Miniforge (Conda) is installed
call conda --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Miniforge is not installed or not set up in the PATH.
    exit /b 1
) else (
    echo Miniforge is installed.
    call conda --version
    for /f "tokens=*" %%i in ('conda info --base') do set CONDA_BASE=%%i
    set CONDA_ENVS_PATH=!CONDA_BASE!\envs
    echo !CONDA_ENVS_PATH!
)

:: Save platform family to env var
for /f "delims=" %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -File "get_platform.ps1"') do set PLATFORM=%%i
echo Detected platform: %PLATFORM%

:: Save driver information for PBI postprocessing
call powershell -NoProfile -ExecutionPolicy Bypass -File "get_driver.ps1"

:: Save system info for PBI postprocessing
call powershell -NoProfile -ExecutionPolicy Bypass -File "get_systeminfo.ps1"

:: Save WW info
for /f "delims=" %%i in ('python utils/ww_printer.py') do set "currentWW=%%i"
echo %currentWW%

:: Set general env config
set PYTHONIOENCODING=utf-8
git config --system core.longpaths true

:: ############### SETUP ENV VAR COMPLETE ###############
echo ############### SETUP ENV VAR COMPLETE ###############
echo.

if "%EXECUTE_MODE%"=="manual" (
    echo ############### SKIPPING ARGUMENTS VALIDATION ###############
    echo EXECUTE_MODE is set to manual inside .env file.
    echo.
)
if "%EXECUTE_MODE%"=="" (
    echo ############### VALIDATE ARGUMENTS ###############
    echo.

    python validation\validate.py %PLATFORM% %POSITIONAL_ARGS%

    if !ERRORLEVEL! NEQ 0 (
        echo Validation failed.
        echo.
        goto errorExit
    ) else (
        echo Validation passed.
        echo.
    )
)


echo ############### EXECUTING TEST ###############

if "%1"=="openvino_benchmark_app" (
    @REM Environment setup
    echo EXECUTING TEST: openvino_benchmark_app
    echo.
    call conda create -n openvino_benchmark_app python=3.10* -y
	call conda activate openvino_benchmark_app
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    python -m pip install --upgrade pip
    pip install openvino openvino-dev onnx torch torchvision tensorflow protobuf==3.20.3
    python -c "from openvino.runtime import Core; print(Core().available_devices)"

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

	call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ computer_vision %2 --dest-dir temp\omz_models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    copy uncategorized\openvino_benchmark_app\benchmark.py temp /y
    xcopy /y utils\ temp\utils_aiml_intel\
    cd temp

    @REM Execute test
    python benchmark.py --model %2 --hint %3 --iteration %4

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="openvino_model_zoo" (
    @REM Environment setup
    echo EXECUTING TEST: openvino_model_zoo
    echo.
    call conda create -n openvino_model_zoo python=3.10* -y
	call conda activate openvino_model_zoo
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    git clone https://github.com/openvinotoolkit/open_model_zoo.git temp
    copy uncategorized\openvino_model_zoo\benchmark.py temp\demos\gpt2_text_prediction_demo\python
    
    python -m pip install --upgrade pip
    pip install openvino openvino-dev onnx torch tokenizers
    python -c "from openvino.runtime import Core; print(Core().available_devices)"

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml
    
    if "%2"=="gpt2" (
        xcopy /y utils\ temp\demos\gpt2_text_prediction_demo\python\utils_aiml_intel\
        cd temp\demos\gpt2_text_prediction_demo\python
        omz_downloader --list models.lst
        omz_converter --list models.lst
    )

    @REM Execute test
    python benchmark.py --model %2 --num_iter %3

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="ov_llm_bench" (
    @REM Environment setup
    echo EXECUTING TEST: ov_llm_bench
    echo.
    call conda activate ov_llm_bench
    @REM if !ERRORLEVEL! NEQ 0 (
    call conda create -n ov_llm_bench python=3.10* -y
    call conda activate ov_llm_bench
    @REM ) else (
    @REM     echo Conda environment exists, skipping environment creation.
    @REM )
    
	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if not exist "temp" (
        git clone --depth 1 --branch 2025.1.0.0 https://github.com/openvinotoolkit/openvino.genai.git temp
    ) else (
        echo Temp directory exists, skipping temp folder creation.
    )

    pip install update --upgrade
    pip install -r temp\tools\llm_bench\requirements.txt

    if "%7"=="nightly" (
        @REM pip install openvino-genai==2025.1.0rc1 openvino-tokenizers==2025.1.0rc1 openvino==2025.1.0rc1 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release 
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    ) else (
        pip uninstall openvino openvino_tokenizers openvino-genai openvino_telemetry optimum optimum-intel -y
        pip install openvino openvino_tokenizers openvino-genai openvino_telemetry optimum optimum-intel
    )

    pip install numpy==1.26.4
    pip install sentencepiece
    pip install pyyaml

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

    call var_retrival.bat %2
    echo HF_name: !hf_name!
    echo sub_model: !sub_model!

    if not exist temp\llm_bench\python\models\%2 (
        if "%7"=="nightly" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/openvino-nightly/llm %2 --dest-dir temp\tools\llm_bench\models\%2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        ) else (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/openvino/llm %2 --dest-dir temp\tools\llm_bench\models\%2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        )
    ) else (
        echo Skipping model download since model directory exists.
    )

    copy /y scripts\large_language_model\openvino\benchmark.py temp\tools\llm_bench
    @REM TODO: Double check need of this 
    xcopy /E /I /Y scripts\large_language_model\prompts\openvino\ temp\tools\llm_bench\prompts  
    xcopy /y utils\ temp\tools\llm_bench\utils_aiml_intel\
    cd temp\tools\who_what_benchmark

    pip install .
    cd ../../..

    @REM Get goldel asset
    call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/openvino/llm %2 --dest-dir temp\tools\llm_bench\golden\%2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    
    cd temp\tools\llm_bench
    @REM wwb --base-model %hf_name% --gt-data gt.csv --model-type text --hf
    wwb --target-model models\%2 --gt-data golden\%2\gt.csv --model-type text --genai --device %5 --output wwb_generated

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM Execute test
    @REM Modify logging to append from wwb so logged files are cumulative
    python benchmark.py -m models\%2 --prompt_file "prompts\%2\%3.jsonl" --infer_count %4 --device %5 --num_iters %6 -rj results-INT4.json --genai

    
    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="dml_llm_bench" (
    @REM Environment setup
    echo EXECUTING TEST: dml_llm_bench
    echo.
    call conda create -n dml_llm_bench python=3.10* -y
	call conda activate dml_llm_bench
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs
    
    pip install onnxruntime-genai-directml==0.5.2 pandas psutil tqdm
    pip install pyyaml

    @REM choices [
    @REM     llama_v2
    @REM     llama_v3
    @REM     llama_v3.2-1B-instruct
    @REM     llama_v3.2-3B-instruct
    @REM     phi_v2
    @REM     phi_v3
    @REM     phi_v3.5
    @REM     mistral_v0.1
    @REM     mistral_v0.3
    @REM     qwen_v2
    @REM     gemma_v1
    @REM ]

    if not exist temp\%2 (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models %2-genai --dest-dir temp\%2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    ) else (
        echo Skipping model download since model directory exists.
    )


    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml
    copy scripts\large_language_model\directml\* temp
    xcopy /E /I /Y scripts\large_language_model\prompts\directml\ temp\prompts
    xcopy /y utils\ temp\utils_aiml_intel\
    cd temp

    @REM Execute test
    python run_llm_io_binding.py -i %2 -l %3 -g %4 -w 2 -r 8 -mo

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="llm_bench" (
    @REM Environment setup
    echo EXECUTING TEST: llm_bench
    echo.
	call conda activate llm_bench
	@REM if !ERRORLEVEL! NEQ 0 (
	@REM 	call conda create -n llm_bench python=3.10* -y
	@REM 	call conda activate llm_bench
	@REM 	call conda install libuv -y
		
	@REM 	if "%2"=="ipex" (
    @REM         pip install ipex-llm[xpu] --pre --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --proxy=proxy-dmz.intel.com:912
	@REM 		pip install -r scripts\large_language_model\pytorch\requirements_ipex.txt
	@REM 	)    
	@REM 	if "%2"=="cuda" (
	@REM 		pip install -r scripts\large_language_model\pytorch\requirements_cuda.txt
	@REM 	)
	@REM ) else (
	@REM 	echo Conda environment exists, skipping environment creation.
	@REM )

    call conda create -n llm_bench python=3.10* -y
    call conda activate llm_bench
    call conda install libuv -y

    if "%2"=="ipex" (
        pip install ipex-llm[xpu] --pre --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --proxy=proxy-dmz.intel.com:912
        pip install -r scripts\large_language_model\pytorch\requirements_ipex.txt
    )    
    if "%2"=="cuda" (
        pip install -r scripts\large_language_model\pytorch\requirements_cuda.txt
    )
	
	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs
	
	if not exist "temp" (
		git clone https://github.com/intel-analytics/ipex-llm.git temp
		cd temp
		git checkout 493cbd9a3642287089a979f3fce7f81f8d07e64a
		cd ..
	) else (
		echo Temp directory exists, skipping temp folder creation.
	)

    @REM ipex_llm benchmark script requirements
    pip install transformers==4.38.0

    if "%3"=="llava_v1.5" (
        git clone -b v1.1.1 --depth=1 https://github.com/haotian-liu/LLaVA
        set LLAVA_REPO_DIR=%cd%\LLaVA
        pip install einops
        pip install transformers==4.31.0
    )
    if "%2"=="ipex" (
        set SYCL_CACHE_PERSISTENT=1
        set IPEX_LLM_LOW_MEM=1
    )


    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml
    
    if "%3"=="llama-2-7b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Llama-2-7b-chat-hf" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v2 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Llama-2-7b-chat-hf -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v2 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Llama-2-7b-chat-hf -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "meta-llama/Llama-2-7b-chat-hf" --input_token %4 --output_token %5
    )
    if "%3"=="llama-3-8b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Meta-Llama-3-8B" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Meta-Llama-3-8B -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Meta-Llama-3-8B -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "meta-llama/Meta-Llama-3-8B" --input_token %4 --output_token %5
    )
    if "%3"=="qwen-1.0-7b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Qwen-7B-Chat" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.0 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen-7B-Chat -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.0 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen-7B-Chat -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "Qwen/Qwen-7B-Chat" --input_token %4 --output_token %5
    )
    if "%3"=="qwen-1.5-7b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Qwen1.5-7B-Chat" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.5 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen1.5-7B-Chat -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.5 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen1.5-7B-Chat -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "Qwen/Qwen1.5-7B-Chat" --input_token %4 --output_token %5
    )
    if "%3"=="qwen-2.0-7b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Qwen2-7B-Instruct" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v2.0_7b --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen2-7B-Instruct -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v2.0_7b --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Qwen2-7B-Instruct -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "Qwen/Qwen2-7B-Instruct" --input_token %4 --output_token %5
    )
    if "%3"=="phi-2" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\phi-2" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_2 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\phi-2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_2 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\phi-2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "microsoft/phi-2" --input_token %4 --output_token %5
    )
    if "%3"=="phi-3" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Phi-3-mini-128k-instruct" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Phi-3-mini-128k-instruct -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Phi-3-mini-128k-instruct -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "microsoft/Phi-3-mini-128k-instruct" --input_token %4 --output_token %5
    )
    if "%3"=="mistral-7b-v0.1" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\Mistral-7B-v0.1" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_mistral_v0.1 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Mistral-7B-v0.1 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_mistral_v0.1 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\Mistral-7B-v0.1 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "mistralai/Mistral-7B-v0.1" --input_token %4 --output_token %5
    )
    if "%3"=="llava_v1.5" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\llava-v1.5-7b" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llava_v1.5 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\llava-v1.5-7b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llava_v1.5 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\llava-v1.5-7b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "liuhaotian/llava-v1.5-7b" --input_token %4 --output_token %5
    )
    if "%3"=="gemma" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\gemma-7b" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_gemma --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\gemma-7b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
				call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_gemma --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\gemma-7b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
		) else (
            echo Skipping model download since model directory exists.
        )
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "google/gemma-7b" --input_token %4 --output_token %5
    )
    if "%3"=="chatglm3-6b" (
        if not exist "temp\python\llm\dev\benchmark\all-in-one\hub\chatglm3-6b" (
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_chatglm3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\chatglm3-6b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			if !ERRORLEVEL! NEQ 0 (
				rmdir "C:\Users\%USERNAME%\.cache" /s /q
                call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_chatglm3 --dest-dir temp\python\llm\dev\benchmark\all-in-one\hub\chatglm3-6b -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
			)
        ) else (
            echo Skipping model download since model directory exists.
        )
        pip install transformers==4.37.2
        python scripts\large_language_model\pytorch\configurator.py --api %2 --model "THUDM/chatglm3-6b" --input_token %4 --output_token %5
    )
	
    copy /y temp\python\llm\dev\benchmark\all-in-one\config.yaml logs
    copy /y scripts\large_language_model\pytorch\run.py temp\python\llm\dev\benchmark\all-in-one
    xcopy /y utils\ temp\python\llm\dev\benchmark\all-in-one\utils_aiml_intel\
    cd temp\python\llm\dev\benchmark\all-in-one

    @REM Execute test
    python run.py
    if NOT "%6"=="no_delete" (
        rmdir hub /s /q
    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="ipex_llm_bmg" (
    echo EXECUTING TEST: ipex_llm_bmg
    echo.
    call conda create -n ipex_llm_bmg python=3.10* -y
	call conda activate ipex_llm_bmg
    call conda install pkg-config libuv -y
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs


    call var_retrival.bat %2
    echo HF_name: !hf_name!
    echo sub_model: !sub_model!


    @REM ipex internal repo
    call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ipex_internal 3_18_2025 --dest-dir temp -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory

    @REM model download
    @REM call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file models %2 --dest-dir temp/frameworks.ai.pytorch.ipex-gpu-master/examples/gpu/llm/inference/models/%2 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory

    echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
    if %PLATFORM%==BMG (
        python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
        python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
    )
    if %PLATFORM%==ACM (
        python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
        python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
    )
    if %PLATFORM%==ARL (
        python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
        python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
    )
    if %PLATFORM%==LNL (
        python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
        python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
    )
    if %PLATFORM%==MTL (
        python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
        python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
    )
    if %PLATFORM%==PTL (
        :: Require custom steps for IPEX PTL build
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
        pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
        pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
        call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
    )

    @REM transformers
    call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file transformers 05_06_2025 --dest-dir temp -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory

    pip install numpy==1.26.4
    pip install "temp\transformers-4.48.3-py3-none-any.whl"
    pip install neural-compressor==3.3
    pip install datasets==3.4.1 accelerate==1.5.2 einops==0.8.1 diffusers==0.32.2 optimum==1.24.0 pydantic==2.10.6 numba tbb
    python -m pip install git+https://github.com/huggingface/optimum-intel.git@8fa4ebd9fb45349b16334a38409a4cfb40c2bc68
    pip install pyyaml


    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

    copy /y scripts\large_language_model\ipex\run_generation_woq.py temp\frameworks.ai.pytorch.ipex-gpu-master\examples\gpu\llm\inference
    xcopy /E /I /Y scripts\large_language_model\prompts\ipex\ temp\frameworks.ai.pytorch.ipex-gpu-master\examples\gpu\llm\inference\prompts
    xcopy /y utils\ temp\frameworks.ai.pytorch.ipex-gpu-master\examples\gpu\llm\inference\utils_aiml_intel\
    cd temp\frameworks.ai.pytorch.ipex-gpu-master\examples\gpu\llm\inference

    set IPEX_COMPUTE_ENG=0
    set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

    if "%2"=="chatglm_v3.0_6b" (
        python -u run_generation_woq.py --benchmark -m !hf_name! --sub-model-name !sub_model! --num-beams 1 --num-iter 4 --num-warmup 1 --batch-size 1 --input-tokens %3 --max-new-tokens %4 --device xpu --ipex --dtype float16 --token-latency
    )
    if "%2"=="minicpm_v1.0_1b" (
        python -u run_generation_woq.py --benchmark -m !hf_name! --sub-model-name !sub_model! --num-beams 1 --num-iter 4 --num-warmup 1 --batch-size 1 --input-tokens %3 --max-new-tokens %4 --device xpu --ipex --dtype float16 --token-latency
    )
    if "%2" NEQ "chatglm_v3.0_6b" (
        python -u run_generation_woq.py --benchmark -m !hf_name! --sub-model-name !sub_model! --num-beams 1 --num-iter 4 --num-warmup 1 --batch-size 1 --input-tokens %3 --max-new-tokens %4 --device xpu --ipex --dtype float16 --token-latency --use-static-cache --use-hf-code
    )
    if "%2" NEQ "minicpm_v1.0_1b" (
        python -u run_generation_woq.py --benchmark -m !hf_name! --sub-model-name !sub_model! --num-beams 1 --num-iter 4 --num-warmup 1 --batch-size 1 --input-tokens %3 --max-new-tokens %4 --device xpu --ipex --dtype float16 --token-latency --use-static-cache --use-hf-code
    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="asr" (
    echo EXECUTING TEST: asr
    echo.
    call conda create -n asr python=3.10* -y
	call conda activate asr
    call conda install libuv -y
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if "%2"=="ipex" (
        call conda install pkg-config -y
        @REM call gta-asset.exe pull gfx-e2eval-sw-fm\gfx-e2eval-sw-fm\AI_ML_Automation\misc\ipex private_drops rc3 --dest-dir temp\ipex_drops -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        @REM cd temp\ipex_drops
        @REM pip install "torch-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchaudio-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchvision-0.20.1+cxx11.abi-cp310-cp310-win_amd64.whl" "intel_extension_for_pytorch-2.5.10+xpu-cp310-cp310-win_amd64.whl"
        @REM cd ..\..
        echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
        if %PLATFORM%==BMG (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ACM (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ARL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==LNL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==MTL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==PTL (
            :: Require custom steps for IPEX PTL build
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
            pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
            pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
            call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
        )

        pip install transformers
		pip install setuptools==69.5.1
    )
    if "%2"=="cuda" (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if "%2"=="openvino" (
        pip install openvino openvino-tokenizers openvino-genai
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install torch
    )
    if "%2"=="openvino-nightly" (
        pip uninstall openvino openvino_tokenizers openvino_telemetry -y
        @REM pip install openvino-genai==2025.1.0rc1 openvino-tokenizers==2025.1.0rc1 openvino==2025.1.0rc1 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install torch
    )

    pip install librosa datasets accelerate transformers funasr torchaudio
    pip install numpy==1.26.4
    pip install pyyaml

    @REM download asset
    call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/assets/ audio librispeech_long --dest-dir temp/ -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory

    if "%3"=="paraformer_zh" (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/automatic_speech_recognition openvino paraformer_zh --dest-dir temp/openvino -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    )

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

    copy /y scripts\automatic_speech_recognition\ temp\
    xcopy /y utils\ temp\utils_aiml_intel\
    cd temp

    @REM Execute test
    python asr_benchmark.py --api %2 --model %3

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="bert" (
    echo EXECUTING TEST: bert
    echo.
    call conda create -n bert_env python=3.10* -y
	call conda activate bert_env
    call conda install libuv -y
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if "%2"=="openvino" (
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    )
    if "%2"=="ipex" (
        call conda install pkg-config -y
        @REM call gta-asset.exe pull gfx-e2eval-sw-fm\gfx-e2eval-sw-fm\AI_ML_Automation\misc\ipex private_drops rc3 --dest-dir temp\ipex_drops -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        @REM cd temp\ipex_drops
        @REM pip install "torch-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchaudio-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchvision-0.20.1+cxx11.abi-cp310-cp310-win_amd64.whl" "intel_extension_for_pytorch-2.5.10+xpu-cp310-cp310-win_amd64.whl"
        @REM cd ..\..
        echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
        if %PLATFORM%==BMG (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ACM (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ARL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==LNL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==MTL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==PTL (
            :: Require custom steps for IPEX PTL build
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
            pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
            pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
            call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
        )

        pip install transformers
		pip install setuptools==69.5.1
    )
    if "%2"=="cuda" (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if "%4"=="nightly" (
        @REM pip install openvino-genai==2025.1.0rc1 openvino-tokenizers==2025.1.0rc1 openvino==2025.1.0rc1 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    )
    
    pip install numpy==1.26.4
    pip install pyyaml

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

    copy /y scripts\large_language_model\bert\bert_sample.py temp\
    xcopy /y utils\ temp\utils_aiml_intel\
    cd temp

    @REM Execute test
    python bert_sample.py --model bert --input_token %3 --api %2

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="computer_vision" (
    echo EXECUTING TEST: computer_vision
    echo.
    call conda create -n computer_vision python=3.10* -y
	call conda activate computer_vision
    call conda install libuv -y
    mkdir temp
    mkdir logs
    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if "%3"=="openvino" (
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
        pip install onnx onnxruntime
        pip install --upgrade timm
    )
    if "%3"=="ipex" (
        call conda install pkg-config -y
        @REM call gta-asset.exe pull gfx-e2eval-sw-fm\gfx-e2eval-sw-fm\AI_ML_Automation\misc\ipex private_drops rc3 --dest-dir temp\ipex_drops -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        @REM cd temp\ipex_drops
        @REM pip install "torch-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchaudio-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchvision-0.20.1+cxx11.abi-cp310-cp310-win_amd64.whl" "intel_extension_for_pytorch-2.5.10+xpu-cp310-cp310-win_amd64.whl"
        @REM cd ..\..
        echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
        if %PLATFORM%==BMG (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ACM (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ARL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==LNL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==MTL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==PTL (
            :: Require custom steps for IPEX PTL build
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
            pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
            pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
            call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
        )

		pip install setuptools==69.5.1
    )
    if "%3"=="cuda" (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if "%3"=="directml" (
        pip install optimum[exporters]
        pip install onnxruntime-directml
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    )
    if "%3"=="openvino-nightly" (
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip uninstall openvino openvino_tokenizers openvino_telemetry -y
        @REM pip install openvino-genai==2025.1.0rc1 openvino-tokenizers==2025.1.0rc1 openvino==2025.1.0rc1 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
        pip install onnx onnxruntime 
    )

    pip install datasets accelerate super_image diffusers transformers pillow
    pip install numpy==1.26.4
    pip install pyyaml

    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml

    if "%2"=="resnet50" (
        copy /y scripts\computer_vision\resnet50_benchmark.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp
        call optimum-cli export onnx --model microsoft/resnet-50 onnx

        @REM Execute test
        python resnet50_benchmark.py --model resnet50 --api %3 --num_iter %4
    )
    if "%2"=="openpose" (
        pip install controlnet_aux
        copy /y scripts\computer_vision\openpose_benchmark.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python openpose_benchmark.py --model openpose --api %3 --num_iter %4
    )
    if "%2"=="edsr" (
        pip install huggingface_hub==0.25.2
        copy /y scripts\computer_vision\edsr_benchmark.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python edsr_benchmark.py --model edsr --api %3 --upscale %4
    )
    if "%2"=="clip" (
        copy /y scripts\computer_vision\clip_benchmark.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python clip_benchmark.py --model clip --api %3 --num_iter %4
    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )

)
if "%1"=="base_sd" (
    echo EXECUTING TEST: base_sd
    echo.
    echo Force deleting possible %CONDA_ENVS_PATH%\base_sd due to CondaValueError
    rmdir /S /Q %CONDA_ENVS_PATH%\base_sd
    call conda create -n base_sd python=3.10* -y
	call conda activate base_sd
    call conda install libuv -y

	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
        echo Purging contents inside logs directory to prevent duplicate logging in automation
        del /q /f logs\*
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if not exist "temp" (
        mkdir temp
    ) else (
        echo Temp directory exists, skipping temp folder creation.
    )

    pip install pyyaml sentencepiece

    if "%2"=="openvino" (
        pip install -r scripts\stable_diffusion\requirements_openvino.txt
        @REM pip install torch==2.3.1
        @REM pip install openvino
        @REM pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        @REM if "%3"=="xl" (
        @REM     pip install optimum==1.21.4 optimum-intel==1.18.3 transformers==4.43.4
        @REM )
    )
    if "%2"=="openvino-nightly" (
        pip install optimum[openvino]
        pip uninstall openvino openvino_tokenizers openvino_telemetry -y
        @REM pip install openvino-genai==2025.1.0rc1 openvino-tokenizers==2025.1.0rc1 openvino==2025.1.0rc1 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install diffusers transformers accelerate numpy==1.26.4
        @REM pip install torch==2.3.1
        @REM if "%3"=="xl" (
        @REM     pip install optimum==1.21.4 optimum-intel==1.18.3 transformers==4.43.4
        @REM )
    )
    if "%2"=="ipex" (
        call conda install pkg-config -y
        @REM call gta-asset.exe pull gfx-e2eval-sw-fm\gfx-e2eval-sw-fm\AI_ML_Automation\misc\ipex private_drops rc3 --dest-dir temp\ipex_drops -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        @REM cd temp\ipex_drops
        @REM pip install "torch-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchaudio-2.5.1+cxx11.abi-cp310-cp310-win_amd64.whl" "torchvision-0.20.1+cxx11.abi-cp310-cp310-win_amd64.whl" "intel_extension_for_pytorch-2.5.10+xpu-cp310-cp310-win_amd64.whl"
        @REM cd ..\..
        echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
        if %PLATFORM%==BMG (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ACM (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ARL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==LNL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==MTL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==PTL (
            :: Require custom steps for IPEX PTL build
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
            pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
            pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
            call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
        )

        pip install transformers diffusers accelerate sentencepiece protobuf
		pip install setuptools==69.5.1
        @REM pip install -r scripts\stable_diffusion\requirements_ipex.txt
        @REM pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0
        @REM pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    )
    if "%2"=="cuda" (
        pip install -r scripts\stable_diffusion\requirements_cuda.txt
        @REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    if "%2"=="directml" (
        ECHO Executing DirectML
        if not exist "temp" (
            git clone https://github.com/microsoft/Olive.git temp --branch v0.5.0
        ) else (
            echo Temp directory exists, deleting temp folder.
            rd /s /q temp
            git clone https://github.com/microsoft/Olive.git temp --branch v0.5.0
        )

        pip install olive-ai==0.5.0

        if "%3"=="v1.5" (
            pip install -r temp\examples\stable_diffusion\requirements.txt
        )
        if "%3"=="v2.0" (
            pip install -r temp\examples\stable_diffusion\requirements.txt
        )
        if "%3"=="v2.1" (
            pip install -r temp\examples\stable_diffusion\requirements.txt
            @REM pip install transformers==4.42.4
        )
        if "%3"=="xl" (
            pip install olive-ai==0.4.0
            pip install -r temp\examples\directml\stable_diffusion_xl\requirements.txt
            pip install tf-keras
            pip install optimum==1.22.0
        )
        pip install numpy==1.26.4
        pip install torch==2.3.1
        pip install opencv-python
        pip install --upgrade transformers

        @REM Export conda environment
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml
        
        if "%3"=="v1.5" (
            set model="runwayml/stable-diffusion-v1-5"
            set inference_steps=20
            mkdir temp\examples\stable_diffusion\models
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion directml_v1.5 --dest-dir temp\examples\stable_diffusion\models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            copy scripts\stable_diffusion\directml\stable_diffusion.py temp\examples\stable_diffusion\stable_diffusion.py /y
            copy scripts\stable_diffusion\directml\ort_optimization_util.py temp\examples\stable_diffusion\ort_optimization_util.py /y
            xcopy /y utils\ temp\examples\stable_diffusion\utils_aiml_intel\
            cd temp\examples\stable_diffusion
        )
        if "%3"=="v2.0" (
            set model="stabilityai/stable-diffusion-2"
            set inference_steps=20
            mkdir temp\examples\stable_diffusion\models
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion directml_v2.0 --dest-dir temp\examples\stable_diffusion\models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            copy scripts\stable_diffusion\directml\stable_diffusion.py temp\examples\stable_diffusion\stable_diffusion.py /y
            copy scripts\stable_diffusion\directml\ort_optimization_util.py temp\examples\stable_diffusion\ort_optimization_util.py /y
            xcopy /y utils\ temp\examples\stable_diffusion\utils_aiml_intel\
            cd temp\examples\stable_diffusion
        )
        if "%3"=="v2.1" (
            set model="stabilityai/stable-diffusion-2-1"
            set inference_steps=20
            mkdir temp\examples\stable_diffusion\models
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion directml_v2.1 --dest-dir temp\examples\stable_diffusion\models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            copy scripts\stable_diffusion\directml\stable_diffusion.py temp\examples\stable_diffusion\stable_diffusion.py /y
            copy scripts\stable_diffusion\directml\ort_optimization_util.py temp\examples\stable_diffusion\ort_optimization_util.py /y
            xcopy /y utils\ temp\examples\stable_diffusion\utils_aiml_intel\
            cd temp\examples\stable_diffusion
        )
        if "%3"=="xl" (
            set model="stabilityai/stable-diffusion-xl-base-1.0"
            set inference_steps=20
            mkdir temp\examples\directml\stable_diffusion_xl\models
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion directml_xl --dest-dir temp\examples\directml\stable_diffusion_xl\models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            copy scripts\stable_diffusion\directml\stable_diffusion_xl.py temp\examples\directml\stable_diffusion_xl\stable_diffusion.py /y
            xcopy /y utils\ temp\examples\directml\stable_diffusion_xl\utils_aiml_intel\
            cd temp\examples\directml\stable_diffusion_xl
        )
        
        @REM Execute test
        python stable_diffusion.py --model_id !model! --width %4 --height %5 --num_inference_steps !inference_steps! --num_images %6 --batch_size 1
    )

    if "%2" NEQ "directml" (
        @REM Fix version for optimum regression 10/10
        pip install numpy==1.26.4
        pip install peft pandas opencv-python
        
        @REM Export conda environment
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml

        copy /y scripts\stable_diffusion\base_sd.py temp\
        copy /y scripts\stable_diffusion\sd_hijack_utils.py temp\
        copy /y scripts\stable_diffusion\xpu_specific.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python base_sd.py --api %2 --model %3 --height %4 --width %5 --num_iter %6
    )


    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="trt_llm_bench" (
    echo EXECUTING TEST: trt_llm_bench
    echo.
    git clone https://github.com/NVIDIA/TensorRT-LLM.git --depth 1 --branch v0.11.0
    powershell ./setup_env.ps1 -skipPython
    call conda create -n trt_llm_bench python=3.10* -y
	call conda activate trt_llm_bench
    call conda install pyarrow -y
    pip uninstall -y tensorrt tensorrt_libs tensorrt_bindings
    pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
    pip install tensorrt_llm==0.12.0 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121/torch/
    python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)

if "%1" == "msft" (
    echo EXECUTING TEST: msft
    echo.
    @REM Results will be in custom folders created with the format Model_Perf_results_GPU_<date>_<time>\ as a .csv
    echo Creating and preparing testing environment...
    conda create -n msft python=3.9* -y
	conda activate msft
    mkdir temp
    mkdir logs

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    echo Attempting to download assets from artifactory...
    @REM Download assets from artifactory
    if not exist "%~dp0\temp\benchmark_repo" (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc msft benchmark_repo --dest-dir temp\benchmark_repo -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    ) else (
        echo benchmark_repo Asset is already exists.Skipping download.
    )
    if not exist "%~dp0\temp\models" (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc msft models --dest-dir temp\models -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    ) else (
        echo models Asset is already exists.Skipping download.
    )
    if not exist "C:\OpenVINO_Custom" (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc msft ov_custom --dest-dir C:\OpenVINO_Custom -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    ) else (
        echo C:\OpenVINO_Custom Asset is already exists.Skipping download.
    )
    
    echo Starting %2 testing...
	if "%2" == "native_ov" (
		cd %~dp0temp\benchmark_repo\OV_Full_Stack_Tests
		setup_ov_env.bat C:\OpenVINO_Custom
		cd Performance
		call run_perf.bat %~dp0temp\models %~dp0csv\model_list.csv L NO TIME 60 F32
        cd %~dp0
		call Powershell.exe -ExecutionPolicy RemoteSigned -File .\scripts\copy_logs.ps1 %~dp0temp\benchmark_repo\OV_Full_Stack_Tests\Performance\Perf_Analysis_Outputs
	)

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)

if "%1" == "clpeak" (
    echo EXECUTING TEST: clpeak
    echo.
    call conda create -n clpeak_bench python=3.10* -y 
    call conda activate clpeak_bench 
    call pip install pandas
    
    mkdir temp
    mkdir logs

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs
    
    echo Attempting to download assets from artifactory...
    @REM Download assets from artifactory
    if not exist "%~dp0\temp\clpeak" (
        call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc clpeak --dest-dir temp\clpeak -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    ) else (
        echo clpeak Asset is already exists.Skipping download.
    )
    @REM Export conda environment
    call conda env export > logs\environment.txt
    call conda env export > logs\environment.yml
    
    copy /y scripts\clpeak_bench.py temp\
    xcopy /y utils\ temp\utils_aiml_intel\
    cd temp
    
    python clpeak_bench.py %2
    
    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )
    
    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)

if "%1"=="quantize" (
    echo EXECUTING TEST: quantize
    echo.

    :: Check if asset exists
    call gta-asset.exe size gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/%2/%3 %4 %currentWW% -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
	@REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        echo Golden asset already exists in gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/%2/%3 %4 %currentWW%
        echo Deleting asset to re-upload
        call gta-asset.exe delete gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/%2/%3 %4 %currentWW% -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        echo Asset deleted
    )

    echo Force deleting possible %CONDA_ENVS_PATH%\quantize due to CondaValueError
    rmdir /S /Q %CONDA_ENVS_PATH%\quantize
    call conda create -n quantize python=3.10* -y
	call conda activate quantize

	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
        echo Purging contents inside logs directory to prevent duplicate logging in automation
        del /q /f logs\*
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    call var_retrival.bat %4
    echo HF_name: !hf_name!
    echo sub_model: !sub_model!

    pip install pyyaml python-dateutil pandas

    if "%2"=="openvino" (
        if not exist "temp" (
            git clone --depth 1 --branch 2025.1.0.0 https://github.com/openvinotoolkit/openvino.genai.git temp
        ) else (
            echo Temp directory exists, skipping temp folder creation.
        )
        cd temp\tools\llm_bench
        pip install update --upgrade
        pip install -r requirements.txt
        pip uninstall openvino openvino_tokenizers openvino-genai openvino_telemetry optimum optimum-intel -y
        pip install openvino openvino_tokenizers openvino-genai openvino_telemetry optimum optimum-intel

        :: Quantize models
        optimum-cli export openvino -m !hf_name! --weight-format int4 --ratio 1.0 --sym --group-size 128 --trust-remote-code --task !task! %4
        @REM Raise non-zero error if hit
        if !ERRORLEVEL! NEQ 0 (
            echo Encountered error during test execution
            goto errorExit
        )

        cd ../../..
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml
        call conda env export > temp\tools\llm_bench\%4\environment.txt
        call gta-asset.exe push gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/%2/%3 %4 %currentWW% temp\tools\llm_bench\%4 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        del /q /f temp\tools\llm_bench\%4
        del /q /f C:\Users\%USERNAME%\.cache\huggingface\*
    )

    if "%2"=="openvino-nightly" (
        if not exist "temp" (
            git clone https://github.com/openvinotoolkit/openvino.genai.git temp
        ) else (
            echo Temp directory exists, skipping temp folder creation.
        )
        cd temp\tools\llm_bench
        pip install update --upgrade
        pip install -r requirements.txt
        pip uninstall openvino openvino_tokenizers openvino-genai openvino_telemetry -y
        pip install --upgrade --pre openvino-genai==2025.2.0.dev20250515 openvino-tokenizers==2025.2.0.dev20250515 openvino==2025.2.0.dev20250515 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

        :: Quantize models
        optimum-cli export openvino -m !hf_name! --weight-format int4 --ratio 1.0 --sym --group-size 128 --trust-remote-code --task !task! %4
        @REM Raise non-zero error if hit
        if !ERRORLEVEL! NEQ 0 (
            echo Encountered error during test execution
            goto errorExit
        )

        cd ../../..
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml
        call conda env export > temp\tools\llm_bench\%4\environment.txt
        call gta-asset.exe push gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/%2/%3 %4 %currentWW% temp\tools\llm_bench\%4 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        del /q /f temp\tools\llm_bench\%4
        del /q /f C:\Users\%USERNAME%\.cache\huggingface\*
    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="golden" (
    echo EXECUTING TEST: golden
    echo.

    :: Check if asset exists
    call gta-asset.exe size gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/%2/%3 %4 %currentWW% -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        echo Golden asset already exists in gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/%2/%3/%4/%currentWW%
        echo Deleting asset to re-upload
        call gta-asset.exe delete gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/%2/%3 %4 %currentWW% -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        echo Asset deleted
        @REM goto passExit
    )

    echo Force deleting possible %CONDA_ENVS_PATH%\golden due to CondaValueError
    rmdir /S /Q %CONDA_ENVS_PATH%\golden
    call conda create -n golden python=3.10* -y
	call conda activate golden

	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
        echo Purging contents inside logs directory to prevent duplicate logging in automation
        del /q /f logs\*
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    call var_retrival.bat %4
    echo HF_name: !hf_name!
    echo sub_model: !sub_model!

    pip install pyyaml

    if "%2"=="openvino" (
        if not exist "temp" (
            git clone --depth 1 --branch 2025.1.0.0 https://github.com/openvinotoolkit/openvino.genai.git temp
        ) else (
            echo Temp directory exists, skipping temp folder creation.
        )
        cd temp\tools\who_what_benchmark
        pip install .

        :: Generate LLM golden outputs
        if "%3"=="llm" (
            mkdir %4
            wwb --base-model !hf_name! --gt-data %4\gt.csv --model-type text --hf
        )
        @REM Raise non-zero error if hit
        if !ERRORLEVEL! NEQ 0 (
            echo Encountered error during test execution
            goto errorExit
        )

        cd ../../..
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml
        call conda env export > temp\tools\who_what_benchmark\%4\environment.txt
        call gta-asset.exe push gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/%2/%3 %4 %currentWW% temp\tools\who_what_benchmark\%4 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        del /q /f temp\tools\who_what_benchmark\%4
        del /q /f C:\Users\%USERNAME%\.cache\huggingface\*
    )

    if "%2"=="openvino-nightly" (
        if not exist "temp" (
            git clone https://github.com/openvinotoolkit/openvino.genai.git temp
        ) else (
            echo Temp directory exists, skipping temp folder creation.
        )
        cd temp\tools\who_what_benchmark
        set PIP_PRE=1
        set PIP_EXTRA_INDEX_URL=https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install .

        :: Generate LLM golden outputs
        if "%3"=="llm" (
            mkdir %4
            wwb --base-model !hf_name! --gt-data %4\gt.csv --model-type text --hf
        )
        @REM Raise non-zero error if hit
        if !ERRORLEVEL! NEQ 0 (
            echo Encountered error during test execution
            goto errorExit
        )

        cd ../../..
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml
        call conda env export > temp\tools\who_what_benchmark\%4\environment.txt
        call gta-asset.exe push gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/golden/openvino/%3 %4 %currentWW% temp\tools\who_what_benchmark\%4 -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
        del /q /f temp\tools\who_what_benchmark\%4
        del /q /f C:\Users\%USERNAME%\.cache\huggingface\*
    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )
)
if "%1"=="microbench" (
    echo EXECUTING TEST: microbench
    echo.
    echo Force deleting possible %CONDA_ENVS_PATH%\microbench due to CondaValueError
    rmdir /S /Q %CONDA_ENVS_PATH%\microbench
    call conda create -n microbench python=3.10* -y
	call conda activate microbench
    call conda install libuv -y

	if not exist "logs" (
		mkdir logs
	) else (
		echo Logs directory exists, skipping logs folder creation.
        echo Purging contents inside logs directory to prevent duplicate logging in automation
        del /q /f logs\*
	)

    move /Y driverInfo.* logs
    move /Y deviceID.* logs
    move /Y SystemInfo.* logs

    if not exist "temp" (
        mkdir temp
    ) else (
        echo Temp directory exists, skipping temp folder creation.
    )

    if "%2"=="ipex" (
        call conda install pkg-config -y
        echo Executing IPEX TEST WITH PLATFORM = %PLATFORM%
        if %PLATFORM%==BMG (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ACM (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==ARL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==LNL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==MTL (
            python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --proxy=proxy-dmz.intel.com:912
            python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/  --proxy=proxy-dmz.intel.com:912
        )
        if %PLATFORM%==PTL (
            :: Require custom steps for IPEX PTL build
            call gta-asset.exe pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/misc/ipex_file ptl_builds WW17 --dest-dir ipex_ww17_builds -u %ARTIFACTORY_USERNAME% -p %ARTIFACTORY_PASSWORD% --root-url https://gfx-assets.intel.com/artifactory
            pip install "ipex_ww17_builds\torch-2.8.0a0+git836955b-cp310-cp310-win_amd64.whl"
            pip install pytorch_triton_xpu==3.3.0+git0bcc8265 --index-url https://download.pytorch.org/whl/nightly/xpu --proxy=proxy-dmz.intel.com:912
            pip install "ipex_ww17_builds\intel_extension_for_pytorch-2.8.10+gitc67ab61-cp310-cp310-win_amd64.whl"
            call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
            call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
        )

        @REM Export conda environment
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml

        copy /y scripts\microbench\ipex_microbench.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python ipex_microbench.py
    )
    if "%2"=="openvino" (
        pip install openvino

        @REM Export conda environment
        call conda env export > logs\environment.txt
        call conda env export > logs\environment.yml

        copy /y scripts\microbench\ov_microbench.py temp\
        xcopy /y utils\ temp\utils_aiml_intel\
        cd temp

        @REM Execute test
        python ov_microbench.py

    )

    @REM Raise non-zero error if hit
    if !ERRORLEVEL! NEQ 0 (
        echo Encountered error during test execution
        goto errorExit
    )

    @REM goto passing exit
    if !ERRORLEVEL! EQU 0 (
        goto passExit
    )

)

:passExit
echo ############### TEST EXECUTION COMPLETED WITH NO ERRORS ###############
exit /b %ERRORLEVEL%

:errorExit
echo ############### TEST EXECUTION COMPLETED WITH ERRORS CODE %ERRORLEVEL% ###############
exit /b %ERRORLEVEL%

:: Helper function
:showHelp
echo Usage: %~nx0 [option]
echo.
echo Options:
for %%o in (%validOptions%) do (
    echo   %%o
)
exit /b 0

ENDLOCAL
