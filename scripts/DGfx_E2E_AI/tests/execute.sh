#!/bin/bash

# add test scripts for Ubuntu, only openvino_benchmark_app and ov_llm_bench were verified.

ENV_FILE=${HOME}/.env

goto_exit() {
    echo The Exit code is $1
    exit $1
}

if [[ -f "${ENV_FILE}" ]]; then
	ARTIFACTORY_USERNAME=$(grep  ARTIFACTORY_USERNAME ${ENV_FILE}| awk -F'=' '{print $2}')
	ARTIFACTORY_PASSWORD=$(grep  ARTIFACTORY_PASSWORD ${ENV_FILE}| awk -F'=' '{print $2}')
else
	echo Please provide .env file containing artifactory credentials
	exit 1
fi

if [[ -z $ARTIFACTORY_USERNAME ]]; then
    echo Please provide .env file containing artifactory credentials
	exit 1
fi

if [[ $# -eq 0 ]]; then
    echo No argument provided. Please provide argument to execute AI ML tests.
    exit 1
fi

if ! [[ -f "gta-asset" ]]; then
    wget https://gfx-assets.sh.intel.com/artifactory/gta-fm/gta-asset/latest/linux/artifacts/gta-asset
    chmod +x ./gta-asset
fi

export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
export no_proxy="localhost, 127.0.0.1, ::1, *.intel.com"

source /home/gta/conda/etc/profile.d/conda.sh
# source oneapi env and check GPU device
source /opt/intel/oneapi/setvars.sh
sycl-ls
if [[ $? -ne 0 ]]; then
    echo "no L0/OpenCL device found"
    exit 1
fi

if [[ "$1" == "openvino_benchmark_app" ]]; then
    # Environment setup
    conda activate openvino_benchmark_app
    if [[ $? -ne 0 ]]; then
        conda create -n openvino_benchmark_app python=3.10* -y
        conda activate openvino_benchmark_app
    fi
    conda info --envs
    mkdir temp
    mkdir logs

    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    python -m pip install --upgrade pip
    pip install openvino openvino-dev onnx torch torchvision tensorflow protobuf==3.20.3
    python -c "from openvino.runtime import Core; print(Core().available_devices)"

    # Export conda environment
    conda env export > logs/environment.txt

	# ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ computer_vision $2 --dest-dir temp/omz_models -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
    ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights computer_vision "$2" --dest-dir temp/omz_models -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
    cp uncategorized/openvino_benchmark_app/benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python benchmark.py --model $2 --hint $3 --iteration $4
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi

fi

if [[ "$1" == "openvino_model_zoo" ]]; then
    # Environment setup
    conda activate openvino_model_zoo
    if [[ $? -ne 0 ]]; then
        conda create -n openvino_model_zoo python=3.10* -y
        conda activate openvino_model_zoo
    fi
    conda info --envs
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt
    git clone https://github.com/openvinotoolkit/open_model_zoo.git temp
    cp uncategorized/openvino_model_zoo/benchmark.py temp/demos/gpt2_text_prediction_demo/python
    
    python -m pip install --upgrade pip
    pip install openvino openvino-dev onnx torch tokenizers
    python -c "from openvino.runtime import Core; print(Core().available_devices)"

    # Export conda environment
    conda env export > logs/environment.txt
    
    if [[ "$2" == "gpt2" ]]; then
        cp utils/ temp/demos/gpt2_text_prediction_demo/python/utils_aiml_intel/
        cd temp/demos/gpt2_text_prediction_demo/python
        omz_downloader --list models.lst
        omz_converter --list models.lst
    fi

    # Execute test
    python benchmark.py --model $2 --num_iter $3

    # Raise non-zero error if hit
    if [[ $? -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi

if [[ "$1" == "A1111" ]]; then
    # Environment setup
    # conda create -n A1111 python=3.10* -y
	# conda activate A1111

    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    pip install psutil

    # Export conda environment
    # conda env export > logs/environment.txt

    if [[ "$2" == "OpenVINO" ]]; then
        git clone https://github.com/openvinotoolkit/stable-diffusion-webui.git temp
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v1.5/OV/A1111/webui-user-openvino.bat temp/webui-user.bat 
    fi
    if [[ "$2" == "IPEX" ]]; then
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git temp
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v1.5/IPEX/A1111/webui-user-ipex.bat temp/webui-user.bat 
    fi
    if [[ "$2" == "CUDA" ]]; then
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git temp
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v1.5/CUDA/A1111/webui-user-cuda.bat temp/webui-user.bat 
    fi

    if [[ "$3" == "v1.5" ]]; then
	    ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion v1.5 --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v1.5/IPEX/A1111/A1111_benchmark.py temp 
    fi
    if [[ "$3" == "v2.0" ]]; then
	    ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion v2.0 --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v2.0/IPEX/A1111/A1111_benchmark.py temp 
    fi
    if [[ "$3" == "v2.1" ]]; then
	    ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion v2.1 --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_v2.1/IPEX/A1111/A1111_benchmark.py temp 
    fi
    if [[ "$3" == "xl" ]]; then
        ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion xl --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion sdxl_vae --dest-dir temp/models/VAE -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_XL/IPEX/A1111/A1111_benchmark.py temp 
    fi
    if [[ "$3" == "turbo" ]]; then
        ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion turbo --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_Turbo/IPEX/A1111/A1111_benchmark.py temp 
    fi
    if [[ "$3" == "xl_turbo" ]]; then
        ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ stable_diffusion xl_turbo --dest-dir temp/models/Stable-diffusion -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
        cp Foundational_Models/Generative_Creator_AI/Stable_Diffusion_XL_Turbo/IPEX/A1111/A1111_benchmark.py temp 
    fi

    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python A1111_benchmark.py --api $2 --ckpt $3 --num_iter $4
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi

if [[ "$1" == "ov_llm_bench" ]]; then
    # Environment setup
    conda activate ov_llm_bench 
    if [[ $? -ne 0 ]]; then
        conda create -n ov_llm_bench python=3.10* -y
        conda activate ov_llm_bench
    fi
    conda info --envs

	mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if ! [[ -f "temp/llm_bench/python/requirements.txt" ]]; then
        sudo rm -rf temp
        git clone https://github.com/openvinotoolkit/openvino.genai.git temp
        cd temp
        git checkout 87a1a3b0d98f95c242bc59c75f9804c58c4834e1
        cd ..

        #export GIT_CLONE_PROTECTION_ACTIVE=false
        pip install update --upgrade
        pip install -r temp/llm_bench/python/requirements.txt
        if [[ $? -ne 0 ]]; then
            echo pip install failed, please setup proxy or huggingface mirrors if in PRC
            rm temp/llm_bench/python/requirements.txt
            exit 1
        fi
        pip install transformers==4.40.0
        pip install torch==2.3.1
    else
        echo Temp directory exists, skipping temp folder creation.
    fi
    if [[ "$5" == "nightly" ]]; then
            #pip uninstall openvino -y
            pip install openvino-nightly
    fi

    python -c "from openvino.runtime import Core; print(Core().available_devices)"
    if [[ $? -ne 0 ]]; then
        echo failed to load openvino.runtime
        exit 1
    fi
    # Export conda environment
    conda env export > logs/environment.txt

    # choices [
    #     llama-2-7b
    #     llama-3-8b
    #     qwen-1.0
    #     qwen-1.5-7b
    #     qwen-2.0
    #     phi-2
    #     phi-3
    #     mistral-7b-v0.1
    #     gemma-v1
    #     chatglm3-6b
    # ]

    if ! [[ -d "temp/llm_bench/python/models/$2/INT4" ]]; then
        ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models $2-INT4 --dest-dir temp/llm_bench/python/models/$2/INT4 -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
    else
        echo Skipping model download since model directory exists.
    fi

    cp scripts/large_language_model/openvino/benchmark.py temp/llm_bench/python/
    cp -r utils/ temp/llm_bench/python/utils_aiml_intel/
    cd temp/llm_bench/python

    # WA for failure: couldn't find module utils.model_utils
    touch utils/__init__.py
    # Execute test
    python benchmark.py -m models/$2/INT4 --prompt_file $3 --infer_count $4 -rj results-INT4.json 
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi


if [[ "$1" == "llm_bench" ]]; then
    # Environment setup
	conda activate llm_bench
    if [[ $? -ne 0 ]]; then
        conda create -n llm_bench python=3.10* -y
        conda activate llm_bench
    fi
    conda info --envs
	conda install libuv -y

	if [[ "$2" == "ipex" ]]; then
			pip install -r scripts/large_language_model/pytorch/requirements_ipex.txt
	elif [[ "$2" == "cuda" ]]; then
			pip install -r scripts/large_language_model/pytorch/requirements_cuda.txt
	else
		echo Conda environment exists, skipping environment creation.
	fi
	
	if ! [[ -d "logs" ]]; then
		mkdir logs
 	else
		echo Logs directory exists, skipping logs folder creation.
	fi
	
	if ! [[ -d "temp" ]]; then
		git clone https://github.com/intel-analytics/ipex-llm.git temp
	else
		echo Temp directory exists, skipping temp folder creation.
	fi

   # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    # ipex_llm benchmark script requirements
    pip install transformers==4.38.0

    export PYTHONIOENCODING=utf-8

    if [[ "$3" == "llava_v1.5" ]]; then
        git clone -b v1.1.1 --depth=1 https://github.com/haotian-liu/LLaVA
        set LLAVA_REPO_DIR=%cd%/LLaVA
        pip install einops
        pip install transformers==4.31.0
    fi
    if [[ "$2" == "ipex" ]]; then
        export SYCL_CACHE_PERSISTENT=1
        export IPEX_LLM_LOW_MEM=1
    fi


    # Export conda environment
    conda env export > logs/environment.txt
    
    if [[ "$3" == "llama-2-7b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Llama-2-7b-chat-hf" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v2 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Llama-2-7b-chat-hf -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v2 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Llama-2-7b-chat-hf -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "meta-llama/Llama-2-7b-chat-hf" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "llama-3-8b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Meta-Llama-3-8B" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Meta-Llama-3-8B -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llama_v3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Meta-Llama-3-8B -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "meta-llama/Meta-Llama-3-8B" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "qwen-1.0-7b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Qwen-7B-Chat" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.0 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen-7B-Chat -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.0 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen-7B-Chat -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "Qwen/Qwen-7B-Chat" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "qwen-1.5-7b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Qwen1.5-7B-Chat" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.5 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen1.5-7B-Chat -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v1.5 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen1.5-7B-Chat -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "Qwen/Qwen1.5-7B-Chat" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "qwen-2.0-7b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Qwen2-7B-Instruct" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v2.0_7b --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen2-7B-Instruct -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_qwen_v2.0_7b --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Qwen2-7B-Instruct -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "Qwen/Qwen2-7B-Instruct" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "phi-2" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/phi-2" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_2 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/phi-2 -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_2 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/phi-2 -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "microsoft/phi-2" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "phi-3" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Phi-3-mini-128k-instruct" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Phi-3-mini-128k-instruct -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_phi_3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Phi-3-mini-128k-instruct -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "microsoft/Phi-3-mini-128k-instruct" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "mistral-7b-v0.1" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/Mistral-7B-v0.1" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_mistral_v0.1 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Mistral-7B-v0.1 -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_mistral_v0.1 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/Mistral-7B-v0.1 -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "mistralai/Mistral-7B-v0.1" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "llava_v1.5" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/llava-v1.5-7b" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llava_v1.5 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/llava-v1.5-7b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_llava_v1.5 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/llava-v1.5-7b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "liuhaotian/llava-v1.5-7b" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "gemma" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/gemma-7b" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_gemma --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/gemma-7b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
				./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_gemma --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/gemma-7b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
		else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "google/gemma-7b" --input_token $4 --output_token $5
    fi
    if [[ "$3" == "chatglm3-6b" ]]; then
        if ! [[ -d "temp/python/llm/dev/benchmark/all-in-one/hub/chatglm3-6b" ]]; then
            ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_chatglm3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/chatglm3-6b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			if [[ $? -ne 0 ]]; then
				rmdir "C:/Users/%USERNAME%/.cache" /s /q
                ./gta-asset pull gfx-e2eval-sw-fm/gfx-e2eval-sw-fm/AI_ML_Automation/weights/ large_language_models hf_chatglm3 --dest-dir temp/python/llm/dev/benchmark/all-in-one/hub/chatglm3-6b -u $ARTIFACTORY_USERNAME -p $ARTIFACTORY_PASSWORD --root-url https://gfx-assets.fm.intel.com/artifactory
			fi
        else
            echo Skipping model download since model directory exists.
        fi
        python scripts/large_language_model/pytorch/configurator.py --api $2 --model "THUDM/chatglm3-6b" --input_token $4 --output_token $5
    fi
	
    cp temp/python/llm/dev/benchmark/all-in-one/config.yaml logs
    cp scripts/large_language_model/pytorch/run.py temp/python/llm/dev/benchmark/all-in-one
    cp -r utils/ temp/python/llm/dev/benchmark/all-in-one/utils_aiml_intel/
    cd temp/python/llm/dev/benchmark/all-in-one

    # Execute test
    python run.py
    ret=$?
    if ! [[ "$6" == "no_delete" ]]; then
        rm -r hub
    fi
    
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi

if [[ "$1" == "whisper_large" ]]; then
    conda create -n whisper python=3.10* -y
	conda activate whisper
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt


    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$4" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install librosa datasets accelerate transformers
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Foundational_Models/Natural_Language_Processing/Whisper_Large_v3/whisper_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python whisper_benchmark.py --api $2 --num_iter $3
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "whisper_medium" ]]; then
    conda create -n whisper python=3.10* -y
	conda activate whisper
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt


    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$4" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install librosa datasets accelerate transformers
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Foundational_Models/Natural_Language_Processing/Whisper_Medium/whisper_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python whisper_benchmark.py --api $2 --num_iter $3
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "whisper_base" ]]; then
    conda create -n whisper python=3.10* -y
	conda activate whisper
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt


    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$4" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install librosa datasets accelerate transformers
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Foundational_Models/Natural_Language_Processing/Whisper_Base/whisper_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python whisper_benchmark.py --api $2 --num_iter $3
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "clip" ]]; then
    conda create -n clip_env python=3.10* -y
	conda activate clip_env
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install transformers
    fi
    if [[ "$2" == "dml" ]]; then
        pip install optimum[exporters]
        pip install onnxruntime-directml transformers pillow
        pip install torch==2.3.1
    fi
    if [[ "$3" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install datasets
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Scale_Models/Computer_Vision/CLIP_Large_Patch/clip_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python clip_benchmark.py --model clip --api $2
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "edsr" ]]; then
    conda create -n edsr_env python=3.10* -y
	conda activate edsr_env
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$2" == "dml" ]]; then
        pip install optimum[exporters]
        pip install onnxruntime-directml transformers pillow
        pip install torch==2.3.1
    fi
    if [[ "$4" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install datasets super_image
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Scale_Models/Computer_Vision/EDSR/edsr_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python edsr_benchmark.py --model edsr --api $2 --upscale $3
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "openpose" ]]; then
    conda create -n openpose_env python=3.10* -y
	conda activate openpose_env
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$2" == "dml" ]]; then
        pip install optimum[exporters]
        pip install onnxruntime-directml transformers pillow
        pip install torch==2.3.1
    fi
    if [[ "$3" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install datasets controlnet_aux diffusers
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Scale_Models/Computer_Vision/OpenPose/openpose_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python openpose_benchmark.py --model openpose --api $2
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "resnet50" ]]; then
    conda create -n resnet50_env python=3.10* -y
	conda activate resnet50_env
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install transformers
    fi
    if [[ "$2" == "dml" ]]; then
        pip install optimum[exporters]
        pip install onnxruntime-directml transformers pillow
        pip install torch==2.3.1
    fi
    if [[ "$3" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi

    pip install datasets
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Scale_Models/Computer_Vision/Resnet_50/resnet50_benchmark.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp
    optimum-cli export onnx --model microsoft/resnet-50 onnx

    # Execute test
    python resnet50_benchmark.py --model resnet50 --api $2
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "bert" ]]; then
    conda create -n bert_env python=3.10* -y
	conda activate bert_env
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install openvino
        pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
        pip install nncf
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers
		pip install setuptools==69.5.1
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    if [[ "$4" == "nightly" ]]; then
        pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install torch==2.3.1
    fi
    
    pip install numpy==1.26.4

    # Export conda environment
    conda env export > logs/environment.txt

    cp Scale_Models/Large_Language_Models/BERT/bert_sample.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp

    # Execute test
    python bert_sample.py --model bert --input_token $3 --api $2
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi
if [[ "$1" == "base_sd" ]]; then
    conda activate base_sd
    if [[ $? -ne 0 ]]; then
        conda create -n base_sd python=3.10* -y
        conda activate base_sd
    fi
    conda info --envs
    
    conda install libuv -y
    mkdir temp
    mkdir logs
    # Linux Driver info
    uname -r | tee logs/driver_info.txt
    apt list --installed | apt list --installed | grep -e intel-opencl-icd -e mesa-vulkan-drivers -e intel-i915-dkms -e intel-media-va-driver-non-free| tee -a logs/driver_info.txt

    if [[ "$2" == "openvino" ]]; then
        pip install -r scripts/stable_diffusion/requirements_openvino.txt
        pip install torch==2.3.1
        # pip install openvino
        # pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
    fi
    if [[ "$2" == "openvino-nightly" ]]; then
        pip install optimum[openvino]
        pip -y uninstall openvino openvino_tokenizers openvino_telemetry
        pip install openvino-nightly
        # pip install --upgrade --pre openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        pip install diffusers transformers accelerate numpy==1.26.4
        pip install torch==2.3.1
    fi
    if [[ "$2" == "ipex" ]]; then
        conda install pkg-config -y
        python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
        pip install transformers diffusers accelerate sentencepiece protobuf
		pip install setuptools==69.5.1
        # pip install -r scripts/stable_diffusion/requirements_ipex.txt
        # pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0
        # pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    fi
    if [[ "$2" == "cuda" ]]; then
        pip install -r scripts/stable_diffusion/requirements_cuda.txt
        # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi

    # pip install numpy==1.26.4
    # pip install diffusers transformers accelerate
    
    # Export conda environment
    conda env export > logs/environment.txt

    cp scripts/stable_diffusion/base_sd.py temp/
    cp scripts/stable_diffusion/sd_hijack_utils.py temp/
    cp scripts/stable_diffusion/xpu_specific.py temp/
    cp -r utils/ temp/utils_aiml_intel/
    cd temp
    python base_sd.py --api openvino-nightly --model xl  --height 768 --width 768 --num_iter 5
    # Execute test
    python base_sd.py --api $2 --model $3 --height $4 --width $5 --num_iter $6
    ret=$?
    # Raise non-zero error if hit
    if [[ $ret -ne 0 ]]; then
        echo Encountered error during test execution
        goto_exit $ret
    fi
fi

echo Test execution completed with no errors




