import argparse
import os
import platform
import subprocess

from pathlib import Path

IS_WINDOWS = platform.system() == 'Windows'
SERVER_ROOT = "http://ov-share-05.sclab.intel.com/cv_bench_cache/WW32_llm-optimum_2024.4.0-16283-41691a36b90"
DOWNLOAD_URL_LIST = [
    f'{SERVER_ROOT}/baichuan2-7b-chat/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/chatglm3-6b/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/chatglm3-6b/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/gemma-7b-it/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/gemma-7b-it/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/glm-4-9b/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/llama-2-7b-chat-hf/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/llama-2-7b-chat-hf/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/llama-3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/llama-3-8b/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/minicpm-1b-sft/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/mistral-7b-v0.1/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/mistral-7b-v0.1/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/phi-2/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/phi-2/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/phi-3-mini-4k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/phi-3-mini-4k-instruct/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/qwen-7b-chat/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
    f'{SERVER_ROOT}/qwen-7b-chat/pytorch/ov/OV_FP16-4BIT_MAXIMUM/',
    f'{SERVER_ROOT}/qwen2-7b/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
]


SERVER_ROOT = "http://ov-share-05.sclab.intel.com/cv_bench_cache/WW32_llm-optimum_2024.4.0-16283-41691a36b90"
DOWNLOAD_OTHER_URL_LIST = [
    f'{SERVER_ROOT}/lcm-dreamshaper-v7/pytorch/ov/OV_FP16-4BIT_DEFAULT/',
]

def main():
    parser = argparse.ArgumentParser(description="download models" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output directory', type=Path, default='c:\dev\models' if IS_WINDOWS else '/var/www/html/models')
    args = parser.parse_args()

    for url in DOWNLOAD_URL_LIST + DOWNLOAD_OTHER_URL_LIST:
        wget_cmd = f'wget -q --show-progress --no-proxy -c -r --compression=auto --reject=html,tmp -nH --cut-dirs=1 --no-parent -P {args.output} {url}'
        subprocess.call(wget_cmd, shell=True)

if __name__ == "__main__":
    main()
