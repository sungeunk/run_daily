import argparse
import os
import platform
import subprocess

from pathlib import Path

IS_WINDOWS = platform.system() == 'Windows'
MODEL_DATE = os.environ["MODEL_DATE"]

DOWNLOAD_URL_LIST = [
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/stable-diffusion-v1-5/pytorch/dldt/FP16/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/stable-diffusion-v1-5/pytorch/dldt/compressed_weights/OV_FP16-INT8_ASYM/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/stable-diffusion-v2-1/pytorch/dldt/FP16/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/stable-diffusion-v2-1/pytorch/dldt/compressed_weights/OV_FP16-INT8_ASYM/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/lcm-dreamshaper-v7/pytorch/dldt/FP16/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/llama-2-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/chatglm3-6b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/qwen-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/llama-3-8b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/mistral-7b-v0.1/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/phi-2/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/phi-3-mini-4k-instruct/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/gemma-7b-it/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
]

DOWNLOAD_URL_MAXIMUM_LIST = [
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/llama-2-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_MAXIMUM/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/chatglm3-6b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_MAXIMUM/',
    f'http://ov-share-05.sclab.intel.com/cv_bench_cache/{MODEL_DATE}/mistral-7b-v0.1/pytorch/dldt/compressed_weights/OV_FP16-4BIT_MAXIMUM/',
]

# c:\dev\models\WW26_llm_2024.3.0-15805-6138d624dc1\stable-diffusion-v1-5\pytorch\dldt\FP16\

def main():
    parser = argparse.ArgumentParser(description="download models" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output directory', type=Path, default='c:\dev\models' if IS_WINDOWS else '/mnt/shared/models')
    args = parser.parse_args()

    for url in DOWNLOAD_URL_LIST + DOWNLOAD_URL_MAXIMUM_LIST:
        wget_cmd = f'wget -q --show-progress --no-proxy -c -r --compression=auto --reject=html,tmp -nH --cut-dirs=1 --no-parent -P {args.output} {url}'
        subprocess.call(wget_cmd, shell=True)

if __name__ == "__main__":
    main()
