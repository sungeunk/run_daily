import argparse
import platform
import subprocess

from pathlib import Path

IS_WINDOWS = platform.system() == 'Windows'

DOWNLOAD_URL_LIST = [
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/stable-diffusion-v1-5/pytorch/dldt/FP16/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/stable-diffusion-v1-5/pytorch/dldt/compressed_weights/OV_FP16-INT8_ASYM/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/stable-diffusion-v2-1/pytorch/dldt/FP16/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/stable-diffusion-v2-1/pytorch/dldt/compressed_weights/OV_FP16-INT8_ASYM/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/lcm-dreamshaper-v7/pytorch/dldt/FP16/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/llama-2-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/chatglm3-6b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/qwen-7b-chat/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/llama-3-8b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/mistral-7b-v0.1/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/phi-2/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/phi-3-mini-4k-instruct/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
    'http://ov-share-05.sclab.intel.com/cv_bench_cache/WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2/gemma-7b-it/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/',
]

# c:\dev\models\WW22_llm_2024.2.0-15519-5c0f38f83f6-releases_2024_2\stable-diffusion-v1-5\pytorch\dldt\FP16\

def main():
    parser = argparse.ArgumentParser(description="download models" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output directory', type=Path, default='c:\dev\models' if IS_WINDOWS else '/mnt/shared/models')
    args = parser.parse_args()

    for url in DOWNLOAD_URL_LIST:
        wget_cmd = f'wget -q --show-progress --no-proxy -c -r --compression=auto --reject=html,tmp -nH --cut-dirs=1 --no-parent -P {args.output} {url}'
        subprocess.call(wget_cmd, shell=True)

if __name__ == "__main__":
    main()
