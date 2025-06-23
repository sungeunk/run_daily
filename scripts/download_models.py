import argparse
import asyncio
import platform

from pathlib import Path
from tqdm.asyncio import tqdm

IS_WINDOWS = platform.system() == 'Windows'

def generate_download_url_list(args):
    SERVER_ROOT = 'http://ov-share-05.sclab.intel.com/cv_bench_cache'
    MODEL_COMMIT = args.model_commit

    # models
    MODEL_LIST = [
        'baichuan2-7b-chat',
        'chatglm3-6b',
        'gemma-7b-it',
        'glm-4-9b-chat-hf',
        'llama-2-7b-chat-hf',
        'llama-3.1-8b-instruct', # old: 'llama-3-8b',
        'minicpm-1b-sft',
        'minicpm-v-2_6',
        'mistral-7b-v0.1',
        'phi-3.5-mini-instruct', # old: 'phi-2',
        'phi-3-mini-4k-instruct',
        'qwen-7b-chat',
        'qwen2-7b-instruct',     # old: 'qwen2-7b',
    ]
    MODEL_PRECISION_LIST = [
        'pytorch/ov/OV_FP16-4BIT_DEFAULT',
    ]
    DOWNLOAD_URL_LIST = [ f'{SERVER_ROOT}/{MODEL_COMMIT}/{model}/{pre}/' for model in MODEL_LIST for pre in MODEL_PRECISION_LIST ]


    # stable-diffusion models
    SD_MODEL_LIST = [
        'stable-diffusion-v1-5',
        'stable-diffusion-v2-1',
        'stable-diffusion-3.5-large-turbo',
        # 'stable-diffusion-xl-1.0-inpainting-0.1',
        'lcm-dreamshaper-v7',
    ]
    SD_MODEL_PRECISION_LIST = [
        'pytorch/ov/FP16',
    ]
    SD_DOWNLOAD_URL_LIST = [ f'{SERVER_ROOT}/{MODEL_COMMIT}/{model}/{pre}/' for model in SD_MODEL_LIST for pre in SD_MODEL_PRECISION_LIST ]

    return DOWNLOAD_URL_LIST + SD_DOWNLOAD_URL_LIST

def convert_cmd_for_popen(cmd):
    return cmd.split() if IS_WINDOWS else cmd

async def async_download_files(url_list, output):
    semaphore = asyncio.Semaphore(5)

    async def async_download_file(url, output):
        async with semaphore:
            wget_cmd = f'wget -q --no-proxy -c -r --compression=auto --reject=html,tmp -nH --cut-dirs=1 --no-parent -P {output} {url}'
            print(f'call: {wget_cmd}')
            if IS_WINDOWS:
                process = await asyncio.create_subprocess_exec(*convert_cmd_for_popen(wget_cmd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            else:
                process = await asyncio.create_subprocess_shell(wget_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            print(f'done: {wget_cmd}')

    await tqdm.gather(
        *[async_download_file(url, output) for url in url_list]
    )

def main():
    parser = argparse.ArgumentParser(description="download models" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output directory', type=Path, default='c:\dev\models' if IS_WINDOWS else '/var/www/html/models')
    parser.add_argument('--model_commit', help='model commit name at http://ov-share-05.sclab.intel.com/cv_bench_cache/. ex) WW14_llm-optimum_2025.2.0-18615', type=str, default='WW14_llm-optimum_2025.2.0-18615')
    args = parser.parse_args()

    download_url_list = generate_download_url_list(args)
    asyncio.run(async_download_files(download_url_list, args.output))

if __name__ == "__main__":
    main()
