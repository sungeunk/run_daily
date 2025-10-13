import argparse
import asyncio
import platform

from pathlib import Path
from tqdm.asyncio import tqdm

IS_WINDOWS = platform.system() == 'Windows'

def generate_download_url_list(args):
    SERVER_ROOT = 'http://ov-share-04.sclab.intel.com/cv_bench_cache'
    MODEL_COMMIT = args.model_commit

    # models
    MODEL_LIST = [
        ['hbonet-0.25', 'onnx', 'FP16'],
        ['hbonet-0.25', 'onnx', 'FP32'],
        ['east_resnet_v1_50', 'tf', 'FP16'],
        ['east_resnet_v1_50', 'tf', 'FP16-INT8'],
        ['person-attributes-recognition-crossroad-0238', 'onnx', 'FP16-INT8'],
        ['shufflenet', 'onnx', 'FP16'],
        ['shufflenet', 'onnx', 'FP16-INT8'],
        ['unet-2d', 'onnx', 'FP16'],
        ['unet-2d', 'onnx', 'FP16-INT8'],
    ]
    def convert_framework(framework):
        dicts = {
            "onnx": "onnx/onnx",
            "tf":   "tf/tf_frozen",
        }
        return dicts.get(framework, None)

    def convert_precision(precision):
        dicts = {
            "FP16":      "FP16/1/ov",
            "FP32":      "FP32/1/ov",
            "FP16-INT8": "FP16/INT8/1/ov/optimized/",
            "INT8":      "FP16/INT8/1/ov/optimized/",
            "FP32-INT8": "FP32/INT8/1/ov/optimized/",
        }
        return dicts.get(precision, None)

    DOWNLOAD_URL_LIST = []
    for MODEL in MODEL_LIST:
        model_name = MODEL[0]
        framework_url = convert_framework(MODEL[1])
        precision_url = convert_precision(MODEL[2])
        DOWNLOAD_URL_LIST.append(f'{SERVER_ROOT}/{MODEL_COMMIT}/{model_name}/{framework_url}/{precision_url}')

    return DOWNLOAD_URL_LIST

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
    parser.add_argument('-o', '--output', help='output directory', type=Path, default='.' if IS_WINDOWS else '.')
    parser.add_argument('--model_commit', help='model commit name at http://ov-share-04.sclab.intel.com/cv_bench_cache/. ex) WW32_static_2025.3.0-19691', type=str, default='WW32_static_2025.3.0-19691')
    args = parser.parse_args()

    download_url_list = generate_download_url_list(args)
    asyncio.run(async_download_files(download_url_list, args.output))

if __name__ == "__main__":
    main()
