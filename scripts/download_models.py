import argparse
import itertools
import json
import os
import platform
import re
import subprocess
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

IS_WINDOWS = platform.system() == 'Windows'

# Regex to capture percentage from wget's progress output
# e.g., " 34%[=======>         ] 34,567,890  12.3MB/s  eta 5s" -> "34"
PROGRESS_RE = re.compile(r'\s*(\d+)%')


# Util Functions
def generate_url(server, model_commit, model_name, framework, precision):
    SERVER_DICT = {
        'ov-share-04':'http://ov-share-04.sclab.intel.com/cv_bench_cache',
        'ov-share-05':'http://ov-share-05.sclab.intel.com/cv_bench_cache'
    }
    FRAMEWORK_DICT = {
        "onnx": "onnx/onnx",
        "tf": "tf/tf_frozen",
        "tf2": "tf2/tf2_saved_model",
        "paddle": "paddle/paddle",
        "pytorch": "pytorch/pytorch",
    }
    PRECISION_DICT = {
        "FP16": "FP16/1/ov",
        "FP32": "FP32/1/ov",
        "FP16-INT8": "FP16/INT8/1/ov/optimized/",
        "FP32-INT8": "FP32/INT8/1/ov/optimized/",
    }

    try:
        common_url = f'{SERVER_DICT.get(server)}/{model_commit}/{model_name}'
        if server == 'ov-share-04':
            return common_url + '/' + f'{FRAMEWORK_DICT.get(framework)}/{PRECISION_DICT.get(precision)}/'
        elif server == 'ov-share-05':
            return common_url + '/' + f'{framework}/ov/{precision}/'
        else:
            return ''
    except Exception as e:
        print(e)
        return ''

def generate_download_target_list(args):
    target_list = []
    with open(args.config_json) as file:
        json_data = json.load(file)

        server = json_data['server']
        commit = json_data['cache_commit']
        models = json_data['model']
        for model in models:
            for model_attr in itertools.product(model['name'], model['framework'], model['precision']):
                url = generate_url(server, commit, model_attr[0], model_attr[1], model_attr[2])
                if url:
                    target_list.append((f'{model_attr[0]}|{model_attr[1]}|{model_attr[2]}', url))

    return target_list

def download_target(args, target, position):
    """
    Downloads files using the wget command-line tool and shows progress.
    Can perform recursive downloads if RECURSIVE is set to True.
    NOTE: This function requires 'wget' to be installed and accessible in the system's PATH.

    Args:
        target (tuple): The URL to download from.
            target[0]: display name
            target[1]: url
        position (int): The line number for the tqdm progress bar to display on.
    """
    display_name = target[0]
    url = target[1]
    
    # Setup the progress bar for this specific download
    # position+1 is to avoid overlapping with the main progress bar at position 0
    pbar = tqdm(
        total=100, # Progress will be from 0 to 100 percent
        desc=f"{display_name}", # Truncate/pad name for alignment
        position=position + 1,
        unit='%',
        leave=False # Clear the progress bar when done
    )

    try:
        # Base wget command arguments
        command = ['wget', '--progress=bar:force:noscroll', '-c', '-r',
                   '--compression=auto', '--reject=html,tmp', '-nH', '--cut-dirs=1', '--no-parent',
                   '-P', args.download_dir, url]

        # Start the wget process
        process = subprocess.Popen(
            command,
            stderr=subprocess.PIPE,  # wget progress is written to stderr
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Read stderr line by line in real-time to parse progress
        # For recursive downloads, this will show progress for the CURRENT file
        for line in process.stderr:
            match = PROGRESS_RE.search(line)
            if match:
                percentage = int(match.group(1))
                # Update the progress bar to the current percentage
                pbar.update(percentage - pbar.n)

        process.wait()  # Wait for the process to complete

        # Check the final return code
        if process.returncode == 0:
            pbar.update(100 - pbar.n) # Ensure bar is full on success
            pbar.close()
            return (target, "Success")
        else:
            pbar.close()
            # Try to get the error output for a concise message
            error_output, _ = process.communicate()
            # Filter empty lines and get the last piece of info
            error_lines = [line for line in error_output.strip().split('\n') if line]
            last_error_line = error_lines[-1] if error_lines else "Unknown wget error"
            return (target, f"Failed: {last_error_line}")

    except FileNotFoundError:
        pbar.close()
        # This error is critical, as it means wget is not installed.
        return (target, "Failed: 'wget' command not found. Please install it.")
    except Exception as e:
        pbar.close()
        return (target, f"Failed: An unexpected error occurred - {e}")


# template: http://{SERVER}.intel.com/cv_bench_cache/{MODEL_COMMIT}/{MODEL_NAME}/{FRAMEWORK}/{PRECISION}
# ex) http://ov-share-04.iotg.sclab.intel.com/cv_bench_cache/WW23_static_2025.2.0-19136-RC2/began/onnx/onnx/FP16/INT8/1/ov/optimized/
def main():
    parser = argparse.ArgumentParser(description='download model files' , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--download_dir', help='download directory', type=Path, default='.' if IS_WINDOWS else '.')
    parser.add_argument('-cj', '--config_json', help='user config json file path', type=Path, default=None)
    parser.add_argument('-mw', '--max_workers', help='count of workers', type=int, default=5)
    parser.add_argument('--for_daily', help='download models for daily. ignore config_json & download_dir', action='store_true')
    args = parser.parse_args()

    if args.for_daily:
        args.config_json = os.path.join(*[Path(__file__).resolve().parent, 'sample/daily_WW43_llm-optimum_2025.4.0-20264.json'])
        args.download_dir = 'c:/dev/models/daily' if IS_WINDOWS else '/var/www/html/models/daily'

    download_target_list = generate_download_target_list(args)

    os.makedirs(args.download_dir, exist_ok=True)

    success_count = 0
    failure_count = 0
    failure_targets = []

    # Use ThreadPoolExecutor to manage a pool of threads for concurrent downloads
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit download tasks to the executor pool
        # The position is calculated using (i % MAX_WORKERS) to reuse the same lines
        # on the console for new downloads, keeping the display compact.
        future_to_target = {
            executor.submit(download_target, args, target, i % args.max_workers): target 
            for i, target in enumerate(download_target_list)
        }

        # Create a main progress bar to track overall progress
        main_pbar = tqdm(as_completed(future_to_target), total=len(download_target_list), desc="Overall Progress", unit="job", position=0)

        for future in main_pbar:
            target, result = future.result()
            if "Success" in result:
                success_count += 1
            else:
                failure_targets.append(target)
                failure_count += 1

    # Add newlines to move cursor below the space used by individual progress bars
    print("\n" * (args.max_workers + 1))
    print("--- Download Complete ---")
    print(f"Total jobs: {len(download_target_list)}, Success: {success_count}, Failed: {failure_count}")
    print(f'Failed urls')
    print('\n'.join(f'\t{target[1]}' for target in failure_targets))

if __name__ == "__main__":
    main()
