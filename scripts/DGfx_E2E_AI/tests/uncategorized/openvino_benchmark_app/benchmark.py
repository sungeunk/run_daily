import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from pprint import pformat

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="define which model to run benchmark")
parser.add_argument("--hint", type=str, choices=["latency", "throughput"], help="define performance hints to benchmark with")
parser.add_argument("-n", "--iteration", type=int, default=4, help="define number of iterations to run per precision")
args = parser.parse_args()

def is_parent_directory_present(parent_dir_name):
    current_path = Path.cwd()  # Get the current working directory
    # Iterate through parent directories until reaching the root directory ('/')
    while current_path != current_path.parent:  # Check if current_path is not the root
        if current_path.name == parent_dir_name:
            return True  # Found the directory
        current_path = current_path.parent  # Move to the parent directory
    # Check the root directory
    if current_path.name == parent_dir_name:
        return True  # Found the directory
    return False

def traverse_up_directories(num_levels):
    current_path = Path.cwd()  # Get the current working directory
    target_path = current_path.resolve().parents[num_levels - 1] if num_levels <= len(current_path.parts) else None
    return target_path

def parse_benchmark_output(benchmark_output: str):
    """Prints the output from benchmark_app in human-readable format"""
    throughput_loc = benchmark_output.find("Throughput")
    fps_loc = benchmark_output.find("FPS")
    
    for substring in benchmark_output[throughput_loc:fps_loc+3].split(" "):
        try:
            fps = float(substring)
        except ValueError:
            pass

    print(f"FPS: {fps}")
    return fps

def execute_benchmark_loop(model, iter_count):
    batch_size_list = [1, 16, 32, 64, 128]

    if model == "faster-rcnn-resnet101-coco-sparse-60-0001":
        precision_list = ["FP16", "FP32", "FP16-INT8"]
    else:
        precision_list = ["FP16", "FP32"]

    run_dict = {}

    for precision in precision_list:
        for bs in batch_size_list:
            run_dict[f"{precision}_BS{bs}"] = {}
            run_dict[f"{precision}_BS{bs}"]["average"] = 0
            run_dict[f"{precision}_BS{bs}"]["runs"] = []
            fps_list = []
            for iteration in range(iter_count):
                fps_list = execute_benchmark(bs, iteration, fps_list, model, precision)
            logger.info(f"Model: {model}, Precision: {precision}, Batch Size: {bs}, Average FPS: {np.round(np.average(fps_list),2)}")
            run_dict[f"{precision}_BS{bs}"]["average"] = np.round(np.average(fps_list),2)
            run_dict[f"{precision}_BS{bs}"]["average_exclude"] = np.round(np.average(fps_list[1:]),2)
            run_dict[f"{precision}_BS{bs}"]["runs"] = fps_list

    for line in pformat(run_dict, depth=60).split('\n'):
        logger.info(line)

    with open(result_json, "w") as f:
        json.dump(run_dict, f)

def execute_benchmark(bs, iteration, fps_list, model, precision):
    base_path = Path("omz_models")
    if sys.platform == 'linux':
        split_char= '/'
    else:
        split_char = "\\"
    front = [i for i in base_path.glob(f"*{split_char}{model}")][0]
    model_path = [i for i in (front / f"{precision}").glob("*.xml")][0]


    time.sleep(5)
    print(f"Model: {model} ({model_path}), Precision: {precision}, Batch Size: {bs}, Iteration: {iteration}")
    output = subprocess.Popen(["benchmark_app", "-m", model_path, "-d", "GPU", "-api", "async", "-hint", args.hint,"-b", str(bs), "-t", "15"], stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,)
    out, err = output.communicate()
    fps_list.append(parse_benchmark_output(out.decode()))
    time.sleep(5)

    return fps_list
    
if __name__ == "__main__":
    cwd = Path.cwd()
    if is_parent_directory_present("gtax-client"):
        test_path = traverse_up_directories(-3)
        log_path = test_path / "logs"
        Path(log_path).mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path("logs")
        Path(log_path).mkdir(parents=True, exist_ok=True)

    result_json = log_path / "result_payload.json"

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        filename=log_path / "parsed_result.log",
        filemode="w",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    logger = logging.getLogger(__name__)
    logger.info("Setup logging at parsed_result.log")

    logger.info("############################################################")
    logger.info("                      Starting logging                      ")
    logger.info("############################################################")
    if sys.platform.startswith('win'):
        command = "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty DriverVersion"
        driver_version = subprocess.run(['powershell', "-Command", command], capture_output=True, text=True)
        driver_version = driver_version.stdout.strip()
        logger.info(f"Driver Version: {driver_version}")
    output = subprocess.check_output('pip list', shell=True)
    output = output.decode("utf-8")
    logger.info(output)

    logger.info("##################  Result Logging Start  ##################")
    
    try:
        iter_count = args.iteration
        model = args.model
        execute_benchmark_loop(model, iter_count)
    finally:
        logger.info("Deleting omz_models folder")
        shutil.rmtree("omz_models")
        logger.info("Completed deleting omz_models folder")
    