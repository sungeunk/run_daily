import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
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

cwd = Path.cwd()
# if running via gtax
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

models = ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-xl-base-1.0"]
if args.model == "gpt2":
    model_id = "gpt2"
elif args.model == "v2.0":
    model_id = models[1]
elif args.model == "v2.1":
    model_id = models[2]
elif args.model == "xl":
    model_id = models[3]
else:
    print("Invalid checkpoint defined")
    logger.info("Invalid checkpoint defined")
    sys.exit(1)


run_dict = {}
parameter = [(256,128), (512,128), (768,128), (1024,128), (256,256), (512,256), (768,256), (1024,256)]
precision_list = ["FP16", "FP32"]
input_text = "Describe the solar system in great detailDescribe the solar system in great detail"

try:
    for precision in precision_list:
        for param in parameter:
            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"] = {}
            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["average"] = 0
            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["average_exclude"] = 0
            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["runs"] = []

            for _ in range(args.num_iter):
                seq, gen = param
                print(f"Starting with model_id: {model_id}, ({seq}x{gen})")
                logger.info(f"Starting with model_id: {model_id}, ({seq}x{gen})")
                result = subprocess.run(["python", "gpt2_text_prediction_demo.py", f"--model=public\\gpt-2\\{precision}\\gpt-2.xml", "--vocab=public\\gpt-2\\gpt2\\vocab.json", "--merges=public\\gpt-2\\gpt2\\merges.txt", "--max_seq_len", str(seq), "--max_sample_token_num", str(gen), "-d", "GPU", "-i", input_text], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)         
                contents = result.stdout
                init_text = "requests were processed in"
                temp_idx = contents.find(init_text)
                temp_front_idx = contents[temp_idx-50:temp_idx+50].find("[ INFO ]")
                temp_back_idx = contents[temp_idx-50:temp_idx+50][temp_front_idx:].find(init_text)
                front_text = contents[temp_idx-50:temp_idx+50][temp_front_idx:][:len(init_text)+temp_back_idx]
                back_text = "per request)"
                back_length = len(back_text)
                front_idx = contents.find(front_text)
                back_idx = contents[front_idx:].find(back_text)
                full_text = contents[front_idx:front_idx+back_idx+back_length]
                perf_front = full_text.find("(")
                perf_back = full_text.find("sec per request")
                performance = np.round(1/float(full_text[perf_front+1:perf_back]),2)
                time_front = full_text.find("in ")
                time_back = full_text.find("sec (")
                time_taken = float(full_text[time_front+3:time_back])
                

                run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["runs"].append(performance)

                print(f"Completed with model_id: {model_id}, {precision}_seq{param[0]}_gen{param[1]}, Time Taken: {time_taken}, Performance: {performance} t/s")
                logger.info(f"Completed with model_id: {model_id}, {precision}_seq{param[0]}_gen{param[1]}, Time Taken: {time_taken}, Performance: {performance} t/s")

            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["average"] = np.round(np.average(run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["runs"]),2)
            run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["average_exclude"] = np.round(np.average(run_dict[f"{precision}_seq{param[0]}_gen{param[1]}"]["runs"][1:]),2)

        with open(result_json, "w") as f:
            json.dump(run_dict, f)

except Exception as e:
    print(f"Encountered error: {e}")
    logger.critical(f"Encountered error: {e}")

    sys.exit(1)

finally:
    print("Test Completed")
    logger.info("Test Completed")

