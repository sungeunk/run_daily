import argparse
import json
import sys
import time
from pathlib import Path

from utils_aiml_intel.setup_logging import (get_gtax_test_dir,
                                            update_log_details)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["bert"], help="Define LLM model to use")
parser.add_argument("--input_token", type=str, default=256)
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
parser.add_argument("--api", type=str, choices=["openvino", "ipex", "cuda"], help="Define which API to run")
args = parser.parse_args()

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

update_log_details()
test_path = get_gtax_test_dir()
log_path = test_path / "logs"

prompt_path = test_path / "Foundational_Models" / "Large_Language_Models" / "prompt"
prompt_path_list = [i for i in prompt_path.glob("*.txt")]

selected_prompt_file = ""
for i in prompt_path_list:
    if str(args.input_token) in str(i):
        selected_prompt_file = i
    
assert selected_prompt_file != "", "Selected input token size invalid"

prompt = open(selected_prompt_file).read()

if args.api == "ipex":
    import intel_extension_for_pytorch as ipex
    device = "xpu"
elif args.api == "cuda":
    device = "cuda"
else:
    device = "GPU"
    
if args.model == "bert":
    model_id = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_id)
    if args.api != "openvino":
        model = BertModel.from_pretrained(model_id)
        model.half()
        model.to(device)
    else:
        from optimum.intel.openvino import OVModelForMaskedLM
        model_path = Path(model_id.replace('/', '_'))
        ov_config = {"CACHE_DIR": ""}

        if not model_path.exists():
            model = OVModelForMaskedLM.from_pretrained(
                model_id, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
            )
            model.half()
            model.save_pretrained(model_path)
        else:
            model = OVModelForMaskedLM.from_pretrained(
                model_path, ov_config=ov_config, compile=False
            )

        model.to(device)
        model.compile()

encoded_input = tokenizer(prompt, return_tensors='pt')

def measure_perf(model, n=4):
    fps = []
    avg_latency = []
    first_latency = []
    subsequent_latency = []
    for run in range(n):
        start = time.perf_counter()
        img_idx = 0
        first_latency_run = 0
        while time.perf_counter() - start < 15:
            latency_run = []
            subsequent_latency_run = []
            with torch.no_grad():
                start_latency = time.perf_counter()
                if args.api == "openvino":
                    logits = model(**encoded_input)
                else:
                    encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
                    encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
                    encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
                    logits = model(**encoded_input)
                stop_latency = time.perf_counter()
            img_latency = np.round((stop_latency-start_latency)*1000,2)
            if img_idx == 0:
                first_latency_run = img_latency
            else:
                subsequent_latency_run.append(img_latency)
            latency_run.append(img_latency)
            img_idx += 1
            

        if run == 0:
            print(f"Warm-up run {run} completed with FPS: {img_idx}, Latency: {np.average(latency_run)} ms")
        else:
            print(f"Run {run} completed with FPS: {img_idx}, Latency: {np.average(latency_run)} ms")
            fps.append(img_idx)
            first_latency.append(first_latency_run)
            subsequent_latency.append(np.average(subsequent_latency_run))
            avg_latency.append(np.average(latency_run))
    return np.round(np.mean(fps),2), np.round(np.mean(avg_latency),2), np.round(np.mean(first_latency),2), np.round(np.mean(subsequent_latency),2)
    
perf_fps, perf_latency, first_token_latency, subsequent_token_latency = measure_perf(model, args.num_iter)

print(f"Mean FPS: {perf_fps}, Mean latency: {perf_latency}, Mean first token latency: {first_token_latency}, Mean second+ latency: {subsequent_token_latency}")

dic = {}
dic["mean_fps"] = perf_fps
dic["perf_latency"] = perf_latency
dic["first_token_latency"] = first_token_latency
dic["subsequent_token_latency"] = subsequent_token_latency

result_json = log_path / "execution_results.json"
with open(result_json, "w") as f:
    json.dump(dic, f)

from result_logger import log_result
log_result(dic, log_path)

import pprint

summary_data = pprint.pformat(dic)
summary_path = log_path / "summary_output.txt"
with open(summary_path, "w") as f:
    f.write(summary_data)
