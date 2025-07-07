import argparse
import json
import sys
import time
from pathlib import Path

from utils_aiml_intel.mem_logging_functions import MemLogger
from utils_aiml_intel.metrics import BenchmarkRecord
from utils_aiml_intel.result_logger import log_result
from utils_aiml_intel.setup_logging import (get_gtax_test_dir,
                                            update_log_details)
from utils_aiml_intel.tools import get_target_pip_package_version

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["openpose"], help="Define LLM model to use")
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
parser.add_argument("--api", type=str, choices=["ipex", "cuda"], help="Define which API to run")
args = parser.parse_args()

import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from controlnet_aux.processor import Processor
from datasets import load_dataset
from diffusers.utils import load_image
from torchvision.transforms import Compose, Resize
from torchvision.transforms.functional import pil_to_tensor

update_log_details()
test_path = get_gtax_test_dir()
log_path = test_path / "logs"

if args.api == "ipex":
    import intel_extension_for_pytorch as ipex
    device = "xpu"
elif args.api == "cuda":
    device = "cuda"
else:
    device = "GPU"

pre_inference_start = time.perf_counter()

if args.model == "openpose":
    model_id = "lllyasviel/Annotators"
    # processor = Processor("openpose")
    if args.api != "openvino":
        model = OpenposeDetector.from_pretrained(model_id)
        # model.half()
        model.to(device)
    else:
        from optimum.intel.openvino import OVModelForImageClassification
        model_id = "lllyasviel/sd-controlnet-openpose"
        model_path = Path(model_id.replace('/', '_'))
        ov_config = {"CACHE_DIR": ""}

        if not model_path.exists():
            model = OVModelForImageClassification.from_pretrained(
                model_id, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
            )
            model.half()
            model.save_pretrained(model_path)
        else:
            model = OVModelForImageClassification.from_pretrained(
                model_path, ov_config=ov_config, compile=False
            )

        model.to(device)
        model.compile()

pre_inference_stop = time.perf_counter()
pre_inference_time = pre_inference_stop - pre_inference_start

_transforms = Compose([Resize(512)])

def transforms(examples):
    examples["pixel_values"] = pil_to_tensor(_transforms(examples['image'][0]))
    return examples

dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
dataset.set_transform(transforms)

def measure_perf(model, n=4):
    fps = []
    latency = []
    for run in range(n):
        start = time.perf_counter()
        img_idx = 0
        while time.perf_counter() - start < 15:
            latency_run = []
            input = dataset[img_idx]['pixel_values']
            with torch.no_grad():
                start_latency = time.perf_counter()
                if args.api == "openvino":
                    logits = model(input.unsqueeze(0).half()).logits
                else:
                    logits = model(input)
                stop_latency = time.perf_counter()
            img_latency = stop_latency-start_latency
            latency_run.append(np.round(img_latency*1000,2))
            img_idx += 1
            

        if run == 0:
            print(f"Warm-up run {run} completed with FPS: {img_idx}, Latency: {np.average(latency_run)} ms")
        else:
            print(f"Run {run} completed with FPS: {img_idx}, Latency: {np.average(latency_run)} ms")
            fps.append(img_idx)
            latency.append(np.average(latency_run))
    return np.round(np.mean(fps),2), np.round(np.mean(latency),2)
    
perf_fps, perf_latency = measure_perf(model, args.num_iter)

print(f"Mean FPS: {perf_fps}, Mean latency: {perf_latency}")

filename = "execution_results.csv"

if args.api == "ipex":
    package_name, package_version = get_target_pip_package_version(["intel_extension_for_pytorch"])
elif args.api == "openvino" or args.api == "openvino-nightly":
    package_name, package_version = get_target_pip_package_version(["openvino"])
elif args.api == "cuda":
    package_name, package_version = get_target_pip_package_version(["torch"])

records = []
record = BenchmarkRecord(model_id, "fp16", package_name, "gpu", package_name, package_version)
record.config.batch_size = 1
record.config.customized["Warm Up"] = 1
record.config.customized["Iteration"] = args.num_iter
record.metrics.customized["Pre Inference Time (s)"] = round(pre_inference_time,2)
record.metrics.customized["Frames Per Second (FPS)"] = round(perf_fps, 2)
record.metrics.customized["Latency (ms)"] = round(perf_latency, 2)
records.append(record)
    
BenchmarkRecord.save_as_csv(log_path / filename, records)
BenchmarkRecord.save_as_json(log_path / filename, records)
BenchmarkRecord.save_as_txt(log_path / filename, records)
log_result([record.to_dict() for record in records][0], log_path)