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
parser.add_argument("--model", type=str, choices=["resnet50"], help="Define LLM model to use")
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
parser.add_argument("--api", type=str, choices=["openvino", "openvino-nightly", "ipex", "cuda", "directml"], help="Define which API to run")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--precision", type=str, default="fp16", help="Define precision to run. Currently tested on OV.")
args = parser.parse_args()

import numpy as np
import torch
from transformers import ResNetForImageClassification

update_log_details()
test_path = get_gtax_test_dir()
log_path = test_path / "logs"


if args.api == "ipex":
    import intel_extension_for_pytorch as ipex
    device = "xpu"
elif args.api == "cuda":
    device = "cuda"
elif args.api == "directml":
    import onnxruntime
else:
    device = "GPU"

pre_inference_start = time.perf_counter()

if args.model == "resnet50":
    model_id = "microsoft/resnet-50"
    if args.api == "openvino" or args.api == "openvino-nightly":
        from optimum.intel.openvino import OVModelForImageClassification
        if args.precision == "int8":
            model = OVModelForImageClassification.from_pretrained(model_id, device="GPU", export=True, compile=False, load_in_8bit=True)
        else:
            model = OVModelForImageClassification.from_pretrained(model_id, device="GPU", export=True, compile=False)
        model.reshape(args.batch_size, 3, 224, 224)
        model.compile()
    elif args.api == "directml":
        model = onnxruntime.InferenceSession("onnx/model.onnx", providers=['DmlExecutionProvider'])
    else:
        model = ResNetForImageClassification.from_pretrained(model_id)
        model.half()
        model.to(device)

pre_inference_stop = time.perf_counter()
pre_inference_time = pre_inference_stop - pre_inference_start

def measure_perf(model, n=4):
    fps = []
    latency = []
    input = torch.randn((args.batch_size, 3, 224, 224))
    print(f"Input dimension: {input.shape}")
    for run in range(n):
        start = time.perf_counter()
        img_idx = 0
        while time.perf_counter() - start < 15:
            latency_run = []
            if args.api == "directml":
                start_latency = time.perf_counter()
                ort_outputs = model.run([], {'pixel_values':input.numpy()})[0]
                stop_latency = time.perf_counter()
            else:
                with torch.no_grad():
                    start_latency = time.perf_counter()
                    if args.api == "openvino" or args.api == "openvino-nightly":
                        logits = model(input.half()).logits
                    else:
                        logits = model(input.half().to(device)).logits
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
elif args.api == "directml":
    package_name, package_version = get_target_pip_package_version(["onnxruntime-directml"])

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