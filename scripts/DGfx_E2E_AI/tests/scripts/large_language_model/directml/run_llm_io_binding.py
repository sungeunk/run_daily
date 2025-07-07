# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This is an end-to-end benchmarking script for any ONNX model for Non-CUDA backends.
#
# Prerequisites: 
# 0) Install onnxruntime-genai and onnxruntime
#
# 1) Use builder.py to build the desired ONNX model
#
# 2) Run this script with the desired arguments. Run benchmark_e2e_DML.py -h for help.

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import psutil
from tqdm import tqdm
from utils_aiml_intel.mem_logging_functions import MemLogger
from utils_aiml_intel.metrics import BenchmarkRecord
from utils_aiml_intel.result_logger import log_result
from utils_aiml_intel.setup_logging import (get_gtax_test_dir,
                                            update_log_details)
from utils_aiml_intel.tools import get_target_pip_package_version

update_log_details()
test_path = get_gtax_test_dir()
log_path = test_path / "logs"

peak_cpu_memory = 0.0
peak_gpu_memory = 0.0
gpu_memory_command = r'(((Get-Counter "\GPU Process Memory(*)\Local Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
peak_memory_lock = threading.Lock()
stop_monitoring = False

try:
    subprocess.run(['powershell', '-Command', gpu_memory_command], check=True)
    IS_GPU_SYSTEM = True
except Exception:
    IS_GPU_SYSTEM = False

# Monitor the GPU memory usage
def monitor_gpu_memory():
    global peak_gpu_memory

    while not stop_monitoring:
        result = subprocess.run(['powershell', '-Command', gpu_memory_command], capture_output=True).stdout.decode("ascii")
        memory_usage = float(result.strip().replace(',', '.'))
        gpu_memory = memory_usage
        current_peak = round(gpu_memory / 1024**3, 2)
        print(f"current_peak: {current_peak}")
        with peak_memory_lock:
            peak_gpu_memory = max(current_peak, peak_gpu_memory)
        time.sleep(0.1)


# Monitor the CPU memory usage
def monitor_cpu_memory():
    global peak_cpu_memory

    while not stop_monitoring:
        current_used_memory = psutil.virtual_memory().used
        with peak_memory_lock:
            peak_cpu_memory = round(max(peak_cpu_memory, current_used_memory) / 1024**3, 2)
        time.sleep(0.1)


def get_target_pip_package_version(target_pip_package_name_list):
    # get package name and version
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [
            f"{i.key}=={i.version}"
            for i in installed_packages
            if i.key in target_pip_package_name_list
        ]
    )

    pkg_name = ""
    pkg_version = ""
    if installed_packages_list:
        pkg_name = installed_packages_list[0].split("==")[0]
        pkg_version = installed_packages_list[0].split("==")[1]
    return pkg_name, pkg_version

def get_model_info_from_genai_config(model_input_folder):
    genai_config_file_path = os.path.join(model_input_folder, "genai_config.json")
    genai_config_file = open(genai_config_file_path)
    genai_config = json.load(genai_config_file)
    model_info = {}
    model_info['context_length'] = genai_config['model']['context_length']
    model_info["execution_provider"] = "cpu"
    if len(genai_config["model"]["decoder"]["session_options"]["provider_options"]) > 0:
        model_info["execution_provider"] = list(genai_config["model"]["decoder"]["session_options"]["provider_options"][0].keys())[0]
    genai_config_file.close()
    return model_info

def save_results(args, results, filename, print_memory_usage=False):
    import pandas as pd

    columns=[
    "Batch Size",
    "Prompt Length",
    "Tokens Generated",
    "Max Length",
    "Pre Inference Time (s)", 
    "1st Token Generation Latency (ms)",
    "2nd+ Token Generation Latency (ms)",
    "2nd+ Token Generation Throughput (tps)",
    "Wall Clock Throughput (tps)",
    "Wall Clock Time (s)",
    "Generated Output",
    "Peak GPU Memory Usage (Local) (GB)",
    "Peak GPU Memory Usage (Shared) (GB)",
    "Peak GPU Memory Usage (Dedicated) (GB)",
    "Peak GPU Memory Usage (NonLocal) (GB)",
    "Peak GPU Memory Usage (Committed) (GB)",
    "Peak GPU Engine Usage (Compute) (%)",
    "Peak GPU Engine Usage (Copy) (%)",
    "Peak Sys Memory Usage (Committed) (GB)",
    "Peak Sys Memory Usage (Committed) (%)",
    "Peak Sys Memory Limit (Committed) (GB)"
    ]

    if print_memory_usage:
        if IS_GPU_SYSTEM:
            columns.append("peak_gpu_memory (GiB)")
        else:
            columns.append("peak_cpu_memory(GiB)")

    df = pd.DataFrame(
        results,
        columns=columns,
    )
    # df = df.transpose()  # This line swaps the rows and columns
    
    genai_package_name, genai_package_version = get_target_pip_package_version(["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime-genai-directml"])
    model_info = get_model_info_from_genai_config(args.input_folder)
    
    records = []
    for _, row in df.iterrows():
        record = BenchmarkRecord(args.input_folder, args.precision, "onnxruntime-genai", model_info["execution_provider"], genai_package_name, genai_package_version )
        record.config.batch_size = row["Batch Size"]
        record.config.customized["Prompt Length"] = row["Prompt Length"]
        record.config.customized["Tokens Generated"] = row["Tokens Generated"]
        record.config.customized["Max Length"] = row["Max Length"]
        record.metrics.customized["Pre Inference Time (s)"] = row["Pre Inference Time (s)"]
        record.metrics.customized["First token latency (ms)"] = row["1st Token Generation Latency (ms)"]
        record.metrics.customized["Second+ token latency (ms)"] = row["2nd+ Token Generation Latency (ms)"]
        record.metrics.customized["Second+ token throughput (tokens/s)"] = row["2nd+ Token Generation Throughput (tps)"]  
        record.metrics.customized["Wall Clock throughput (tokens/s)"] = row["Wall Clock Throughput (tps)"]
        record.metrics.customized["Wall Clock Time (s)"] = row["Wall Clock Time (s)"]
        record.metrics.customized["Generated Output"] = row["Generated Output"]
        record.metrics.customized["Peak GPU Memory Usage (Local) (GB)"] = row["Peak GPU Memory Usage (Local) (GB)"]
        record.metrics.customized["Peak GPU Memory Usage (Shared) (GB)"] = row["Peak GPU Memory Usage (Shared) (GB)"]
        record.metrics.customized["Peak GPU Memory Usage (Dedicated) (GB)"] = row["Peak GPU Memory Usage (Dedicated) (GB)"]
        record.metrics.customized["Peak GPU Memory Usage (NonLocal) (GB)"] = row["Peak GPU Memory Usage (NonLocal) (GB)"]
        record.metrics.customized["Peak GPU Memory Usage (Committed) (GB)"] = row["Peak GPU Memory Usage (Committed) (GB)"]
        record.metrics.customized["Peak GPU Engine Usage (Compute) (%)"] = row["Peak GPU Engine Usage (Compute) (%)"]
        record.metrics.customized["Peak GPU Engine Usage (Copy) (%)"] = row["Peak GPU Engine Usage (Copy) (%)"]
        record.metrics.customized["Peak Sys Memory Usage (Committed) (GB)"] = row["Peak Sys Memory Usage (Committed) (GB)"]
        record.metrics.customized["Peak Sys Memory Usage (Committed) (%)"] = row["Peak Sys Memory Usage (Committed) (%)"]
        record.metrics.customized["Peak Sys Memory Limit (Committed) (GB)"] = row["Peak Sys Memory Limit (Committed) (GB)"]
        if print_memory_usage:
            if IS_GPU_SYSTEM:
                record.metrics.customized["Peak GPU Memory (GB)"] = row["peak_gpu_memory (GiB)"]
            else:
                record.metrics.customized["Peak CPU Memory (GB)"] = row["peak_cpu_memory(GiB)"]
        
        records.append(record)
        
    BenchmarkRecord.save_as_csv(log_path / filename, records)
    BenchmarkRecord.save_as_json(log_path / filename, records)
    BenchmarkRecord.save_as_txt(log_path / filename, records)
    log_result([record.to_dict() for record in records][0], log_path)
    

def run_benchmark_memory(args, batch_size, prompt_length, generation_length, max_length):
    """
    This function is to run benchmark and print the momory usage
    """
    global stop_monitoring

    if IS_GPU_SYSTEM:
        monitor_thread = threading.Thread(target=monitor_gpu_memory)
    else:
        monitor_thread = threading.Thread(target=monitor_cpu_memory)
    
    monitor_thread.start()

    metrics = run_benchmark(args, batch_size, prompt_length, generation_length, max_length)

    stop_monitoring = True
    monitor_thread.join()

    if IS_GPU_SYSTEM:
        metrics.append(peak_gpu_memory)
    else:
        metrics.append(peak_cpu_memory)
    
    return metrics

def run_benchmark(args, batch_size, prompt_length, generation_length, max_length):

    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Get tokenizer, and model
    model_load_start_time = time.perf_counter()
    if args.verbose: print("Loading model... ")
    model=og.Model(f'{args.input_folder}')
    if args.multimodal:
        processor = model.create_multimodal_processor()
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)

 
    # Generate prompt
    if args.input_file:
        with open(args.input_file) as f:
            data = f.read()
            prompt = [data.replace("\n", " ")]
            print(prompt)
    elif args.task:
        pass # prompt read inside next multimodal block
    else:
        prompt_path = test_path / "temp" / "prompts" / str(args.input_folder)
        prompt_path_list = [i for i in prompt_path.glob("*.txt")]

        selected_prompt_file = ""
        for i in prompt_path_list:
            if str(args.prompt_lengths[0]) in str(i.name):
                selected_prompt_file = i
            
        assert selected_prompt_file != "", "Selected input token size invalid"
        print(selected_prompt_file)
        data = open(selected_prompt_file).read()
        prompt = [data.replace("\n", " ")]
        print(prompt)
  
    if args.multimodal:
        if args.task:
            cwd = Path.cwd()
            task_asset_path = cwd / "input_assets"
            task_assets = [i for i in task_asset_path.glob(f"{args.task}*")]
            assert len(task_assets) == 2, f"Task asset length != 2. {task_assets}"
            for asset in task_assets:
                if asset.suffix == ".txt":
                    prompt_path = asset
                elif asset.suffix == ".png":
                    image_path = asset
                else:
                    raise RuntimeError(f"Unsupported asset extension {asset.suffix}. Please review asset: {asset}")

            with open(prompt_path) as f:
                data = f.read()
                prompt = [data.replace("\n", " ")]
            #prompt = "<|image_1|>\n" + prompt[0]
            prompt = "<|user|>\n<|image_1|>\n" + prompt[0] + "<|end|>\n<|assistant|>\n"

            print(f"Predefined task selected: {args.task}")
            print(f"Image: {image_path}")
            print(f"Prompt: {prompt}")

            image = og.Images.open(str(image_path))
            inputs = processor(prompt, images=image)
            
        else:
            image = og.Images.open(args.image_path)
            inputs = processor(prompt, images=image)
    else:
        tokens = tokenizer.encode_batch(prompt)
        tokens = [tk[0:prompt_length] for tk in tokens]
        print("Prompt length = ", len(tokens[0]))

    params = og.GeneratorParams(model)
    if args.multimodal:
        params.set_inputs(inputs)
    else:
        params.input_ids = tokens
    do_sample = args.top_k > 1 or (args.top_p != 1.0 and args.top_p > 0.0)
    params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length, random_seed=1337)

    if args.use_graph_capture:
        params.try_graph_capture_with_max_batch_size(batch_size)
    model_load_end_time = time.perf_counter()

    tokenize_times = []
    prompt_times = []
    first_token_gen_times = []
    token_gen_times = []
    sampling_times = []
    wall_clock_times = []
    if args.verbose: print(f"Running benchmark for batch size = {batch_size}, prompt length = {prompt_length}")
    with MemLogger() as mem_session:
        for _ in tqdm(range(args.warmup + num_repetitions)):
            wall_clock_start_time = time.time()

            # Measure tokenization[Not enabled]
            if args.multimodal:
                tokenize_start_time = time.perf_counter()
                inputs = processor(prompt, images=image)
                tokenize_end_time = time.perf_counter()
                tokenize_times.append(tokenize_end_time - tokenize_start_time)
            else:
                tokenize_start_time = time.perf_counter()
                tokens = tokenizer.encode_batch(prompt)
                tokenize_end_time = time.perf_counter()
                tokenize_times.append(tokenize_end_time - tokenize_start_time)
            
                tokens = [tk[0:prompt_length] for tk in tokens] # cut specific length of tokens from all tokens

            params = og.GeneratorParams(model)
            if args.multimodal:
                tokenizer_stream = processor.create_stream()
                inputs = processor(prompt, images=image)
                params.set_inputs(inputs)
            else:
                params.input_ids = tokens
            params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length, random_seed=1337)

            generator = og.Generator(model, params)

            # Measure prompt processing
            prompt_start_time = time.perf_counter()
            generator.compute_logits()
            prompt_end_time = time.perf_counter()
            if _ > args.warmup - 1:
                prompt_times.append(prompt_end_time - prompt_start_time)

            sampling_start_time = time.perf_counter()
            generator.generate_next_token()
            sampling_end_time = time.perf_counter()
            sampling_times.append(sampling_end_time - sampling_start_time)

            # Measure token generation
            i = 1
            output_text = []
            print("=" * 10, "Start generation", "=" * 10)
            while not generator.is_done() and i < generation_length:
                # Run inference
                token_gen_start_time = time.perf_counter()
                generator.compute_logits()
                token_gen_end_time = time.perf_counter()

                sampling_start_time = time.perf_counter()
                generator.generate_next_token()
                sampling_end_time = time.perf_counter()
                
                token_gen_times.append(token_gen_end_time - token_gen_start_time)
                sampling_times.append(sampling_end_time - sampling_start_time)
                i += 1
                
                if args.print_model_output:
                    new_token = generator.get_next_tokens()[0]
                    try:
                        if args.multimodal:
                            output = tokenizer_stream.decode(new_token)
                        else:
                            output = tokenizer.decode(new_token)                    
                        print(output, end='', flush=True)
                        output_text.append(output)
                    except UnicodeDecodeError:
                        print("Skipping unicode decode error")

            wall_clock_end_time = time.time()
            wall_clock_times.append(wall_clock_end_time - wall_clock_start_time)
            print(" ")
            print("=" * 10, "End generation", "=" * 10)

            # if args.print_model_output: print(tokenizer.decode(generator.get_sequence(0)))

            # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
            del generator
            
    mem_session.print_summary()

    # Calculate tokenization metrics
    avg_tokenization_latency_s = sum(tokenize_times) / len(tokenize_times)
    avg_tokenization_latency_ms = avg_tokenization_latency_s * 1000
    avg_per_token_tokenization_latency_ms = avg_tokenization_latency_ms / prompt_length
    avg_tokenization_thrpt = batch_size * (1000 / avg_per_token_tokenization_latency_ms)

    # Calculate prompt processing metrics
    avg_prompt_latency_s = sum(prompt_times) / len(prompt_times)
    avg_prompt_latency_ms = round(avg_prompt_latency_s * 1000, 2)
    avg_per_token_prompt_latency_ms = avg_prompt_latency_ms / prompt_length
    avg_per_token_prompt_thrpt = batch_size * (1000 / avg_per_token_prompt_latency_ms)
    print(f"All 1st token latency {prompt_times}")
    print(f"Length 1st token latency {len(prompt_times)}")
    print(f"Average 1st Token Latency: {avg_prompt_latency_ms} ms")
    
    # COMBINING THE TOKENIZATION AND PROMPT PROCESSING TO A SINGLE PRE-INFERENCE TIMING
    avg_per_token_pre_inference_latency_ms = round((((model_load_end_time - model_load_start_time) * 1000) + (avg_per_token_tokenization_latency_ms + avg_per_token_prompt_latency_ms) / 2) / 1000, 2)
    print(f"Pre Inference Time: {avg_per_token_pre_inference_latency_ms} s")
    print()
    
    # Calculate token generation input prep metrics
    avg_token_gen_latency_s = sum(token_gen_times) / len(token_gen_times)
    avg_token_gen_latency_ms = round(avg_token_gen_latency_s * 1000, 2)
    avg_token_gen_thrpt = round(batch_size * (1 / avg_token_gen_latency_s), 2)
    print(f"Length 2nd+ token latency {len(token_gen_times)}")
    print(f"Average 2nd+ Token Latency (per token): {avg_token_gen_latency_ms} ms")
    print(f"Average 2nd+ Token Throughput (per token): {avg_token_gen_thrpt} tokens/s")
    print()
    
    # Calculate wall clock time
    avg_wall_clock_time = round(sum(wall_clock_times) / len(wall_clock_times), 2)
    avg_wall_clock_thrpt = round(batch_size * (max_length / avg_wall_clock_time), 2)
    print(f"Average Wall Clock Time: {avg_wall_clock_time} s")
    print(f"Average Wall Clock Throughput: {avg_wall_clock_thrpt} tokens/s")
    
    generated_output = "".join(output_text)

    metrics = [
        batch_size, # batch size
        prompt_length, # prompt length
        generation_length, # tokens generated
        max_length, # max length
        avg_per_token_pre_inference_latency_ms, # pre-inference (s)
        avg_prompt_latency_ms, # 1st token latency (ms)
        avg_token_gen_latency_ms, # 2+ token latency (ms)
        avg_token_gen_thrpt, # 2+ token throughput (tokens/s)
        avg_wall_clock_thrpt, # wall clock throughput (tokens/s)
        avg_wall_clock_time, # wall clock time (s)
        generated_output, # last iteration generated output
        mem_session.peak_gpu_memory_process_local,
        mem_session.peak_gpu_memory_process_shared,
        mem_session.peak_gpu_memory_process_dedicated,
        mem_session.peak_gpu_memory_process_nonlocal,
        mem_session.peak_gpu_memory_process_committed,
        mem_session.peak_gpu_memory_engine_utilization_compute,
        mem_session.peak_gpu_memory_engine_utilization_copy,
        mem_session.peak_sys_memory_committed_bytes,
        mem_session.memory_percentage,
        mem_session.peak_sys_memory_commit_limit
    ]
    return metrics


def main(args):
    all_csv_metrics = []
    
    for batch_size in args.batch_sizes:
        for l, prompt_length in enumerate(args.prompt_lengths):
            for g, gen_length in enumerate(args.generation_lengths):
                if args.max_lengths:
                    m = l * len(args.generation_lengths) + g
                    max_length = args.max_lengths[m]
                else:
                    max_length = prompt_length + gen_length
                model_info = get_model_info_from_genai_config(args.input_folder)
                if model_info['context_length'] < max_length:
                    context_length = model_info['context_length']
                    raise Exception(f"Input Prompt Length {args.prompt_lengths} + Generated Output Length {args.generation_lengths} shouldn't be greater than Model Context Length {context_length}" )
                print(f"Args: batch_size = {batch_size}, prompt_length = {prompt_length}, tokens = {gen_length}, max_length = {max_length}")
                if args.print_memory_usage:
                    metrics = run_benchmark_memory(args, batch_size, prompt_length, gen_length, max_length)
                else:
                    metrics = run_benchmark(args, batch_size, prompt_length, gen_length, max_length)
                all_csv_metrics.append(metrics)
    # Add metrics to CSV
    if args.verbose: print("Adding results to CSV")
    filename = args.output

    if args.print_memory_usage:
        if IS_GPU_SYSTEM:
            print(f"-------------------* Peak GPU Memory Usage: {peak_gpu_memory} GiB *-------------------")
        else:
            print(f"-------------------* Peak CPU Memory Usage: {peak_cpu_memory} GiB *-------------------")
        save_results(args, all_csv_metrics, filename, print_memory_usage=True)
    else:
        save_results(args, all_csv_metrics, filename)

def str2intlist(value):
    return [int(v) for v in value.split(',')]

def str2strlist(value):
    return [str(v) for v in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-im', '--image_path', type=str, help='Path to the image')
    parser.add_argument('-mlm', '--multimodal', action='store_true', help='Enable/disable Multimodal')
    parser.add_argument('-t', '--task', type=str, help='Fixed task list coverage', choices=['summarization', 'localization', 'ocr', 'multi_object_detection'])
    parser.add_argument('-b', '--batch_sizes', type=str2intlist, default=[1], help='Number of sequences to generate in parallel')
    parser.add_argument('-l', '--prompt_lengths', type=str2intlist, required=True, help='Number of tokens for prompt')
    parser.add_argument('-g', '--generation_lengths', type=str2intlist, required=True, help='Number of tokens to generate after prompt')
    parser.add_argument('-m', '--max_lengths', type=str2intlist, default=[], help='Max length buffer sizes... User should supply one for every combination of Prompt and Generation length')
    parser.add_argument('-r', '--repetitions', type=int, default=10, help='Number of times to repeat the benchmark')
    parser.add_argument('-w', '--warmup', type=int, default=5, help='Number of warmup runs before benchmarking')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-p', '--top_p', type=float, default=1.0, help='Top p probability to sample with')
    parser.add_argument('-o', '--output', type=str, default='execution_results.csv', help='Output CSV file name or path (with .csv extension)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print extra information')
    parser.add_argument('-mo', '--print_model_output', action='store_true', help='Print model output')
    parser.add_argument('-pm', '--print_memory_usage', default=False, help='Print memory footprint')
    parser.add_argument('-gc', '--use_graph_capture', action='store_true', help='Use the graph capture feature for CUDA or DML')
    parser.add_argument('-mn', '--model_name', type=str, default='model_name', help='Model name defined by users')
    parser.add_argument('-pr', '--precision', type=str, default='fp16', help='Model precision for metrics info')
    parser.add_argument('-if', '--input_file', type=str, help='File to load sample prompts from')
    args = parser.parse_args()
    main(args)
