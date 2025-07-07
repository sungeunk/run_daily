import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from utils_aiml_intel.mem_logging_functions import MemLogger
from utils_aiml_intel.metrics import BenchmarkRecord
from utils_aiml_intel.result_logger import log_result
from utils_aiml_intel.setup_logging import (get_gtax_test_dir,
                                            update_log_details)
from utils_aiml_intel.tools import get_target_pip_package_version

parser = argparse.ArgumentParser()
parser.add_argument("--api", type=str, choices=["openvino", "openvino-nightly", "ipex", "cuda"], help="Define which API to run")
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
parser.add_argument("--model", type=str, default="whisper_large_v3")
args = parser.parse_args()


def extract_input_features(processor, sample):
    #Length of 30 second audio sample
    num_samples_30s = sample["audio"]["sampling_rate"]*30
    #Trimming the audio
    trimmed_audio = sample["audio"]["array"][:num_samples_30s]    
    input_features = processor(
            trimmed_audio,
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
            ).input_features
    return input_features


def measure_perf(model, processor, sample, n, api):
    timers = []
    # transcribed_texts = []
    if processor == "paraformer": # paraformer with in-built processing
        input_features = str(sample)
        print("Warm-up")
        output = model.generate(input_features)
        print("Measuring...")
        for _ in tqdm(range(n), desc="Measuring performance"):
            start = time.perf_counter()*1000
            output = model.generate(input_features)
            end = time.perf_counter()*1000
            timers.append(end - start)
            print(f"Translation took: {end - start}ms")
            transcription = output[0]['text']
            # transcribed_texts.append(transcription)
            print(transcription)

    else:
        input_features = extract_input_features(processor, sample)
        
        print("Warm-up")
        if api == "ipex":
            output = model.generate(input_features.half().to("xpu"))
        elif api == "cuda":
            output = model.generate(input_features.half().to("cuda"))
        else:
            output = model.generate(input_features.half())
        print("Measuring...")
        for _ in tqdm(range(n), desc="Measuring performance"):
            start = time.perf_counter()*1000
            if api == "ipex":
                output = model.generate(input_features.half().to("xpu"))
            elif api == "cuda":
                output = model.generate(input_features.half().to("cuda"))
            else:
                output = model.generate(input_features.half())
            end = time.perf_counter()*1000
            timers.append(end - start)
            print(f"Translation took: {end - start}ms")
            transcription = processor.batch_decode(output, skip_special_tokens=True)
            # transcribed_texts.append(transcription)
            print(transcription)
    return np.mean(timers), transcription


def main():
    update_log_details()
    test_path = get_gtax_test_dir()
    log_path = test_path / "logs"
    
    if args.model == "whisper_tiny":
        model_id = "openai/whisper-tiny"
    elif args.model == "whisper_base":
        model_id = "openai/whisper-base"
    elif args.model == "whisper_small":
        model_id = "openai/whisper-small"
    elif args.model == "whisper_medium":
        model_id = "openai/whisper-base"
    elif args.model == "whisper_large_v3":
        model_id = "openai/whisper-large-v3"
    elif args.model == "paraformer_large":
        model_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    elif args.model == "paraformer_zh":
        model_id = "paraformer-zh"

    # check if audiofile from artifactory exists
    cwd = Path.cwd()
    audio_file = cwd / "librispeech_long.wav"
    if audio_file.exists() and args.model == "paraformer_large" or args.model == "paraformer_zh":
        sample = audio_file
    else:
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]
        
    with MemLogger() as mem_session:
        if args.api == "openvino" or args.api == "openvino-nightly":
            if args.model != "paraformer_zh":
                from optimum.intel.openvino import OVModelForSpeechSeq2Seq

                device = "GPU" 
                processor = AutoProcessor.from_pretrained(model_id)

                model_path = Path(model_id.replace('/', '_'))
                ov_config = {"CACHE_DIR": ""}

                # if not model_path.exists():
                ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
                    model_id, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
                )
                ov_model.half()
                ov_model.save_pretrained(model_path)
            # else:
            pre_inference_start = time.perf_counter()
            if args.model == "paraformer_zh":
                import supplementary as sp
                from funasr import AutoModel
                punc_model = "ct-punc"
                ov_model = AutoModel(model=model_id, vad_model="fsmn-vad", punc_model=punc_model)
                ov_model.model = sp.NewSeacoParaformer(ov_model.model)
                ov_model.punc_model = sp.NewCTTransformer(ov_model.punc_model)
                processor = "paraformer"
                sample = f"{ov_model.model_path}/example/asr_example.wav"
            else:
                ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
                    model_path, ov_config=ov_config, compile=False
                )

                ov_model.to(device)
                ov_model.compile()
            pre_inference_stop = time.perf_counter()
            pre_inference_time = pre_inference_stop-pre_inference_start
            print(f"pre-inference time: {pre_inference_time}s")

            performance_metric, transcribed_text = measure_perf(ov_model, processor, sample, args.num_iter, args.api)
            print(f"Mean openvino {model_id} generation time: {performance_metric:.3f}ms")
        elif args.api == "ipex" or args.api == "cuda":
            if args.api == "ipex":
                import intel_extension_for_pytorch as ipex
                device = "xpu" # the device to load the model onto
            else:
                device = "cuda"

            pre_inference_start = time.perf_counter()
            
            if args.model == "paraformer_large":
                from funasr import AutoModel
                model = AutoModel(model=model_id, device=device)
                processor = "paraformer"
            else:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
                )
                model.to(device)
                processor = AutoProcessor.from_pretrained(model_id)
                
            pre_inference_stop = time.perf_counter()
            pre_inference_time = pre_inference_stop-pre_inference_start
            print(f"pre-inference time: {pre_inference_time}s")

            performance_metric, transcribed_text = measure_perf(model, processor, sample, args.num_iter, args.api)
            print(f"Mean torch {model_id} generation time: {performance_metric:.3f}ms")
        
    mem_session.print_summary()
        
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
    record.metrics.customized["Time to transcribe audio (ms)"] = round(performance_metric, 2)
    record.metrics.customized["Wall Clock throughput (tokens/s)"] = round(89 / performance_metric, 2)
    record.metrics.customized["Generated Output"] = transcribed_text
    record.metrics.customized["Peak GPU Memory Usage (Local) (GB)"] = mem_session.peak_gpu_memory_process_local
    record.metrics.customized["Peak GPU Memory Usage (Shared) (GB)"] = mem_session.peak_gpu_memory_process_shared
    record.metrics.customized["Peak GPU Memory Usage (Dedicated) (GB)"] = mem_session.peak_gpu_memory_process_dedicated
    record.metrics.customized["Peak GPU Memory Usage (NonLocal) (GB)"] = mem_session.peak_gpu_memory_process_nonlocal
    record.metrics.customized["Peak GPU Memory Usage (Committed) (GB)"] = mem_session.peak_gpu_memory_process_committed
    record.metrics.customized["Peak GPU Engine Usage (Compute) (%)"] = mem_session.peak_gpu_memory_engine_utilization_compute
    record.metrics.customized["Peak GPU Engine Usage (Copy) (%)"] = mem_session.peak_gpu_memory_engine_utilization_copy
    record.metrics.customized["Peak Sys Memory Usage (Committed) (GB)"] = mem_session.peak_sys_memory_committed_bytes
    record.metrics.customized["Peak Sys Memory Usage (Committed) (%)"] = mem_session.memory_percentage
    record.metrics.customized["Peak Sys Memory Limit (Committed) (GB)"] = mem_session.peak_sys_memory_commit_limit
    records.append(record)
        
    BenchmarkRecord.save_as_csv(log_path / filename, records)
    BenchmarkRecord.save_as_json(log_path / filename, records)
    BenchmarkRecord.save_as_txt(log_path / filename, records)
    log_result([record.to_dict() for record in records][0], log_path)


if __name__ == "__main__":
    main()