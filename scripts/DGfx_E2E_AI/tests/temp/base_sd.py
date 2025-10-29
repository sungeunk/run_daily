import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from utils_aiml_intel.generated_image_validators import blank_image_validator
from utils_aiml_intel.mem_logging_functions import MemLogger
from utils_aiml_intel.metrics import BenchmarkRecord
from utils_aiml_intel.result_logger import log_result
from utils_aiml_intel.setup_logging import (get_gtax_test_dir,
                                            update_log_details)
from utils_aiml_intel.tools import get_target_pip_package_version

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="GPU", help="device")
parser.add_argument("--model_root", type=str, help="model store path")
parser.add_argument("--model", type=str, help="Define SD model to use")
parser.add_argument("--feature", type=str, default="txt2img", help="Define txt2img, img2img, upscale")
parser.add_argument("--num_iter", type=int, default=4, help="Define number of iterations to run")
parser.add_argument("--num_warm", type=int, default=1, help="Define number of warm-up iterations")
parser.add_argument("--api", type=str, help="Define backend API to use")
parser.add_argument("--prompt", type=str, help="Define prompt to generate", default="sailing ship in storm by Rembrandt")
parser.add_argument("--height", type=int, help="Define the image height dimension")
parser.add_argument("--width", type=int, help="Define the image width dimension")
parser.add_argument("--batch_size", type=int, default=1, help="Define the number of batch images for inference")
parser.add_argument("--precision", type=str, default="fp16", help="Define precision to run. Currently tested on OV.")
parser.add_argument("--demo", action="store_true")
parser.add_argument("--performance_target", type=float, help="Define target performance in seconds per image")
args = parser.parse_args()


def get_inputs(prompt, batch_size=1, num_inference_steps=20, guidance_scale=7.5, image=None):
    if args.api == "ipex":
        device = "xpu"
    elif args.api == "cuda":
        device = "cuda"
    generator = [torch.Generator(device).manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps, "height": args.height, "width": args.width, "guidance_scale": guidance_scale, "image": image}   

def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    total_imgs = len(imgs)
    grid_length = np.ceil(np.sqrt(total_imgs)).astype('int')
    grid = Image.new('RGB', size=(grid_length*w, grid_length*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%grid_length*w, i//grid_length*h))
    return grid  

def main():
    update_log_details()
    test_path = get_gtax_test_dir()
    log_path = test_path / "logs"

    if args.model == "v1.5":
        model_id = "nmkd/stable-diffusion-1.5-fp16"
        num_steps = 20
        gscale = 7.5
    elif args.model == "v2.0":
        model_id = "stabilityai/stable-diffusion-2"
        num_steps = 20
        gscale = 7.5
    elif args.model == "v2.1":
        model_id = "stabilityai/stable-diffusion-2-1"
        num_steps = 20
        gscale = 7.5
    elif args.model == "v3.0":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        num_steps = 20
        gscale = 7.5
    elif args.model == "lcm":
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        num_steps = 4
        gscale = 7.5
    elif args.model == "xl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        num_steps = 20
        gscale = 7.5
    elif args.model == "turbo":
        model_id = "stabilityai/sd-turbo"
        num_steps = 1
        gscale = 0.0
    elif args.model == "xl_turbo":
        model_id = "stabilityai/sdxl-turbo"
        num_steps = 1
        gscale = 0.0
    elif args.model == "xl_lcm":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        num_steps = 4
        gscale = 2.0
    elif args.model == "flux.1_schnell":
        model_id = "black-forest-labs/FLUX.1-schnell"
        num_steps = 4
        gscale = 0.0
        
    pre_inference_start = time.perf_counter()

    if args.api == "openvino" or args.api == "openvino-nightly":
        if args.model == "v1.5" or args.model == "v2.0" or args.model == "v2.1":
            from optimum.intel import OVStableDiffusionPipeline
            if args.precision == "int8":
                from optimum.intel import OVWeightQuantizationConfig
                quantization_config = OVWeightQuantizationConfig(bits=8)
                pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, device=args.device, export=True, quantization_config=quantization_config, ov_config={"INFERENCE_PRECISION_HINT": "f16"})
            else:
                pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, device=args.device, export=True, ov_config={"INFERENCE_PRECISION_HINT": "f16"})
        elif args.model == "lcm":
            from optimum.intel import OVLatentConsistencyModelPipeline
            if args.precision == "int8":
                from optimum.intel import OVWeightQuantizationConfig
                quantization_config = OVWeightQuantizationConfig(bits=8)
                pipeline = OVLatentConsistencyModelPipeline.from_pretrained(model_id, device=args.device, export=True, quantization_config=quantization_config, ov_config={"INFERENCE_PRECISION_HINT": "f16"}, dynamic_shapes=False)
            else:
                pipeline = OVLatentConsistencyModelPipeline.from_pretrained(model_id, device=args.device, export=True, ov_config={"INFERENCE_PRECISION_HINT": "f16"}, dynamic_shapes=False)
        elif args.model == "xl":
            from optimum.intel import OVStableDiffusionXLPipeline
            if args.precision == "int8":
                from optimum.intel import OVWeightQuantizationConfig
                quantization_config = OVWeightQuantizationConfig(bits=8)
                pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id, device=args.device, export=True, quantization_config=quantization_config)
            else:
                model_dir = os.path.join(*[args.model_root, f'stable-diffusion-{args.model}'])
                if Path(model_dir).exists():
                    pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=args.device, export=False, compile=False)
                else:
                    pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id, device=args.device, export=True, compile=True)
                    pipeline.save_pretrained(model_dir)
        elif args.model == "xl_lcm":
            import gc

            from diffusers import AutoPipelineForText2Image, LCMScheduler
            from optimum.exporters.openvino import export_from_model
            from optimum.intel.openvino import OVStableDiffusionXLPipeline

            output_model_path = Path("sdxl-lcm-ov")
            
            if not output_model_path.exists():
                pipe = AutoPipelineForText2Image.from_pretrained(model_id)
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights(adapter_id)
                pipe.fuse_lora()

                export_from_model(pipe, output_model_path, task="text-to-image", stateful=False)
                del pipe
                gc.collect()
            
            pipeline = OVStableDiffusionXLPipeline.from_pretrained(output_model_path, device=args.device)
            
        elif args.model == "v3.0":
            model_dir = os.path.join(*[args.model_root, f'stable-diffusion-{args.model}'])
            from optimum.intel import OVStableDiffusion3Pipeline
            if Path(model_dir).exists():
                pipeline = OVStableDiffusion3Pipeline.from_pretrained(model_dir, device=args.device, export=False, compile=False, ov_config={"INFERENCE_PRECISION_HINT": "f16"})
            else:
                pipeline = OVStableDiffusion3Pipeline.from_pretrained(model_id, device=args.device, compile=True, ov_config={"INFERENCE_PRECISION_HINT": "f16"})
                pipeline.save_pretrained(model_dir)

        elif args.model == "flux.1_schnell":
            from optimum.intel import OVFluxPipeline
            pipeline = OVFluxPipeline.from_pretrained(model_id, device=args.device, export=False, ov_config={"INFERENCE_PRECISION_HINT": "f16"})

        pipeline.reshape(batch_size=-1, height=args.height, width=args.width, num_images_per_prompt=1)
        pipeline.half()
        pipeline.compile()
        
    else:
        if args.api == "ipex":
            import intel_extension_for_pytorch as ipex
            device = "xpu"
        elif args.api == "cuda":
            device = "cuda"

        generator = torch.Generator('cpu').manual_seed(1337)
        if args.model == "v1.5" or args.model == "v2.0" or args.model == "v2.1":
            from diffusers import StableDiffusionPipeline
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        elif args.model == "v3.0":
            from diffusers import StableDiffusion3Pipeline
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        elif args.model == "lcm":
            from diffusers import DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        elif args.model == "xl":
            from diffusers import StableDiffusionXLPipeline
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        elif args.model == "turbo" or args.model == "xl_turbo":
            from diffusers import AutoPipelineForText2Image
            pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        elif args.model == "xl_lcm":
            from diffusers import AutoPipelineForText2Image, LCMScheduler
            pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(device)
            pipeline.load_lora_weights(adapter_id)
            pipeline.fuse_lora()
        elif args.model == "flux.1_schnell":
            from diffusers import FluxPipeline
            pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline = pipeline.to(device)
        
    pre_inference_stop = time.perf_counter()
    pre_inference_time = pre_inference_stop - pre_inference_start
        
    time_list = []
    total_start = time.time()

    if args.demo:
        print("Warm-up run")
        _ = pipeline(args.prompt, num_inference_steps=num_steps, guidance_scale=gscale).images[0]

        while True:
            try:
                text = input("Text prompt: ")
                if not text:
                    print("Error, input cannot be empty")
                    continue
            
                measure_start = time.time()
                if args.api == "ipex" or args.api == "cuda":
                    image = pipeline(text, generator=generator, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                    image_path = log_path / f"SD_IPEX_{args.model}_{args.height}x{args.width}_step{num_steps}_prompt{text}.png"
                elif args.api == "openvino" or args.api == "openvino-nightly":
                    image = pipeline(text, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                    image_path = log_path / f"SD_OV_{args.model}_{args.height}x{args.width}_step{num_steps}_prompt{text}.png"
                measure_stop = time.time()
                image.show()

                image_path = log_path / f"SD_{args.model}_{args.height}x{args.width}_step{num_steps}_prompt{text}.png"
                image.save(image_path)

                print(f"Model ID: {args.model}")
                print(f"Height: {args.height}")
                print(f"Width: {args.width}")
                print(f"Inference steps: {num_steps}")
                print(f"Guidance scale: {gscale}")
                print(f"Prompt: {text}")
                print(f"Performance: {measure_stop - measure_start} s/img")
                print("\n")
            except KeyboardInterrupt:
                print("\n")
                print("Ctrl-C detected, Terminating program.")
                break

    else:
        with MemLogger() as mem_session:
            if args.performance_target:
                print(f"Performance target set at: {args.performance_target} s/img")
            for i in range(args.num_warm + args.num_iter):
                if args.api == "ipex" or args.api == "cuda":
                    if args.feature == "txt2img":
                        
                        if args.batch_size == 1:
                            measure_start = time.time()
                            image = pipeline(args.prompt, generator=generator, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                            measure_stop = time.time()
                        else:
                            measure_start = time.time()
                            image = pipeline(**get_inputs(args.prompt, batch_size=args.batch_size, num_inference_steps=num_steps, guidance_scale=gscale)).images
                            measure_stop = time.time()
                            image = image_grid(image)
                        
                        image_path = log_path / f"SD_{args.model}_txt2img_{args.height}x{args.width}_bs{args.batch_size}_step{num_steps}_iter{i}.png"
                        image.save(image_path)
                        if args.performance_target:
                            elapsed_time = measure_stop - measure_start
                            if elapsed_time < args.performance_target:
                                print(f"Current performance {elapsed_time} s/img is faster than target performance {args.performance_target} s/img. Sleeping for {np.round(args.performance_target - elapsed_time,2)} s")
                                time.sleep(args.performance_target - elapsed_time)
                            else:
                                print(f"Current performance {elapsed_time} s/img is slower than target performance {args.performance_target} s/img. Proceeding without sleep")
                        if i > args.num_warm - 1:
                            time_list.append(measure_stop - measure_start)

                            
                    elif args.feature == "img2img":
                        if i == 0:
                            src_img = pipeline(args.prompt, generator=generator, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                            src_img.save("img2img_source.png")
                            from diffusers import \
                                StableDiffusionImg2ImgPipeline
                            pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
                            pipeline = pipeline.to("xpu")
                            img2img_prompt = "titanic"

                        if args.batch_size == 1:
                            measure_start = time.time()
                            image = pipeline(prompt=args.prompt, image=src_img, generator=generator, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                            measure_stop = time.time()
                        else:
                            measure_start = time.time()
                            image = pipeline(**get_inputs(prompt=img2img_prompt, batch_size=args.batch_size, num_inference_steps=num_steps, guidance_scale=gscale, image=src_img)).images
                            measure_stop = time.time()
                            image = image_grid(image)                    

                        # measure_start = time.time()
                        # image = pipeline(img2img_prompt, image=image, generator=generator, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                        # measure_stop = time.time()
                        image_path = log_path / f"SD_{args.model}_img2img_{args.height}x{args.width}_step{num_steps}_iter{i}.png"
                        image.save(image_path)
                        if args.performance_target:
                            elapsed_time = measure_stop - measure_start
                            if elapsed_time < args.performance_target:
                                print(f"Current performance {elapsed_time} s/img is faster than target performance {args.performance_target} s/img. Sleeping for {np.round(args.performance_target - elapsed_time,2)} s")
                                time.sleep(args.performance_target - elapsed_time)
                            else:
                                print(f"Current performance {elapsed_time} s/img is slower than target performance {args.performance_target} s/img. Proceeding without sleep")
                        if i > args.num_warm - 1:
                            time_list.append(measure_stop - measure_start)


                elif args.api == "openvino" or args.api == "openvino-nightly":
                    measure_start = time.time()
                    image = pipeline(args.prompt, height=args.height, width=args.width, num_inference_steps=num_steps, guidance_scale=gscale).images[0]
                    measure_stop = time.time()
                    image_path = log_path / f"SD_{args.model}_{args.height}x{args.width}_step{num_steps}_iter{i}.png"
                    image.save(image_path)
                    if args.performance_target:
                        elapsed_time = measure_stop - measure_start
                        if elapsed_time < args.performance_target:
                            print(f"Current performance {elapsed_time} s/img is faster than target performance {args.performance_target} s/img. Sleeping for {np.round(args.performance_target - elapsed_time,2)} s")
                            time.sleep(args.performance_target - elapsed_time)
                        else:
                            print(f"Current performance {elapsed_time} s/img is slower than target performance {args.performance_target} s/img. Proceeding without sleep")
                    if i > args.num_warm - 1:
                        time_list.append(measure_stop - measure_start)
        
            total_stop = time.time()
            total_runtime = total_stop - total_start
            total_seconds_per_image = total_runtime / (args.num_iter + args.num_warm)
        
        mem_session.print_summary()

        print(time_list)
        print(np.average(time_list))
        
        blank_image_validator(log_path)
            
        filename = "execution_results.csv"
        
        if args.api == "ipex":
            package_name, package_version = get_target_pip_package_version(["intel_extension_for_pytorch"])
        elif args.api == "openvino" or args.api == "openvino-nightly":
            package_name, package_version = get_target_pip_package_version(["openvino"])
        elif args.api == "cuda":
            package_name, package_version = get_target_pip_package_version(["torch"])

        records = []
        record = BenchmarkRecord(model_id, args.precision, package_name, args.device, package_name, package_version)
        record.config.batch_size = args.batch_size
        record.config.customized["Warm Up"] = args.num_warm
        record.config.customized["Iteration"] = args.num_iter
        record.config.customized["Width"] = args.width
        record.config.customized["Height"] = args.height
        record.config.customized["Inference Steps"] = num_steps
        record.config.customized["Guidance Scale"] = gscale
        record.metrics.customized["Pre Inference Time (s)"] = round(pre_inference_time,2)
        record.metrics.customized["Wall Clock throughput (img/s)"] = round(total_seconds_per_image, 2)
        record.metrics.customized["Wall Clock Time (s)"] = round(total_runtime, 2)
        record.metrics.customized["Performance All Runs"] = time_list
        record.metrics.customized["Seconds per image (s/img)"] = round(np.average(time_list), 2)
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
        BenchmarkRecord.print_for_test_e2e_plugin(records)
        log_result([record.to_dict() for record in records][0], log_path)

if __name__ == "__main__":
    main()