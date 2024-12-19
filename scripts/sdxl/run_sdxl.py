import argparse
from pathlib import Path
from optimum.intel.openvino import OVStableDiffusionXLPipeline
import gc
import time
import numpy as np

parser = argparse.ArgumentParser(description="""Run perf test for generative models""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--device', help='Target device', type=str, required=False, default="GPU")
parser.add_argument('-m', '--model_dir', help='Model dir', type=str, required=False, default="C:\dev\models\daily\sdxl_1_0_ov\FP16")
args = parser.parse_args()

batch_size = 1
num_images_per_prompt = 1
height = 768
width = 768
nsteps = 20
prompt = "cute cat 4k, high-res, masterpiece, best quality, soft lighting, dynamic angle"

text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(args.model_dir, compile=False, ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR":""})
text2image_pipe.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
text2image_pipe.to(args.device)
text2image_pipe.vae_encoder = None
text2image_pipe.compile()

start = time.time()
image = text2image_pipe(prompt, num_inference_steps=nsteps, height=height, width=width, num_images_per_prompt=num_images_per_prompt, generator=np.random.RandomState(314)).images[0]
image.save("cat.png")
end = time.time()
elapsed_time = (end - start) * 1000
print(f"pipeline: {elapsed_time} ms")
