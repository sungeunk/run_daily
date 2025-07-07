import argparse
import copy
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Define HF LLM model path to use")
parser.add_argument("--input_token", type=str, help="Define input token size")
parser.add_argument("--output_token", type=str, help="Define output token size")
parser.add_argument("--num_iter", type=str, default="4", help="Define number of iterations to run")
parser.add_argument("--api", type=str, choices=["ipex", "cuda"], help="Helps select which backend framework to use")
args = parser.parse_args()

cwd = Path.cwd()
original_config_path = cwd / "scripts" / "large_language_model" / "pytorch" / "config.yaml"
with open(str(original_config_path), 'r') as file:
    conf = yaml.safe_load(file)
reference = copy.deepcopy(conf)

for repo in conf['repo_id']:
    if args.model not in repo:
        reference['repo_id'].remove(repo)
if len(reference['repo_id']) == 0:
    raise Exception("IPEX configurator bug. Likely missing options in .yaml file that needs to be added.")

in_out_pairs = f"{args.input_token}-{args.output_token}"
for pairs in conf['in_out_pairs']:
    if in_out_pairs not in pairs:
        reference['in_out_pairs'].remove(pairs)
if len(reference['in_out_pairs']) == 0:
    raise Exception("IPEX configurator bug. Likely missing options in .yaml file that needs to be added.")

if args.api == "ipex":
    reference['test_api'].remove("transformer_int4_gpu_cuda_win")
else:
    reference['test_api'].remove("transformer_int4_fp16_gpu_win")
if len(reference['test_api']) == 0:
    raise Exception("IPEX configurator bug. Likely missing options in .yaml file that needs to be added.")

payload_path = cwd / "temp/python/llm/dev/benchmark/all-in-one/config.yaml"
with open(str(payload_path), "w") as f:
    yaml.dump(reference, f)