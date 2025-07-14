#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *

class TestStableDiffusionGenai(TestTemplate):
    CONFIG_MAP = {
        # ('stable-diffusion-v1-5', ModelConfig.FP16): [{}],
        # ('stable-diffusion-v2-1', ModelConfig.FP16): [{}],
        ('lcm-dreamshaper-v7', ModelConfig.FP16): [{}],
        ('flux.1-schnell', ModelConfig.OV_FP16_4BIT_DEFAULT): [{'prompt':'prompts/32_1024/flux.1-schnell.jsonl'}],
    }

    def __get_configs():
        ret_configs = {}
        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_configs[(key_tuple[0], key_tuple[1], __class__)] = config_list
        return ret_configs

    def get_command_spec(args) -> dict:
        cfg = GlobalConfig()
        ret_dict = {}

        for key_tuple, config_list in __class__.__get_configs().items():
            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{cfg.MODEL_DATE}/{key_tuple[0]}/pytorch/ov/{key_tuple[1]}')
                APP_PATH = convert_path(f'{cfg.PWD}/openvino.genai/tools/llm_bench/benchmark.py')
                prompt = config.get('prompt', f'prompts/multimodal/{key_tuple[0]}.jsonl')
                PROMPT_PATH = convert_path(f'{cfg.PWD}/{prompt}')

                cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device} -mc 1 -n 1 --genai -pf {PROMPT_PATH} --output_dir {args.output_dir}'
                ret_dict[key_tuple] = [{CmdItemKey.cmd: cmd}]

        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        data_dict = {}
        for line in output.splitlines():
            match_obj = re.search(r'\[warm-up\]', line)
            if match_obj:   # ignore warm-up inference
                continue

            match_obj = re.search(r'Input params: Batch_size=(\d+), steps=(\d+), width=(\d+), height=(\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                data_dict['Batch_size'] = int(values[0])
                data_dict['steps'] = int(values[1])
                data_dict['size'] = f'{int(values[2])}x{int(values[3])}'

            match_obj = re.search(r'guidance_scale=(\d+.\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                data_dict['guidance_scale'] = float(values[0])

            match_obj = re.search(r'Input token size: (\d+), Infer count: (\d+), Generation Time: (\d+.\d+)s,', line)
            if match_obj != None:
                values = match_obj.groups()
                data_dict['Input token size'] = int(values[0])
                data_dict['Infer count'] = int(values[1])
                data_dict['Generation Time'] = float(values[2])
                item = {}
                item[CmdItemKey.DataItemKey.perf] = [data_dict['Generation Time'], data_dict['Batch_size'], data_dict['steps'], data_dict['size'], data_dict['Input token size'], data_dict['Infer count']]
                ret_list.append(item)

        return ret_list

    def generate_report(result_root) -> str:
        def __get_inf(data_item:dict, index):
            try:
                value = data_item[CmdItemKey.DataItemKey.perf][index]
                return f'{value:.02f}' if is_float(value) else value
            except:
                return ''

        take_time = 0
        raw_data_list = []
        for key_tuple in __class__.__get_configs().keys():
            for cmd_item in result_root.get(key_tuple, []):
                take_time += cmd_item.get(CmdItemKey.process_time, 0)

                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data = []
                    for i in range(len(data_item[CmdItemKey.DataItemKey.perf])):
                        raw_data.append(__get_inf(data_item, i))
                    raw_data_list.append([key_tuple[0], key_tuple[1]] + raw_data)

        if len(raw_data_list):
            headers = ['model', 'precision', 'pipeline time(s)', 'Batch_size', 'steps', 'size', 'Input token size', 'Infer count']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='right', numalign='right')
            return f'[RESULT] stable_diffusion_genai / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
