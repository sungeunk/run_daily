#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *


class TestStableDiffusionGenai(TestTemplate):
    CONFIG_MAP = {
        ('stable-diffusion-v1-5', ModelConfig.FP16):                 [{PROMPT_TYPE_KEY: PROMPT_TYPE_MULTIMODAL}],
        ('stable-diffusion-v2-1', ModelConfig.FP16):                 [{PROMPT_TYPE_KEY: PROMPT_TYPE_MULTIMODAL}],
        ('lcm-dreamshaper-v7',    ModelConfig.FP16):                 [{PROMPT_TYPE_KEY: PROMPT_TYPE_MULTIMODAL}],
        ('flux.1-schnell',        ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        ('whisper-large-v3',      ModelConfig.OV_FP16_4BIT_DEFAULT): [{PROMPT_TYPE_KEY: PROMPT_TYPE_MULTIMODAL}],
    }

    def __get_configs():
        ret_configs = {}
        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_configs[(key_tuple[0], key_tuple[1], __class__)] = config_list
        return ret_configs

    def get_command_spec(args) -> dict:
        cfg = GlobalConfig()
        APP_PATH = convert_path(f'{cfg.PWD}/openvino.genai/tools/llm_bench/benchmark.py')
        ret_dict = {}

        for key_tuple, config_list in __class__.__get_configs().items():
            ret_dict.setdefault(key_tuple, [])

            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{cfg.MODEL_DATE}/{key_tuple[0]}/pytorch/ov/{key_tuple[1]}')
                cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device} -mc 1 -n 1 --genai --output_dir {args.output_dir}'

                prompt_type = config.get(PROMPT_TYPE_KEY, PROMPT_TYPE_DEFAULT)
                PROMPT_PATH = convert_path(f'{cfg.PWD}/prompts/{prompt_type}/{key_tuple[0]}.jsonl')
                cmd += f' -pf {PROMPT_PATH}'

                ret_dict[key_tuple] = [{CmdItemKey.cmd: cmd}]

        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        data_dict = {}
        for line in output.splitlines():
            match_obj = re.search(r'\[warm-up\]', line)
            if match_obj:   # ignore warm-up inference
                continue

            def __add_value_to_dict(result_dict, key, parse_str, line):
                match_obj = re.search(parse_str, line)
                if match_obj != None:
                    values = match_obj.groups()
                    result_dict[key] = float(values[0]) if is_float(values[0]) else int(values[0])

            __add_value_to_dict(data_dict, 'Batch_size', r'Batch_size=(\d+)', line)
            __add_value_to_dict(data_dict, 'steps', r'steps=(\d+)', line)
            __add_value_to_dict(data_dict, 'width', r'width=(\d+)', line)
            __add_value_to_dict(data_dict, 'height', r'height=(\d+)', line)
            __add_value_to_dict(data_dict, 'guidance_scale', r'guidance_scale=(\d+.\d+)', line)
            __add_value_to_dict(data_dict, 'Input token size', r'Input token size: (\d+)', line)
            __add_value_to_dict(data_dict, 'Output token size', r'Output size: (\d+)', line)
            __add_value_to_dict(data_dict, 'Infer count', r'Infer count: (\d+)', line)
            __add_value_to_dict(data_dict, 'Generation Time', r'Generation Time: (\d+.\d+)s', line)

            match_obj = re.search(r'\[(\d+)\]\[P(\d+)\] start: ', line)
            if match_obj != None:
                width = data_dict.get('width', '')
                height = data_dict.get('height', '')
                size = f'{int(width)}x{int(height)}' if width and height else ''

                item = {}
                item[CmdItemKey.DataItemKey.perf] = [data_dict.get('Generation Time'),
                                                     data_dict.get('Batch_size', ''),
                                                     data_dict.get('steps', ''),
                                                     size,
                                                     data_dict.get('Input token size', ''),
                                                     data_dict.get('Output token size', ''),
                                                     data_dict.get('Infer count', '')]
                ret_list.append(item)
                data_dict = {}

        return ret_list

    def generate_report(result_root) -> str:
        def __get_inf(data_item:dict, index):
            try:
                value = data_item[CmdItemKey.DataItemKey.perf][index]
                if is_float(value):
                    return f'{value:.02f}' if index == 0 else int(value)
                else:
                    return value
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
            headers = ['model', 'precision', 'pipeline time(s)', 'Batch_size', 'steps', 'size', 'Input token size', 'Output token size', 'Infer count']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='right', numalign='right')
            return f'[RESULT] stable_diffusion_genai / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
