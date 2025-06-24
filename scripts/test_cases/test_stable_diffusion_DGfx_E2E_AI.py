#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *

class TestStableDiffusion(TestTemplate):
    APP_GENAI_PATH = 'openvino.genai/tools/llm_bench/benchmark.py'
    APP_BASE_SD_PATH = ''

    CONFIG_MAP_GENAI = {
        ('stable-diffusion-v1-5', ModelConfig.FP16): [{'model': 'stable-diffusion-v1-5/pytorch/ov', 'app_path': APP_GENAI_PATH, 'prompt_path': 'prompts/multimodal/stable-diffusion-v1-5.jsonl'}],
        ('stable-diffusion-v2-1', ModelConfig.FP16): [{'model': 'stable-diffusion-v2-1/pytorch/ov', 'app_path': APP_GENAI_PATH, 'prompt_path': 'prompts/multimodal/stable-diffusion-v2-1.jsonl'}],
        ('stable-diffusion-3.5-large-turbo', ModelConfig.FP16): [{'model': 'stable-diffusion-3.5-large-turbo/pytorch/ov', 'app_path': APP_GENAI_PATH, 'prompt_path': 'prompts/multimodal/stable-diffusion-3.5-large-turbo.jsonl'}],
        ('lcm-dreamshaper-v7', ModelConfig.FP16): [{'model': f'lcm-dreamshaper-v7/pytorch/ov', 'app_path': APP_GENAI_PATH, 'prompt_path': 'prompts/multimodal/lcm-dreamshaper-v7.jsonl'}],
    }
    CONFIG_MAP_BASE_SD = {
        ('stable-diffusion-xl-1.0-inpainting-0.1', ModelConfig.FP16): [{'model': f'stable-diffusion-xl-1.0-inpainting-0.1/pytorch/ov', 'app_path': APP_BASE_SD_PATH}],
        ('stable-diffusion_v3.0', ModelConfig.FP16): [{'model': f'daily/stable-diffusion_v3.0', 'app_path': APP_BASE_SD_PATH}],
    }

    def __get_configs(cfg = None):
        ret_configs = {}
        for key_tuple, config_list in __class__.CONFIG_MAP_GENAI.items():
            if cfg != None:
                for config in config_list:
                    config['model'] = f'{cfg.MODEL_DATE}/{config['model']}/{key_tuple[1]}'
            ret_configs[(key_tuple[0], key_tuple[1], __class__)] = config_list
        for key_tuple, config_list in __class__.CONFIG_MAP_BASE_SD.items():
            ret_configs[(key_tuple[0], key_tuple[1], __class__)] = config_list
        return ret_configs

    def get_command_spec(args) -> dict:
        cfg = GlobalConfig()
        ret_dict = {}

        for key_tuple, config_list in __class__.__get_configs(cfg).items():
            ret_dict[key_tuple] = []
            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{config["model"]}')
                APP_PATH = convert_path(f'{cfg.PWD}/{config["app_path"]}')
                cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'
                ret_dict[key_tuple].append({CmdItemKey.cmd: cmd})
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        for line in output.splitlines():
            match_obj = re.search(f'pipeline: (\d+.\d+) ms', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[CmdItemKey.DataItemKey.perf] = [float(values[0])]
                ret_list.append(item)

            match_obj = re.search(f'time in seconds:\ +(\d+.\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[CmdItemKey.DataItemKey.perf] = [float(values[0])]
                ret_list.append(item)

        return ret_list

    def generate_report(result_root) -> str:
        def __get_inf(data_item:dict, index):
            try:
                return f'{data_item[CmdItemKey.DataItemKey.perf][index]:.02f}'
            except:
                return ''

        take_time = 0
        raw_data_list = []
        for key_tuple in __class__.__get_configs().keys():
            for cmd_item in result_root.get(key_tuple, []):
                take_time += cmd_item.get(CmdItemKey.process_time, 0)

                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], __get_inf(data_item, 0)])

        if len(raw_data_list):
            headers = ['model', 'precision', 'pipeline time(ms)']
            floatfmt = ['', '', '', '.2f']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='right', numalign='right')
            return f'[RESULT] stable_diffusion / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
