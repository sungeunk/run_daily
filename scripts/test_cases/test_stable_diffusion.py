#!/usr/bin/env python3

import re

from common_utils import *
from .test_template import *

class TestStableDiffusion(TestTemplate):
    MODEL_DATE_WW32_LLM = 'WW32_llm_2024.4.0-16283-41691a36b90'
    CONFIG_MAP = {
        ('SD 1.5', ModelConfig.FP16): [{'model': 'daily/sd_15_ov', 'app_path': 'bin/sd/stable_diffusion.exe'}],
        ('SD 1.5', ModelConfig.INT8): [{'model': 'daily/sd_15_ov', 'app_path': 'bin/sd/stable_diffusion.exe'}],
        ('SD 2.1', ModelConfig.FP16): [{'model': 'daily/sd_21_ov', 'app_path': 'bin/sd/stable_diffusion.exe'}],
        ('SD 2.1', ModelConfig.INT8): [{'model': 'daily/sd_21_ov', 'app_path': 'bin/sd/stable_diffusion.exe'}],
        ('Stable-Diffusion LCM', ModelConfig.FP16): [{'model': f'{MODEL_DATE_WW32_LLM}/lcm-dreamshaper-v7/pytorch/dldt', 'app_path': 'bin/lcm/lcm_dreamshaper.exe'}],
        ('Stable Diffusion XL', ModelConfig.FP16): [{'model': f'daily/sdxl_1_0_ov/FP16', 'app_path': 'scripts/sdxl/run_sdxl.py'}],
        ('SD 3.0 Dynamic', ModelConfig.MIXED): [{'model': f'stable-diffusion-3', 'app_path': 'scripts/stable-diffusion/run_sd3_ov_daily.py', 'dynamic': True}],
        ('SD 3.0 Static', ModelConfig.MIXED): [{'model': f'stable-diffusion-3', 'app_path': 'scripts/stable-diffusion/run_sd3_ov_daily.py', 'dynamic': False}],
    }

    def get_command_list(args) -> dict:
        cfg = GlobalConfig()
        ret_dict = {}

        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_dict[key_tuple] = []
            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{config["model"]}')
                APP_PATH = convert_path(f'{args.working_dir}/{config["app_path"]}')

                dynamic = config.get("dynamic", None)
                if dynamic != None:
                    result_filename = convert_path(f'{args.output_dir}/daily.{cfg.NOW}.sd.{"dynamic" if dynamic else "static"}.png')
                    cmd = f'python {APP_PATH} -m {MODEL_PATH} {"--dynamic" if dynamic else ""} -d {args.device} --result_img {result_filename}'
                elif APP_PATH.endswith('.py'):
                    cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'
                else:
                    cmd = f'{APP_PATH} -m {MODEL_PATH} -d {args.device} -t {key_tuple[1]}'

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

        raw_data_list = []
        for key_tuple in __class__.CONFIG_MAP.keys():
            for cmd_item in result_root.get(key_tuple, []):
                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], __get_inf(data_item, 0)])

        if len(raw_data_list):
            headers = ['model', 'precision', 'pipeline time(ms)']
            floatfmt = ['', '', '', '.2f']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='right', numalign='right')
            return '[RESULT] stable_diffusion\n' + tabulate_str + '\n'
        else:
            return ''

    def is_included(model_name) -> bool:
        for key_tuple in __class__.CONFIG_MAP.keys():
            if model_name == key_tuple[0]:
                return True
        return False

    def is_class_name(name) -> bool:
        return __class__.__name__ == name
