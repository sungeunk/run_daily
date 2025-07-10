#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *

class TestStableDiffusionDGfxE2eAi(TestTemplate):
    CONFIG_MAP = {
        ('stable-diffusion-v3.0', ModelConfig.FP16): [{'model_name':'v3.0', 'height':512, 'width':512}],
        ('stable-diffusion-xl', ModelConfig.FP16): [{'model_name':'xl', 'height':768, 'width':768}],
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
            WORK_PATH = convert_path(f'{cfg.PWD}/scripts/DGfx_E2E_AI/tests')
            APP_PATH = convert_path(f'temp/base_sd.py')
            MODEL_ROOT_PATH = convert_path(f'{args.model_dir}')

            ret_dict[key_tuple] = []
            for config in config_list:
                cmd = f'python {APP_PATH} --device {args.device} --api openvino-nightly --model {config.get("model_name")} --height {config.get("height")} --width {config.get("width")} --num_warm 1 --num_iter 1 --model_root {MODEL_ROOT_PATH}'
                ret_dict[key_tuple].append({CmdItemKey.cmd: cmd, CmdItemKey.work_dir: WORK_PATH})

        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        data_dict = {}
        for line in output.splitlines():
            idx = line.find('testResult')
            if idx > 0:
                line = line[idx:]

                # Overall Score:  {'config': {'batch_size': 1, 'precision': 'fp16', 'warmup_runs': 1, 'measured_runs': 10,
                # 'model_info': {'full_name': 'stabilityai/stable-diffusion-3-medium-diffusers'},
                # 'Warm Up': 1, 'Iteration': 1, 'Width': 512, 'Height': 512, 'Inference Steps': 20, 'Guidance Scale': 7.5},
                # 'metadata': {'device': 'GPU.1'},
                # 'metrics': {'Pre Inference Time (s)': 27.55, 'Wall Clock throughput (img/s)': 2.79, 'Wall Clock Time (s)': 5.57, 'Performance All Runs': [2.4369802474975586], 'Seconds per image (s/img)': np.float64(2.44)}, 'logging': {}, 'trigger_date': '2025-07-07 21:16:08',
                # 'testResult': {'config': {'batch_size': 1, 'precision': 'fp16', 'warmup_runs': 1, 'measured_runs': 10, 'model_info': {'full_name': 'stabilityai/stable-diffusion-3-medium-diffusers'}, 'Warm Up': 1, 'Iteration': 1, 'Width': 512, 'Height': 512, 'Inference Steps': 20, 'Guidance Scale': 7.5},
                # 'metadata': {'device': 'GPU.1'},
                # 'metrics': {'Pre Inference Time (s)': 27.55, 'Wall Clock throughput (img/s)': 2.79, 'Wall Clock Time (s)': 5.57, 'Performance All Runs': [2.4369802474975586], 'Seconds per image (s/img)': np.float64(2.44)}, 'logging': {}, 'trigger_date': '2025-07-07 21:16:08'}}

                match_obj = re.search(r'\'batch_size\': (\d+)', line)
                if match_obj != None:
                    values = match_obj.groups()
                    data_dict['Batch_size'] = int(values[0])
                    
                match_obj = re.search(r'\'Width\': (\d+), \'Height\': (\d+)', line)
                if match_obj != None:
                    values = match_obj.groups()
                    data_dict['size'] = f'{int(values[0])}x{int(values[1])}'

                match_obj = re.search(r'\'Inference Steps\': (\d+)', line)
                if match_obj != None:
                    values = match_obj.groups()
                    data_dict['steps'] = int(values[0])

                match_obj = re.search(r'\'Seconds per image \(s/img\)\': np.float64\((\d+.\d+)\)', line)
                if match_obj != None:
                    values = match_obj.groups()
                    data_dict['pipeline'] = float(values[0])

                    item = {}
                    item[CmdItemKey.DataItemKey.perf] = [data_dict['pipeline'], data_dict['Batch_size'], data_dict['steps'], data_dict['size']]
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
            headers = ['model', 'precision', 'pipeline time(s)', 'Batch_size', 'steps', 'size']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='right', numalign='right')
            return f'[RESULT] stable_diffusion_DGfx_E2E_AI / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
