#!/usr/bin/env python3

import re

from common_utils import *
from .test_template import *

class TestWhisperBase(TestTemplate):
    CONFIG_MAP = {
        ('Whisper base', ModelConfig.UNKNOWN): [{'model': 'whisper-base-nonstateful', 'app_path': 'scripts/whisper/optimum_notebook/non_stateful/run_model.py'}],
    }

    def get_command_list(args) -> dict:
        ret_dict = {}

        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_dict[key_tuple] = []
            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{config["model"]}')
                APP_PATH = convert_path(f'{args.working_dir}/{config["app_path"]}')
                ret_dict[key_tuple].append({CmdItemKey.cmd: f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'})
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        for line in output.splitlines():
            match_obj = re.search(f'tps : (\d+.\d+)', line)
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
            return '[RESULT] whisper_base\n' + tabulate_str + '\n'
        else:
            return ''

    def is_included(model_name) -> bool:
        for key_tuple in __class__.CONFIG_MAP.keys():
            if model_name == key_tuple[0]:
                return True
        return False

    def is_class_name(name) -> bool:
        return __class__.__name__ == name
