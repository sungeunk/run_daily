#!/usr/bin/env python3

import re

from common_utils import *
from .test_template import *

class TestChatSample(TestTemplate):
    MODEL_DATE = 'WW44_llm-optimum_2024.5.0-17246-44b86a860ec'
    CONFIG_MAP = {
        (ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT): [{'app_path': 'openvino.genai/samples/python/chat_sample/chat_sample.py'}],
    }

    def get_command_list(args) -> dict:
        ret_dict = {}

        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_dict[key_tuple] = []
            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{__class__.MODEL_DATE}/{key_tuple[0]}/pytorch/ov/{key_tuple[1]}')
                APP_PATH = convert_path(f'{args.working_dir}/{config["app_path"]}')
                ret_dict[key_tuple].append({ResultKey.cmd: f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'})

        return ret_dict

    def parse_output(output) -> list[dict]:
        pass

    def generate_tabulate_table(key_tuple, cmd_item_list) -> str:
        pass

    def generate_report(result_root) -> str:
        report_str = ''
        raw_data_list = []
        for key_tuple in __class__.CONFIG_MAP.keys():
            for cmd_item in result_root.get(key_tuple, []):
                report_str += cmd_item[ResultKey.raw_log]

        if len(report_str):
            return '[RESULT] benchmark\n' + report_str + '\n'
        else:
            return ''

    def is_included(model_name) -> bool:
        for key_tuple in __class__.CONFIG_MAP.keys():
            if model_name == key_tuple[0]:
                return True
        return False
