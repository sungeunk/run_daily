#!/usr/bin/env python3

import re

from common_utils import *
from .test_template import *

class TestChatSample(TestTemplate):
    MODEL_DATE = 'WW44_llm-optimum_2024.5.0-17246-44b86a860ec'
    CONFIG_MAP = {
        (ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT): [{'app_path': 'openvino.genai/samples/python/chat_sample/chat_sample.py'}],
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
            ret_dict[key_tuple] = []
            for config in config_list:
                APP_PATH = convert_path(f'{config["app_path"]}')
                MODEL_PATH = convert_path(f'{args.model_dir}/{__class__.MODEL_DATE}/{key_tuple[0]}/pytorch/ov/{key_tuple[1]}')
                cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'
                ret_dict[key_tuple].append({CmdItemKey.cmd: cmd})

        return ret_dict

    def parse_output(args, output) -> list[dict]:
        return []

    def generate_report(result_root) -> str:
        report_str = ''
        raw_data_list = []
        for key_tuple in __class__.__get_configs().keys():
            for cmd_item in result_root.get(key_tuple, []):
                report_str += cmd_item[CmdItemKey.raw_log]

        if len(report_str):
            return '[RESULT] chat_sample\n' + report_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
