#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *

class TestMeasuredUsageCpp(TestTemplate):
    CONFIG_MAP = {
        ('qwen_usage', ModelConfig.INT8): [
            { 'select_inputs': 0 },
            { 'select_inputs': 1 },
            { 'select_inputs': 2 },
            { 'select_inputs': 3 },
            { 'select_inputs': 4 },
            { 'select_inputs': 5 },
            { 'select_inputs': 6 },
            { 'select_inputs': 7 },
        ]
    }

    def __get_configs():
        ret_configs = {}
        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_configs[(key_tuple[0], key_tuple[1], __class__)] = config_list
        return ret_configs

    def get_command_spec(args) -> dict:
        cfg = GlobalConfig()
        ret_dict = {}
        APP_PATH = convert_path(f'{cfg.BIN_DIR}/qwen/main{".exe" if is_windows() else ""}')
        MODEL_PATH = convert_path(f'{args.model_dir}/ww52-qwen-bkm-stateful/modified_openvino_model.xml')
        TOKENIZER_PATH = convert_path(f'{args.model_dir}/ww52-qwen-bkm-stateful/qwen.tiktoken')
        OUT_TOKEN_LEN = 256

        for key_tuple, config_list in __class__.__get_configs().items():
            ret_dict[key_tuple] = []
            for config in config_list:
                ret_dict[key_tuple].append({
                    CmdItemKey.cmd: f'{APP_PATH} -m {MODEL_PATH} -t {TOKENIZER_PATH} -d {args.device} -l en --stateful -mcl {OUT_TOKEN_LEN} -f --select_inputs {config["select_inputs"]}',
                    CmdItemKey.test_config: { CmdItemKey.TestConfigKey.mem_check: True },
                })
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        _1st_inf = -1
        _2nd_inf = -1
        is_sentence = False
        sentence = ''
        for line in output.splitlines():
            match_obj = re.search(f'First inference took (\d+.\d+) ms', line)
            if match_obj != None:
                _1st_inf = float(match_obj.groups()[0])
                is_sentence = True
                continue

            match_obj = re.search(f'Average other token latency: (\d+.\d+) ms', line)
            if match_obj != None:
                _2nd_inf = float(match_obj.groups()[0])
                is_sentence = False
                continue

            match_obj = re.search(f'Input num tokens: (\d+), output num tokens: (\d+), ', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[CmdItemKey.DataItemKey.in_token] = int(values[0])
                item[CmdItemKey.DataItemKey.out_token] = int(values[1])
                item[CmdItemKey.DataItemKey.perf] = [_1st_inf, _2nd_inf]
                item[CmdItemKey.DataItemKey.generated_text] = sentence
                ret_list.append(item)
                sentence = ''
                continue

            if is_sentence:
                sentence += line
                continue

        return ret_list

    def generate_report(result_root) -> str:
        def __get_inf(item:dict, index):
            try:
                return f'{item[CmdItemKey.DataItemKey.perf][index]:.02f}'
            except:
                return ''

        take_time = 0
        raw_data_list = []
        index = 0
        for key_tuple in __class__.__get_configs().keys():
            for cmd_item in result_root.get(key_tuple, []):
                take_time += cmd_item.get(CmdItemKey.process_time, 0)

                for result_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([index, result_item.get(CmdItemKey.DataItemKey.in_token, 0), result_item.get(CmdItemKey.DataItemKey.out_token, 0),
                                        __get_inf(result_item, 0), __get_inf(result_item, 1),
                                        cmd_item.get(CmdItemKey.peak_cpu_usage_percent, 0),
                                        cmd_item.get(CmdItemKey.peak_mem_usage_size, 0),
                                        cmd_item.get(CmdItemKey.peak_mem_usage_percent, 0)])
                    index += 1

        if len(raw_data_list):
            headers = ['index', 'in token', 'out token', '1st inf', '2nd inf', 'CPU (%)', 'Memory', 'Memory (%)']
            floatfmt = ['', '', '', '.2f', '.2f', '.2f', '', '.2f']
            tabulate_str =  tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='right', numalign='right')
            return f'[RESULT] measured usage(cpp) / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
