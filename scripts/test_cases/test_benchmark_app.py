#!/usr/bin/env python3

import re
import time

from common_utils import *
from .test_template import *

class TestBenchmarkapp(TestTemplate):
    CONFIG_MAP = {
        ('Resnet50', ModelConfig.INT8): [
            { 'model': 'models/resnet_v1.5_50/resnet_v1.5_50_i8.xml', 'batch': 1 },
            { 'model': 'models/resnet_v1.5_50/resnet_v1.5_50_i8.xml', 'batch': 64 },
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
        for key_tuple, config_list in __class__.__get_configs().items():
            ret_dict[key_tuple] = []
            for config in config_list:
                ret_dict[key_tuple].append({
                    CmdItemKey.cmd: ['python', '-c', "from openvino.tools.benchmark.main import main; main()", '-m', f'{convert_path(config["model"])}', '-b', f'{config["batch"]}', '-d', f'{args.device}', '-hint', 'none', '-nstreams', '2', '-nireq', '4', '-t', '10'],
                    CmdItemKey.test_config: {
                        CmdItemKey.TestConfigKey.batch: config['batch']
                    }
                })
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        for line in output.splitlines():
            match_obj = re.search(f'Throughput: +(\d+.\d+) FPS', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[CmdItemKey.DataItemKey.perf] = [float(values[0])]
                ret_list.append(item)
                break

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
                batch = cmd_item[CmdItemKey.test_config]['batch']
                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], batch, __get_inf(data_item, 0)])

        if len(raw_data_list):
            headers = ['model', 'precision', 'batch', 'throughput(fps)']
            floatfmt = ['', '', '', '.2f']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='right', numalign='right')
            return f'[RESULT] benchmark_app(cpp) / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
