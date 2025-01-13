#!/usr/bin/env python3

import re
import time

from statistics import geometric_mean

from common_utils import *
from .test_template import *

class TestBenchmark(TestTemplate):
    CONFIG_MAP = {
        (ModelName.baichuan2_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.glm_4_9b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.llama_3_8b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.minicpm_1b_sft, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.mistral_7b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.phi_2, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.phi_3_mini_4k_instruct, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.gemma_7b_it, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.qwen_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.qwen2_7b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
    }

    def __name__() -> str:
        return 'TestBenchmark'

    def get_command_spec(args) -> dict:
        cfg = GlobalConfig()
        APP_PATH = convert_path(f'{args.working_dir}/openvino.genai/tools/llm_bench/benchmark.py')
        ret_dict = {}
        for key_tuple, config_list in __class__.CONFIG_MAP.items():
            ret_dict.setdefault(key_tuple, [])

            for config in config_list:
                MODEL_PATH = convert_path(f'{args.model_dir}/{cfg.MODEL_DATE}/{key_tuple[0]}/pytorch/ov/{key_tuple[1]}')
                PROMPT_PATH = convert_path(f'{args.working_dir}/prompts/32_1024/{key_tuple[0]}.jsonl')
                cmd = f'python {APP_PATH} -m {MODEL_PATH} -pf {PROMPT_PATH} -d {args.device} -mc 1 -ic {cfg.out_token_length} -n {cfg.benchmark_iter_num} {"--genai" if args.genai else "" }'
                ret_dict[key_tuple].append({CmdItemKey.cmd: cmd})
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        cfg = GlobalConfig()
        ret_list = []
        in_token_size = 0
        out_token_size = 0
        input_text = ''
        sentence_map = {}
        generated_text = None

        for line in output.splitlines():
            if generated_text:
                match_obj = re.search(f'\[ ([\S]+) \] ', line)
                if match_obj != None:
                    sentence_map[in_token_size] = generated_text.replace(input_text, '')
                    generated_text = None
                else:
                    generated_text += line
                continue

            match_obj = re.search(f'\] Input token size: (\d+), Output size: (\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                in_token_size = int(values[0])
                out_token_size = int(values[1])
                continue

            values = None
            match_obj1 = re.search(f'\[{cfg.benchmark_iter_num}\]\[[A-Z0-9]+\] First token latency: (\d+.\d+) ms\/token, other tokens latency: (\d+.\d+) ms\/token', line)
            match_obj2 = re.search(f'\[{cfg.benchmark_iter_num}\]\[[A-Z0-9]+\] First token latency: (\d+.\d+) ms\/token', line)
            if match_obj1 != None:
                values = match_obj1.groups()
            elif match_obj2 != None:
                values = match_obj2.groups()

            if values != None:
                item = {}
                item[CmdItemKey.DataItemKey.in_token] = in_token_size
                item[CmdItemKey.DataItemKey.out_token] = out_token_size
                item[CmdItemKey.DataItemKey.perf] = [float(values[0]), float(values[1])] if len(values) == 2 else [float(values[0])]
                item[CmdItemKey.DataItemKey.generated_text] = sentence_map[in_token_size]
                ret_list.append(item)
                input_text = ''
                continue

            match_obj = re.search(f'\] Input text: ([\S ]+)', line)
            if match_obj != None:
                input_text = match_obj.groups()[0]
                continue

            match_obj = re.search(f'\] Generated:([\S ]+)', line)
            if match_obj != None:
                generated_text = match_obj.groups()[0]
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
        for key_tuple in __class__.CONFIG_MAP.keys():
            for cmd_item in result_root.get(key_tuple, []):
                take_time += cmd_item.get(CmdItemKey.process_time, 0)

                for result_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1],
                                          result_item[CmdItemKey.DataItemKey.in_token],
                                          result_item[CmdItemKey.DataItemKey.out_token],
                                          __get_inf(result_item, 0), __get_inf(result_item, 1)])

        if len(raw_data_list):
            value_dict_1st = {}
            value_dict_2nd = {}
            value_dict_1st[32]   = [ float(raw_data[4]) for raw_data in raw_data_list if len(raw_data) == 6 and raw_data[2] == 32 and is_float(raw_data[4]) ]
            value_dict_1st[1024] = [ float(raw_data[4]) for raw_data in raw_data_list if len(raw_data) == 6 and raw_data[2] == 1024 and is_float(raw_data[4]) ]
            value_dict_2nd[32]   = [ float(raw_data[5]) for raw_data in raw_data_list if len(raw_data) == 6 and raw_data[2] == 32 and is_float(raw_data[5]) ]
            value_dict_2nd[1024] = [ float(raw_data[5]) for raw_data in raw_data_list if len(raw_data) == 6 and raw_data[2] == 1024 and is_float(raw_data[5]) ]

            raw_data_list.append(['','','','','-','-'])
            raw_data_list.append(['Success count', '', '', '', len(value_dict_1st[32]) + len(value_dict_1st[1024]), len(value_dict_2nd[32]) + len(value_dict_2nd[1024])])
            raw_data_list.append(['geomean (token:  32)', '', '', '', f'{geometric_mean(value_dict_1st[32]):.02f}', f'{geometric_mean(value_dict_2nd[32]):.02f}'])
            raw_data_list.append(['geomean (token:1024)', '', '', '', f'{geometric_mean(value_dict_1st[1024]):.02f}', f'{geometric_mean(value_dict_2nd[1024]):.02f}'])

            headers = ['model', 'precision', 'in token', 'out token', '1st inf', '2nd inf']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='right')
            return f'[RESULT] benchmark (python) / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_included(model_name) -> bool:
        for key_tuple in __class__.CONFIG_MAP.keys():
            if model_name == key_tuple[0]:
                return True
        return False

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
