#!/usr/bin/env python3

import re
import time

from statistics import geometric_mean

from common_utils import *
from .test_template import *


BENCHMARK_MODE = 'BENCHMARK_MODE'
BENCHMARK_MODE_LLM = 'LLM'
BENCHMARK_MODE_VLM = 'VLM'
BENCHMARK_MODE_DEFAULT = BENCHMARK_MODE_LLM

class TestBenchmark(TestTemplate):

    CONFIG_MAP = {
        (ModelName.baichuan2_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.glm_4_9b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        ('llama-3.1-8b-instruct', ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.minicpm_1b_sft, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.mistral_7b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        ('phi-3.5-mini-instruct', ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.phi_3_mini_4k_instruct, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.gemma_7b_it, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.qwen_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        (ModelName.qwen2_7b, ModelConfig.OV_FP16_4BIT_DEFAULT): [{}],
        ('minicpm-v-2_6', ModelConfig.OV_FP16_4BIT_DEFAULT): [{BENCHMARK_MODE:BENCHMARK_MODE_VLM, 'prompt': 'What is on this image?', 'media': convert_path("res/cat-448x448.png")}],
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
                cmd = f'python {APP_PATH} -m {MODEL_PATH} -d {args.device} -mc 1 -ic {cfg.out_token_length} -n {cfg.benchmark_iter_num} {"--optimum" if args.optimum else "" }'
                if not args.prompt_permutation:
                    cmd += f' --disable_prompt_permutation'
                if not args.continuous_batch:
                    cmd += f' --load_config {convert_path("res/config_wa.json")}'

                TEST_MODE = config.get(BENCHMARK_MODE, BENCHMARK_MODE_DEFAULT)
                if TEST_MODE == BENCHMARK_MODE_LLM:
                    PROMPT_PATH = convert_path(f'{cfg.PWD}/prompts/32_1024/{key_tuple[0]}.jsonl')
                    cmd += f' -pf {PROMPT_PATH}'
                elif TEST_MODE == BENCHMARK_MODE_VLM:
                    cmd += f' -p "{config["prompt"]}" --media {config["media"]}'

                ret_dict[key_tuple].append({CmdItemKey.cmd: cmd})
        return ret_dict

    def parse_output(args, output) -> list[dict]:
        ret_list = []
        generated_text = None
        prompt_id = 0

        for line in output.splitlines():
            if generated_text:
                match_obj = re.search(fr'\[ ([\S]+) \] ', line)
                if match_obj != None:
                    ret_list[prompt_id][CmdItemKey.DataItemKey.generated_text] = generated_text
                    generated_text = None
                else:
                    generated_text += line
                continue

            match_obj = re.search(r'prompt nums: (\d+)', line)
            if match_obj != None:
                for i in range(0, int(match_obj.groups()[0])):
                    ret_list.append({})
                continue

            match_obj = re.search(r'\[\w(\d+)\] Input token size: (\d+), Output size: (\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                ret_list[int(values[0])][CmdItemKey.DataItemKey.in_token] = int(values[1])
                ret_list[int(values[0])][CmdItemKey.DataItemKey.out_token] = int(values[2])
                continue

            values = None
            match_obj1 = re.search(r'\[\d+\]\[\w(\d+)\] First token latency: (\d+.\d+) ms\/token, other tokens latency: (\d+.\d+) ms\/token', line)
            match_obj2 = re.search(r'\[\d+\]\[\w(\d+)\] First token latency: (\d+.\d+) ms\/token', line)
            if match_obj1 != None:
                values = match_obj1.groups()
            elif match_obj2 != None:
                values = match_obj2.groups()

            if values != None:
                new_perf = [float(values[1]), float(values[2])] if len(values) == 3 else [float(values[1])]
                old_perf = ret_list[int(values[0])].get(CmdItemKey.DataItemKey.perf, None)
                if old_perf == None or geometric_mean(new_perf) < geometric_mean(old_perf):
                    ret_list[int(values[0])][CmdItemKey.DataItemKey.perf] = new_perf
                continue

            match_obj = re.search(r'\[warm-up\]\[\w(\d+)\] Generated:([\S ]+)', line)
            if match_obj != None:
                values = match_obj.groups()
                prompt_id = int(values[0])
                generated_text = values[1]
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
        for key_tuple in __class__.__get_configs().keys():
            for cmd_item in result_root.get(key_tuple, []):
                take_time += cmd_item.get(CmdItemKey.process_time, 0)

                if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                    for result_item in cmd_item.get(CmdItemKey.data_list, []):
                        raw_data_list.append([key_tuple[0], key_tuple[1],
                                            result_item[CmdItemKey.DataItemKey.in_token],
                                            result_item[CmdItemKey.DataItemKey.out_token],
                                            __get_inf(result_item, 0), __get_inf(result_item, 1)])

        if len(raw_data_list):
            SHORT_TOKEN_ID = 0
            LONG_TOKEN_ID = 1
            value_dict_1st = {SHORT_TOKEN_ID:[], LONG_TOKEN_ID:[]}
            value_dict_2nd = {SHORT_TOKEN_ID:[], LONG_TOKEN_ID:[]}

            for raw_data in raw_data_list:
                print(f'raw_data: {raw_data}, is_float([raw_data[4]]): {is_float(raw_data[4])}, is_float([raw_data[5]]): {is_float(raw_data[5])}')
                if len(raw_data) != 6:
                    continue

                token_id = LONG_TOKEN_ID if int(raw_data[2]) > 100 else SHORT_TOKEN_ID
                if is_float(raw_data[4]):
                    value_dict_1st[token_id].append(float(raw_data[4]))
                if is_float(raw_data[5]):
                    value_dict_2nd[token_id].append(float(raw_data[5]))

            def __get_geomean(data):
                return geometric_mean(data) if len(data) > 0 else 0

            raw_data_list.append(['','','','','-','-'])
            raw_data_list.append(['Success count', '', '', '', len(value_dict_1st[SHORT_TOKEN_ID]) + len(value_dict_1st[LONG_TOKEN_ID]), len(value_dict_2nd[SHORT_TOKEN_ID]) + len(value_dict_2nd[LONG_TOKEN_ID])])
            raw_data_list.append(['geomean (token:short)', '', '', '', f'{__get_geomean(value_dict_1st[SHORT_TOKEN_ID]):.02f}', f'{__get_geomean(value_dict_2nd[SHORT_TOKEN_ID]):.02f}'])
            raw_data_list.append(['geomean (token:long)',  '', '', '', f'{__get_geomean(value_dict_1st[LONG_TOKEN_ID]):.02f}',  f'{__get_geomean(value_dict_2nd[LONG_TOKEN_ID]):.02f}'])

            headers = ['model', 'precision', 'in token', 'out token', '1st inf', '2nd inf']
            tabulate_str = tabulate(raw_data_list, tablefmt="github", headers=headers, stralign='right')
            return f'[RESULT] benchmark (python) / process_time: {time.strftime("%H:%M:%S", time.gmtime(take_time))}\n' + tabulate_str + '\n'
        else:
            return ''

    def is_class_name(name) -> bool:
        return compare_class_name(__class__, name)
