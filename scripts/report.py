#!/usr/bin/env python3

import logging as log
import pickle
import re
import string
import time

from io import StringIO
from statistics import geometric_mean
from tabulate import tabulate

# import class/method from local
from common_utils import *
from test_cases.test_template import *
from download_ov_nightly import generate_manifest

# import all class in test_cases directory
from test_cases.test_benchmark import TestBenchmark
from test_cases.test_benchmark_app import TestBenchmarkapp
from test_cases.test_chat_sample import TestChatSample
from test_cases.test_measured_usage_cpp import TestMeasuredUsageCpp
from test_cases.test_stable_diffusion_genai import TestStableDiffusionGenai
from test_cases.test_stable_diffusion_DGfx_E2E_AI import TestStableDiffusionDGfxE2eAi
from test_cases.test_whisper_base import TestWhisperBase


def convert_url(filename) -> str:
    cfg = GlobalConfig()
    return f'{cfg.BACKUP_SERVER}/daily/{platform.node()}/{filename}'

def save_result_file(filepath, result_root):
    try:
        with open(filepath, 'wb') as fos:
            pickle.dump(result_root, fos)
    except Exception as e:
        log.error(f'Report::save: {e}')

def load_result_file(filepath) -> dict:
    try:
        with open(filepath, 'rb') as fis:
            return pickle.load(fis)
    except Exception as e:
        log.error(f'load_result_file: {filepath}')
        log.error(f'reason: {e}')
        return {}

def pass_value(value:float):
    return value
def fps_to_ms(value:float):
    return 1000 / value
def sec_to_ms(value:float):
    return value * 1000

def generate_csv_raw_data(result_root) -> list:
    def __get_inf(item:dict, index, convert=pass_value):
        try:
            return f'{convert(float(item[CmdItemKey.DataItemKey.perf][index])):.2f}'
        except:
            return ''

    # raw_data format: [ model_name, model_precision, in_token, out_token, latency(ms) ]
    def raw_data_for_benchmark(key_tuple, args={}):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                for result_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '1st', __get_inf(result_item, 0)])
                    raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '2nd', __get_inf(result_item, 1)])

        while len(raw_data_list) < args.get('data_num', 4): raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_benchmarkapp(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                batch = cmd_item[CmdItemKey.test_config]['batch']
                for result_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], '', '', f'batch:{batch}', __get_inf(result_item, 0, fps_to_ms)])

        while len(raw_data_list) < 2: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_measure_usage(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                peak_mem_usage_size = sizestr_to_num(cmd_item[CmdItemKey.peak_mem_usage_size])
                peak_mem_usage_percent = cmd_item[CmdItemKey.peak_mem_usage_percent]

                for result_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], 'memory size', peak_mem_usage_size])
                    raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], 'memory percent', peak_mem_usage_percent])

        while len(raw_data_list) < 16: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_stablediffusion(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], '', '', 'pipeline', __get_inf(data_item, 0, sec_to_ms)])

        print(f'raw_data_list: {raw_data_list}')
        while len(raw_data_list) < 1: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_whisperbase(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            if cmd_item.get(CmdItemKey.return_code, -1) == 0:
                for data_item in cmd_item.get(CmdItemKey.data_list, []):
                    raw_data_list.append([key_tuple[0], key_tuple[1], '', '', 'ms/token', __get_inf(data_item, 0, fps_to_ms)])

        while len(raw_data_list) < 1: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    MODEL_REPORT_CONFIG = [
        [(ModelName.baichuan2_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [("glm-4-9b-chat-hf", ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.gemma_7b_it, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('llama-3.1-8b-instruct', ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.minicpm_1b_sft, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.mistral_7b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('phi-3.5-mini-instruct', ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.phi_3_mini_4k_instruct, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.qwen_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('qwen2-7b-instruct', ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('minicpm-v-2_6', ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark, {'data_num':2}],
        [('qwen_usage', ModelConfig.INT8, TestMeasuredUsageCpp), raw_data_for_measure_usage],
        [('Resnet50', ModelConfig.INT8, TestBenchmarkapp), raw_data_for_benchmarkapp],
        [('Whisper base', ModelConfig.UNKNOWN, TestWhisperBase), raw_data_for_whisperbase],
        [('stable-diffusion-v1-5', ModelConfig.FP16, TestStableDiffusionGenai), raw_data_for_stablediffusion],
        [('stable-diffusion-v2-1', ModelConfig.FP16, TestStableDiffusionGenai), raw_data_for_stablediffusion],
        [('stable-diffusion-v3.0', ModelConfig.FP16, TestStableDiffusionDGfxE2eAi), raw_data_for_stablediffusion],
        [('stable-diffusion-xl', ModelConfig.FP16, TestStableDiffusionDGfxE2eAi), raw_data_for_stablediffusion],
        [('lcm-dreamshaper-v7', ModelConfig.FP16, TestStableDiffusionGenai), raw_data_for_stablediffusion],
    ]

    table = []
    for config in MODEL_REPORT_CONFIG:
        key_tuple = config[0]
        raw_data_func = config[1]

        if len(config) == 3:
            table.extend(raw_data_func(key_tuple, config[2]))
        else:
            table.extend(raw_data_func(key_tuple))
    return table

# input: raw_data_table  << get from generate_csv_raw_data()
# return (geomean, success_count)
def get_static_info_from_raw_data(raw_data_table) -> tuple[int, int]:
    success_count = 0
    value_list = []
    for item in raw_data_table:
        if len(item) == 6 and is_float(item[5]):
            success_count += 1
            if item[0] != 'qwen_usage':
                value_list.append(float(item[5]))

    geomean = geometric_mean(value_list) if len(value_list) else 0
    return geomean, success_count

def generate_csv_table(result_root, format_number = True) -> list:
    csv_table = generate_csv_raw_data(result_root)
    geomean, success_count = get_static_info_from_raw_data(csv_table)

    # formating table for report
    if format_number:
        for item in csv_table:
            if len(item) == 6 and item[0] == 'qwen_usage':
                if item[4] == 'memory percent':
                    item[5] = f'{item[5]:.2f} %'
                elif item[4] == 'memory size':
                    item[5] = sizeof_fmt(item[5])

    def add_table_for_llm(table, token_size, exec):
        __value_list = [ float(raw_list[5]) for raw_list in table if len(raw_list) == 6 and is_float(raw_list[5]) and raw_list[2] == token_size and raw_list[4] == exec ]
        if len(__value_list):
            __geomean = geometric_mean(__value_list)
            table.append([f'geomean (LLM/{exec}/{token_size:4})', '', '', '', '', f'{float(__geomean):.2f}'])
        else:
            table.append([f'geomean (LLM/{exec}/{token_size:4})', '', '', '', '', 0])

    csv_table.append(['', '', '', '', '', ''])
    csv_table.append(['Success count', '', '', '', '', success_count])
    csv_table.append(['geomean', '', '', '', '', f'{float(geomean):.2f}'])
    add_table_for_llm(csv_table, 32, '2nd')
    add_table_for_llm(csv_table, 32, '1st')
    add_table_for_llm(csv_table, 1024, '2nd')
    add_table_for_llm(csv_table, 1024, '1st')

    return csv_table

def generate_csv_tabulate_str(csv_table):
    tabulate_str = tabulate(csv_table, tablefmt="github", headers=['model', 'precision', 'in', 'out', 'exec', 'latency(ms)'], floatfmt='.2f', stralign='right', numalign='right')
    return f'[Result] csv table\n' + tabulate_str

def calculate_score(ref, target):
    if len(ref) == 0 or len(target) == 0:
        return 0
    words_ref = set(re.sub('[' + string.punctuation + ']', '', ref).split())
    words_target = set(re.sub('[' + string.punctuation + ']', '', target).split())
    intersection = words_ref.intersection(words_target)
    union = words_ref.union(words_target)
    try:
        return len(intersection) / len(union)
    except:
        return 0

def __get_data_list(parent_list, index):
    if len(parent_list) > index:
        return parent_list[index].get(CmdItemKey.data_list, [])
    else:
        return []

def __get_data_item(parent_list, index):
    if len(parent_list) > index:
        return parent_list[index]
    else:
        return None

def compare_result_item_map(fos, callback, this_result_root, ref_map={}):
    SKIP_KEY_LIST = [
        'qwen_usage',
        'Whisper base',
        'stable-diffusion-v1-5',
        'stable-diffusion-v2-1',
        'lcm-dreamshaper-v7',
        'Resnet50',
    ]

    for this_key_tuple, this_cmd_item_list in this_result_root.items():
        if this_key_tuple[0] in SKIP_KEY_LIST:
            continue

        ref_cmd_item_list = ref_map.get(this_key_tuple, [])
        if len(ref_cmd_item_list) == 0:
            continue

        for y in range(0, max(len(this_cmd_item_list), len(ref_cmd_item_list))):
            this_data_list = __get_data_list(this_cmd_item_list, y)
            ref_data_list = __get_data_list(ref_cmd_item_list, y)

            for x in range(0, max(len(this_data_list), len(ref_data_list))):
                this_item = __get_data_item(this_data_list, x)
                ref_item = __get_data_item(ref_data_list, x)
                if this_item != None:
                    callback(fos, this_key_tuple, this_item, ref_item)

# tabulate_data: [model, precision, token, iou]
def generate_compared_text_summary(tabulate_data, key_tuple:tuple, this_item:dict, ref_item:dict):
    this_text = this_item.get(CmdItemKey.DataItemKey.generated_text, '')
    if len(this_text) == 0:
        return

    in_token = this_item.get(CmdItemKey.DataItemKey.in_token, 0)
    if ref_item == None:
        return

    ref_text = ref_item.get(CmdItemKey.DataItemKey.generated_text, '')
    iou = calculate_score(this_text, ref_text)

    if iou < 0.2:
        tabulate_data.append([key_tuple[0], key_tuple[1], in_token, f'{iou:.2f}'])

def print_compared_text(fos, key_tuple:tuple, this_item:dict, ref_item:dict):
    LIMIT_TEXT_LENGTH = 256
    this_text = this_item.get(CmdItemKey.DataItemKey.generated_text, '')
    if len(this_text) == 0:
        fos.write(f'{key_tuple} has no generated text.\n')
        return

    in_token = this_item.get(CmdItemKey.DataItemKey.in_token, 0)
    if ref_item == None:
        fos.write(f'[TEXT][{key_tuple}][{in_token}]\n')
        fos.write(f'\t[this] {this_text}\n')
        return

    ref_text = ref_item.get(CmdItemKey.DataItemKey.generated_text, '')
    iou = calculate_score(this_text, ref_text)

    this_text = this_text if len(this_text) < LIMIT_TEXT_LENGTH else this_text[0:LIMIT_TEXT_LENGTH]
    this_text = this_text.replace("<s>", "_s_")     # WA: remove '<s>'. It will replace to cancel line in outlook.
    ref_text = ref_text if len(ref_text) < LIMIT_TEXT_LENGTH else ref_text[0:LIMIT_TEXT_LENGTH]
    ref_text = ref_text.replace("<s>", "_s_")

    sts_str = ('OK' if iou > 0.2 else 'DIFF') if iou > 0 else 'ERR'
    if iou <= 0.2 :
        fos.write(f'[TEXT][{key_tuple}][{in_token}][{sts_str}][iou:{iou:0.2f}]\n')
        fos.write(f'\t[this] {this_text}\n')
        fos.write(f'\t[ref ] {ref_text}\n\n')

def get_test_list(target:str=''):
    if target == '':
        try:
            target = GlobalConfig().test_filter
        except Exception as e:
            pass

    all_test_list = [
        TestBenchmark,
        TestMeasuredUsageCpp,
        # TestStableDiffusion,
        TestStableDiffusionGenai,
        TestStableDiffusionDGfxE2eAi,
        TestWhisperBase,
        TestBenchmarkapp,
        TestChatSample,
    ]
    target_list = target.split(',')
    test_list = []
    if target == '':
        test_list = all_test_list
    else:
        for target in target_list:
            for test_class in all_test_list:
                if test_class.is_class_name(target):
                    test_list.append(test_class)
                    break
    return test_list

def generate_report_for_each_test(result_root):
    all_reports_str = ''
    test_class_list = get_test_list()
    for test_class in test_class_list:
        report_str = test_class.generate_report(result_root)
        if len(report_str) > 0:
            all_reports_str += report_str + '\n'
            
    return all_reports_str

def generate_summary(args, PROCESS_TIME) -> str:
    cfg = GlobalConfig()
    summary_table_data = []
    summary_table_data.append(['Purpose', f'{args.description}'])
    summary_table_data.append(['TOTAL TASK TIME', f'{time.strftime("%H:%M:%S", time.gmtime(PROCESS_TIME))}'])
    summary_table_data.append(['OpenVINO', f'{cfg.OV_VERSION}'])
    summary_table_data.append(['Report', f'{convert_url(cfg.REPORT_FILENAME)}'])
    summary_table_data.append(['RawLog', f'{convert_url(cfg.RAW_FILENAME)}'])
    summary_table_data.append(['PipRequirements', f'{convert_url(cfg.PIP_FREEZE_FILENAME)}'])

    RESULT_SD3_DYNAMIC_PATH = convert_path(f'{args.output_dir}/{cfg.RESULT_SD3_DYNAMIC_FILENAME}')
    if exists_path(RESULT_SD3_DYNAMIC_PATH):
        summary_table_data.append(['SD3.0 dynamic', f'{convert_url(cfg.RESULT_SD3_DYNAMIC_FILENAME)}'])

    RESULT_SD3_STATIC_PATH = convert_path(f'{args.output_dir}/{cfg.RESULT_SD3_STATIC_FILENAME}')
    if exists_path(RESULT_SD3_STATIC_PATH):
        summary_table_data.append(['SD3.0 static', f'{convert_url(cfg.RESULT_SD3_STATIC_FILENAME)}'])

    if args.ref_report != None:
        summary_table_data.append(['Reference Report', f'{convert_url(os.path.basename(args.ref_report))}'])
    summary_table = tabulate(summary_table_data, tablefmt="youtrack")
    return '[ Summary ]\n' + summary_table

def generate_versions() -> str:
    cfg = GlobalConfig()

    # C:\dev\sungeunk\run_daily\openvino_nightly\latest_ov_setup_file.txt
    # C:\dev\sungeunk\run_daily\openvino_nightly\2025.0.0.17826_a8dfb18f\setupvars.bat
    LATEST_OV_FILEPATH = convert_path(f'{cfg.PWD}/openvino_nightly/latest_ov_setup_file.txt')
    try:
        with open(LATEST_OV_FILEPATH, 'rt') as fis1:
            SETUP_FILEPATH = fis1.readline()
            MANIFEST_FILEPATH = convert_path(os.path.join(os.path.dirname(SETUP_FILEPATH), 'manifest.yml'))
            return '[ manifest ]\n' + generate_manifest(MANIFEST_FILEPATH)
    except:
        return ''

def generate_error_table(result_root) -> str:
    raw_data_list = []
    for key_tuple, cmd_item_list in result_root.items():
        for cmd_item in cmd_item_list:
            returncode = cmd_item.get(CmdItemKey.return_code, -1)
            if returncode != 0:
                raw_data_list.append([key_tuple[2].__name__, key_tuple[0], key_tuple[1], returncode])

    if len(raw_data_list):
        return '[ Error ] \n' + tabulate(raw_data_list, tablefmt="github", headers=['TestClass', 'Model', 'Precision', 'returncode'])
    else:
        return ''

def generate_report_str(args, result_root:dict, PROCESS_TIME) -> str:
    out = StringIO()
    csv_table = generate_csv_table(result_root)
    csv_tabulate_str = generate_csv_tabulate_str(csv_table)
    summary_tabulate = generate_summary(args, PROCESS_TIME)
    versions_table = generate_versions()

    #
    # System info
    #
    APP = os.path.join('scripts', 'device_info.py')
    system_info, returncode = call_cmd(args, f'python {APP}', shell=False, verbose=False)

    #
    # Generate Report
    #
    out.write('<pre>\n')
    out.write(f'{summary_tabulate}\n\n')

    # Error list
    error_str = generate_error_table(result_root)
    if len(error_str) > 0:
        out.write(f'{error_str}\n\n')

    # Error table for generated text
    result_ref_map = load_result_file(replace_ext(args.ref_report, "pickle")) if args.ref_report else {}
    generated_text_table = []
    compare_result_item_map(generated_text_table, generate_compared_text_summary, result_root, result_ref_map)
    if len(generated_text_table):
        tabulate_str = tabulate(generated_text_table, tablefmt="github", headers=['model', 'precision', 'in_token', 'iou'])
        out.write(f'[ Error table for generated text ]\n')
        out.write(tabulate_str + '\n\n')

    if len(csv_tabulate_str) > 0:
        out.write(csv_tabulate_str + '\n\n')

    test_report_str = generate_report_for_each_test(result_root)
    if len(test_report_str) > 0:
        out.write(test_report_str + '\n\n')

    # Version & system
    out.write(f'{versions_table}\n\n')
    out.write(f'{system_info}\n\n')

    # Print all generated text
    compare_result_item_map(out, print_compared_text, result_root, result_ref_map)
    out.write(f'\n\n')

    # command list
    last_cmd = ''
    for key, cmd_item_list in result_root.items():
        for cmd_item in cmd_item_list:
            cmd = cmd_item.get(CmdItemKey.cmd, '')
            if last_cmd != cmd:
                out.write(f'[CMD][{key}] {cmd}\n')
                last_cmd = cmd

    out.write('\n</pre>')

    return out.getvalue()

def generate_mail_title_suffix(result_root:dict):
    csv_table = generate_csv_raw_data(result_root)
    geomean, success_count = get_static_info_from_raw_data(csv_table)
    return f'({float(geomean):.2f}/{success_count})'
