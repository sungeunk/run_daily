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
from test_cases.test_stable_diffusion import TestStableDiffusion
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
def ms_to_sec(value:float):
    return value / 1000

def generate_ccg_table(result_root):
    def __get_inf(item:dict, index, convert=pass_value):
        try:
            return f'{convert(float(item[CmdItemKey.DataItemKey.perf][index])):.2f}'
        except:
            return 'N/A'

    def find_result_item(result_root, key_tuple, cb_cmd_item=None, cb_result_item=None):
        try:
            for cmd_item in result_root.get(key_tuple, []):
                if cb_cmd_item == None or cb_cmd_item(cmd_item):
                    for result_item in cmd_item[CmdItemKey.data_list]:
                        if cb_result_item == None or cb_result_item(result_item):
                            return result_item
        except Exception as e:
            log.error(f'find_result_item: {key_tuple}: {e}')
        return {}

    table = []

    result_item = find_result_item(result_root, ('Resnet50', ModelConfig.INT8, TestBenchmarkapp), lambda item: item.get(CmdItemKey.test_config, {}).get('batch', 0) == 1)
    table.append(['Resnet50 INT8 bs=1', 'fps', __get_inf(result_item, 0), __get_inf(result_item, 0, fps_to_ms)])
    result_item = find_result_item(result_root, ('Resnet50', ModelConfig.INT8, TestBenchmarkapp), lambda item: item.get(CmdItemKey.test_config, {}).get('batch', 0) == 64)
    table.append(['Resnet50 INT8 bs=64', 'fps', __get_inf(result_item, 0), __get_inf(result_item, 0, fps_to_ms)])

    result_item = find_result_item(result_root, ('SD 1.5', ModelConfig.INT8, TestStableDiffusion))
    table.append(['SD 1.5 INT8', 'static, second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])
    result_item = find_result_item(result_root, ('SD 1.5', ModelConfig.FP16, TestStableDiffusion))
    table.append(['SD 1.5 FP16', 'static, second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])
    result_item = find_result_item(result_root, ('SD 2.1', ModelConfig.INT8, TestStableDiffusion))
    table.append(['SD 2.1 INT8', 'static, second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])
    result_item = find_result_item(result_root, ('SD 2.1', ModelConfig.FP16, TestStableDiffusion))
    table.append(['SD 2.1 FP16', 'static, second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])
    result_item = find_result_item(result_root, ('Stable Diffusion XL', ModelConfig.FP16, TestStableDiffusion))
    table.append(['Stable Diffusion XL FP16', 'second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])
    result_item = find_result_item(result_root, ('Stable-Diffusion LCM', ModelConfig.FP16, TestStableDiffusion))
    table.append(['Stable-Diffusion LCM FP16', 'static, second per image (s)', __get_inf(result_item, 0, ms_to_sec), __get_inf(result_item, 0)])

    MODEL_CONFIG = [
        [(ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'llama2-7b INT4 DEFAULT', 1024],
        [(ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'llama2-7b INT4 DEFAULT', 32],
        [(ModelName.llama_3_8b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'llama3-8b INT4 DEFAULT', 1024],
        [(ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'chatGLM3-6b INT4 DEFAULT', 1024],
        [(ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'chatGLM3-6b INT4 DEFAULT', 32],
        [(ModelName.qwen_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'Qwen-7b INT4 DEFAULT', 1024],
        [(ModelName.phi_3_mini_4k_instruct, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'Phi-3-mini INT4 DEFAULT', 1024],
        [(ModelName.gemma_7b_it, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'Gemma-7B INT4 DEFAULT', 1024],
        [(ModelName.mistral_7b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'mistral-7B INT4 DEFAULT', 1024],
    ]
    for config in MODEL_CONFIG:
        result_item = find_result_item(result_root, config[0], None, lambda item: item.get(CmdItemKey.DataItemKey.in_token, 0) == config[2])
        table.append([config[1], f'{config[2]}/{result_item.get(CmdItemKey.DataItemKey.out_token)}, 1st token latency (ms)', __get_inf(result_item, 0), __get_inf(result_item, 0)])
        table.append([config[1], f'{config[2]}/{result_item.get(CmdItemKey.DataItemKey.out_token)}, 2nd token avg (ms)', __get_inf(result_item, 0), __get_inf(result_item, 0)])

    result_item = find_result_item(result_root, ('Whisper base', ModelConfig.UNKNOWN, TestWhisperBase))
    table.append(['Whisper base', 'tokens/second', __get_inf(result_item, 0), __get_inf(result_item, 0, fps_to_ms)])
    result_item = {}
    table.append(['Stable-Diffusion3 (bs=1, FP16, 1024x1024, 28steps)', '', __get_inf(result_item, 0), __get_inf(result_item, 0, fps_to_ms)])

    MODEL_CONFIG = [
        [(ModelName.qwen2_7b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'qwen2-7b INT4 DEFAULT', 1024],
        [(ModelName.phi_2, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'Phi-2 INT4 DEFAULT', 1024],
        [(ModelName.minicpm_1b_sft, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), 'minicpm-1b-sft INT4 DEFAULT', 1024],
    ]
    for config in MODEL_CONFIG:
        result_item = find_result_item(result_root, config[0], None, lambda item: item.get(CmdItemKey.DataItemKey.in_token, 0) == config[2])
        table.append([config[1], f'{config[2]}/{result_item.get(CmdItemKey.DataItemKey.out_token)}, 1st token latency (ms)', __get_inf(result_item, 0), __get_inf(result_item, 0)])
        table.append([config[1], f'{config[2]}/{result_item.get(CmdItemKey.DataItemKey.out_token)}, 2nd token avg (ms)', __get_inf(result_item, 0), __get_inf(result_item, 0)])

    value_list = [ float(raw_list[3]) for raw_list in table if len(raw_list) == 4 and is_float(raw_list[3]) ]
    success_count = len(value_list)
    geomean = 0
    if len(value_list):
        geomean = geometric_mean(value_list)

    table.append([])
    table.append(['Success count', '', '', success_count])
    table.append(['geomean', '', '', f'{float(geomean):.2f}'])

    table_str = tabulate(table, tablefmt="github", stralign='right',
                        headers=['KPI Model', 'description', 'value', 'ms'], floatfmt=['', '', '.2f', '.2f'])
    return f'[Result] ccg table\n{table_str}'

def generate_csv_raw_data(result_root) -> list:
    def __get_inf(item:dict, index, convert=pass_value):
        try:
            return f'{convert(float(item[CmdItemKey.DataItemKey.perf][index])):.2f}'
        except:
            return ''

    # raw_data format: [ model_name, model_precision, in_token, out_token, latency(ms) ]
    def raw_data_for_benchmark(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            for result_item in cmd_item.get(CmdItemKey.data_list, []):
                raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '1st', __get_inf(result_item, 0)])
                raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '2nd', __get_inf(result_item, 1)])

        while len(raw_data_list) < 4: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_benchmarkapp(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            batch = cmd_item[CmdItemKey.test_config]['batch']
            for result_item in cmd_item.get(CmdItemKey.data_list, []):
                raw_data_list.append([key_tuple[0], key_tuple[1], '', '', f'batch:{batch}', __get_inf(result_item, 0, fps_to_ms)])

        while len(raw_data_list) < 2: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_qwen(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            for result_item in cmd_item.get(CmdItemKey.data_list, []):
                raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '1st', __get_inf(result_item, 0)])
                raw_data_list.append([key_tuple[0], key_tuple[1], result_item[CmdItemKey.DataItemKey.in_token], result_item[CmdItemKey.DataItemKey.out_token], '2nd', __get_inf(result_item, 1)])

        while len(raw_data_list) < 16: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_stablediffusion(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            for data_item in cmd_item.get(CmdItemKey.data_list, []):
                raw_data_list.append([key_tuple[0], key_tuple[1], '', '', 'pipeline', __get_inf(data_item, 0)])

        while len(raw_data_list) < 1: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    def raw_data_for_whisperbase(key_tuple):
        raw_data_list = []
        for cmd_item in result_root.get(key_tuple, []):
            for data_item in cmd_item.get(CmdItemKey.data_list, []):
                raw_data_list.append([key_tuple[0], key_tuple[1], '', '', 'ms/token', __get_inf(data_item, 0, fps_to_ms)])

        while len(raw_data_list) < 1: raw_data_list.append([key_tuple[0], key_tuple[1]])
        return raw_data_list

    MODEL_REPORT_CONFIG = [
        [(ModelName.baichuan2_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.chatglm3_6b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.glm_4_9b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.gemma_7b_it, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.llama_2_7b_chat_hf, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.llama_3_8b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.minicpm_1b_sft, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.mistral_7b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.phi_2, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.phi_3_mini_4k_instruct, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [(ModelName.qwen_7b_chat, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('qwen_usage', ModelConfig.INT8, TestMeasuredUsageCpp), raw_data_for_qwen],
        [(ModelName.qwen2_7b, ModelConfig.OV_FP16_4BIT_DEFAULT, TestBenchmark), raw_data_for_benchmark],
        [('Resnet50', ModelConfig.INT8, TestBenchmarkapp), raw_data_for_benchmarkapp],
        [('SD 1.5', ModelConfig.FP16, TestStableDiffusion), raw_data_for_stablediffusion],
        [('SD 1.5', ModelConfig.INT8, TestStableDiffusion), raw_data_for_stablediffusion],
        [('SD 2.1', ModelConfig.FP16, TestStableDiffusion), raw_data_for_stablediffusion],
        [('SD 2.1', ModelConfig.INT8, TestStableDiffusion), raw_data_for_stablediffusion],
        [('Stable Diffusion XL', ModelConfig.FP16, TestStableDiffusion), raw_data_for_stablediffusion],
        [('Stable-Diffusion LCM', ModelConfig.FP16, TestStableDiffusion), raw_data_for_stablediffusion],
        [('Whisper base', ModelConfig.UNKNOWN, TestWhisperBase), raw_data_for_whisperbase],
        [('SD 3.0 Dynamic', ModelConfig.MIXED, TestStableDiffusion), raw_data_for_stablediffusion],
        [('SD 3.0 Static', ModelConfig.MIXED, TestStableDiffusion), raw_data_for_stablediffusion],
    ]

    table = []
    for config in MODEL_REPORT_CONFIG:
        key_tuple = config[0]
        raw_data_func = config[1]
        table.extend(raw_data_func(key_tuple))
    return table

def generate_csv_table(result_root) -> tuple[str, int, int]:
    table = generate_csv_raw_data(result_root)

    value_list = [ float(raw_list[5]) for raw_list in table if len(raw_list) == 6 and is_float(raw_list[5]) ]
    success_count = len(value_list)
    geomean = 0
    if len(value_list):
        geomean = geometric_mean(value_list)

    def add_table_for_llm(table, token_size, exec):
        __value_list = [ float(raw_list[5]) for raw_list in table if len(raw_list) == 6 and is_float(raw_list[5]) and raw_list[2] == token_size and raw_list[4] == exec ]
        __success_count = len(value_list)
        if len(__value_list):
            __geomean = geometric_mean(__value_list)
            table.append([f'geomean (LLM/{exec}/{token_size:4})', '', '', '', '', f'{float(__geomean):.2f}'])
        else:
            table.append([f'geomean (LLM/{exec}/{token_size:4})', '', '', '', '', 0])

    table.append(['', '', '', '', '', '-'])
    table.append(['Success count', '', '', '', '', success_count])
    table.append(['geomean', '', '', '', '', f'{float(geomean):.2f}'])
    add_table_for_llm(table, 32, '2nd')
    add_table_for_llm(table, 32, '1st')
    add_table_for_llm(table, 1024, '2nd')
    add_table_for_llm(table, 1024, '1st')

    tabulate_str = tabulate(table, tablefmt="github", headers=['model', 'precision', 'in', 'out', 'exec', 'latency(ms)'], floatfmt='.2f', stralign='right', numalign='right')
    return f'[Result] csv table\n' + tabulate_str, geomean, success_count

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

def compare_result_item_map(fos, callback, this_result_root, ref_map={}):
    for key_tuple, cmd_item_list in this_result_root.items():
        for cmd_item in cmd_item_list:
            for data_item in cmd_item.get(CmdItemKey.data_list, []):
                callback(fos, key_tuple, data_item, None)

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

    sts_str = ('OK' if iou > 0.5 else 'DIFF') if iou > 0 else 'ERR'

    fos.write(f'[TEXT][{key_tuple}][{in_token}][{sts_str}][iou:{iou:0.2f}]\n')
    if iou == 1:
        fos.write(f'\t[this] {this_text}\n')
    else:
        fos.write(f'\t[this] {this_text}\n')
        fos.write(f'\t[ref ] {ref_text}\n')
    fos.write(f'\n')

def get_test_list(target:str=''):
    if target == '':
        try:
            target = GlobalConfig().test_filter
        except Exception as e:
            pass

    all_test_list = [
        TestBenchmark,
        TestMeasuredUsageCpp,
        TestStableDiffusion,
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
    ccg_tabulate = ''#generate_ccg_table(result_root)
    csv_tabulate, csv_geomean, csv_success_cnt = generate_csv_table(result_root)
    summary_tabulate = generate_summary(args, PROCESS_TIME)
    versions_table = generate_versions()

    #
    # System info
    #
    APP = os.path.join('scripts', 'device_info.py')
    system_info, returncode = call_cmd(args, f'python {APP}', shell=True, verbose=False)

    #
    # Generate Report
    #
    out.write('<pre>\n')
    out.write(f'{summary_tabulate}\n\n')

    # Error list
    error_str = generate_error_table(result_root)
    if len(error_str) > 0:
        out.write(f'{error_str}\n\n')

    if len(ccg_tabulate) > 0:
        out.write(ccg_tabulate + '\n\n')
    if len(csv_tabulate) > 0:
        out.write(csv_tabulate + '\n\n')

    test_report_str = generate_report_for_each_test(result_root)
    if len(test_report_str) > 0:
        out.write(test_report_str + '\n\n')

    # Version & system
    out.write(f'{versions_table}\n\n')
    out.write(f'{system_info}\n\n')

    # generated text
    result_ref_map = load_result_file(replace_ext(args.ref_report, "pickle")) if args.ref_report else {}
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
    csv_tabulate, csv_geomean, csv_success_cnt = generate_csv_table(result_root)
    return f'({float(csv_geomean):.2f}/{csv_success_cnt})'
