#!/usr/bin/env python3

import argparse
import datetime as dt
import enum
from enum import auto
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import platform
import psutil
import re
import string
import subprocess
import sys
import threading
import time

from glob import glob
from openvino.runtime import get_version
from pathlib import Path
from statistics import geometric_mean
from tabulate import tabulate

from device_info import get_external_ip_address, python_packages



################################################
# Key
################################################

class StrEnum(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class ResultKey(StrEnum):
    id = auto()
    cmd = auto()
    return_code = auto()
    raw_log = auto()
    generated_text = auto()
    in_token = auto()
    out_token = auto()
    perf = auto()
    peak_cpu_usage_percent = auto()
    peak_mem_usage_percent = auto()
    peak_mem_usage_size = auto()


################################################
# Global variable
################################################
NOW = dt.datetime.now().strftime("%Y%m%d_%H%M")
IS_WINDOWS = platform.system() == 'Windows'
PWD = os.path.abspath('.')
REPORT_JSON_FILENAME = f'daily.{NOW}.{get_version().replace("/", "_")}.json'
REPORT_FILENAME = f'daily.{NOW}.{get_version().replace("/", "_")}.report'
RAW_FILENAME = f'daily.{NOW}.{get_version().replace("/", "_")}.raw'
PIP_FREEZE_FILENAME = f'daily.{NOW}.requirements.txt'
RESULT_SD3_DYNAMIC_FILENAME = f'daily.{NOW}.sd.dynamic.png'
RESULT_SD3_STATIC_FILENAME = f'daily.{NOW}.sd.static.png'
OUT_TOKEN_LEN = 256
TRY_EXEC_NUM=1
BENCHMARK_ITER_NUM=3
BACKUP_SERVER='http://dg2raptorlake.ikor.intel.com/daily'
MODEL_DATE = 'WW44_llm-optimum_2024.5.0-17246-44b86a860ec'
MODEL_DATE_WW32_LLM = 'WW32_llm_2024.4.0-16283-41691a36b90'


################################################
# logging
################################################
log = logging.getLogger()



################################################
# Utils
################################################
def sizeof_fmt(num):
    for unit in ("", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    raise Exception(f'Out of bound!!! size({num})')

def get_file_info(path):
    filesize = os.path.getsize(path)

    with open(path, 'rb') as fis:
        hash = hashlib.md5(fis.read()).hexdigest()

    return filesize, hash

def exists_path(path):
    try:
        return os.path.exists(path)
    except:
        return False

def get_now():
    print(f'{NOW}')

def convert_path(args, filename):
    return os.path.join(*[f'{args.output_dir}', filename])

def convert_url(filename):
    return f'{BACKUP_SERVER}/{platform.node()}/{filename}'

#
# WA: At subprocess.popen(cmd, ...), the cmd should be string on ubuntu or be string array on windows.
#
def convert_cmd_for_popen(cmd):
    return cmd.split() if IS_WINDOWS else cmd

def send_mail(report_path, recipients, title, suffix_title=''):
    MAIL_TITLE = f'[{platform.node()}/{NOW}] {title} {suffix_title}'
    MAIL_TO = recipients

    if IS_WINDOWS:
        try:
            ID_RSA_PATH = f'{os.environ["USERPROFILE"]}\\.ssh\\id_rsa'
            MAIL_RELAY_SERVER = os.environ["MAIL_RELAY_SERVER"]
            cmd = f'ssh -i {ID_RSA_PATH} {MAIL_RELAY_SERVER} \"mail --content-type=text/html -s \\\"{MAIL_TITLE}\\\" {MAIL_TO} \" < {report_path}'
        except Exception as e:
            log.error(f'Exception: {str(e)}')
            return
    else:
        cmd = f'cat {report_path} | mail --content-type=text/html -s \"{MAIL_TITLE}\" {MAIL_TO}'

    subprocess.call(cmd, shell=True)

def backup_files(args, files):
    try:
        MAIL_RELAY_SERVER = os.environ["MAIL_RELAY_SERVER"]
    except Exception as e:
        log.error(f'Exception: {str(e)}')
        return

    REMOTE_PATH=f'{MAIL_RELAY_SERVER}:/var/www/html/daily/{platform.node()}/'
    if IS_WINDOWS:
        log.info(f'backup files to {REMOTE_PATH}')
        for file in files:
            if exists_path(file):
                log.info(f'  ok: {file}')
                call_cmd(args=args, cmd=f'scp.exe {file} {REMOTE_PATH}', shell=True, verbose=False)
            else:
                log.error(f'  failed: could not find {file}')


class HWDataKey(enum.Enum):
    TIMESTAMP = 0
    MEM_USAGE_PERCENT = 1
    MEM_USAGE_SIZE = 2
    CPU_USAGE_PERCENT = 3

class HWResourceTracker(threading.Thread):
    def __init__(self, process = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = process
        self.running = True
        self.data = []
        self.period_time = 0.1  # unit: seconds
        self.lock = threading.Lock()

    def set_property(self, props):
        self.period_time = props.get('period_time', 0.1)

    def run(self):
        while self.running:
            time.sleep(self.period_time)
            timestamp_ms = round(time.time()*1000)

            if (self.process != None):
                if self.process.is_running():
                    self._append_data(timestamp_ms,
                                 self.process.memory_percent(),
                                 self.process.memory_info().rss,
                                 self.process.cpu_percent())
            else:
                memory_usage_dict = dict(psutil.virtual_memory()._asdict())
                self._append_data(timestamp_ms,
                             memory_usage_dict['percent'],
                             memory_usage_dict['used'],
                             psutil.cpu_percent())

    def _append_data(self, *argv):
        self.lock.acquire()
        self.data.append([ value for value in argv ])
        self.lock.release()

    def _get_usage_info(self):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ key.name for key in HWDataKey ])
        df_max = df.max()
        df_min = df.min()
        df_mean = df.mean()
        self.lock.release()

        return df_mean[HWDataKey.CPU_USAGE_PERCENT.name], \
               df_max[HWDataKey.MEM_USAGE_SIZE.name] - df_min[HWDataKey.MEM_USAGE_SIZE.name], \
               df_max[HWDataKey.MEM_USAGE_PERCENT.name] - df_min[HWDataKey.MEM_USAGE_PERCENT.name]

    def get_data_min_max_last(self, key):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ dataKey.name for dataKey in HWDataKey ])
        df_max = df.max()
        df_min = df.min()
        df_last = df.iloc[-1]
        self.lock.release()

        return df_min[key.name], df_max[key.name], df_last[key.name]

    def save_graph(self, key, save_path):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ key.name for key in HWDataKey ])
        df.plot(x=HWDataKey.TIMESTAMP.name, y=key.name, kind='line')
        plt.savefig(save_path)
        self.lock.release()

    def stop(self):
        if self.is_alive():
            self.running = False
            self.join()
            self.running = True

        return self._get_usage_info()

# MainKey: test case name
# SubKey: 'in_token'
# ItemKey: 'cmd', 'generated_text', 'raw_log', 'in_token', 'out_token', 'peak_cpu_usage_percent', 'peak_mem_usage_percent', 'peak_mem_usage_size'
class Report:
    def __init__(self):
        self.data = {}

    def load(self, filepath: Path) -> dict:
        with open(filepath) as f:
            self.data = json.load(f)

    def save(self, filepath: Path) -> bool:
        try:
            with open(filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
                return True
        except Exception as e:
            log.error(f'Report::save: {e}')
            return False

    def add_result(self, key: str, results: list[dict]):
        if key in self.data:
            self.data[key].append(results)
        else:
            self.data[key] = results

    def get_result(self, key: str) -> list[dict]:
        if key in self.data:
            return self.data[key]
        else:
            return []


################################################
# Test class
################################################
class AppClassChatglm3():
    @staticmethod
    def get_model_path(args, origin=False):
        MODEL_FILE_NAME = 'openvino_model.xml' if origin else 'modified_openvino_model.xml'
        return os.path.join(*[f'{args.model_dir}', f'{MODEL_DATE}', 'chatglm3-6b', 'pytorch', 'ov', 'OV_FP16-4BIT_DEFAULT', MODEL_FILE_NAME]), \
               os.path.join(*[f'{args.model_dir}', 'chatglm3-6b_openvino_tokenizer', 'openvino_tokenizer.xml']), \
               os.path.join(*[f'{args.model_dir}', 'chatglm3-6b_openvino_tokenizer', 'openvino_detokenizer.xml'])

    @staticmethod
    def get_executor_path(args):
        return os.path.join(*[args.bin_dir, 'chatglm', 'chatglm' + ('.exe' if IS_WINDOWS else '')])

    @staticmethod
    def get_cmd(args, key=None, index=-1):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, TOKENIZER_PATH, DETOKENIZER_PATH = __class__.get_model_path(args)
        cmd = f'{APP_PATH} -m {MODEL_PATH} -token {TOKENIZER_PATH} -detoken {DETOKENIZER_PATH} -d {args.device} --output_fixed_len {OUT_TOKEN_LEN}'
        if index >= 0:
            cmd += f' --select_inputs {index}'
        return cmd

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)
        sentences_map = __class__.parse_generated_sentences(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode
            item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token), '')

        return result_list

    @staticmethod
    def convert_model(args):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, TOKENIZER_PATH, DETOKENIZER_PATH = __class__.get_model_path(args)
        if not exists_path(MODEL_PATH):
            MODEL_PATH, TOKENIZER_PATH, DETOKENIZER_PATH = __class__.get_model_path(args, True)
            cmd = f'{APP_PATH} -m {MODEL_PATH} -token {TOKENIZER_PATH} -detoken {DETOKENIZER_PATH} -d {args.device} --reduce_logits'
            call_cmd(args, cmd)

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        for line in output.splitlines():
            # {index}, {input token}, {output token}, {1st inf time(ms)}, {2nd inf time(ms)}
            match_obj = re.search(f'(\d+), (\d+), (\d+), (\d+.\d+), (\d+.\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = int(values[0])
                item[ResultKey.in_token] = int(values[1])
                item[ResultKey.out_token] = int(values[2])
                item[ResultKey.perf] = [float(values[3]), float(values[4])]
                ret.append(item)

        if len(ret) == 0:
            #         input lenghth 1564
            # First token took 616.589 ms
            # To improve the performance of your game, there are several steps you can take to optimize its speed and responsiveness. Here are some suggestions to consider:<br><br>1. Update your graphics card drivers: Drivers are essential for your graphics card to communicate with your computer. If you have outdated drivers, it can affect the game's performance. Make sure you have the latest drivers installed by visiting the manufacturer's website or using a driver update utility.<br><br>2. Close background applications: Background applications can consume resources and slow down your game's performance. Close unnecessary programs, such as browser tabs
            # Other Avg inference took total 3378.28 ms token num 127 first 616.589 ms  avg 26.6007 ms
            index = 0
            in_token_len = 0
            for line in output.splitlines():
                match_obj = re.search(f'input lenghth (\d+)', line)
                if match_obj != None:
                    in_token_len = int(match_obj.groups()[0])
                    continue

                match_obj = re.search(f'Other Avg inference took total (\d+.\d+) ms token num (\d+) first (\d+.\d+) ms  avg (\d+.\d+) ms', line)
                if match_obj != None:
                    values = match_obj.groups()
                    item = {}
                    item[ResultKey.id] = index
                    item[ResultKey.in_token] = in_token_len
                    item[ResultKey.out_token] = int(values[1])
                    item[ResultKey.perf] = [float(values[2]), float(values[3])]
                    ret.append(item)
                    index += 1
                    continue

        return ret

    @staticmethod
    def parse_generated_sentences(output):
        input_len = 0
        index = 0
        sentence = ''
        sentence_map = {}
        is_sentence = False
        for line in output.splitlines():
            match_obj = re.search(f'input lenghth (\d+)', line)
            if (match_obj != None):
                input_len = int(match_obj.groups()[0])
                continue
            match_obj = re.search(f'First token took ', line)
            if (match_obj != None):
                is_sentence = True
                continue
            match_obj = re.search(f'Other Avg inference took total ', line)
            if (match_obj != None):
                is_sentence = False
                sentence_map[input_len] = sentence
                sentence = ''
                index += 1
                continue

            if is_sentence:
                sentence += line
        return sentence_map

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref):
        ret_str = ''
        if data_this != None and len(data_this) > 0:
            EnabledMemCheck = ResultKey.peak_cpu_usage_percent in data_this[0].keys()
            def __get_inf(item:dict, index):
                try:
                    return item[ResultKey.perf][index]
                except:
                    return 0

            if EnabledMemCheck:
                headers = ['index', 'in token', 'out token', '1st inf', '2nd inf', 'CPU (%)', 'Memory', 'Memory (%)']
                floatfmt = ['', '', '', '.2f', '.2f', '.2f', '', '.2f']
                raw_data_list = []
                for item in data_this:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.in_token, 0), item.get(ResultKey.out_token, 0),
                                          __get_inf(item, 0), __get_inf(item, 1),
                                          item.get(ResultKey.peak_cpu_usage_percent, 0), item.get(ResultKey.peak_mem_usage_size, 0), item.get(ResultKey.peak_mem_usage_percent, 0)])
            else:
                headers = ['index', 'in token', 'out token', '1st inf', '2nd inf']
                floatfmt = ['', '', '', '.2f', '.2f']
                raw_data_list = []
                for item in data_this:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.in_token, 0), item.get(ResultKey.out_token, 0), __get_inf(item, 0), __get_inf(item, 1)])

            ret_str = tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='left', numalign='right')
        return ret_str

class AppClassChatglm3MeasuredUsage():
    @staticmethod
    def execute(args, key=None):
        ret_list = []
        remove_cache(args)
        for index in range(8):
            cmd = AppClassChatglm3.get_cmd(args, key, index)
            log.info(f'{cmd}')

            tracker = HWResourceTracker()
            tracker.start()
            output, returnCode = call_cmd(args, cmd)
            cpu_usage_percent, mem_usage_size, mem_usage_percent = tracker.stop()

            result_list = __class__.parse_output(output)
            sentences_map = __class__.parse_generated_sentences(output)
            ret_list += result_list

            index = 0
            for item in result_list:
                item[ResultKey.id] = index
                item[ResultKey.cmd] = cmd
                item[ResultKey.raw_log] = output
                item[ResultKey.return_code] = returnCode
                item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token), '')
                item[ResultKey.peak_cpu_usage_percent] = cpu_usage_percent
                item[ResultKey.peak_mem_usage_size] = sizeof_fmt(mem_usage_size)
                item[ResultKey.peak_mem_usage_percent] = mem_usage_percent
                index += 1

        return ret_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        return AppClassChatglm3.parse_output(output)

    @staticmethod
    def parse_generated_sentences(output):
        return AppClassChatglm3.parse_generated_sentences(output)

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref):
        return AppClassChatglm3.generate_tabulate(data_this, data_ref)

class AppClassQwen():
    @staticmethod
    def get_model_path(args, origin=False):
        MODEL_FILE_NAME = 'openvino_model.xml' if origin else 'modified_openvino_model.xml'
        # return os.path.join(*[f'{args.model_dir}', f'{MODEL_DATE}', 'qwen-7b-chat', 'pytorch', 'dldt', 'compressed_weights', 'OV_FP16-4BIT_DEFAULT', MODEL_FILE_NAME]), \
        #        os.path.join(*[f'{args.model_dir}', f'{MODEL_DATE}', 'qwen-7b-chat', 'pytorch', 'dldt', 'compressed_weights', 'OV_FP16-4BIT_DEFAULT', 'qwen.tiktoken'])
        return os.path.join(*[f'{args.model_dir}', 'ww52-qwen-bkm-stateful', MODEL_FILE_NAME]), \
               os.path.join(*[f'{args.model_dir}', 'ww52-qwen-bkm-stateful', 'qwen.tiktoken'])

    @staticmethod
    def get_executor_path(args):
        return os.path.join(*[args.bin_dir, 'qwen', 'main' + ('.exe' if IS_WINDOWS else '')])

    @staticmethod
    def get_cmd(args, key=None, index=-1):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, TOKENIZER_PATH = __class__.get_model_path(args)
        cmd = f'{APP_PATH} -m {MODEL_PATH} -t {TOKENIZER_PATH} -d {args.device} -l en --stateful -mcl {OUT_TOKEN_LEN} -f'
        if index >= 0:
            cmd += f' --select_inputs {index}'
        return cmd

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)
        sentences_map = __class__.parse_generated_sentences(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode
            item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token), '')

        return result_list

    @staticmethod
    def convert_model(args):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, TOKENIZER_PATH = __class__.get_model_path(args)
        if not exists_path(MODEL_PATH):
            MODEL_PATH, TOKENIZER_PATH = __class__.get_model_path(args, True)
            cmd = f'{APP_PATH} -m {MODEL_PATH} -t {TOKENIZER_PATH} -d {args.device} -l en --convert_kv_fp16'
            call_cmd(args, cmd)

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        _1st_inf = -1
        _2nd_inf = -1
        index = 0
        for line in output.splitlines():
            match_obj = re.search(f'First inference took (\d+.\d+) ms', line)
            if match_obj != None:
                _1st_inf = float(match_obj.groups()[0])
                continue

            match_obj = re.search(f'Average other token latency: (\d+.\d+) ms', line)
            if match_obj != None:
                _2nd_inf = float(match_obj.groups()[0])
                continue

            match_obj = re.search(f'Input num tokens: (\d+), output num tokens: (\d+), ', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.in_token] = int(values[0])
                item[ResultKey.out_token] = int(values[1])
                item[ResultKey.perf] = [_1st_inf, _2nd_inf]
                ret.append(item)
                index += 1
                continue

        return ret

    @staticmethod
    def parse_generated_sentences(output):
        input_len = 0
        index = 0
        sentence = ''
        sentence_map = {}
        is_sentence = False
        for line in output.splitlines():
            match_obj = re.search(f'Input token length: (\d+)', line)
            if (match_obj != None):
                input_len = int(match_obj.groups()[0])
                continue
            match_obj = re.search(f'First inference took ', line)
            if (match_obj != None):
                is_sentence = True
                continue
            match_obj = re.search(f'Other inference took in total: ', line)
            if (match_obj != None):
                is_sentence = False
                sentence_map[input_len] = sentence
                sentence = ''
                index += 1
                continue

            if is_sentence:
                sentence += line
        return sentence_map

    @staticmethod
    def generate_tabulate(data_this, data_ref):
        return AppClassChatglm3.generate_tabulate(data_this, data_ref)


class AppClassQwenMeasuredUsage():
    @staticmethod
    def execute(args, key=None):
        ret_list = []
        remove_cache(args)
        for index in range(8):
            cmd = AppClassQwen.get_cmd(args, key, index)
            log.info(f'{cmd}')

            tracker = HWResourceTracker()
            tracker.start()
            output, returnCode = call_cmd(args, cmd)
            cpu_usage_percent, mem_usage_size, mem_usage_percent = tracker.stop()

            result_list = __class__.parse_output(output)
            sentences_map = __class__.parse_generated_sentences(output)
            ret_list += result_list

            for item in result_list:
                item[ResultKey.id] = index
                item[ResultKey.cmd] = cmd
                item[ResultKey.raw_log] = output
                item[ResultKey.return_code] = returnCode
                item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token, ''), '')
                item[ResultKey.peak_cpu_usage_percent] = cpu_usage_percent
                item[ResultKey.peak_mem_usage_size] = sizeof_fmt(mem_usage_size)
                item[ResultKey.peak_mem_usage_percent] = mem_usage_percent
                index += 1

        return ret_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        return AppClassQwen.parse_output(output)

    @staticmethod
    def parse_generated_sentences(output):
        return AppClassQwen.parse_generated_sentences(output)

    @staticmethod
    def generate_tabulate(data_this, data_ref):
        return AppClassChatglm3.generate_tabulate(data_this, data_ref)

class NotFoundModelException(Exception):
    def __init__(self, message):
        super().__init__(message)

class AppClassGenai():
    CONFIG_MAP = {}
    MODEL_CONFIG = [
        # ('chatGLM3-6b INT4 prompts', f'{MODEL_DATE}\\chatglm3-6b\pytorch\ov',            'res\\chatglm3-6b.jsonl'),
        # ('chatGLM3-6b INT4 usage',   f'{MODEL_DATE}\\chatglm3-6b\pytorch\ov',            'res\\chatglm3-6b.jsonl',
        #     {'command_args_list':['-pi 0', '-pi 1', '-pi 2', '-pi 3', '-pi 4', '-pi 5', '-pi 6', '-pi 7'],
        #      'keep_cache': True, 'measure_usage': True, 'config_json':'res\\config.json'}),
        # ('Qwen-7b INT4 prompts',     f'{MODEL_DATE}\\qwen-7b-chat\pytorch\ov',           'res\\qwen.jsonl'),
        # ('Qwen-7b INT4 usage',       f'{MODEL_DATE}\\qwen-7b-chat\pytorch\ov',           'res\\qwen.jsonl',
        #     {'command_args_list':['-pi 0', '-pi 1'],
        #      'keep_cache': True, 'measure_usage': True, 'config_json':'res\\config.json'}),
        ('baichuan2-7b-chat INT4',   f'{MODEL_DATE}\\baichuan2-7b-chat\pytorch\ov',      'prompts\\32_1024\\baichuan2-7b-chat.jsonl'),
        ('chatGLM3-6b INT4',         f'{MODEL_DATE}\\chatglm3-6b\pytorch\ov',            'prompts\\32_1024\\chatglm3-6b.jsonl'),
        ('glm-4-9b INT4',            f'{MODEL_DATE}\\glm-4-9b-chat\pytorch\ov',          'prompts\\32_1024\\glm-4-9b.jsonl'),
        ('llama2-7b INT4',           f'{MODEL_DATE}\\llama-2-7b-chat-hf\pytorch\ov',     'prompts\\32_1024\\llama-2-7b-chat.jsonl'),
        ('llama3-8b INT4',           f'{MODEL_DATE}\\llama-3-8b\pytorch\ov',             'prompts\\32_1024\\llama-3-8b.jsonl'),
        # ('llama3-8b INT4 (DQGS:1M)', f'{MODEL_DATE}\\llama-3-8b\pytorch\ov',             'prompts\\32_1024\\llama-3-8b.jsonl', {'config_json':'res\\llama3-8b.json'}),
        ('minicpm-1b-sft INT4',      f'{MODEL_DATE}\\minicpm-1b-sft\pytorch\ov',         'prompts\\32_1024\\minicpm-1b-sft.jsonl'),
        ('mistral-7B INT4',          f'{MODEL_DATE}\\mistral-7b-v0.1\pytorch\ov',        'prompts\\32_1024\\mistral-7b-v0.1.jsonl'),
        ('Phi-2 INT4',               f'{MODEL_DATE}\\phi-2\pytorch\ov',                  'prompts\\32_1024\\phi2-2.7b.jsonl'),
        ('Phi-3-mini INT4',          f'{MODEL_DATE}\\phi-3-mini-4k-instruct\pytorch\ov', 'prompts\\32_1024\\phi-3-mini-4k-instruct.jsonl'),
        ('Gemma-7B INT4',            f'{MODEL_DATE}\\gemma-7b-it\pytorch\ov',            'prompts\\32_1024\\gemma-7b.jsonl'),
        ('Qwen-7b INT4',             f'{MODEL_DATE}\\qwen-7b-chat\pytorch\ov',           'prompts\\32_1024\\qwen-7b-chat.jsonl'),
        ('qwen2-7b INT4',            f'{MODEL_DATE}\\qwen2-7b\pytorch\ov',               'prompts\\32_1024\\qwen2-7b.jsonl'),
    ]

    for name, model, prompt, *opt in MODEL_CONFIG:
        for model_type in ['MAXIMUM', 'DEFAULT']:
            suffix_name = f' {model_type}' if model_type == 'DEFAULT' else ''
            key = f'{name}{suffix_name}'
            model_path = f'{model}\OV_FP16-4BIT_{model_type}'
            CONFIG_MAP[key] = (model_path, prompt, opt[0]) if opt else (model_path, prompt, {})

    @staticmethod
    def get_model_path(args, key):
        model, prompt, config_map = __class__.CONFIG_MAP.get(key, None)
        return os.path.join(*[f'{args.model_dir}', model]), \
               os.path.join(*[args.working_dir, prompt]), \
               config_map

    @staticmethod
    def get_executor_path(args):
        return os.path.join(*[args.working_dir, 'openvino.genai', 'tools', 'llm_bench', 'benchmark.py'])

    @staticmethod
    def get_cmd(args, key) -> tuple[list[str], dict]:
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, PROMPT_PATH, config_map = __class__.get_model_path(args, key)
        if not exists_path(MODEL_PATH):
            raise NotFoundModelException(MODEL_PATH)

        cmd_list = []
        cmd = f'python {APP_PATH} -m {MODEL_PATH} -pf {PROMPT_PATH} -d {args.device} -mc 1 -ic {OUT_TOKEN_LEN} -n {BENCHMARK_ITER_NUM} {"--genai" if args.genai else "" }'
        json_config = config_map.get('config_json', None)
        if json_config:
            cmd += f' --load_config {os.path.join(*[args.working_dir, json_config])}'

        command_args_list = config_map.get('command_args_list', [])
        if command_args_list:
            for command_args in command_args_list:
                cmd_list.append(cmd + f' {command_args}')
        else:
            cmd_list = [cmd]
        return cmd_list, config_map

    @staticmethod
    def execute(args, key=None):
        ret_list = []
        cmd_list, config = __class__.get_cmd(args, key)

        remove_cache(args)
        for cmd in cmd_list:
            log.info(f'{cmd}')
            if not config.get('keep_cache', False):
                remove_cache(args)

            if config.get('measure_usage', False):
                tracker = HWResourceTracker()
                tracker.start()
            output, returnCode = call_cmd(args, cmd)
            if config.get('measure_usage', False):
                cpu_usage_percent, mem_usage_size, mem_usage_percent = tracker.stop()

            if returnCode == 0:
                result_list = __class__.parse_output(output)
                sentences_map = __class__.parse_generated_sentences(output)
                ret_list += result_list
                for item in result_list:
                    item[ResultKey.cmd] = cmd
                    item[ResultKey.raw_log] = output
                    item[ResultKey.return_code] = returnCode
                    item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token, ''), '')

                    if config.get('measure_usage', False):
                        item.peak_cpu_usage_percent = cpu_usage_percent
                        item.peak_mem_usage_size = sizeof_fmt(mem_usage_size)
                        item.peak_mem_usage_percent = mem_usage_percent
            else:
                errorItem = {}
                errorItem[ResultKey.cmd] = cmd
                errorItem[ResultKey.raw_log] = output
                errorItem[ResultKey.return_code] = returnCode
                ret_list += [errorItem]

        return ret_list

    @staticmethod
    def parse_generated_sentences(output):
        in_token_size = 0
        input_text = ''
        generated_text = None
        sentence_map = {}
        for line in output.splitlines():
            if generated_text:
                match_obj = re.search(f'\[ ([\S]+) \] ', line)
                if match_obj != None:
                    sentence_map[in_token_size] = generated_text.replace(input_text, '')
                    generated_text = None
                else:
                    generated_text += line
                continue

            match_obj = re.search(f'Input token size: ([0-9]+)', line)
            if match_obj != None:
                in_token_size = int(match_obj.groups()[0])
                continue

            match_obj = re.search(f'Input text: ([\S ]+)', line)
            if match_obj != None:
                input_text = match_obj.groups()[0]
                continue

            match_obj = re.search(f'\] Generated:([\S ]+)', line)
            if match_obj != None:
                generated_text = match_obj.groups()[0]
                continue

        return sentence_map

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        index = 0
        in_token_size = 0
        out_token_size = 0
        for line in output.splitlines():
            # [ INFO ] [3][P0] Input token size: 32, Output size: 256, Infer count: 256, Tokenization Time: 0.17ms, Detokenization Time: 0.12ms, Generation Time: 3.63s, Latency: 14.18 ms/token
            # [ INFO ] [3][P0] First token latency: 27.35 ms/token, other tokens latency: 14.12 ms/token, len of tokens: 256 * 1
            # [ INFO ] [3][P0] First infer latency: 25.11 ms/infer, other infers latency: 12.88 ms/infer, inference count: 256
            # [ INFO ] [3][P0] Result MD5:['35db56cf1cb1267323bc500d8a4af5f4']
            # [ INFO ] [3][P0] start: 2024-11-08T05:54:18.663800, end: 2024-11-08T05:54:22.296733
            # [ INFO ] [3][P1] Input token size: 1024, Output size: 256, Infer count: 256, Tokenization Time: 1.47ms, Detokenization Time: 0.42ms, Generation Time: 4.10s, Latency: 16.03 ms/token
            # [ INFO ] [3][P1] First token latency: 268.78 ms/token, other tokens latency: 15.03 ms/token, len of tokens: 256 * 1
            # [ INFO ] [3][P1] First infer latency: 234.29 ms/infer, other infers latency: 13.67 ms/infer, inference count: 256
            # [ INFO ] [3][P1] Result MD5:['6415d1a8d30446bbbd19f8d87c95a5ee']
            # [ INFO ] [3][P1] start: 2024-11-08T05:54:22.296733, end: 2024-11-08T05:54:26.403995
            match_obj = re.search(f'\[{BENCHMARK_ITER_NUM}\]\[[A-Z0-9]+\] Input token size: (\d+), Output size: (\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                in_token_size = int(values[0])
                out_token_size = int(values[1])
                continue

            match_obj = re.search(f'\[{BENCHMARK_ITER_NUM}\]\[[A-Z0-9]+\] First token latency: (\d+.\d+) ms\/token, other tokens latency: (\d+.\d+) ms\/token', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.in_token] = in_token_size
                item[ResultKey.out_token] = out_token_size
                item[ResultKey.perf] = [float(values[0]), float(values[1])]
                ret.append(item)
                index += 1
                continue

            match_obj = re.search(f'\[{BENCHMARK_ITER_NUM}\]\[[A-Z0-9]+\] First token latency: (\d+.\d+) ms\/token', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.in_token] = in_token_size
                item[ResultKey.out_token] = out_token_size
                item[ResultKey.perf] = [float(values[0]), 0]
                ret.append(item)
                index += 1
                continue

        return ret

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref):
        return AppClassChatglm3.generate_tabulate(data_this, data_ref)

class AppClassWhisper():
    CONFIG_MAP = {
        'Whisper base': 'whisper-base-nonstateful',
    }

    @staticmethod
    def get_model_path(args, key):
        model = __class__.CONFIG_MAP.get(key, None)
        if model != None:
            return os.path.join(*[f'{args.model_dir}', model])
        return None

    @staticmethod
    def get_executor_path(args):
        return os.path.join(*[args.working_dir, 'scripts', 'whisper', 'optimum_notebook', 'non_stateful', 'run_model.py'])

    @staticmethod
    def get_cmd(args, key):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH = __class__.get_model_path(args, key)

        return f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)
        sentences_map = __class__.parse_generated_sentences(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode
            item[ResultKey.generated_text] = sentences_map.get(item.get(ResultKey.in_token, ''), '')

        return result_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        index = 0
        for line in output.splitlines():
            # tps : 68.708
            match_obj = re.search(f'tps : (\d+.\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.perf] = [float(values[0])]
                ret.append(item)
                index += 1

        return ret

    @staticmethod
    def parse_generated_sentences(output):
        sentence = ''
        sentence_map = {}
        lines = output.splitlines()
        index = 0
        while index < len(lines):
            match_obj = re.search(f'^(\d+)$', lines[index])
            if (match_obj != None):
                sentence += lines[index + 2]
                index += 2

            match_obj = re.search(f'average time = ', lines[index])
            if (match_obj != None):
                sentence_map[0] = sentence
                break

            index += 1

        return sentence_map

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref) -> str:
        if data_this != None and len(data_this) > 0:
            headers = ['index', 'tokens/s']
            floatfmt = ['', '.2f']
            raw_data_list = []
            for item in data_this:
                try:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.perf)[0]])
                except:
                    raw_data_list.append([0, 0])
            return tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='left', numalign='right')
        return ''

class AppClassGenaiCppStableDiffusion():
    CONFIG_MAP = {
        'SD 1.5 FP16': ('FP16', f'daily\sd_15_ov'),
        'SD 1.5 INT8': ('INT8', f'daily\sd_15_ov'),
        'SD 2.1 FP16': ('FP16', f'daily\sd_21_ov'),
        'SD 2.1 INT8': ('INT8', f'daily\sd_21_ov'),
        'Stable-Diffusion LCM FP16': ('FP16', f'{MODEL_DATE_WW32_LLM}\lcm-dreamshaper-v7\pytorch\dldt'),
        'Stable Diffusion XL FP16': ('FP16', f'daily\sdxl_1_0_ov\FP16'),
    }

    @staticmethod
    def get_model_path(args, key):
        precision, model = __class__.CONFIG_MAP.get(key, None)
        if model != None:
            return precision, os.path.join(*[f'{args.model_dir}', model])
        return None

    @staticmethod
    def get_executor_path(args, key):
        if key == 'Stable-Diffusion LCM FP16':
            return os.path.join(*[args.bin_dir, 'lcm', 'lcm_dreamshaper.exe'])
        elif key == 'Stable Diffusion XL FP16':
            return os.path.join(*[args.working_dir, 'scripts', 'sdxl', 'run_sdxl.py'])
        else:
            return os.path.join(*[args.bin_dir, 'sd', 'stable_diffusion.exe'])

    @staticmethod
    def get_cmd(args, key):
        APP_PATH = __class__.get_executor_path(args, key)
        PRECISION, MODEL_PATH = __class__.get_model_path(args, key)

        if key == 'Stable Diffusion XL FP16':
            return f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'
        else:
            return f'{APP_PATH} -m {MODEL_PATH} -d {args.device} -t {PRECISION}'

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode

        return result_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        index = 0
        for line in output.splitlines():
            # 06:16:52:INFO: Loading and compiling text encoder: 4488.8 ms
            # 06:17:01:INFO: Loading and compiling UNet: 9786.81 ms
            # 06:17:02:INFO: Loading and compiling VAE decoder: 966.16 ms
            # 06:17:02:INFO: Loading and compiling tokenizer: 95.4802 ms
            # 06:17:05:INFO: Running Stable Diffusion pipeline: 2238.85 ms
            match_obj = re.search(f'pipeline: (\d+.\d+) ms', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.perf] = [float(values[0])]
                ret.append(item)
                index += 1

        return ret

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref) -> str:
        if data_this != None and len(data_this) > 0:
            headers = ['index', 'pipeline Time']
            floatfmt = ['', '.2f']
            raw_data_list = []
            index = 0
            for item in data_this:
                try:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.perf)[0]])
                except:
                    raw_data_list.append([index, 0])
                index += 1
            return tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='left', numalign='right')
        return ''


class AppClassPythonStableDiffusion():
    CONFIG_MAP = {
        'SD 3.0 Dynamic': (f'stable-diffusion-3', True),
        'SD 3.0 Static': (f'stable-diffusion-3', False),
    }

    @staticmethod
    def get_model_path(args, key):
        model, dynamic = __class__.CONFIG_MAP.get(key, None)
        if model != None:
            return os.path.join(*[f'{args.model_dir}', model]), dynamic
        return None

    @staticmethod
    def get_executor_path(args):
        return os.path.join(*[args.working_dir, 'scripts', 'stable-diffusion', 'run_sd3_ov_daily.py'])

    @staticmethod
    def get_cmd(args, key):
        APP_PATH = __class__.get_executor_path(args)
        MODEL_PATH, DYNAMIC = __class__.get_model_path(args, key)
        if not exists_path(MODEL_PATH):
            raise NotFoundModelException(MODEL_PATH)

        result_filename = convert_path(args, RESULT_SD3_DYNAMIC_FILENAME if DYNAMIC else RESULT_SD3_STATIC_FILENAME)
        return f'python {APP_PATH} -m {MODEL_PATH} {"--dynamic" if DYNAMIC else ""} -d {args.device} --result_img {result_filename}'

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode

        return result_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        index = 0
        for line in output.splitlines():
            # time in seconds:  15.823483300046064
            match_obj = re.search(f'time in seconds:\ +(\d+.\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = index
                item[ResultKey.perf] = [float(values[0])]
                ret.append(item)
                index += 1

        return ret

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref) -> str:
        if data_this != None and len(data_this) > 0:
            headers = ['index', 'pipeline Time']
            floatfmt = ['', '.2f']
            raw_data_list = []
            index = 0
            for item in data_this:
                try:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.perf)[0]])
                except:
                    raw_data_list.append([index, 0])
                index += 1
            return tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='left', numalign='right')
        return ''


class AppClassBenchmark():
    CONFIG_MAP = {
        'Resnet50 INT8 bs=1': { 'model': 'models\\resnet_v1.5_50\\resnet_v1.5_50_i8.xml', 'batch': 1 },
        'Resnet50 INT8 bs=64': { 'model': 'models\\resnet_v1.5_50\\resnet_v1.5_50_i8.xml', 'batch': 64 },
    }

    @staticmethod
    def get_cmd(args, key):
        config = __class__.CONFIG_MAP.get(key, None)
        return f'{args.benchmark_app} -m {config["model"]} -b {config["batch"]} -d {args.device} --hint none -nstreams 2 -nireq 4 -t 10'

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        output, returnCode = call_cmd(args, cmd)
        result_list = __class__.parse_output(output)

        for item in result_list:
            item[ResultKey.cmd] = cmd
            item[ResultKey.raw_log] = output
            item[ResultKey.return_code] = returnCode

        return result_list

    @staticmethod
    def parse_output(output) -> list[dict]:
        ret = []
        batch_size = 0
        for line in output.splitlines():
            # Model batch size: 1
            # [ INFO ] Throughput:          1647.22 FPS
            match_obj = re.search(f'Model batch size: (\d+)', line)
            if match_obj != None:
                batch_size = int(match_obj.groups()[0])
                continue

            match_obj = re.search(f'Throughput: +(\d+.\d+) FPS', line)
            if match_obj != None:
                values = match_obj.groups()
                item = {}
                item[ResultKey.id] = batch_size
                item[ResultKey.perf] = [float(values[0])]
                ret.append(item)
                break

        return ret

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref):
        if data_this != None and len(data_this) > 0:
            headers = ['batch', 'throughput(fps)']
            floatfmt = ['', '.2f']
            raw_data_list = []
            index = 0
            for item in data_this:
                try:
                    raw_data_list.append([item.get(ResultKey.id, 0), item.get(ResultKey.perf)[0]])
                except:
                    raw_data_list.append([index, 0])
                index += 1
            return tabulate(raw_data_list, tablefmt="github", headers=headers, floatfmt=floatfmt, stralign='left', numalign='right')
        return ''

class AppClassSmokeTest():
    CONFIG_MAP = {
        'chat_sample llama-2-7b-chat DEFAULT': f'{MODEL_DATE}\\llama-2-7b-chat-hf\pytorch\ov\OV_FP16-4BIT_DEFAULT',
    }

    @staticmethod
    def get_model_path(args, key):
        model = __class__.CONFIG_MAP.get(key, None)
        if model != None:
            return os.path.join(*[f'{args.model_dir}', model])
        return None

    @staticmethod
    def get_executor_path(args, key):
        return os.path.join(*[args.working_dir, 'openvino.genai', 'samples', 'python', 'chat_sample', 'chat_sample.py'])

    @staticmethod
    def get_cmd(args, key):
        APP_PATH = __class__.get_executor_path(args, key)
        MODEL_PATH = __class__.get_model_path(args, key)
        return f'python {APP_PATH} -m {MODEL_PATH} -d {args.device}'

    @staticmethod
    def execute(args, key=None):
        cmd = __class__.get_cmd(args, key)
        log.info(f'{cmd}')

        remove_cache(args)
        time.sleep(args.sleep_bf_test)
        output, returnCode = call_cmd(args, cmd, shell=True)

        item = {}
        item[ResultKey.cmd] = cmd
        item[ResultKey.raw_log] = output
        item[ResultKey.return_code] = returnCode
        return [item]

    @staticmethod
    def generate_tabulate(data_this: list[dict], data_ref):
        return data_this[0].get(ResultKey.raw_log, '')



################################################
# Main
################################################

def call_cmd(args, cmd:str, shell=False, verbose=True) -> tuple[str, int]:
    out_log = ''
    returncode = -1

    try:
        with subprocess.Popen(convert_cmd_for_popen(cmd),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell, text=True, encoding='UTF-8', errors="ignore") as proc:
            try:
                out_log, error = proc.communicate(timeout=args.timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired as e:
                log.error(f'Timeout: {str(e)}')
                log.log('try to kill this process...')
                proc.kill()
                log.log('try to kill this process...done.')
                out_log, error = proc.communicate()
                returncode = proc.returncode
                log.log(f'returncode: {returncode}')
    except Exception as e:
        log.error(f'Exception: {str(e)}')

    if returncode != 0:
        log.error(f'returncode: {returncode}')
        log.error(f'out_log: {out_log}')
    else:
        if verbose:
            log.info(f'returncode: {returncode}')
            log.info(f'out_log: {out_log}')

    return out_log, returncode

def load_result_json(filepath) -> dict:
    if not exists_path(filepath):
        return {}

    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        log.error(f'load_result_json: {filepath}')
        log.error(f'--> {e}')
        return {}

def get_ov_version_from_report(report_path):
    if exists_path(report_path):
        with open(report_path, 'r', encoding='utf8') as fis:
            for line in fis.readlines():
                # OpenVINO: 2024.2.0-15519-5c0f38f83f6-releases/2024/2
                # 2025.0.0-17598-329670f2266
                match_obj = re.search(f'([\d]+.[\d]+.[\d]+-[\d]+-[\da-z\-\/]+)', line)
                if match_obj != None:
                    return match_obj.groups()[0]
    return None

def remove_cache(args):
    if exists_path(args.cache_dir):
        for file in glob(os.path.join(args.cache_dir, '*.cl_cache')):
            os.remove(file)
        for file in glob(os.path.join(args.cache_dir, '*.blob')):
            os.remove(file)
        time.sleep(1)

def pass_value(value:float):
    return value
def fps_to_ms(value:float):
    return 1000 / value
def ms_to_sec(value:float):
    return value / 1000

def generate_ccg_table(result_map):

    MODEL_REPORT_CONFIG = [
        # [key, description, token_index, perf_index, converter]
        ['Resnet50 INT8 bs=1', 'fps', 0, 0, pass_value, fps_to_ms],
        ['Resnet50 INT8 bs=64', 'fps', 0, 0, pass_value, fps_to_ms],
        ['SD 1.5 INT8', 'static, second per image (s)', 0, 0, ms_to_sec],
        ['SD 1.5 FP16', 'static, second per image (s)', 0, 0, ms_to_sec],
        ['SD 2.1 INT8', 'static, second per image (s)', 0, 0, ms_to_sec],
        ['SD 2.1 FP16', 'static, second per image (s)', 0, 0, ms_to_sec],
        ['Stable-Diffusion LCM FP16', 'static, second per image (s)', 0, 0, ms_to_sec],
        ['Stable Diffusion XL FP16', 'second per image (s)', 0, 0, ms_to_sec],
        ['llama2-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['llama2-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['llama2-7b INT4 DEFAULT', f'32/{OUT_TOKEN_LEN}, 1st token latency (ms)', 0, 0],
        ['llama2-7b INT4 DEFAULT', f'32/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 0, 1],
        ['llama3-8b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['llama3-8b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['chatGLM3-6b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['chatGLM3-6b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['chatGLM3-6b INT4 DEFAULT', f'32/{OUT_TOKEN_LEN}, 1st token latency (ms)', 0, 0],
        ['chatGLM3-6b INT4 DEFAULT', f'32/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 0, 1],
        ['Qwen-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['Qwen-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['Phi-3-mini INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['Phi-3-mini INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['Gemma-7B INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['Gemma-7B INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['mistral-7B INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['mistral-7B INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['Whisper base', 'tokens/second', 0, 0, pass_value, fps_to_ms],
        ['Stable-Diffusion3 (bs=1, FP16, 1024x1024, 28steps)', ''],
        ['qwen2-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['qwen2-7b INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['Phi-2 INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['Phi-2 INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['minicpm-1b-sft INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 1st token latency (ms)', 1, 0],
        ['minicpm-1b-sft INT4 DEFAULT', f'1024/{OUT_TOKEN_LEN}, 2nd token avg (ms)', 1, 1],
        ['llama3-llava-next-8b', ''],
        ['llama3-llava-next-8b', ''],
        ['Whisper Large v3', ''],
        ['SD 3.0 Dynamic', 'seconds', 0, 0, ms_to_sec],
        ['SD 3.0 Static', 'seconds', 0, 0, ms_to_sec],
    ]

    value_list = []
    table = []

    for config in MODEL_REPORT_CONFIG:
        key = config[0]
        description = config[1]
        try:
            result_item_list = result_map.get(key, None)
            item = result_item_list[config[2]]
            value = item.get(ResultKey.perf, [0, 0])[config[3]]
            value_list.append(value)
            convert_value = config[4] if len(config) >= 5 else pass_value
            convert_latency = config[5] if len(config) >= 6 else pass_value
            table.append([key, description, f'{convert_value(value):.2f}', f'{convert_latency(value):.2f}'])
        except:
            table.append([key, description, 'N/A', ''])

    geomean = 0
    if len(value_list):
        try:
            geomean = geometric_mean(value_list)
        except:
            geomean = 0

    table.append([])
    table.append(['Success count', '', '', len(value_list)])
    table.append(['geomean', '', '', f'{float(geomean):.2f}'])

    return tabulate(table, tablefmt="github", stralign='right',
                    headers=['KPI Model', 'description', 'value', 'ms'], floatfmt=['', '', '.2f', '.2f'])

def generate_csv_table(result_map):
    MODEL_REPORT_CONFIG = [
        # [key, data_num, perf_count, converter]
        ['baichuan2-7b-chat INT4 DEFAULT', 2, 2],
        ['chatglm3', 10, 2],
        ['chatGLM3-6b INT4', 2, 2],
        ['chatGLM3-6b INT4 DEFAULT', 2, 2],
        ['chatglm3_usage', 8, 2],
        ['glm-4-9b INT4 DEFAULT', 2, 2],
        ['Gemma-7B INT4', 2, 2],
        ['Gemma-7B INT4 DEFAULT', 2, 2],
        ['llama2-7b INT4', 2, 2],
        ['llama2-7b INT4 DEFAULT', 2, 2],
        ['llama3-8b INT4', 2, 2],
        ['llama3-8b INT4 DEFAULT', 2, 2],
        ['minicpm-1b-sft INT4 DEFAULT', 2, 2],
        ['mistral-7B INT4', 2, 2],
        ['mistral-7B INT4 DEFAULT', 2, 2],
        ['Phi-2 INT4', 2, 2],
        ['Phi-2 INT4 DEFAULT', 2, 2],
        ['Phi-3-mini INT4', 2, 2],
        ['Phi-3-mini INT4 DEFAULT', 2, 2],
        ['qwen', 8, 2],
        ['Qwen-7b INT4', 2, 2],
        ['Qwen-7b INT4 DEFAULT', 2, 2],
        ['qwen_usage', 8, 2],
        ['qwen2-7b INT4 DEFAULT', 2, 2],
        ['Resnet50 INT8 bs=1', 1, 1, fps_to_ms],
        ['Resnet50 INT8 bs=64', 1, 1, fps_to_ms],
        ['SD 1.5 FP16', 1, 1],
        ['SD 1.5 INT8', 1, 1],
        ['SD 2.1 FP16', 1, 1],
        ['SD 2.1 INT8', 1, 1],
        ['Stable Diffusion XL FP16', 1, 1],
        ['Stable-Diffusion LCM FP16', 1, 1],
        ['Whisper base', 1, 1, fps_to_ms],
        ['SD 3.0 Dynamic', 1, 1],
        ['SD 3.0 Static', 1, 1],
    ]

    value_list = []
    table = []
    for config in MODEL_REPORT_CONFIG:
        key = config[0]
        data_count = config[1]
        raw_line = data_count * config[2]
        converter = config[3] if len(config) == 4 else pass_value
        result_item_list = result_map.get(key, None)

        for i in range(0, data_count):
            try:
                item = result_item_list[i]
                for latency in item.get(ResultKey.perf):
                    raw_line -= 1
                    table.append([key, item.get(ResultKey.in_token, ''), item.get(ResultKey.out_token, ''), f'{float(converter(latency)):.2f}'])
                    if isinstance(latency, float):
                        value_list.append(latency)
            except:
                continue

        for i in range(0, raw_line):
            table.append([key, '', '', ''])

    success_count = len(value_list)
    geomean = 0
    if len(value_list):
        try:
            geomean = geometric_mean(value_list)
        except:
            geomean = 0

    table.append([])
    table.append(['Success count', '', '', success_count])
    table.append(['geomean', '', '', f'{float(geomean):.2f}'])

    return tabulate(table, tablefmt="github", headers=['model', 'in', 'out', 'latency(ms)'], floatfmt='.2f', stralign='right', numalign='right'), geomean, success_count

def replace_ext(filepath, new_ext):
    filepath_str = str(filepath)
    index = filepath_str.rfind('.')
    return filepath_str[0:index] + '.' + new_ext

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

def print_report(report):
    result_map = load_result_json(replace_ext(report, "json"))
    ov_ccg_tabulate = generate_ccg_table(result_map)
    ov_csv_tabulate, csv_geomean, csv_success_cnt = generate_csv_table(result_map)

    table_map = {}
    for key, test_class in get_test_list():
        table_map[key] = test_class.generate_tabulate(result_map.get(key), None)

    for key, value in table_map.items():
        if value != None:
            log.info(f'[RESULT][{key}]\n')
            log.info(f'{value}\n\n')
    if len(ov_ccg_tabulate) > 0:
        log.info(ov_ccg_tabulate + '\n\n')
    if len(ov_csv_tabulate) > 0:
        log.info(ov_csv_tabulate + '\n\n')

def get_test_list(target:str='all'):
    test_single_run_list = []
    if target == 'all':
        test_single_run_list = [
            # ('chatglm3', AppClassChatglm3),
            # ('qwen', AppClassQwen),
            # ('chatglm3_usage', AppClassChatglm3MeasuredUsage),
            ('qwen_usage', AppClassQwenMeasuredUsage),
        ]
        for key in AppClassGenai.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassGenai))
        for key in AppClassGenaiCppStableDiffusion.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassGenaiCppStableDiffusion))
        for key in AppClassPythonStableDiffusion.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassPythonStableDiffusion))
        for key in AppClassWhisper.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassWhisper))
        for key in AppClassBenchmark.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassBenchmark))
        # for key in AppClassSmokeTest.CONFIG_MAP.keys():
        #     test_single_run_list.append((key, AppClassSmokeTest))
    elif target == 'llm_only':
        for key in AppClassGenai.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassGenai))
    elif target == 'sd_only':
        for key in AppClassGenaiCppStableDiffusion.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassGenaiCppStableDiffusion))
        for key in AppClassPythonStableDiffusion.CONFIG_MAP.keys():
            test_single_run_list.append((key, AppClassPythonStableDiffusion))
    elif target == 'cpp_only':
        test_single_run_list = [
            # ('chatglm3', AppClassChatglm3),
            # ('qwen', AppClassQwen),
            # ('chatglm3_usage', AppClassChatglm3MeasuredUsage),
            ('qwen_usage', AppClassQwenMeasuredUsage),
        ]
    elif target == 'cpp_usage_only':
        test_single_run_list = [
            # ('chatglm3_usage', AppClassChatglm3MeasuredUsage),
            ('qwen_usage', AppClassQwenMeasuredUsage),
        ]
    else:
        raise Exception(f'target: {target} is not supported.')
    return test_single_run_list

def compare_result_item_map(fos, callback, this_map, ref_map={}):
    for key, this_result_list in this_map.items():
        ref_result_list = ref_map.get(key, [])
        for this_item in this_result_list:
            this_item_in_token = this_item.get(ResultKey.in_token, -1)
            ref_item = None
            for item in ref_result_list:
                ref_item_in_token = item.get(ResultKey.in_token, -2)
                if this_item_in_token == ref_item_in_token:
                    ref_item = item
                    break
            callback(fos, key, this_item, ref_item)

def print_compared_text(fos, key:str, this_item:dict, ref_item:dict):
    LIMIT_TEXT_LENGTH = 256
    this_text = this_item.get(ResultKey.generated_text, '')
    if len(this_text) == 0:
        log.warning(f'{key} has no generated text.')
        return

    in_token = this_item.get(ResultKey.in_token, 0)
    if ref_item == None:
        fos.write(f'[TEXT][{key}][{in_token}]\n')
        fos.write(f'\t[this] {this_text}\n')
        return

    ref_text = ref_item.get(ResultKey.generated_text, '')
    iou = calculate_score(this_text, ref_text)

    this_text = this_text if len(this_text) < LIMIT_TEXT_LENGTH else this_text[0:LIMIT_TEXT_LENGTH]
    this_text = this_text.replace("<s>", "_s_")     # WA: remove '<s>'. It will replace to cancel line in outlook.
    ref_text = ref_text if len(ref_text) < LIMIT_TEXT_LENGTH else ref_text[0:LIMIT_TEXT_LENGTH]
    ref_text = ref_text.replace("<s>", "_s_")

    sts_str = ('OK' if iou > 0.5 else 'DIFF') if iou > 0 else 'ERR'

    fos.write(f'[TEXT][{key}][{in_token}][{sts_str}][iou:{iou:0.2f}]\n')
    if iou == 1:
        fos.write(f'\t[this] {this_text}\n')
    else:
        fos.write(f'\t[this] {this_text}\n')
        fos.write(f'\t[ref ] {ref_text}\n')
    fos.write(f'\n')

def generate_report(args, result_map:dict, PROCESS_TIME) -> tuple[Path, str]:
    #
    # Generate tabulate tables
    #
    table_map = {}
    for key, test_class in get_test_list(args.model_target):
        table_str = test_class.generate_tabulate(result_map.get(key), None)
        if len(table_str) > 0:
            table_map[key] = table_str

    ov_ccg_tabulate = generate_ccg_table(result_map)
    ov_csv_tabulate, csv_geomean, csv_success_cnt = generate_csv_table(result_map)
    suffix_title = f'({float(csv_geomean):.2f}/{csv_success_cnt})'

    #
    # System info
    #
    APP = os.path.join('scripts', 'device_info.py')
    system_info, returncode = call_cmd(args, f'python {APP}', shell=True, verbose=False)

    #
    # python packages
    #
    PIP_FREEZE_PATH = convert_path(args, PIP_FREEZE_FILENAME)
    with open(PIP_FREEZE_PATH, 'w', encoding='utf-8') as fos:
        fos.write(f'{python_packages()}')

    #
    # summary
    #
    summary_table_data = []
    summary_table_data.append(['Purpose', f'{args.description}'])
    summary_table_data.append(['TOTAL TASK TIME', f'{time.strftime("%H:%M:%S", time.gmtime(PROCESS_TIME))}'])
    summary_table_data.append(['OpenVINO', f'{get_version()}'])
    summary_table_data.append(['Report', f'{convert_url(REPORT_FILENAME)}'])
    summary_table_data.append(['RawLog', f'{convert_url(RAW_FILENAME)}'])
    summary_table_data.append(['PipRequirements', f'{convert_url(PIP_FREEZE_FILENAME)}'])

    RESULT_SD3_DYNAMIC_PATH = convert_path(args, RESULT_SD3_DYNAMIC_FILENAME)
    if exists_path(RESULT_SD3_DYNAMIC_PATH):
        summary_table_data.append(['SD3.0 dynamic', f'{convert_url(RESULT_SD3_DYNAMIC_FILENAME)}'])

    RESULT_SD3_STATIC_PATH = convert_path(args, RESULT_SD3_STATIC_FILENAME)
    if exists_path(RESULT_SD3_STATIC_PATH):
        summary_table_data.append(['SD3.0 static', f'{convert_url(RESULT_SD3_STATIC_FILENAME)}'])

    if args.ref_report != None:
        summary_table_data.append(['Reference Report', f'{convert_url(os.path.basename(args.ref_report))}'])
    summary_table = tabulate(summary_table_data, tablefmt="youtrack")

    #
    # Generate Report
    #
    REPORT_PATH = convert_path(args, REPORT_FILENAME)
    with open(REPORT_PATH, 'w', encoding='utf-8') as fos:
        fos.write('<pre>\n')
        fos.write(f'{summary_table}\n\n')

        for key, result_list in result_map.items():
            for item in result_list:
                if item.get(ResultKey.return_code, -1) != 0:
                    fos.write(f'[ERROR][{key}] id: {item.get(ResultKey.id, "")}, in_token: {item.get(ResultKey.in_token)}\n')

        for key, klass in get_test_list('llm_only'):
            result_list = result_map.get(key)
            if result_list == None:
                continue

            for item in result_list:
                in_token = item.get(ResultKey.in_token, 0)
                if not in_token in [32, 1024]:
                    fos.write(f'[Changed:InputToken][{key}] in_token: {in_token}\n')

        fos.write('\n\nTest results[unit: ms]\n')
        for key, value in table_map.items():
            fos.write(f'[RESULT][{key}]\n')
            fos.write(f'{value}\n\n')
        if len(ov_ccg_tabulate) > 0:
            fos.write(ov_ccg_tabulate + '\n\n')
        if len(ov_csv_tabulate) > 0:
            fos.write(ov_csv_tabulate + '\n\n')
        fos.write(system_info + '\n\n')

        # generated text
        result_ref_map = load_result_json(replace_ext(args.ref_report, "json"))
        compare_result_item_map(fos, print_compared_text, result_map, result_ref_map)
        fos.write(f'\n\n')

        # command list
        last_cmd = ''
        for key, result_list in result_map.items():
            for item in result_list:
                item_cmd = item.get(ResultKey.cmd, '')
                if last_cmd != item_cmd:
                    fos.write(f'[CMD][{key}] {item_cmd}\n')
                    last_cmd = item_cmd
        fos.write('\n</pre>')
    return REPORT_PATH, suffix_title

def run_daily(args):
    test_single_run_list = get_test_list(args.model_target)

    #
    # Run all tests
    #
    result_this_map = {}

    task_start_time = time.time()
    for key, test_class in test_single_run_list:
        # remove all cache before run each tests
        remove_cache(args)

        log.info(f'Run {key}...')
        try:
            result_this_map[key] = test_class.execute(args, key)
        except NotFoundModelException as e:
            log.error(f'NotFoundModelException: {e}\n')
            continue
        log.info(f'Run {key}... Done\n')
    PROCESS_TIME = time.time() - task_start_time

    # save results to json
    REPORT_JSON_PATH = convert_path(args, REPORT_JSON_FILENAME)
    try:
        with open(REPORT_JSON_PATH, 'w') as f:
            json.dump(result_this_map, f, indent=4)
    except Exception as e:
        log.error(f'Report::save: {e}')

    return generate_report(args, result_this_map, PROCESS_TIME)

def main_setting(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory
    os.chdir(args.working_dir)

    if not IS_WINDOWS:
        # WA: this is needed to run an executable without './'.
        os.environ["PATH"] = f'{PWD}:{os.environ["PATH"]}'

    # set log
    log.setLevel(logging.INFO)

    stream_hander = logging.StreamHandler()
    stream_hander.setLevel(logging.INFO)
    log.addHandler(stream_hander)

    # Redirect stdout/stderr to file
    RAW_PATH = convert_path(args, RAW_FILENAME)

    file_handler = logging.FileHandler(filename=RAW_PATH, mode='w', encoding='utf8')
    formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s: %(message)s', datefmt='%I:%M:%S')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser(description="Run daily check for LLM" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--benchmark_app', help='benchmark_app(cpp) path', type=Path, default=os.path.join(*[PWD, 'bin', 'benchmark_app', 'benchmark_app']))
    parser.add_argument('-cd', '--cache_dir', help='cache directory', type=Path, default=os.path.join(*[PWD, 'llm-cache']))
    parser.add_argument('--convert_models', help='', action='store_true')
    parser.add_argument('-d', '--device', help='target device', type=Path, default='GPU')
    parser.add_argument('--description', help='add description for report', type=str, default='LLM')
    parser.add_argument('-m', '--model_dir', help='root directory for models', type=Path, default=os.path.join(*['c:\\', 'dev', 'models']))
    parser.add_argument('--mail', help='sending mail recipient list. Mail recipients can be separated by comma.', type=str, default='')
    parser.add_argument('--model_target', help='Set mode: all, llm_only, sd_only, cpp_only, cpp_usage_only', type=str, default='all')
    parser.add_argument('--ref_report', help='reference report to compare performance', type=Path, default=None)
    parser.add_argument('--genai', help='enable genai option for llm benchmark', action='store_true')
    parser.add_argument('--test', help='run tests with short config', action='store_true')
    parser.add_argument('--this_report', help='target report to compare performance', type=Path, default=None)
    parser.add_argument('--timeout', help='set timeout [unit: seconds].', type=int, default=300)
    parser.add_argument('--sleep_bf_test', help='set sleep time [unit: seconds].', type=int, default=1)
    parser.add_argument('--repeat', help='Do not use.', type=int, default=1)
    parser.add_argument('-o', '--output_dir', help='output directory to store log files', type=Path, default=os.path.join(*[PWD, 'output']))
    parser.add_argument('-w', '--working_dir', help='working directory', type=Path, default=PWD)
    parser.add_argument('--bin_dir', help='binary directory', type=Path, default=os.path.join(*[PWD, 'bin']))
    args = parser.parse_args()

    main_setting(args)

    if args.test:
        global OUT_TOKEN_LEN
        global BENCHMARK_ITER_NUM
        OUT_TOKEN_LEN = 32
        BENCHMARK_ITER_NUM = 1
        if exists_path(args.this_report):
            print_report(args.this_report)
            return 0

    # Convert models for chatglm/qwen cpp
    if args.convert_models:
        for test_class in [AppClassChatglm3, AppClassQwen]:
            test_class.convert_model(args)
        return 0

    if args.this_report and args.ref_report:
        result_this_map = load_result_json(replace_ext(args.this_report, "json"))
        result_ref_map = load_result_json(replace_ext(args.ref_report, "json"))
        compare_result_item_map(sys.stdout, print_compared_text, result_this_map, result_ref_map)
    else:
        # Run test
        REPORT_PATH, suffix_title = run_daily(args)
        log.info(f'Report: {REPORT_PATH}')

        # Print and send report
        if exists_path(REPORT_PATH):
            backup_list = [ convert_path(args, filename) for filename in [RAW_FILENAME, RESULT_SD3_DYNAMIC_FILENAME, RESULT_SD3_STATIC_FILENAME, REPORT_JSON_FILENAME] ]
            backup_files(args, [REPORT_PATH] + backup_list)

            # Send mail
            if args.mail:
                send_mail(REPORT_PATH, args.mail, args.description, suffix_title)



if __name__ == "__main__":
    main()
