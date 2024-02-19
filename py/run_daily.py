#!/usr/bin/env python3

import abc
import argparse
import copy
import cpuinfo
import datetime as dt
import enum
import hashlib
import socket
import logging
import os
import pickle
import platform
import psutil
import pyopencl
import re
import subprocess
import sys
import time

from glob import glob
from openvino.runtime import get_version
from pathlib import Path
from statistics import mean
from tabulate import tabulate
from threading import Thread



################################################
# Global variable
################################################
NOW = dt.datetime.now().strftime("%Y%m%d_%H%M")
IS_WINDOWS = platform.system() == 'Windows'
PWD = os.path.abspath('.')
HOSTNAME = socket.gethostname()
IP_ADDRESS = socket.gethostbyname(HOSTNAME)



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

def get_system_info():
    opencl_info = {}
    GPU_NUM = 0
    for plat in pyopencl.get_platforms():
        for dev in plat.get_devices(pyopencl.device_type.CPU):
            opencl_info['CPU.NAME'] = dev.get_info(pyopencl.device_info.NAME)
            opencl_info['CPU.DRIVER_VERSION'] = dev.get_info(pyopencl.device_info.DRIVER_VERSION)
            opencl_info['CPU.GLOBAL_MEM_SIZE'] = dev.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE)
        for dev in plat.get_devices(pyopencl.device_type.GPU):
            opencl_info[f'GPU{GPU_NUM}.NAME'] = dev.get_info(pyopencl.device_info.NAME)
            opencl_info[f'GPU{GPU_NUM}.DRIVER_VERSION'] = dev.get_info(pyopencl.device_info.DRIVER_VERSION)
            opencl_info[f'GPU{GPU_NUM}.GLOBAL_MEM_SIZE'] = dev.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE)
            GPU_NUM += 1

    if IS_WINDOWS:
        OS_NAME = f'{platform.system()} {platform.release()}'
    if not IS_WINDOWS:
        output = subprocess.check_output(['lsb_release', '-d'], text=True)
        match_obj = re.search(r'Description:([\w\s\.]+)', output)
        if match_obj != None:
            OS_NAME = match_obj.groups()[0].strip()

    info = []
    info.append(['HOSTNAME', platform.node()])
    info.append(['IP', IP_ADDRESS])
    info.append(['OS', f'{OS_NAME}'])
    info.append(['OpenVINO', f'{get_version()}'])

    if opencl_info.get('CPU.NAME', '') == '':
        info.append(['CPU (NAME)', cpuinfo.get_cpu_info()["brand_raw"]])
        info.append(['CPU (GLOBAL_MEM_SIZE)', sizeof_fmt(round(psutil.virtual_memory().total))])
    else:
        info.append(['OpenCL::CPU (NAME)', opencl_info['CPU.NAME']])
        info.append(['OpenCL::CPU (DRIVER_VERSION)', opencl_info['CPU.DRIVER_VERSION']])
        info.append(['OpenCL::CPU (GLOBAL_MEM_SIZE)', sizeof_fmt(opencl_info['CPU.GLOBAL_MEM_SIZE'])])

    for gpu_id in range(0, GPU_NUM):
         info.append([f'OpenCL::GPU[{gpu_id}] (NAME)', opencl_info[f'GPU{gpu_id}.NAME']])
         info.append([f'OpenCL::GPU[{gpu_id}] (DRIVER_VERSION)', opencl_info[f'GPU{gpu_id}.DRIVER_VERSION']])
         info.append([f'OpenCL::GPU[{gpu_id}] (GLOBAL_MEM_SIZE)', sizeof_fmt(opencl_info[f'GPU{gpu_id}.GLOBAL_MEM_SIZE'])])

    info.append(['DATE(YYYYMMDD_HHMM)', f'{NOW}'])

    return info

def find_process(process, name):
    if process.name() == name:
        return process
    else:
        children_processes = process.children(recursive=True)
        for child in children_processes:
            target = find_process(child, name)
            if target != None:
                return target
    return None

#
# WA: At subprocess.popen(cmd, ...), the cmd should be string on ubuntu or be string array on windows.
#
def convert_cmd_for_popen(cmd):
    return cmd.split() if IS_WINDOWS else cmd

def get_filepath_with_url(args, filename):
    filepath = os.path.join(args.output_dir, filename)
    OUTPUT_DIRNAME = os.path.split(args.output_dir)[-1]
    return f'{filepath} | http://{IP_ADDRESS}/{OUTPUT_DIRNAME}/{filename}'

class HWDataKey(enum.IntEnum):
    MEM_USAGE_PERCENT = 1
    MEM_USAGE_SIZE = 2
    CPU_USAGE_PERCENT = 3

class HWResourceTracker(Thread):
    class Data:
        def __init__(self) -> None:
            self.__data = {}

        def append(self, key, timestamp, value):
            if self.__data.get(key, None) == None:
                self.__data[key] = {}
            self.__data[key][timestamp] = value

        def max(self, key):
            try:
                start_time = self.__get_start_time(key)
                return max(dict(filter(lambda data: data[0] > start_time, self.__data[key].items())).values())
            except:
                return 0

        def min(self, key):
            try:
                start_time = self.__get_start_time(key)
                return min(dict(filter(lambda data: data[0] > start_time, self.__data[key].items())).values())
            except:
                return 0

        def mean(self, key):
            try:
                start_time = self.__get_start_time(key)
                return mean(dict(filter(lambda data: data[0] > start_time, self.__data[key].items())).values())
            except:
                return 0

        def __get_start_time(self, key):
            max_ts = max(self.__data[key].keys())
            min_ts = min(self.__data[key].keys())
            return (max_ts - min_ts) * 0.7 + min_ts


    def __init__(self, process = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = process
        self.running = True
        self.data = HWResourceTracker.Data()
        self.period_time = 0.1  # unit: seconds

    def run(self):
        while self.running:
            time.sleep(self.period_time)
            timestamp_ms = round(time.time()*1000)

            if (self.process != None):
                if self.process.is_running():
                    self.data.append(HWDataKey.CPU_USAGE_PERCENT, timestamp_ms, self.process.cpu_percent())
                    self.data.append(HWDataKey.MEM_USAGE_PERCENT, timestamp_ms, self.process.memory_percent())
                    self.data.append(HWDataKey.MEM_USAGE_SIZE, timestamp_ms, self.process.memory_info().rss)
            else:
                memory_usage_dict = dict(psutil.virtual_memory()._asdict())
                self.data.append(HWDataKey.CPU_USAGE_PERCENT, timestamp_ms, psutil.cpu_percent())
                self.data.append(HWDataKey.MEM_USAGE_PERCENT, timestamp_ms, memory_usage_dict['percent'])
                self.data.append(HWDataKey.MEM_USAGE_SIZE, timestamp_ms, memory_usage_dict['used'])

    def stop(self):
        if self.is_alive():
            self.running = False
            self.join()
            self.running = True

            return self.data.mean(HWDataKey.CPU_USAGE_PERCENT), \
                   self.data.max(HWDataKey.MEM_USAGE_SIZE) - self.data.min(HWDataKey.MEM_USAGE_SIZE), \
                   self.data.max(HWDataKey.MEM_USAGE_PERCENT) - self.data.min(HWDataKey.MEM_USAGE_PERCENT)
        raise Exception

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)

    def flush(self):
        pass

def send_mail(report_path):
    MAIL_TITLE = f'Daily report[{NOW}] from {HOSTNAME}'
    # MAIL_TO = 'nex.nswe.odt.runtime.kor@intel.com vladimir.paramuzov@intel.com'
    MAIL_TO = 'nex.nswe.odt.runtime.kor@intel.com'

    if IS_WINDOWS:
        ID_RSA_PATH = f'{os.environ["USERPROFILE"]}\\.ssh\\id_rsa_sungeunk'
        MAIL_RELAY_SERVER = 'sungeunk@dg2raptorlake.ikor.intel.com'
        cmd = f'ssh -i {ID_RSA_PATH} {MAIL_RELAY_SERVER} \"mail --content-type=text/html -s \\\"{MAIL_TITLE}\\\" {MAIL_TO} \" < {report_path}'
    else:
        cmd = f'cat {report_path} | mail --content-type=text/html -s \"{MAIL_TITLE}\" {MAIL_TO}'

    subprocess.call(cmd, shell=True)



################################################
# Abstract TestCase class
################################################
class ReportItemKey(enum.Enum):
    TEST_CLASS = 'test class'
    REMOVE_CACHE = 'Remove cache'
    STATEFUL = 'Stateful'
    IN = 'In'
    OUT = 'Out'
    ADJUST_PREALLOC = 'Adjust prealloc'
    LOADING = 'loading time(s)'
    CACHE_SIZE = 'cache size'
    FIRST_INFERENCE_LATENCY = '1st inf latency(s)'
    SECOND_INFERENCE_LATENCY = '2nd inf latency(s)'
    MEM_USAGE_SIZE = 'mem usage(size)'
    MEM_USAGE_PERCENT = 'mem usage(%)'
    CPU_USAGE_PERCENT = 'cpu usage(%)'
    SMOKE_TEST = 'smoke test'

class TestConfig:
    def __init__(self, remove_cache = False, stateful = False, in_text_length = 9, out_text_length = 128, adjust_prealloc = False):
        self.REMOVE_CACHE = remove_cache
        self.IN_TEXT_LENGTH = in_text_length
        self.OUT_TEXT_LENGTH = out_text_length
        self.STATEFUL = stateful
        self.ADJUST_PREALLOC = adjust_prealloc

    def __str__(self):
        return ('RemoveCache' if self.REMOVE_CACHE else 'KeepCache') \
                + '/' \
                + ('Stateful' if self.STATEFUL else 'NonStateful') \
                + '/' \
                + 'IN:' + str(self.IN_TEXT_LENGTH) \
                + '/' \
                + 'OUT:' + str(self.OUT_TEXT_LENGTH) \
                + ('/MORE_PREALLOC' if self.ADJUST_PREALLOC else '')

class TestClass(metaclass=abc.ABCMeta):
    def __init__(self, args, conf = TestConfig):
        self.args = args
        self.conf = conf

    def get_test_name(self):
        return self.__class__.__name__ + '/' + str(self.conf)

    def get_model_path(self):
        raise []

    def get_model_info(self):
        info = []
        APP_NAME = self.__class__.__name__
        for path in self.get_model_path():
            PATH_BIN = f'{Path(path).with_suffix(".bin")}'
            info.append([APP_NAME, path] + list(get_file_info(path)))
            info.append([APP_NAME, PATH_BIN] + list(get_file_info(PATH_BIN)))

        return info

    @abc.abstractmethod
    def get_cmd(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_parsers(self):
        raise NotImplemented

    @abc.abstractmethod
    def parse_out_sentense(self, log):
        raise NotImplemented

    @abc.abstractmethod
    def generate_report_item(self, parsed_results):
        raise NotImplemented

    def get_cache_dir(self):
        return self.args.cache_dir

    def remove_cache(self):
        cache_dir = self.get_cache_dir()
        if os.path.exists(cache_dir):
            for file in glob(os.path.join(cache_dir, '*.cl_cache')):
                os.remove(file)
            for file in glob(os.path.join(cache_dir, '*.blob')):
                os.remove(file)
            time.sleep(1)

    def get_cache_size(self):
        cache_dir = self.get_cache_dir()
        file_size = 0
        if os.path.exists(cache_dir):
            for file in glob(os.path.join(cache_dir, '*.cl_cache')):
                file_size += os.path.getsize(file)
            for file in glob(os.path.join(cache_dir, '*.blob')):
                file_size += os.path.getsize(file)
        return file_size

    def run(self):
        if self.get_cmd() == '':
            return {}, ''

        print(f'[{self.get_test_name()}] Start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if self.conf.REMOVE_CACHE:
            print(f'[{self.get_test_name()}] - remove cache: {self.get_cache_dir()}')
            self.remove_cache()
        
        if self.conf.ADJUST_PREALLOC:
            os.environ['OV_GPU_MemPreallocationOptions'] = '20 65536 2 1.0'
        else:
            os.environ['OV_GPU_MemPreallocationOptions'] = ''


        print(f'[{self.get_test_name()}] - Run cmd: {self.get_cmd()}')
        reportItem = {}
        out_sentense = ''
        with subprocess.Popen(convert_cmd_for_popen(self.get_cmd()),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, encoding='utf8') as proc:
            tracker = HWResourceTracker()
            tracker.start()

            out_log, err_log = proc.communicate()

            print(f'[{self.get_test_name()}] - End cmd retcode({proc.returncode})')

            cpu_usage_percent, mem_usage_size, mem_usage_percent = tracker.stop()
            reportItem[ReportItemKey.CPU_USAGE_PERCENT] = cpu_usage_percent
            reportItem[ReportItemKey.MEM_USAGE_SIZE] = mem_usage_size
            reportItem[ReportItemKey.MEM_USAGE_PERCENT] = mem_usage_percent

            reportItem[ReportItemKey.TEST_CLASS] = self.__class__.__name__
            reportItem[ReportItemKey.REMOVE_CACHE] = self.conf.REMOVE_CACHE
            reportItem[ReportItemKey.STATEFUL] = self.conf.STATEFUL
            reportItem[ReportItemKey.IN] = self.conf.IN_TEXT_LENGTH
            reportItem[ReportItemKey.OUT] = self.conf.OUT_TEXT_LENGTH
            reportItem[ReportItemKey.ADJUST_PREALLOC] = self.conf.ADJUST_PREALLOC

            if proc.returncode == 0 and out_log != None:
                results = {}
                for line in out_log.splitlines():
                    for key, regex in self.get_parsers().items():
                        match_obj = regex.search(line)
                        if match_obj != None:
                            results[key] = match_obj.groups()[0]

                out_sentense = self.parse_out_sentense(out_log)

                reportItem.update(self.generate_report_item(results))
                reportItem[ReportItemKey.CACHE_SIZE] = self.get_cache_size()

        print(f'[{self.get_test_name()}] ---------<beginning of out>-----------------')
        print(f'{out_log}')
        print(f'[{self.get_test_name()}] ---------<end of out>-----------------------')
        print(f'[{self.get_test_name()}] ---------<beginning of err>-----------------')
        print(f'{err_log}')
        print(f'[{self.get_test_name()}] ---------<end of err>-----------------------')
        print(f'[{self.get_test_name()}] End')
        return reportItem, out_sentense



################################################
# Test Classes
################################################
class TestChatGLM3(TestClass):
    class RegexKey(enum.Enum):
        READ_MODEL = 'READ_MODEL'
        COMPILE_MODEL = 'COMPILE_MODEL'
        FIRST_INFERENCE_LATENCY = 'FIRST_INFERENCE_LATENCY'
        SECOND_INFERENCE_LATENCY = 'SECOND_INFERENCE_LATENCY'

    def get_model_path(self):
        return os.path.join(*[f'{self.args.model_dir}', 'ChatGLM3-6B-GPTQ-INT4-OV', 'GPTQ_INT4-FP16'])

    def get_cmd(self):
        APP_PATH = os.path.join(*[self.args.working_dir, 'ov_llm_bench', 'benchmark.py'])
        MODEL_DIR = self.get_model_path()
        return f'C:\\dev\\sungeunk\\note\\venv\\Scripts\\python.exe {APP_PATH} -m {MODEL_DIR} -cd {self.args.cache_dir} -pl {self.conf.IN_TEXT_LENGTH} -mnt {self.conf.OUT_TEXT_LENGTH} -d {self.args.device}'

    def get_parsers(self):
        regex_list = {}
        regex_list[self.RegexKey.READ_MODEL] = re.compile(f'read model time: (\d+.\d+) s')
        regex_list[self.RegexKey.COMPILE_MODEL] = re.compile(f'compile model time: (\d+.\d+) s')
        regex_list[self.RegexKey.FIRST_INFERENCE_LATENCY] = re.compile(f'first token time: (\d+.\d+) s')
        regex_list[self.RegexKey.SECOND_INFERENCE_LATENCY] = re.compile(f'Total average generaton speed: (\d+.\d+) tokens/s')
        return regex_list

    def parse_out_sentense(self, log):
        return ''

    def generate_report_item(self, parsed_results):
        out_report_item = {}
        out_report_item[ReportItemKey.LOADING] = float(parsed_results.get(self.RegexKey.READ_MODEL, 0)) + float(parsed_results.get(self.RegexKey.COMPILE_MODEL, 0))
        out_report_item[ReportItemKey.FIRST_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.FIRST_INFERENCE_LATENCY, 0))
        out_report_item[ReportItemKey.SECOND_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.SECOND_INFERENCE_LATENCY, 0))
        return out_report_item

class TestChatGLM3CPP(TestClass):
    class RegexKey(enum.Enum):
        READ_MODEL = 'READ_MODEL'
        COMPILE_MODEL = 'COMPILE_MODEL'
        FIRST_INFERENCE_LATENCY = 'FIRST_INFERENCE_LATENCY'
        SECOND_INFERENCE_LATENCY = 'SECOND_INFERENCE_LATENCY'

    def get_model_path(self):
        # if self.conf.STATEFUL:
        #     return os.path.join(*[f'{self.args.model_dir}', 'ChatGLM3_6B_GPTQ_INT4-FP16_ww50_stateful', 'pytorch', 'dldt', 'GPTQ_INT4-FP16', 'modified_openvino_model.xml']), \
        #         os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'tokenizer.xml']), \
        #         os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'detokenizer.xml'])
        # else:
        #     return os.path.join(*[f'{self.args.model_dir}', 'chatglm3-6b-gptq-int4-single-if_ekaterina', 'pytorch', 'dldt', 'GPTQ_INT4-FP16', 'modified_openvino_model.xml']), \
        #         os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'tokenizer.xml']), \
        #         os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'detokenizer.xml'])
        return os.path.join(*[f'{self.args.model_dir}', 'ww51-chatglm-bkm-stateful', 'modified_openvino_model.xml']), \
               os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'tokenizer.xml']), \
               os.path.join(*[f'{self.args.model_dir}', 'chatglm3_ov_tokenizer', 'detokenizer.xml'])

    def get_cmd(self):
        APP_PATH = os.path.join(*[self.args.working_dir, 'openvino.genai.chatglm3', 'build', 'llm', 'chatglm_cpp', 'chatglm' + ('.exe' if IS_WINDOWS else '')])
        MODEL_PATH, TOKENIZER_PATH, DETOKENIZER_PATH = self.get_model_path()
        INPUT_INDEX_MAP = {9:0, 32:1, 256:2, 512:3, 1024:4, 2048:5, 3000:6}
        INPUT_INDEX = INPUT_INDEX_MAP.get(self.conf.IN_TEXT_LENGTH, 0)

        return f'{APP_PATH} -m {MODEL_PATH} -token {TOKENIZER_PATH} -detoken {DETOKENIZER_PATH} -pi {INPUT_INDEX} --output_fixed_len {self.conf.OUT_TEXT_LENGTH} -d {self.args.device}'

    def get_parsers(self):
        regex_list = {}
        regex_list[self.RegexKey.READ_MODEL] = re.compile(f'Load chatglm tokenizer took (\d+.\d+) ms')
        regex_list[self.RegexKey.COMPILE_MODEL] = re.compile(f'Compile LLM model took (\d+.\d+) ms')
        regex_list[self.RegexKey.FIRST_INFERENCE_LATENCY] = re.compile(f'First token took (\d+.\d+) ms')
        regex_list[self.RegexKey.SECOND_INFERENCE_LATENCY] = re.compile(f'Other Avg inference took total \d+.\d+ ms token num \d+ first \d+.\d+ ms  avg (\d+.\d+) ms')
        return regex_list

    def parse_out_sentense(self, log):
        found_begin_sentense = False
        for line in log.splitlines():
            if found_begin_sentense:
                return line
            if re.search(r'First token took (\d+.\d+) ms', line) != None:
                found_begin_sentense = True
        return ''

    def generate_report_item(self, parsed_results):
        out_report_item = {}
        out_report_item[ReportItemKey.LOADING] = (float(parsed_results.get(self.RegexKey.READ_MODEL, 0)) + float(parsed_results.get(self.RegexKey.COMPILE_MODEL, 0))) / 1000
        out_report_item[ReportItemKey.FIRST_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.FIRST_INFERENCE_LATENCY, 0)) / 1000
        out_report_item[ReportItemKey.SECOND_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.SECOND_INFERENCE_LATENCY, 0)) / 1000
        return out_report_item

class TestQwenCPP(TestClass):
    class RegexKey(enum.Enum):
        READ_MODEL = 'READ_MODEL'
        COMPILE_MODEL = 'COMPILE_MODEL'
        FIRST_INFERENCE_LATENCY = 'FIRST_INFERENCE_LATENCY'
        SECOND_INFERENCE_LATENCY = 'SECOND_INFERENCE_LATENCY'

    def get_model_path(self):
        if self.conf.STATEFUL:
            return os.path.join(*[f'{self.args.model_dir}', 'ww52-qwen-bkm-stateful', 'modified_openvino_model.xml']),
        else:
            return os.path.join(*[f'{self.args.model_dir}', 'ww52-qwen-bkm-stateless', 'modified_openvino_model.xml']),

    def get_cmd(self):
        APP_PATH = os.path.join(*[self.args.working_dir, 'openvino.genai.qwen', 'llm', 'qwen_cpp', 'build', 'bin', 'main' + ('.exe' if IS_WINDOWS else '')])
        MODEL_PATH, = self.get_model_path()

        if self.conf.STATEFUL:
            TOKENIZER_PATH = os.path.join(*[f'{self.args.model_dir}', 'ww52-qwen-bkm-stateful', 'qwen.tiktoken'])
        else:
            TOKENIZER_PATH = os.path.join(*[f'{self.args.model_dir}', 'ww52-qwen-bkm-stateless', 'qwen.tiktoken'])
        INPUT_INDEX_MAP = {9:0, 32:1, 256:2, 512:3, 1024:4, 2048:5, 3000:6}
        INPUT_INDEX = INPUT_INDEX_MAP.get(self.conf.IN_TEXT_LENGTH, 0)

        return f'{APP_PATH} -m {MODEL_PATH} -t {TOKENIZER_PATH} -pi {INPUT_INDEX} -mcl {self.conf.OUT_TEXT_LENGTH} -d {self.args.device} -l en' + (' --stateful' if self.conf.STATEFUL else '')

    def get_parsers(self):
        regex_list = {}
        regex_list[self.RegexKey.READ_MODEL] = re.compile(f'Load Qwen tokenizer took (\d+.\d+) ms')
        regex_list[self.RegexKey.COMPILE_MODEL] = re.compile(f'Compile model took: (\d+.\d+) ms')
        regex_list[self.RegexKey.FIRST_INFERENCE_LATENCY] = re.compile(f'First inference took (\d+.\d+) ms')
        regex_list[self.RegexKey.SECOND_INFERENCE_LATENCY] = re.compile(f'Average other token latency: (\d+.\d+) ms')
        return regex_list

    def parse_out_sentense(self, log):
        ret = ''
        found_begin_sentense = False
        for line in log.splitlines():
            if re.search(r'First inference took (\d+.\d+) ms', line) != None:
                found_begin_sentense = True
                continue
            if re.search(r'Average other token latency: (\d+.\d+) ms', line) != None:
                return ret
            if found_begin_sentense:
                ret += f'{line}\n'

        return ''

    def generate_report_item(self, parsed_results):
        out_report_item = {}
        out_report_item[ReportItemKey.LOADING] = (float(parsed_results.get(self.RegexKey.READ_MODEL, 0)) + float(parsed_results.get(self.RegexKey.COMPILE_MODEL, 0))) / 1000
        out_report_item[ReportItemKey.FIRST_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.FIRST_INFERENCE_LATENCY, 0)) / 1000
        out_report_item[ReportItemKey.SECOND_INFERENCE_LATENCY] = float(parsed_results.get(self.RegexKey.SECOND_INFERENCE_LATENCY, 0)) / 1000
        return out_report_item

class TestEmpty(TestClass):
    def __init__(self):
        super().__init__(None, None)

    def get_test_name(self):
        return ''

    def get_cmd(self):
        return ''

    def get_parsers(self):
        return {}

    def parse_out_sentense(self, log):
        return ''

    def generate_report_item(self, parsed_results):
        return {}

def run_stable_diffusion(args):
    APP = os.path.join(args.gpu_tools_dir, 'check_custom_perf.py')
    CMD = f'python {APP} -a {args.benchmark_app} -m {args.model_dir} -d {args.device} --transformers_sd -v'
    with subprocess.Popen(convert_cmd_for_popen(CMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, encoding='utf8') as proc:
        out_log, err_log = proc.communicate()
        print(f'err: {err_log}')

        if proc.returncode == 0:
            table_str = ''
            for line in out_log.splitlines():
                if len(line) > 0 and line[0] == '|' and line[3] != '-':
                    table_str += (line + '\n')
            table = get_table_from_str(table_str)

            # WA: remove uncessary col/raw
            table.pop()
            table.pop()
            for item in table:
                item.pop()
                item.pop()
                item.pop()
            return table

    return []

def run_263_latent_consistency_from_notebook(args):
    report = []
    APP = os.path.join(args.gpu_tools_dir, '263-latent-consistency-models-image-generation.fp.py')
    RESULT_IMG_FILENAME = f'{NOW}.263-latent-consistency.png'
    RESULT_IMG_PATH = os.path.join(args.output_dir, RESULT_IMG_FILENAME)
    CMD = f'python {APP} -r {RESULT_IMG_PATH} -d {args.device}'
    with subprocess.Popen(convert_cmd_for_popen(CMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, encoding='utf8') as proc:
        out_log, err_log = proc.communicate()
        print(f'err: {err_log}')

        if proc.returncode == 0:
            report.append(['Inference Index', 'Inference Time'])
            for line in out_log.splitlines():
                # match_obj1 = re.search(r'compile_model\(unet\): +(\d+.\d+) s', line)
                match_obj = re.search(r'\[(\d+)\] inference\(unet\): +(\d+.\d+) s', line)
                if match_obj != None:
                    report.append([int(match_obj.groups()[0]), float(match_obj.groups()[1])])
            return report, f'{get_filepath_with_url(args, RESULT_IMG_FILENAME)}'

    return [], ''

def get_table_from_str(table_str):
    table = []
    if len(table_str) > 0:
        for line in table_str.splitlines():
            line_split = [t.strip() for t in line.split('|')[1:-1]]
            table.append(line_split)
    return table

def get_table_from_report(report_filepath):
    def get_table(fis):
        table_str = ''
        while True:
            line = fis.readline()
            if line == '': break
            if line[0:2] == '|-': continue
            elif line[0] == '|': table_str += line
            elif table_str != '': break

        return get_table_from_str(table_str)

    with open(report_filepath, 'r') as fis:
        get_table(fis)  # ignore system info table
        main_table = get_table(fis)
        model_table = get_table(fis)
        sd_table = get_table(fis)
        notebook_263_table = get_table(fis)

    return main_table, model_table, sd_table, notebook_263_table


################################################
# Main
################################################
def main():
    parser = argparse.ArgumentParser(description="Run daily check for chatglm" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_dir', help='root directory for models', type=Path, default=PWD)
    parser.add_argument('-w', '--working_dir', help='working directory', type=Path, default=PWD)
    parser.add_argument('-d', '--device', help='target device', type=Path, default='GPU')
    parser.add_argument('-o', '--output_dir', help='output directory to store log files', type=Path, default=os.path.join(*[PWD, 'output']))
    parser.add_argument('-cd', '--cache_dir', help='cache directory', type=Path, default=os.path.join(*[PWD, 'model_cache']))
    parser.add_argument('--gpu_tools_dir', help='gpu-tools directory', type=Path, default=os.path.join(*[PWD, 'libraries.ai.videoanalyticssuite.gpu-tools']))
    parser.add_argument('--benchmark_app', help='benchmark_app(cpp) path', type=Path, default=None)
    parser.add_argument('--log_to_stdio', help='[DEBUG] enable to write logs to stdout.', action='store_true')
    parser.add_argument('--no_mail', help='disable sending mail', action='store_true')
    parser.add_argument('--ref_pickle', help='reference pickle to compare performance', type=Path, default=None)
    parser.add_argument('--this_pickle', help='target pickle to compare performance', type=Path, default=None)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory
    os.chdir(args.working_dir)

    if not IS_WINDOWS:
        # WA: this is needed to run an executable without './'.
        os.environ["PATH"] = f'{PWD}:{os.environ["PATH"]}'

    # Redirect stdout/stderr to file
    RAW_FILENAME = f'{NOW}.raw'
    RAW_PATH = os.path.join(*[f'{args.output_dir}', RAW_FILENAME])

    if not args.log_to_stdio:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s: %(message)s',
            datefmt='%I:%M:%S',
            filename=RAW_PATH,
            filemode='a'
            )
        log = logging.getLogger()
        sys.stdout = StreamToLogger(log, logging.INFO)
        sys.stderr = StreamToLogger(log, logging.ERROR)

    # Create test cases
    test_list = [
        # [ChatGLM3CPP] - Non-stateful
        # TestChatGLM3CPP(args, TestConfig(remove_cache=True)),
        # TestChatGLM3CPP(args, TestConfig(in_text_length=9, out_text_length=128)),
        # TestChatGLM3CPP(args, TestConfig(in_text_length=256, out_text_length=256)),
        # TestChatGLM3CPP(args, TestConfig(in_text_length=512, out_text_length=512)),
        # TestChatGLM3CPP(args, TestConfig(in_text_length=1024, out_text_length=1024)),
        # TestChatGLM3CPP(args, TestConfig(in_text_length=2048, out_text_length=512)),

        # [ChatGLM3CPP] - Stateful
        TestChatGLM3CPP(args, TestConfig(in_text_length=9, out_text_length=128, stateful=True, remove_cache=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=9, out_text_length=128, stateful=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=256, out_text_length=256, stateful=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=512, out_text_length=512, stateful=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=1024, out_text_length=1024, stateful=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=2048, out_text_length=512, stateful=True)),
        #TestChatGLM3CPP(args, TestConfig(in_text_length=3000, out_text_length=200, stateful=True)),

        TestEmpty(),

        # [QwenCPP] - Non-stateful
        # TestQwenCPP(args, TestConfig(remove_cache=True)),
        # TestQwenCPP(args, TestConfig(in_text_length=9, out_text_length=128)),
        # TestQwenCPP(args, TestConfig(in_text_length=256, out_text_length=256)),
        # TestQwenCPP(args, TestConfig(in_text_length=512, out_text_length=512)),
        # TestQwenCPP(args, TestConfig(in_text_length=1024, out_text_length=1024)),

        # [QwenCPP] - Stateful
        TestQwenCPP(args, TestConfig(in_text_length=9, out_text_length=128, stateful=True, remove_cache=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=9, out_text_length=128, stateful=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=256, out_text_length=256, stateful=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=512, out_text_length=512, stateful=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=1024, out_text_length=1024, stateful=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=2048, out_text_length=512, stateful=True)),
        #TestQwenCPP(args, TestConfig(in_text_length=3000, out_text_length=200, stateful=True)),

        # TestEmpty(),

        # [QwenCPP] - Stateful + adjust_prealloc
        # TestQwenCPP(args, TestConfig(in_text_length=9, out_text_length=128, stateful=True, adjust_prealloc=True)),
        # TestQwenCPP(args, TestConfig(in_text_length=256, out_text_length=256, stateful=True, adjust_prealloc=True)),
        # TestQwenCPP(args, TestConfig(in_text_length=512, out_text_length=512, stateful=True, adjust_prealloc=True)),
        # TestQwenCPP(args, TestConfig(in_text_length=1024, out_text_length=1024, stateful=True, adjust_prealloc=True)),

        # [ChatGLM3] 1. No cache + short prompt
        #            2. Cache + short prompt
        #            3. Cache + long prompt   << not working with 1024 in_text_length on windows
        # TestChatGLM3(args, TestConfig(remove_cache=True)),
        # TestChatGLM3(args, TestConfig()),
        # TestChatGLM3(args, TestConfig(in_text_length=LONG_LEN, out_text_length=LONG_LEN)),
    ]

    task_start_time = time.time()

    # Run all tests
    out_sentenses = ''
    if args.this_pickle == None:
        main_data_this = []
        main_data_this.append([key.value for key in ReportItemKey])
        for test in test_list:
            reportItem, out_sentense = test.run()
            main_data_this.append([reportItem.get(key) for key in ReportItemKey])
            out_sentenses += f'[{test.get_test_name()}] {out_sentense}\n\n'

        sd_data_this = []
        # if args.benchmark_app != None and os.path.exists(args.benchmark_app):
        #     sd_data_this = run_stable_diffusion(args)

        # notebook_263_data_this, notebook_263_result_url = run_263_latent_consistency_from_notebook(args)
        notebook_263_data_this = []
        notebook_263_result_url = ''

        # Get system info
        system_data_this = get_system_info()

        # Get model info. filter out the deplicated test
        unique_test = {}
        for test in test_list:
            if test.conf == None:
                continue
            if unique_test.get((test.__class__, test.conf.STATEFUL), None) == None:
                unique_test[(test.__class__, test.conf.STATEFUL)] = test

        model_data_this = []
        for key, test in unique_test.items():
            info = test.get_model_info()
            if len(info) > 0:
                model_data_this += list(info)
    else:
        # WA: for test to load data
        with open(args.this_pickle, 'rb') as fis:
            this_dict = pickle.load(fis)
            system_data_this = this_dict['system']
            main_data_this = this_dict['main']
            model_data_this = this_dict['model']
            sd_data_this = this_dict['sd']
            notebook_263_data_this = this_dict['notebook_263']


    # save reports to pickle
    if args.this_pickle == None:
        save_dict = {'system': system_data_this, 'main':main_data_this, 'model':model_data_this, 'sd':sd_data_this, 'notebook_263':notebook_263_data_this}
        PICKLE_FILENAME = f'{NOW}.pickle'
        PICKLE_PATH = os.path.join(*[f'{args.output_dir}', PICKLE_FILENAME])
        with open(PICKLE_PATH, 'wb') as fos:
            pickle.dump(save_dict, fos)

    # generate human-readable report
    main_reports = main_data_this
    notebook_263_reports = notebook_263_data_this
    model_reports = model_data_this
    sd_reports = sd_data_this

    if args.ref_pickle != None:
        with open(args.ref_pickle, 'rb') as fis:
            ref_dict = pickle.load(fis)

            if len(main_data_this) > 0:
                main_reports = copy.deepcopy(main_data_this)

                def get_ref_index(data_this, data_ref):
                    for i in range(0, len(data_ref)):
                        if data_this[0:5] == data_ref[i][0:5]:
                            return i
                    return -1

                for i in range(1, len(main_data_this)):
                    data_this = main_data_this[i]
                    if data_this[6] == '' or data_this[6] == None: continue

                    main_reports[i][ 6] = f'{data_this[ 6]:.3f}'
                    main_reports[i][ 7] = f'{sizeof_fmt(data_this[7])}'
                    main_reports[i][ 8] = f'{data_this[ 8]:.3f}'
                    main_reports[i][ 9] = f'{data_this[ 9]:.3f}'
                    main_reports[i][10] = f'{sizeof_fmt(data_this[10])}'
                    main_reports[i][11] = f'{data_this[11]:.1f}'
                    main_reports[i][12] = f'{data_this[12]:.1f}'

                    ref_i = get_ref_index(data_this, ref_dict['main'])
                    if ref_i != -1:
                        data_ref = ref_dict['main'][ref_i]
                        if data_ref[6] != '' and data_ref[6] != None:
                            main_reports[i][ 6] += f' ({data_this[6] - data_ref[6]:.3f})'
                            main_reports[i][ 7] += f' ({sizeof_fmt(data_this[7] - data_ref[7])})'
                            main_reports[i][ 8] += f' ({data_this[8] - data_ref[8]:.3f})'
                            main_reports[i][ 9] += f' ({data_this[9] - data_ref[9]:.3f})'
                            main_reports[i][10] += f' ({sizeof_fmt(data_this[10] - data_ref[10])})'
                            main_reports[i][11] += f' ({data_this[11] - data_ref[11]:.1f})'
                            main_reports[i][12] += f' ({data_this[12] - data_ref[12]:.1f})'

            if len(model_data_this) > 0:
                model_reports = copy.deepcopy(model_data_this)
                model_reports.insert(0, ['test class', 'Model', 'FileSize', 'FileHash', 'Compare (size / hash)'])
                model_data_ref = ref_dict['model']
                for i in range(0, len(model_data_ref)):
                    cache_diff_size = sizeof_fmt(model_data_this[i][2] - model_data_ref[i][2])
                    same_hash = model_data_this[i][3] == model_data_ref[i][3]
                    model_reports[i+1][2] = sizeof_fmt(model_data_this[i][2])
                    model_reports[i+1].append(f'{cache_diff_size} / {"Same" if same_hash else "Diff"}')

            if len(sd_data_this) > 0:
                sd_reports = copy.deepcopy(sd_data_this)
                sd_data_ref = ref_dict['sd']
                for i in range(1, len(sd_data_ref)):
                    sd_reports[i][1] += f' ({float(sd_data_this[i][1]) - float(sd_data_ref[i][1]):.1f})'

            if len(notebook_263_data_this) > 0:
                notebook_263_reports = copy.deepcopy(notebook_263_data_this)
                notebook_263_data_ref = ref_dict['notebook_263']
                for i in range(1, len(notebook_263_data_ref)):
                    notebook_263_reports[i][1] = f'{notebook_263_data_this[i][1]:.3f} ({notebook_263_data_this[i][1] - notebook_263_data_ref[i][1]:.3f})'
    else:
        if len(main_data_this) > 0:
            main_reports = copy.deepcopy(main_data_this)
            for i in range(1, len(main_data_this)):
                if main_data_this[i][6] == '' or main_data_this[i][6] == None: continue

                main_reports[i][ 7] = f'{sizeof_fmt(main_reports[i][ 7])}'
                main_reports[i][10] = f'{sizeof_fmt(main_reports[i][10])}'

        if len(model_data_this) > 0:
            model_reports = copy.deepcopy(model_data_this)
            model_reports.insert(0, ['test class', 'Model', 'FileSize', 'FileHash'])
            for i in range(1, len(model_reports)):
                model_reports[i][2] = f'{sizeof_fmt(model_reports[i][2])}'

    # Gather all cmd
    all_cmd_str = ''
    for test in test_list:
        all_cmd_str += (f'[{test.get_test_name()}] ' + test.get_cmd() + '\n')


    # Print reports as a tabulate
    system_info_tabulate = tabulate(system_data_this, tablefmt="github", floatfmt='.3f', stralign='left', numalign='right')

    main_tabulate = None
    if len(main_reports) > 0:
        main_tabulate = tabulate(main_reports[1:], headers=main_reports[0], tablefmt="github", floatfmt='.3f', stralign='right', numalign='right')

    model_tabulate = None
    if len(model_reports) > 0:
        model_tabulate = tabulate(model_reports[1:], headers=model_reports[0], tablefmt="github", floatfmt='.3f', stralign='right', numalign='right')

    sd_tabulate = None
    if len(sd_reports) > 0:
        sd_tabulate = tabulate(sd_reports[1:], headers=sd_reports[0], tablefmt="github", floatfmt='.3f', stralign='right', numalign='right')

    notebook_263_tabulate = None
    if len(notebook_263_reports) > 0:
        notebook_263_tabulate = tabulate(notebook_263_reports[1:], headers=notebook_263_reports[0], tablefmt="github", floatfmt='.3f', stralign='left', numalign='right')


    # write report file.
    PROCESS_TIME = time.time() - task_start_time
    REPORT_FILENAME = f'{NOW}.report'
    REPORT_PATH = os.path.join(*[f'{args.output_dir}', REPORT_FILENAME])
    with open(REPORT_PATH, 'w', encoding='utf-8') as fos:
        fos.write('<pre>\n')
        fos.write(f'TOTAL TASK TIME: {time.strftime("%H:%M:%S", time.gmtime(PROCESS_TIME))}' + '\n')
        fos.write(f'RawLogFile: {get_filepath_with_url(args, RAW_FILENAME)}\n')
        fos.write(f'ReportFile: {get_filepath_with_url(args, REPORT_FILENAME)}\n\n')
        fos.write('System Info:\n' + system_info_tabulate + '\n\n')
        if main_tabulate != None:
            fos.write('Test results:\n' + main_tabulate + '\n\n')
        if model_tabulate != None:
            fos.write('Models:\n' + model_tabulate + '\n\n')
        if sd_tabulate != None:
            fos.write('stable_diffusion (bencharmk_app):\n' + sd_tabulate + '\n\n')
        if notebook_263_tabulate != None:
            fos.write('notebook_263:\n' + notebook_263_tabulate + '\n\n')
        if len(out_sentenses) > 0:
            fos.write('Out Sentenses:\n' + out_sentenses + '\n\n')
        fos.write('Cmd List:\n' + all_cmd_str)
        fos.write('\n</pre>')

    if os.path.exists(REPORT_PATH):
        if args.log_to_stdio:
            with open(REPORT_PATH, "r", encoding='utf-8') as fin:
                print(fin.read())

        # Send mail
        if not args.no_mail:
            send_mail(REPORT_PATH)



if __name__ == "__main__":
    main()
