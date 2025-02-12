#!/usr/bin/env python3

import argparse
import datetime as dt
import logging as log
import logging.config
import os
import re
import subprocess
import sys
import time

from glob import glob
from pathlib import Path

# check openvino version
try:
    from openvino.runtime import get_version
except:
    logging.warning('could not load openvino.runtime package.')
    def get_version():
        return 'none'

# import class/method from local
from common_utils import *
from device_info import python_packages
from profiling import HWResourceTracker
from report import *
from test_cases.test_template import *



################################################
# Utils
################################################
def remove_cache(args):
    if exists_path(args.cache_dir):
        for file in glob(convert_path(f'{args.cache_dir}/*.cl_cache')):
            os.remove(file)
        for file in glob(convert_path(f'{args.cache_dir}/*.blob')):
            os.remove(file)
        time.sleep(1)



################################################
# Main
################################################

class CmdHelper():
    def __init__(self, cmd_item:dict):
        self.cmd_item = cmd_item
        self.test_config = self.cmd_item.get(CmdItemKey.test_config, {})

    def __enter__(self):
        if self.test_config.get(CmdItemKey.TestConfigKey.mem_check, False):
            self.tracker = HWResourceTracker()
            self.tracker.start()

        self.test_start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.test_config.get(CmdItemKey.TestConfigKey.mem_check, False):
            cpu_usage_percent, mem_usage_size, mem_usage_percent = self.tracker.stop()
            self.cmd_item[CmdItemKey.peak_cpu_usage_percent] = cpu_usage_percent
            self.cmd_item[CmdItemKey.peak_mem_usage_percent] = mem_usage_percent
            self.cmd_item[CmdItemKey.peak_mem_usage_size] = sizeof_fmt(mem_usage_size)

        self.cmd_item[CmdItemKey.process_time] = time.time() - self.test_start_time

def run_daily(args):
    cfg = GlobalConfig()
    test_class_list = get_test_list()
    result_root = {}
    for test_class in test_class_list:
        cmd_dict = test_class.get_command_spec(args)
        if args.test:
            for key_tuple, cmd_list in cmd_dict.items():
                cmd_dict[key_tuple] = cmd_list[:1]
                break
        result_root.update(cmd_dict)

    #
    # Run all tests
    #
    for key_tuple, cmd_item_list in result_root.items():
        log.info(f'Run {key_tuple}...')

        # remove all cache before run each tests
        remove_cache(args)

        for cmd_item in cmd_item_list:
            with CmdHelper(cmd_item) as helper:
                log.info(f'cmd: {cmd_item[CmdItemKey.cmd]}')
                output, return_code = call_cmd(args, cmd_item[CmdItemKey.cmd])

                cmd_item[CmdItemKey.raw_log] = output
                cmd_item[CmdItemKey.return_code] = return_code
                cmd_item[CmdItemKey.data_list] = key_tuple[2].parse_output(args, output)

        log.info(f'Run {key_tuple}... Done\n')

    save_result_file(convert_path(f'{args.output_dir}/{cfg.RESULT_PICKLE_FILENAME}'), result_root)
    return result_root

def set_global_config():
    cfg = GlobalConfig()
    cfg.NOW = dt.datetime.now().strftime("%Y%m%d_%H%M")
    cfg.PWD = Path(__file__).parent.parent
    cfg.OV_VERSION = get_version()

    cfg.BIN_DIR = convert_path(f'{cfg.PWD}/bin')

    pre_daily_filename = f'daily.{cfg.NOW}.{cfg.OV_VERSION.replace("/", "_")}'
    cfg.RESULT_PICKLE_FILENAME = f'{pre_daily_filename}.pickle'
    cfg.REPORT_FILENAME = f'{pre_daily_filename}.report'
    cfg.RAW_FILENAME = f'{pre_daily_filename}.raw'
    cfg.PIP_FREEZE_FILENAME = f'{pre_daily_filename}.requirements.txt'
    cfg.RESULT_SD3_DYNAMIC_FILENAME = f'{pre_daily_filename}.sd.dynamic.png'
    cfg.RESULT_SD3_STATIC_FILENAME = f'{pre_daily_filename}.sd.static.png'
    cfg.BACKUP_FILENAME_LIST = [cfg.RAW_FILENAME, cfg.RESULT_SD3_DYNAMIC_FILENAME, cfg.RESULT_SD3_STATIC_FILENAME, cfg.RESULT_PICKLE_FILENAME, cfg.PIP_FREEZE_FILENAME]
    cfg.BACKUP_SERVER = 'http://dg2raptorlake.ikor.intel.com'
    cfg.MODEL_DATE = ''
    cfg.out_token_length = 256
    cfg.benchmark_iter_num = 3

def update_global_config(args):
    cfg = GlobalConfig()
    if args.test:
        cfg.out_token_length = 32
        cfg.benchmark_iter_num = 1
    cfg.test_filter = args.test_filter
    cfg.MODEL_DATE = args.model_cache

def main_setting(args):
    cfg = GlobalConfig()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory
    os.chdir(cfg.PWD)

    if not is_windows():
        # WA: this is needed to run an executable without './'.
        os.environ["PATH"] = f'{cfg.PWD}:{os.environ["PATH"]}'

    # https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
    logConfig = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s.%(msecs)03d[%(levelname).1s] %(message)s',
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'default': {
                'class' : 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'rawfile': {
                'class' : 'logging.FileHandler',
                'formatter': 'standard',
                'filename': convert_path(f'{args.output_dir}/{cfg.RAW_FILENAME}'),
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'rawfile'],
                'level'   : 'INFO'
            },
        }
    }
    log.config.dictConfig(logConfig)


def main():
    set_global_config()
    cfg = GlobalConfig()

    parser = argparse.ArgumentParser(description="Run daily check for LLM" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--device', help='target device', type=Path, default='GPU')
    parser.add_argument('--description', help='add description for report', type=str, default='LLM')
    parser.add_argument('--mail', help='sending mail recipient list. Mail recipients can be separated by comma.', type=str, default='')
    parser.add_argument('--this_report', help='target report to compare performance', type=Path, default=None)
    parser.add_argument('--ref_report', help='reference report to compare performance', type=Path, default=None)

    parser.add_argument('-cd', '--cache_dir', help='cache directory', type=Path, default=convert_path(f'{cfg.PWD}/llm-cache'))
    parser.add_argument('-m', '--model_dir', help='root directory for models', type=Path, default=convert_path(f'c:/dev/models'))
    parser.add_argument('-o', '--output_dir', help='output directory to store log files', type=Path, default=convert_path(f'{cfg.PWD}/output'))

    # config for test
    parser.add_argument('--model_cache', help='model cache name. It can be found under --model_dir.', type=str, default='WW03_llm-optimum_2025.0.0-17891')
    parser.add_argument('--genai', help='[deprecated] enable genai option for llm benchmark', action='store_true')
    parser.add_argument('--optimum', help='enable optimum option for llm benchmark', action='store_true')
    parser.add_argument('--continuous_batch', help='enable continuous batch pipeline for llm benchmark', action='store_true')
    parser.add_argument('--prompt_permutation', help='enable prompt_permutation for llm benchmark', action='store_true')
    parser.add_argument('--test', help='run tests with short config', action='store_true')
    parser.add_argument('--timeout', help='set timeout [unit: seconds].', type=int, default=1800)
    parser.add_argument('--test_filter', help=f'test class name (delimiter: comma): {[test_class.__name__ for test_class in get_test_list()]}. empty string or all will run all tests', type=str, default='')

    args = parser.parse_args()

    main_setting(args)
    update_global_config(args)

    if args.test and args.this_report:
        result_root = load_result_file(replace_ext(args.this_report, "pickle"))
        report_str = generate_report_str(args, result_root, 0)
        with open(args.this_report, 'w', encoding='utf-8') as fos:
            fos.write(report_str)

        backup_list = glob(replace_ext(args.this_report, "*"))
        backup_files(args, backup_list)
        return 0

    if args.this_report and args.ref_report:
        result_root_this = load_result_file(replace_ext(args.this_report, "pickle"))
        result_root_ref  = load_result_file(replace_ext(args.ref_report, "pickle"))
        compare_result_item_map(sys.stdout, print_compared_text, result_root_this, result_root_ref)
    else:
        # Run test
        total_start_time = time.time()
        result_root = run_daily(args)
        PROCESS_TIME = time.time() - total_start_time

        # create a report file.
        REPORT_PATH = convert_path(f'{args.output_dir}/{cfg.REPORT_FILENAME}')
        with open(REPORT_PATH, 'w', encoding='utf-8') as fos:
            fos.write(generate_report_str(args, result_root, PROCESS_TIME))

        if args.test:
            log.info(f'Report: {REPORT_PATH}')
        else:
            # create a pip freeze file.
            PIP_FREEZE_PATH = convert_path(f'{args.output_dir}/{cfg.PIP_FREEZE_FILENAME}')
            with open(PIP_FREEZE_PATH, 'w', encoding='utf-8') as fos:
                fos.write(f'{python_packages()}')

            # backup report & results
            backup_list = [ convert_path(f'{args.output_dir}/{filename}') for filename in cfg.BACKUP_FILENAME_LIST ]
            backup_files(args, [REPORT_PATH] + backup_list)

            # Send mail
            if args.mail:
                suffix_title = generate_mail_title_suffix(result_root)
                send_mail(REPORT_PATH, args.mail, args.description, suffix_title)



if __name__ == "__main__":
    main()
