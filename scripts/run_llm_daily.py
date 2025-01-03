#!/usr/bin/env python3

import argparse
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
                log.warning('try to kill this process...')
                proc.kill()
                log.warning('try to kill this process...done.')
                out_log, error = proc.communicate()
                returncode = proc.returncode
                log.warning(f'returncode: {returncode}')
    except Exception as e:
        log.error(f'Exception: {str(e)}')

    if returncode != 0:
        log.error(f'returncode: {returncode}')
        log.error(f'out_log:\n{out_log}')
    else:
        if verbose:
            log.info(f'returncode: {returncode}')
            log.info(f'out_log:\n{out_log}')

    return out_log, returncode

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

def get_test_class(test_class_list, model_name):
    for test_class in test_class_list:
        if test_class.is_included(model_name):
            return test_class
    return None



################################################
# Main
################################################

def run_daily(args):
    test_class_list = get_test_list()
    result_root = {}
    for test_class in test_class_list:
        result_root.update(test_class.get_command_list(args))

    #
    # Run all tests
    #
    EnableProfiling = False
    for key_tuple, cmd_item_list in result_root.items():
        log.info(f'Run {key_tuple}...')

        # remove all cache before run each tests
        remove_cache(args)

        for cmd_item in cmd_item_list:
            test_start_time = time.time()
            log.info(f'\tcmd: {cmd_item[CmdItemKey.cmd]}')

            test_config = cmd_item.get(CmdItemKey.test_config, None)
            if test_config != None:
                EnableProfiling = test_config.get(CmdItemKey.TestConfigKey.mem_check, False)

            if EnableProfiling:
                tracker = HWResourceTracker()
                tracker.start()

            output, return_code = call_cmd(args, cmd_item[CmdItemKey.cmd])

            if EnableProfiling:
                cpu_usage_percent, mem_usage_size, mem_usage_percent = tracker.stop()
                cmd_item[CmdItemKey.peak_cpu_usage_percent] = cpu_usage_percent
                cmd_item[CmdItemKey.peak_mem_usage_percent] = mem_usage_percent
                cmd_item[CmdItemKey.peak_mem_usage_size] = sizeof_fmt(mem_usage_size)

            cmd_item[CmdItemKey.raw_log] = output
            cmd_item[CmdItemKey.return_code] = return_code
            cmd_item[CmdItemKey.data_list] = get_test_class(test_class_list, key_tuple[0]).parse_output(args, output)
            cmd_item[CmdItemKey.process_time] = time.time() - test_start_time

        log.info(f'Run {key_tuple}... Done\n')

    save_result_file(convert_path(f'{args.output_dir}/{GlobalConfig().RESULT_PICKLE_FILENAME}'), result_root)
    return result_root

def set_global_config():
    cfg = GlobalConfig()
    cfg.NOW = dt.datetime.now().strftime("%Y%m%d_%H%M")
    cfg.PWD = Path(__file__).parent.parent
    cfg.OV_VERSION = get_version()

    pre_daily_filename = f'daily.{cfg.NOW}.{cfg.OV_VERSION.replace("/", "_")}'
    cfg.RESULT_PICKLE_FILENAME = f'{pre_daily_filename}.pickle'
    cfg.REPORT_FILENAME = f'{pre_daily_filename}.report'
    cfg.RAW_FILENAME = f'{pre_daily_filename}.raw'
    cfg.PIP_FREEZE_FILENAME = f'{pre_daily_filename}.requirements.txt'
    cfg.RESULT_SD3_DYNAMIC_FILENAME = f'{pre_daily_filename}.sd.dynamic.png'
    cfg.RESULT_SD3_STATIC_FILENAME = f'{pre_daily_filename}.sd.static.png'
    cfg.BACKUP_FILENAME_LIST = [cfg.RAW_FILENAME, cfg.RESULT_SD3_DYNAMIC_FILENAME, cfg.RESULT_SD3_STATIC_FILENAME, cfg.RESULT_PICKLE_FILENAME]
    cfg.BACKUP_SERVER = 'http://dg2raptorlake.ikor.intel.com'
    cfg.MODEL_DATE = 'WW44_llm-optimum_2024.5.0-17246-44b86a860ec'
    cfg.out_token_length = 256
    cfg.benchmark_iter_num = 3

def update_global_config(args):
    cfg = GlobalConfig()
    if args.test:
        cfg.out_token_length = 32
        cfg.benchmark_iter_num = 1
    cfg.model_target = args.model_target

def main_setting(args):
    cfg = GlobalConfig()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory
    os.chdir(args.working_dir)

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
    parser.add_argument('--test', help='run tests with short config', action='store_true')
    parser.add_argument('--timeout', help='set timeout [unit: seconds].', type=int, default=300)
    parser.add_argument('--sleep_bf_test', help='set sleep time [unit: seconds].', type=int, default=1)

    parser.add_argument('-cd', '--cache_dir', help='cache directory', type=Path, default=os.path.join(*[cfg.PWD, 'llm-cache']))
    parser.add_argument('-m', '--model_dir', help='root directory for models', type=Path, default=os.path.join(*['c:\\', 'dev', 'models']))
    parser.add_argument('-o', '--output_dir', help='output directory to store log files', type=Path, default=os.path.join(*[cfg.PWD, 'output']))
    parser.add_argument('-w', '--working_dir', help='working directory', type=Path, default=cfg.PWD)
    parser.add_argument('--bin_dir', help='binary directory', type=Path, default=os.path.join(*[cfg.PWD, 'bin']))

    # config for test
    parser.add_argument('--model_target', help='test class name (delimiter: comma)', type=str, default='')
    parser.add_argument('--genai', help='enable genai option for llm benchmark', action='store_true')

    # deprecated
    parser.add_argument('--repeat', help='Do not use.', type=int, default=1)
    parser.add_argument('--benchmark_app', help='benchmark_app(cpp) path', type=Path, default=os.path.join(*[cfg.PWD, 'bin', 'benchmark_app', 'benchmark_app']))
    parser.add_argument('--convert_models', help='', action='store_true')
    args = parser.parse_args()

    main_setting(args)
    update_global_config(args)

    if args.test and args.this_report:
        result_root = load_result_file(replace_ext(args.this_report, "pickle"))
        log.info(generate_report_str(args, result_root, 0))
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
