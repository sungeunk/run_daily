#!/usr/bin/env python3

import hashlib
from io import StringIO
import logging as log
import os
import platform
import re
import subprocess
import time


################################################
# Global variable
################################################

class GlobalConfig:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state

    def __str__(self):
        ret_str = ''
        for key, value in sorted(self.__dict__.items()):
            ret_str += f'{key}: {value}\n'
        return ret_str


################################################
# function
################################################
def backup_files(args, files):
    try:
        MAIL_RELAY_SERVER = os.environ["MAIL_RELAY_SERVER"]
    except Exception as e:
        log.error(f'Exception: {str(e)}')
        return

    REMOTE_PATH=f'{MAIL_RELAY_SERVER}:/var/www/html/daily/{platform.node()}/'
    if is_windows():
        log.info(f'backup files to {REMOTE_PATH}')
        for file in files:
            if exists_path(file):
                log.info(f'  ok: {file}')
                call_cmd(args=args, cmd=f'scp.exe {file} {REMOTE_PATH}', shell=True, verbose=False)
            else:
                log.error(f'  failed: could not find {file}')

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
        log.error(f'out_log: {out_log}')
    else:
        if verbose:
            log.info(f'returncode: {returncode}')
            log.info(f'out_log: {out_log}')

    return out_log, returncode

def compare_class_name(klass, target):
    for class_name in [klass.__name__, klass.__name__[4:]]:
        if target.lower() == class_name.lower():
            return True
    return False

#
# WA: At subprocess.popen(cmd, ...), the cmd should be string on ubuntu or be string array on windows.
#
def convert_cmd_for_popen(cmd: str) -> str:
    # return cmd.split() # if is_windows() else cmd
    return cmd if is_windows() else cmd.split()

def convert_path(path):
    if is_windows():
        return path.replace('/', '\\')
    else:
        return path.replace('\\', '/')

def exists_path(path) -> bool:
    try:
        return os.path.exists(path)
    except:
        return False

def get_file_info(path) -> tuple[int, str]:
    filesize = os.path.getsize(path)

    with open(path, 'rb') as fis:
        hash = hashlib.md5(fis.read()).hexdigest()

    return filesize, hash

def is_float(value):
    try:
        return isinstance(float(value), float)
    except Exception as e:
        return False

def is_str(value):
    try:
        return isinstance(str(value), str)
    except Exception as e:
        return False

def is_windows() -> bool:
    return platform.system() == 'Windows'

def replace_ext(filepath, new_ext):
    filepath_str = str(filepath)
    index = filepath_str.rfind('.')
    return filepath_str[0:index] + '.' + new_ext

def sizeof_fmt(num):
    if not is_float(num):
        return num
    for unit in ("", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    raise Exception(f'Out of bound!!! size({num})')

def sizestr_to_num(str):
    if not is_str(str):
        return float(str)
    value_dict = {"KB":1024, "MB":1024*1024, "GB":1024*1024*1024, "TB":1024*1024*1024*1024}
    match_obj = re.search(r'([\d.]+) ([\w]+)', str)
    if match_obj:
        values = match_obj.groups()
        return float(values[0]) * value_dict.get(values[1], 1)
    return float(str)

def send_mail(report_path, recipients, title, suffix_title=''):
    MAIL_TITLE = f'[{platform.node()}/{GlobalConfig().NOW}] {title} {suffix_title}'
    MAIL_TO = recipients

    if is_windows():
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
