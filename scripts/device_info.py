#!/usr/bin/env python3

import argparse
import netifaces
import platform
import pyopencl
import re
import subprocess

from tabulate import tabulate



################################################
# Global variable
################################################
IS_WINDOWS = platform.system() == 'Windows'

################################################
# Utils
################################################
def sizeof_fmt(num):
    for unit in ("", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    raise Exception(f'Out of bound!!! size({num})')

def add_unit_for_value(key, value):
    if key == pyopencl.device_info.VENDOR_ID:
        return f'{value:x}'
    elif key == pyopencl.device_info.MAX_CLOCK_FREQUENCY:
        return f'{value} MHz'
    elif isinstance(value, int) and value > 10000:
        return sizeof_fmt(value)
    elif key == pyopencl.device_info.GLOBAL_MEM_SIZE or key == pyopencl.device_info.MAX_MEM_ALLOC_SIZE:
        return sizeof_fmt(value)
    else:
        return value

def get_lines(cmd):
    lines = []
    cp = subprocess.run(cmd.split(), capture_output=True, encoding='utf-8')
    for line in cp.stdout.splitlines():
        line = line.rstrip()
        if line == '':
            continue

        lines.append(line)

    return lines if len(lines) > 0 else None

def get_ip_address(interface):
    try:
        addrs = netifaces.ifaddresses(interface)
        ip_info = {}
        if netifaces.AF_INET in addrs:  # Check for IPv4 address
            ip_info['IPv4'] = addrs[netifaces.AF_INET][0]['addr']
        if netifaces.AF_INET6 in addrs:  # Check for IPv6 address
            ip_info['IPv6'] = addrs[netifaces.AF_INET6][0]['addr']
        return ip_info
    except ValueError:
        print("Interface not found or doesn't have an IP address.")
        return {}

def get_external_ip_address():
    INTERNAL_IP = ['192', '127']
    for interface in netifaces.interfaces():
        ip_address = get_ip_address(interface)
        if 'IPv4' in ip_address:
            for exclude_ip in INTERNAL_IP:
                if ip_address["IPv4"].split('.')[0] == exclude_ip:
                    continue
                return ip_address["IPv4"]
    return None

def get_system_info(args):
    info_table = []
    if IS_WINDOWS:
        for line in get_lines('wmic os get Name,CSName /format:list'):
            match_obj = re.search(r'([a-zA-Z]+)=([0-9a-zA-Z() \-]+)', line)
            if match_obj != None:
                info_line = [f'{match_obj.group(1)}', f'{match_obj.group(2)}']
                info_table.append(info_line)
    else:
        for line in get_lines('lsb_release -d'):
            match_obj = re.search(r'[a-zA-Z ]+[: 	]+([0-9a-zA-Z. ]+)', line)
            if match_obj != None:
                info_line = ['OS', match_obj.group(1)]
                info_table.append(info_line)

        for line in get_lines('uname -r'):
            info_line = ['Kernel Version', line]
            info_table.append(info_line)

    ip_address = get_external_ip_address()
    if ip_address != None:
        info_table.append(['IP', ip_address])

    if len(info_table) > 0:
        info_tabulate = tabulate(info_table, headers=['property', 'value'],
                                tablefmt="github", floatfmt='.3f', stralign='left', numalign='right')
        print(f'System Info:')
        print(f'{info_tabulate}')

def get_cpu_info():
    info_table = []
    if IS_WINDOWS:
        for line in get_lines('wmic cpu get name,MaxClockSpeed,NumberOfEnabledCore,NumberOfLogicalProcessors,SystemName /format:list'):
            match_obj = re.search(r'([a-zA-Z]+)=([0-9a-zA-Z() \-]+)', line)
            if match_obj != None:
                info_line = [f'{match_obj.group(1)}', f'{match_obj.group(2)}']
                info_table.append(info_line)
    else:
        for line in get_lines('lscpu'):
            match_obj = re.search(r'([a-zA-Z() ]+):([0-9a-zA-Z() ]+)', line)
            if match_obj != None and match_obj.group(1) in ['Model name', 'CPU max MHz', 'Core(s) per socket', 'CPU(s)']:
                info_line = [f'{match_obj.group(1)}', f'{match_obj.group(2)}']
                info_table.append(info_line)

    if len(info_table) > 0:
        info_tabulate = tabulate(info_table, headers=['property', 'value'],
                                tablefmt="github", floatfmt='.3f', stralign='left', numalign='right')
        print(f'CPU Info:')
        print(f'{info_tabulate}')

def get_gpu_info(args):
    QUERY_REQUIRED_ITEMS = [
        pyopencl.device_info.DRIVER_VERSION,
        pyopencl.device_info.MAX_COMPUTE_UNITS,
        pyopencl.device_info.MAX_CLOCK_FREQUENCY,
        pyopencl.device_info.GLOBAL_MEM_SIZE,
        pyopencl.device_info.LOCAL_MEM_SIZE,
        pyopencl.device_info.MAX_MEM_ALLOC_SIZE,
    ]
    QUERY_ADDITIONAL_ITEMS = [
        pyopencl.device_info.GLOBAL_MEM_CACHE_SIZE,
        pyopencl.device_info.GLOBAL_MEM_CACHE_TYPE,
        pyopencl.device_info.GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE,
        pyopencl.device_info.MAX_PARAMETER_SIZE,
        pyopencl.device_info.LOCAL_MEM_TYPE,
        pyopencl.device_info.MAX_CONSTANT_ARGS,
        pyopencl.device_info.MAX_CONSTANT_BUFFER_SIZE,
        pyopencl.device_info.MAX_GLOBAL_VARIABLE_SIZE,
        pyopencl.device_info.MAX_NUM_SUB_GROUPS,
        pyopencl.device_info.MAX_WORK_GROUP_SIZE,
        pyopencl.device_info.MAX_WORK_ITEM_DIMENSIONS,
        pyopencl.device_info.MAX_WORK_ITEM_SIZES,
        pyopencl.device_info.MIN_DATA_TYPE_ALIGN_SIZE,
    ]

    dev_list = []

    # Get all GPU devices
    for plat in pyopencl.get_platforms():
        dev = plat.get_devices(pyopencl.device_type.GPU)
        if len(dev) > 0:
            dev_list += dev

    info_table = []
    info_line = ['PNPDeviceID']

    # Get PNPDeviceID (device id)
    dev_dic = {}
    if IS_WINDOWS:
        # PNPDeviceID (device id) from wmic
        dev_name = ''
        for line in get_lines('wmic PATH Win32_VideoController get name,PNPDeviceID /format:list'):
            match_obj = re.search(r'Name=([0-9a-zA-Z() ]+)', line)
            if match_obj != None:
                dev_name = match_obj.group(1)

            match_obj = re.search(r'DEV_([0-9a-zA-Z]+)', line)
            if match_obj != None:
                dev_dic[dev_name] = match_obj.group(1)
    else:
        # (venv) sungeunk@dg2raptorlake:~/repo/libraries.ai.videoanalyticssuite.gpu-tools$ clinfo -l
        # Platform #0: Intel(R) OpenCL Graphics
        # `-- Device #0: Intel(R) Arc(TM) A770 Graphics
        # Platform #1: Intel(R) OpenCL Graphics
        # `-- Device #0: Intel(R) UHD Graphics 770
        platform_id_list = []
        for line in get_lines('clinfo -l'):
            match_obj = re.search(r'Platform #([0-9])+:', line)
            if match_obj != None:
                platform_id_list.append(match_obj.group(1))

        for id in platform_id_list:
            device_id = ''
            device_name = ''
            for line in get_lines(f'clinfo -d {id}:0'):
                match_obj = re.search(r'8680([0-9a-zA-Z]+)-', line)
                if match_obj != None:
                    temp = match_obj.group(1)
                    device_id = f'{temp[2:4]}{temp[0:2]}'

                match_obj = re.search(r'Device Name[ ]+([0-9a-zA-Z() ]+)', line)
                if match_obj != None:
                    device_name = match_obj.group(1)
            dev_dic[device_name] = device_id

        for dev in dev_list:
            info_line.append(dev_dic[dev.get_info(pyopencl.device_info.NAME)])
        info_table.append(info_line)

    # Get QUERY_REQUIRED_ITEMS
    for item in QUERY_REQUIRED_ITEMS:
        item_name = pyopencl.device_info.to_string(item)
        info_line = [item_name]
        for dev in dev_list:
            info_line.append(add_unit_for_value(item, dev.get_info(item)))
        info_table.append(info_line)

    if args.get_all:
        for item in QUERY_ADDITIONAL_ITEMS:
            item_name = pyopencl.device_info.to_string(item)
            info_line = [item_name]
            for dev in dev_list:
                info_line.append(add_unit_for_value(item, dev.get_info(item)))
            info_table.append(info_line)

    # print info table
    if len(info_table) > 0:
        info_tabulate = tabulate(info_table, headers=[dev.get_info(pyopencl.device_info.NAME) for dev in dev_list],
                                tablefmt="github", floatfmt='.3f', stralign='left', numalign='right')
        print(f'GPU Info:')
        print(f'{info_tabulate}')

def python_packages() -> str:
    from pip._internal.operations import freeze
    ret = ''
    for pkg in freeze.freeze(): ret += f'{pkg}\n'
    return ret

def main():
    parser = argparse.ArgumentParser(description="query CPU/GPU info." , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--get_all', help='query all information', action='store_true')
    args = parser.parse_args()

    get_system_info(args)
    print('')
    get_cpu_info()
    print('')
    get_gpu_info(args)

if __name__ == "__main__":
    main()
