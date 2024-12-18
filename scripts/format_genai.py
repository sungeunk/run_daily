#!/usr/bin/env python3

import argparse
import pandas as pd
import re
import statistics
import sys

from enum import Enum
from pathlib import Path
from tabulate import tabulate

def parsing_log(handle, args):
    ret = []
    model_name = ''
    ov_version = ''
    temp_item = []
    ts_list = []
    ts_all = []

    for line in handle:
        if args.verbose:
            print(f'{line}', end =' ')

        if not ov_version:
            match_obj = re.search(r'openvino runtime version: ([a-zA-Z0-9-_./]+)', line)
            if match_obj:
                ov_version = match_obj.groups()[0]
                continue

        # print(f'{line}', end =" ")
        match_obj = re.search(r'model_type: ([a-zA-Z0-9-_.]+)', line)
        if match_obj:
            model_name = match_obj.groups()[0]
            continue

        match_obj = re.search(r'SyncInferRequest::infer end:  begin - end = (\d+) us', line)
        if match_obj:
            ts_list.append(float(match_obj.groups()[0])/1000)    # convert from us to ms
            continue

        # [ INFO ] [warm-up][P1] Input token size: 1024, Output size: 192, Infer count: 192, Tokenization Time: 0.51ms, Detokenization Time: 0.83ms, Generation Time: 4.46s, Latency: 23.23 ms/token
        # [ INFO ] [warm-up][P1] First token latency: 796.68 ms/token, other tokens latency: 18.97 ms/token, len of tokens: 192 * 1

        match_obj = re.search(r'\[([\S]+)\]\[P(\d+)\] Input token size: (\d+), Output size: (\d+), Infer count: (\d+), Tokenization Time: (\d+.\d+)ms, Detokenization Time: (\d+.\d+)ms, Generation Time: (\d+.\d+)s, Latency: (\d+.\d+) ms\/token', line)
        if match_obj:
            # temp_item
            # 0: iteration
            # 1: prompt index
            # 2: input token size
            # 3: output size
            # 4: infer count
            # 5: tokenization time
            # 6: detokenization time
            # 7: generation time
            # 8: latency per token time
            temp_item = match_obj.groups()
            continue

        #  [ INFO ] [1][P0] First token latency: 1048.52 ms/token, other tokens latency: 117.78 ms/token, len of tokens: 128 * 1
        # "[1] First token latency: 195.68 ms/token, other tokens latency: 120.94 ms/token, len of tokens: 128 * 1"
        match_obj = re.search(r'First token latency: (\d+.\d+) ms\/token, other tokens latency: ([\d.NA]+)', line)
        # match_obj = re.search(r'First token latency: (\d+.\d+) ms\/token, ', line)
        if match_obj:
            values = match_obj.groups()
            try:
                other_latency = float(values[1])
            except:
                other_latency = values[1]
            ret.append([model_name, temp_item[0], temp_item[2], temp_item[3], temp_item[4], float(values[0]), other_latency, temp_item[5], temp_item[6], float(temp_item[7])*1000, temp_item[8]])
            ts_all.append(ts_list)
            ts_list = []
            continue

    headers = ['model', 'iteration', 'token size', 'out size', 'inf num', '1st inf(ms)', '2nd inf(ms)', 'token(ms)', 'detoken(ms)', 'generation(ms)', 'latency(ms)/token']
    table = tabulate(ret, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')
    print(f'OV Version: {ov_version}')
    print(f'{table}')
    df = pd.DataFrame(ts_all).transpose()
    table1 = tabulate(df, tablefmt="github", floatfmt='.2f', stralign='right', numalign='right')
    print(f'\nraw inf time (xy transposed data)\n{table1}')


# Device Timeline for sdpa_opt_single_token_1941726075318499197_0_0__sa (enqueue 16373) = 1578431119375 ns (queued), 1578431119427 ns (submit), 1578431119479 ns (start), 1578431210624 ns (end)
class ParsedData:
    def __init__(self, values):
        if len(values) == 5:
            self.kernel = str(values[0])
            self.queued_ts = int(values[1])
            self.submit_ts = int(values[2])
            self.start_ts = int(values[3])
            self.end_ts = int(values[4])
        elif len(values) == 2:
            self.kernel = str(values[0])
            self.start_ts = 0
            self.end_ts = int(values[1])

    def get_execution_time(self) -> int:
        return self.end_ts - self.start_ts

    def get_ts(self):
        try:
            return self.queued_ts, self.end_ts
        except:
            return self.end_ts, self.end_ts

def sizeof_fmt(num):
    for unit in ("", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    raise Exception(f'Out of bound!!! size({num})')

def timeof_fmt(time_ns):
    for unit in ("ns", "us", "ms", "s"):
        if abs(time_ns) < 1000.0:
            return f"{time_ns:4.1f} {unit}"
        if unit == "s":
            return f"{time_ns:4.1f} {unit}"
        time_ns /= 1024.0
    raise Exception(f'Out of bound!!! time({time_ns})')


#                                             Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
#                activation_ref_1708410597595680420_0_0__sa,     16,        109266,    0.00%,          6829,          4583,         13958
#                activation_ref_5526270038208258811_0_0__sa,     16,        129265,    0.00%,          8079,          4479,         21562
#         beam_table_update_ref_7445389982782678696_0_0__sa,    512,      16727907,    0.06%,         32671,          2187,        144895
#                                     clEnqueueMemFillINTEL,    468,     148809298,    0.51%,        317968,          1770,       7596562
#                                      clEnqueueMemcpyINTEL,    537,      89615821,    0.31%,        166882,          1250,      17645729
#  concatenation_gpu_simple_ref_2504481582612299751_0_0__sa,     16,        105722,    0.00%,          6607,          3229,         18645
def cliloader_table(parsed_iter_list):
    temp_map_1st = {}
    temp_map_2nd = {}
    total_time_1st = 0
    total_time_2nd = 0

    first_ts = 0
    last_ts = 0

    for data_list in parsed_iter_list[:1]:
        begin_ts,a = data_list[0].get_ts()
        b, end_ts = data_list[-1].get_ts()
        print(f'1st: begin: {begin_ts} us ~ end: {end_ts} us / duration: {(end_ts - begin_ts)/1000000:.2f} ms')

        for item in data_list:
            execution_time = item.get_execution_time()
            total_time_1st += execution_time
            if item.kernel in temp_map_1st:
                temp_map_1st[item.kernel].append(execution_time)
            else:
                temp_map_1st[item.kernel] = [execution_time]

    for data_list in parsed_iter_list[1:]:
        begin_ts,a = data_list[0].get_ts()
        b, end_ts = data_list[-1].get_ts()
        print(f'2nd: begin: {begin_ts} us ~ end: {end_ts} us / duration: {(end_ts - begin_ts)/1000000:.2f} ms')
        for item in data_list:
            execution_time = item.get_execution_time()
            total_time_2nd += execution_time
            if item.kernel in temp_map_2nd:
                temp_map_2nd[item.kernel].append(execution_time)
            else:
                temp_map_2nd[item.kernel] = [execution_time]

    table_data_1st = []
    table_data_2nd = []
    for key in sorted(temp_map_1st.keys()):
        data_list = temp_map_1st[key]
        # print(f'begin: {data_list[0].queued_ts} us ~ end: {data_list[-1].end_ts} us / duration: {(data_list[-1].end_ts - data_list[0].queued_ts)/1000} ms')
        calls = len(data_list)
        average_t = statistics.mean(data_list)
        min_t = min(data_list)
        max_t = max(data_list)
        time_portion = f'{sum(data_list) / total_time_1st * 100:.2f}'
        table_data_1st.append([key, calls, sum(data_list), str(time_portion), int(average_t), min_t, max_t])
    table_data_1st.append(['total', '-', total_time_1st, '-', '-', '-', '-'])

    for key in sorted(temp_map_2nd.keys()):
        data_list = temp_map_2nd[key]
        # print(f'begin: {data_list[0].queued_ts} us ~ end: {data_list[-1].end_ts} us / duration: {(data_list[-1].end_ts - data_list[0].queued_ts)/1000} ms')
        calls = len(data_list)
        average_t = statistics.mean(data_list)
        min_t = min(data_list)
        max_t = max(data_list)
        time_portion = f'{sum(data_list) / total_time_2nd * 100:.2f}'
        table_data_2nd.append([key, calls, sum(data_list), str(time_portion), int(average_t), min_t, max_t])
    table_data_2nd.append(['total', '-', total_time_2nd, '-', '-', '-', '-'])

    headers = ['Function Name', 'Calls', 'Time (ns)', 'Time (%)', 'Average (ns)', 'Min (ns)', 'Max (ns)']
    table_1st = tabulate(table_data_1st, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')
    table_2nd = tabulate(table_data_2nd, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')
    return table_1st, table_2nd

def print_idle_time(parsed_list):
    min_start_ts = 0
    max_end_ts = 0
    for item in sorted(parsed_list, key = lambda item: item.start_ts):
        if min_start_ts == 0:
            min_start_ts = item.start_ts
        else:
            min_start_ts = min(item.start_ts, min_start_ts)

        if max_end_ts == 0:
            max_end_ts = item.end_ts
        else:
            max_end_ts = max(item.end_ts, max_end_ts)
    print(f'total_diff: {timeof_fmt(max_end_ts - min_start_ts)}')

def parsing_cliloader_log(handle):
    model_name = ''
    ov_version = ''
    ov_parsed_data_list = []
    host_parsed_data_list = []
    host_parsed_data_iter_list = []
    device_parsed_data_list = []
    device_parsed_data_iter_list = []
    current_key = None
    genai_result_map = {}
    ov_time_map = {}  # key: (token_size, process_index) / content: ParsedData
    host_time_map = {}  # key: (token_size, process_index) / content: ParsedData
    device_time_map = {}  # key: (token_size, process_index) / content: ParsedData
    start_to_parse_data = False
    last_kernel_name = ''

    # From pretrained time
    # [ 19357 ms ] [ INFO ] [warm-up][P0] Input token size: 33, Output size: 2, Infer count: 2, Tokenization Time: 0.90ms, Detokenization Time: 0.31ms, Generation Time: 2.14s, Latency: 1070.43 ms/token
    # [ 27654 ms ] [ INFO ] [warm-up][P1] Input token size: 1025, Output size: 2, Infer count: 2, Tokenization Time: 4.41ms, Detokenization Time: 0.53ms, Generation Time: 8.29s, Latency: 4145.27 ms/token
    # [ 28616 ms ] [ INFO ] [1][P0] Input token size: 33, Output size: 2, Infer count: 2, Tokenization Time: 0.26ms, Detokenization Time: 0.12ms, Generation Time: 0.96s, Latency: 480.01 ms/token
    # [ 35443 ms ] [ INFO ] [1][P1] Input token size: 1025, Output size: 2, Infer count: 2, Tokenization Time: 1.85ms, Detokenization Time: 0.55ms, Generation Time: 6.82s, Latency: 3411.95 ms/token
    # [ 35920 ms ] [ INFO ] [2][P0] Input token size: 33, Output size: 2, Infer count: 2, Tokenization Time: 0.27ms, Detokenization Time: 0.12ms, Generation Time: 0.48s, Latency: 238.17 ms/token

    # iter / prompt / 1st|2nd

    for line in handle:
        match_obj = re.search(r'model_type: ([a-zA-Z0-9-_.]+)', line)
        if match_obj:
            model_name = match_obj.groups()[0]
            continue

        match_obj = re.search(r'openvino runtime version: ([0-9a-z.\-\/\_]+)', line)
        if match_obj:
            ov_version = match_obj.groups()[0]
            continue

        match_obj = re.search(r'From pretrained time', line)
        if match_obj:
            start_to_parse_data = True
            continue

        if not start_to_parse_data:
            continue

        match_obj = re.search(r'\[([a-z0-9-]+)\]\[P([0-9]+)\] Input token size: ([0-9]+),', line)
        if match_obj != None:
            values = match_obj.groups()
            current_key = (values[0], values[2])
            ov_time_map[current_key] = ov_parsed_data_list
            host_time_map[current_key] = host_parsed_data_iter_list
            device_time_map[current_key] = device_parsed_data_iter_list
            ov_parsed_data_list = []
            host_parsed_data_iter_list = []
            device_parsed_data_iter_list = []
            continue

        # [network::execute] execute(402447) impl(402401)
        match_obj = re.search(r' execute\(([0-9]+)\) impl\(([0-9]+)\)', line)
        if match_obj != None:
            values = match_obj.groups()
            ov_parsed_data_list.append([values[0], values[1]])
            continue

        # Host Time for call 107: clReleaseEvent = 200
        match_obj = re.search(r'Host Time for call [0-9]+: ([a-zA-Z0-9_]+) = ([0-9]+)', line)
        if match_obj != None:
            values = match_obj.groups()
            host_parsed_data_list.append(ParsedData(values))
            continue

        match_obj = re.search(r'Device Timeline for ([a-zA-Z0-9_]+) \([a-z 0-9]+\) = ([0-9]+) ns \(queued\), ([0-9]+) ns \(submit\), ([0-9]+) ns \(start\), ([0-9]+) ns \(end\)', line)
        if match_obj != None:
            values = match_obj.groups()
            # if len(device_parsed_data_list) == 0:
            #     print(f'first line: {line}')

            device_parsed_data_list.append(ParsedData(values))

            kernel_name = values[0]
            if last_kernel_name == 'gemm_kernel' and kernel_name == 'clEnqueueMemcpyINTEL':
                # print(f'KLAIN check point ================================================')
                # print(f'last line: {line}')
                host_parsed_data_iter_list.append(host_parsed_data_list)
                device_parsed_data_iter_list.append(device_parsed_data_list)
                host_parsed_data_list = []
                device_parsed_data_list = []
            last_kernel_name = kernel_name
            continue

        match_obj = re.search(f'First token latency: (\d+.\d+) ms\/token, other tokens latency: (\d+.\d+) ms\/token', line)
        if match_obj != None:
            values = match_obj.groups()
            genai_result_map[current_key] = f'1st: {values[0]} ms/token, other: {values[1]} ms/token'

    # print(f'{genai_result_map.keys()}')
    print(f'model: {model_name}')
    print(f'OpenVINO: {ov_version}')
    # for key in device_time_map.keys():
    #     print(f'key: {key}, {genai_result_map[key]}')
    # for key in ov_time_map.keys():
    #     print(f'key: {key}\nov_time')
    #     for items in ov_time_map[key]:
    #         print(f'\t exec: {timeof_fmt(int(items[0]) * 1000)}, exec_impl: {timeof_fmt(int(items[1]) * 1000)}')
    for key in host_time_map.keys():
        table_1st, table_2nd = cliloader_table(host_time_map[key])
        print(f'key: {key}')
        print(f'host_time(1st):\n{table_1st}\n')
        print(f'host_time(2nd):\n{table_2nd}\n')
    for key in device_time_map.keys():
        print(f'\n\nkey: {key}')
        table_1st, table_2nd = cliloader_table(device_time_map[key])
        print(f'key: {key}, {genai_result_map[key]}')
        print(f'device_time(1st):\n{table_1st}\n')
        print(f'device_time(2nd):\n{table_2nd}\n')


def parsing_check_enqueue_count(handle):
    current_enqueue_cnt = 0
    begin_enqueue_cnt = 0
    in_token_size = 0
    result = []
    for line in handle:
        match_obj = re.search(r'EnqueueCounter: ([0-9]+)', line)
        if match_obj:
            current_enqueue_cnt = int(match_obj.groups()[0])
            continue

        match_obj = re.search(r'\[ INFO \] From pretrained time:', line)
        if match_obj:
            begin_enqueue_cnt = current_enqueue_cnt
            continue

        match_obj = re.search(r'\[warm-up\]\[P[0-9]+\] Input token size: ([0-9]+), ', line)
        if match_obj:
            in_token_size = int(match_obj.groups()[0])
            result.append(['warm-up', in_token_size, begin_enqueue_cnt, current_enqueue_cnt])
            begin_enqueue_cnt = current_enqueue_cnt + 1
            continue

        match_obj = re.search(r'\[([0-9]+)\]\[P[0-9]+\] Input token size: ([0-9]+), ', line)
        if match_obj:
            iter_num = int(match_obj.groups()[0])
            in_token_size = int(match_obj.groups()[1])
            result.append([iter_num, in_token_size, begin_enqueue_cnt, current_enqueue_cnt])
            begin_enqueue_cnt = current_enqueue_cnt + 1
            continue

    headers = ['iter_num', 'token', 'enqueue begin', 'enqueue end']
    table = tabulate(result, tablefmt="github", headers=headers, stralign='right', numalign='right')
    print(f'{table}')

def main():
    parser = argparse.ArgumentParser(description="Run daily check for chatglm" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', help='input log path', type=Path, default=None)
    parser.add_argument('-m', '--mode', help='parsing mode: log, cliloader, enqueue_cnt', type=str, default='log')
    parser.add_argument('--verbose', '-v', help='print the input logs', action='store_true')
    args = parser.parse_args()

    if args.input:
        handle = open(args.input, 'r')
    else:
        handle = sys.stdin

    if args.mode == 'log':
        parsing_log(handle, args)
    elif args.mode == 'cliloader':
        parsing_cliloader_log(handle)
    elif args.mode == 'enqueue_cnt':
        parsing_check_enqueue_count(handle)

if __name__ == "__main__":
    main()

