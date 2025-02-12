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
def cliloader_table(parsed_iter_list_map, merge_2nd=True):
    # 'infer_latency': (unit: ns)
    # 'host_call_total_count': int
    # 'host_call_total_time': (unit: ns)
    # 'host_time_dict': dict
    # 'host_time_table': tabulate str
    # 'device_call_total_count': int
    # 'device_call_total_exec_time': (unit: ns)
    # 'device_time_dict': dict
    # 'device_time_table': tabulate str
    result = []

    # print(f'parsed_iter_list_map[host]: {parsed_iter_list_map["host"]}')
    
    for infer_index in range(len(parsed_iter_list_map['host'])):
        host_data_list = parsed_iter_list_map['host'][infer_index]
        device_data_list = parsed_iter_list_map['device'][infer_index]
        # begin_ts,a = device_data_list[0].get_ts()
        # b, end_ts = device_data_list[-1].get_ts()
        # infer_latency = (end_ts - begin_ts)   >>> need to replace the infer_latency from benchmark log
        # assert(len(host_data_list) == len(device_data_list))
        infer_count = len(device_data_list)

        for i in range(infer_count):
            device_begin_ts, a = device_data_list[i][0].get_ts()
            b, device_end_ts = device_data_list[i][-1].get_ts()
            infer_latency_device = device_end_ts - device_begin_ts

            host_time_dict = {}
            host_call_total_count = 0
            host_call_total_time = 0
            for item in host_data_list[i]:
                execution_time = item.get_execution_time()
                host_call_total_time += execution_time
                host_call_total_count += 1
                if item.kernel in host_time_dict:
                    host_time_dict[item.kernel].append(execution_time)
                else:
                    host_time_dict[item.kernel] = [execution_time]

            prev_end_ts = 0
            device_call_total_exec_delay_time = 0
            device_call_total_queue_delay_time = 0
            device_time_dict = {}
            device_call_total_count = 0
            device_call_total_exec_time = 0
            for item in device_data_list[i]:
                if prev_end_ts == 0:
                    delay_time = 0
                    queue_delay_time = 0
                else:
                    delay_time = item.start_ts - prev_end_ts
                    queue_delay_time = item.queued_ts - prev_end_ts if item.queued_ts > prev_end_ts else 0
                prev_end_ts = item.end_ts
                device_call_total_exec_delay_time += delay_time
                device_call_total_queue_delay_time += queue_delay_time
                execution_time = item.get_execution_time()
                device_call_total_exec_time += execution_time
                device_call_total_count += 1
                if item.kernel in device_time_dict:
                    device_time_dict[item.kernel].append(execution_time)
                else:
                    device_time_dict[item.kernel] = [execution_time]

            result.append({'infer_index': (infer_index, i),
                           'infer_latency_device': infer_latency_device,
                           'device_time_dict': device_time_dict,
                           'device_call_total_count': device_call_total_count,
                           'device_call_total_exec_time': device_call_total_exec_time,
                           'device_call_total_exec_delay_time': device_call_total_exec_delay_time,
                           'device_call_total_queue_delay_time': device_call_total_queue_delay_time,
                           'host_time_dict': host_time_dict,
                           'host_call_total_count': host_call_total_count,
                           'host_call_total_time': host_call_total_time})

    for infer_data in result:
        host_table_data = []
        host_time_dict = infer_data['host_time_dict']
        for key in sorted(host_time_dict.keys()):
            data_list = host_time_dict[key]
            host_table_data.append([key,
                               len(data_list),
                               f'{sum(data_list)/1000:.0f}',
                               f'{sum(data_list) / infer_data["host_call_total_time"] * 100:.2f}',
                               f'{int(statistics.mean(data_list)/1000)}',
                               f'{min(data_list)/1000:.0f}',
                               f'{max(data_list)/1000:.0f}'])

        host_table_data.append(['total', infer_data['host_call_total_count'], int(infer_data['host_call_total_time']/1000), '-', '-', '-', '-'])
        headers = ['Function Name', 'Calls', 'Time (us)', 'Time (%)', 'Average (us)', 'Min (us)', 'Max (us)']
        infer_data['host_time_table'] = tabulate(host_table_data, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')

        device_table_data = []
        device_time_dict = infer_data['device_time_dict']
        for key in sorted(device_time_dict.keys()):
            data_list = device_time_dict[key]
            device_table_data.append([key,
                               len(data_list),
                               f'{sum(data_list)/1000:.0f}',
                               f'{sum(data_list) / infer_data["device_call_total_exec_time"] * 100:.2f}',
                               f'{int(statistics.mean(data_list)/1000)}',
                               f'{min(data_list)/1000:.0f}',
                               f'{max(data_list)/1000:.0f}'])

        device_table_data.append(['total', infer_data['device_call_total_count'], int(infer_data['device_call_total_exec_time']/1000), '-', '-', '-', '-'])
        headers = ['Function Name', 'Calls', 'Time (us)', 'Time (%)', 'Average (us)', 'Min (us)', 'Max (us)']
        infer_data['device_time_table'] = tabulate(device_table_data, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')

    return result

def cliloader_device_tabulate(parsed_iter_list_map, merge_2nd=True):
    def __merge_time(target_dict, device_data_list):
        device_begin_ts, a = device_data_list[0].get_ts()
        b, device_end_ts = device_data_list[-1].get_ts()
        target_dict['infer_latency'] += (device_end_ts - device_begin_ts)

        for item in device_data_list:
            execution_time = item.get_execution_time()
            target_dict['device_call_total_count'] += 1
            target_dict['device_call_total_exec_time'] += execution_time
            if item.kernel in target_dict:
                target_dict[item.kernel].append(execution_time)
            else:
                target_dict[item.kernel] = [execution_time]

    def __generate_tabulate(device_time_dict):
        device_table_data = []
        for key in sorted(device_time_dict.keys()):
            if key in ['infer_latency', 'device_call_total_count', 'device_call_total_exec_time']:
                continue
            data_list = device_time_dict[key]
            device_table_data.append([key,
                               len(data_list),
                               f'{sum(data_list)/1000:.0f}',
                               f'{sum(data_list) / device_time_dict["device_call_total_exec_time"] * 100:.2f}',
                               f'{int(statistics.mean(data_list)/1000)}',
                               f'{min(data_list)/1000:.0f}',
                               f'{max(data_list)/1000:.0f}'])

        device_table_data.append(['total', device_time_dict['device_call_total_count'], int(device_time_dict['device_call_total_exec_time']/1000), '-', '-', '-', '-'])
        headers = ['Function Name', 'Calls', 'Time (us)', 'Time (%)', 'Average (us)', 'Min (us)', 'Max (us)']
        return tabulate(device_table_data, tablefmt="github", headers=headers, floatfmt='.2f', stralign='right', numalign='right')

    for infer_index in range(len(parsed_iter_list_map['device'])):
        device_data_list = parsed_iter_list_map['device'][infer_index]
        infer_count = len(device_data_list)

        device_time_1st = {'device_call_total_count':0, 'device_call_total_exec_time':0, 'infer_latency':0}
        device_time_2nd = {'device_call_total_count':0, 'device_call_total_exec_time':0, 'infer_latency':0}
        for i in range(infer_count):
            __merge_time(device_time_1st if i == 0 else device_time_2nd, device_data_list[i])
        tab_1st = __generate_tabulate(device_time_1st)
        tab_2nd = __generate_tabulate(device_time_2nd)

        title_str = f'warm-up' if infer_index == 0 else f'{infer_index - 1}'
        print(f'[{title_str}]\n{tab_1st}\n\n')
        print(f'[{title_str}]\n{tab_2nd}\n\n')

def cliloader_raw_for_excel(filename, parsed_iter_list_map):
    with open(filename, 'wt') as fos:
        fos.write(f'infer_index,infer_num,kernel,queued_ts(ns),submit_ts(ns),start_ts(ns),end_ts(ns),execute(ms),exec_delay(ms)\n')

        for infer_index in range(len(parsed_iter_list_map['host'])):
            device_data_list = parsed_iter_list_map['device'][infer_index]

            for i in range(len(device_data_list)):
                prev_end_ts = device_data_list[i][0].end_ts
                for item in device_data_list[i]:
                    delay = item.start_ts - prev_end_ts if prev_end_ts < item.start_ts else 0
                    prev_end_ts = item.end_ts
                    fos.write(f'{infer_index},{i},{item.kernel},{item.queued_ts},{item.submit_ts},{item.start_ts},{item.end_ts},{(item.end_ts - item.start_ts)/1000000:03f},{delay/1000000:03f}\n')

def parsing_cliloader_log(handle):
    class ParsingHeader:
        def __init__(self, parsed_data):
            self.parsed_data = parsed_data

        def parse(self, line):
            match_obj = re.search(r'model_type: ([a-zA-Z0-9-_.]+)', line)
            if match_obj:
                self.parsed_data['model_name'] = match_obj.groups()[0]
                return False

            match_obj = re.search(r'openvino runtime version: ([0-9a-z.\-\/\_]+)', line)
            if match_obj:
                self.parsed_data['ov_version'] = match_obj.groups()[0]
                return False

            match_obj = re.search(r'Pipeline initialization time: ([\d\.]+)s', line)
            if match_obj:
                self.parsed_data['init_time'] = match_obj.groups()[0]
                return True

            return False

    class ParsingTime:
        def __init__(self, parsed_data):
            self.parsed_data = parsed_data
            self.check_wait = False
            self.host_parsed_data_list = []
            self.host_parsed_data_list_all = []
            self.device_parsed_data_list = []
            self.device_parsed_data_list_all = []

        def parse(self, line):
            # Host Time for call 1: clEnqueueNDRangeKernel( concatenation_gpu_simple_ref_7709004105888173772_1_0__sa ) = 7900
            match_obj = re.search(r'Host Time for call \d+: ([\w\(\)\_ ]+) = (\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                self.host_parsed_data_list.append(ParsedData(values))

                # need to check last the end of each inference call.
                if values[0] == 'clWaitForEvents':
                    self.check_wait = True

                return False

            # Device Timeline for clEnqueueMemcpyINTEL (enqueue 516) = 3060400582120 ns (queued), 3060401375520 ns (submit), 3060402504375 ns (start), 3060402508177 ns (end)
            match_obj = re.search(r'Device Timeline for ([\w]+) \(enqueue \d+\) = (\d+) ns \(queued\), (\d+) ns \(submit\), (\d+) ns \(start\), (\d+) ns \(end\)', line)
            if match_obj != None:
                values = match_obj.groups()
                self.device_parsed_data_list.append(ParsedData(values))

                if self.check_wait:
                    self.check_wait = False
                    if values[0] == "clEnqueueMemcpyINTEL":
                        # end of inferencing
                        self.host_parsed_data_list_all.append(self.host_parsed_data_list)
                        self.host_parsed_data_list = []
                        self.device_parsed_data_list_all.append(self.device_parsed_data_list)
                        self.device_parsed_data_list = []

                return False

            match_obj = re.search(r'\[([a-z0-9-]+)\]\[P([0-9]+)\] Input token size: ([0-9]+), Output size: \d+, Infer count: (\d+)', line)
            if match_obj != None:
                values = match_obj.groups()
                in_token = int(values[2])
                infer_count = int(values[3])
                key = f'token_{in_token}'
                if not key in self.parsed_data:
                    self.parsed_data[key] = {}
                    self.parsed_data[key]['host'] = []
                    self.parsed_data[key]['device'] = []
                self.parsed_data[key]['host'].append(self.host_parsed_data_list_all)
                self.parsed_data[key]['device'].append(self.device_parsed_data_list_all)
                # print(f'device_parsed_data_list_all: len: {len(self.device_parsed_data_list_all)}')
                # print(f'infer_count: {infer_count}')
                self.host_parsed_data_list_all = []
                self.device_parsed_data_list_all = []
                return False

            return False

    parsed_data = {}
    parser_iter = iter([ ParsingHeader(parsed_data), ParsingTime(parsed_data) ])
    parser = next(parser_iter)
    for line in handle:
        ret = parser.parse(line)
        if ret:
            parser = next(parser_iter)

    print(f'model: {parsed_data["model_name"]}')
    print(f'OpenVINO: {parsed_data["ov_version"]}')

    print(f'{parsed_data.keys()}')
    infer_result_keys = [ key for key in parsed_data.keys() if key.startswith('token')]
    for key in infer_result_keys:
        print(f'infer: {key}')
        cliloader_device_tabulate(parsed_data[key])
        # infer_result_list = cliloader_table(parsed_data[key])
        # cliloader_raw_for_excel(f'cliloader_{key}.csv', parsed_data[key])

        # total_raw_data_list = []
        # for infer_result in infer_result_list:
        #     total_raw_data_list.append([infer_result["infer_index"],
        #                                 infer_result["infer_latency_device"]/1000000,
        #                                 infer_result["device_call_total_exec_time"]/1000000,
        #                                 infer_result["device_call_total_exec_delay_time"]/1000000,
        #                                 infer_result["device_call_total_queue_delay_time"]/1000000,
        #                                 (infer_result["device_call_total_exec_time"] + infer_result["device_call_total_exec_delay_time"] - infer_result["infer_latency_device"])/1000000])
        # headers = ['infer_index', 'A.infer_latency(ms)', 'B.exec_time(ms)', 'C.exec_delay_time(ms)', 'D.queue_delay_time(ms)', 'B + C - A (ms)']
        # table_str = tabulate(total_raw_data_list, tablefmt="github", headers=headers, stralign='right', numalign='right', floatfmt='.2f')
        # print(table_str)


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

def check_mode_for_log(filepath):
    pass
    # CLIntercept (64-bit) is loading...
    
    # Control DumpDir is set to non-default value: c:/dev/sungeunk/run_daily/
    # Control ReportToStderr is set to non-default value: true

    # cliloader mode
    # [required] Control DevicePerformanceTiming is set to non-default value: true
    # [required] Control DevicePerformanceTimelineLogging is set to non-default value: true
    # [optional]Control HostPerformanceTiming is set to non-default value: true

    # ... loading complete.

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

