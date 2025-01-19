
import argparse
import os
import pandas as pd
import re
import streamlit as st

from glob import glob
from datetime import datetime
from pathlib import Path

ROOT_DAILY_REPORT = '/var/www/html/daily'

def get_daily_report_list(directory):
    return glob(os.path.join(directory, '*.report'))

def get_daily_report_data(filelist, num, filter='daily'):
    # key: date
    # content: purpose, filename, workweek
    info_map = {}
    filtered_list = []
    for file in filelist:
        with open(file, 'r', encoding='utf8') as fis:
            line_num = 5
            for line in fis.readlines():
                match_obj = re.search(f'\|[ ]+Purpose[ ]+\| ([a-zA-Z0-9-_ .&\\/\(\)]+)\|', line)
                if match_obj:
                    purpose = match_obj.groups()[0].strip()
                    if filter in purpose:
                        filtered_list.append(file)
                        match_obj = re.search(f'([\_0-9]+)\.(\d+\.\d\.\d)-(\d+-[a-z0-9]+)', file)
                        info_map[match_obj.groups()[0]] = {
                            'filename': file,
                            'commit': match_obj.groups()[2],
                            'purpose': purpose,
                            'workweek': get_weekday(match_obj.groups()[0])}
                        break
                line_num -= 1
                if line_num == 0:
                    break
    filtered_list.sort(reverse=True)
    return filtered_list[0:min(len(filtered_list), num)], info_map

def get_dataframe_ccg_table(filename, need_column=False):
    match_obj = re.search(f'([\_0-9]+)\.(\d+\.\d\.\d)-(\d+-[a-z0-9]+)', filename)
    daily_date = match_obj.groups()[0]
    daily_commit = match_obj.groups()[2]

    table = []
    # table.append(['', '', '', daily_commit] if need_column else [daily_commit])

    with open(filename, 'r', encoding='utf8') as fis:
        while True:
            # readline will return empty str when it is EOF.
            line = fis.readline()
            if line == '': break

            # '| model | in | out | latency(ms) |'
            match_obj = re.search(f'\| +model +\| +in +\| +out +\|', line)
            if match_obj:
                skip_list = ['chatglm3', 'chatGLM3-6b INT4', 'chatglm3_usage', 'Gemma-7B INT4', 'llama2-7b INT4', 'llama3-8b INT4', 'mistral-7B INT4', 'Phi-2 INT4',
                             'Phi-3-mini INT4', 'qwen', 'Qwen-7b INT4']

                while True:
                    line = fis.readline()
                    if line[0] != '|': break

                    # '|      baichuan2-7b-chat |   32 |   256 |         28.76 |'
                    match_obj = re.search(f'\| +([a-zA-Z0-9\-\_.= ]+) +\| +([0-9 ]+) +\| +([0-9 ]+) +\| +([0-9. ]+) +\|', line)
                    if match_obj != None:
                        values = match_obj.groups()

                        # skip data
                        skip_line = False
                        for skip_item in skip_list:
                            if skip_item == values[0].strip():
                                skip_line = True
                                break
                        if skip_line:
                            continue

                        if need_column:
                            table.append([values[0].strip(),
                                        int(values[1]) if values[1].strip() else '',
                                        int(values[2]) if values[2].strip() else '',
                                        f'{float(values[3]):.2f}' if values[3].strip() else ''])
                        else:
                            table.append([f'{float(values[3]):.2f}' if values[3].strip() else ''])
                        continue
                return pd.DataFrame(columns=['model', 'in', 'out', daily_date] if need_column else [daily_date], data=table)

            match_obj = re.search(f'\| +model +\| +precision +\| +in +\| +out +\| +exec +\| +latency\(ms\) +\|', line)
            if match_obj:
                while True:
                    line = fis.readline()
                    if line[0] != '|': break

                    # '|      baichuan2-7b-chat | OV_FP16-4BIT_DEFAULT |   32 |   256 |      1st |         28.76 |'
                    match_obj = re.search(f'\| +([a-zA-Z0-9\-\_.= ]+) +\| +([a-zA-Z0-9\-\_.= ]+) +\| +([0-9 ]+) +\| +([0-9 ]+) +\| +([a-zA-Z0-9: ]+) +\| +([0-9. ]+) +\|', line)
                    if match_obj != None:
                        values = match_obj.groups()
                        if need_column:
                            table.append([values[0].strip(),
                                        values[1].strip(),
                                        int(values[2]) if values[2].strip() else '',
                                        int(values[3]) if values[3].strip() else '',
                                        values[4].strip(),
                                        f'{float(values[5]):.2f}' if values[5].strip() else ''])
                        else:
                            table.append([f'{float(values[5]):.2f}' if values[5].strip() else ''])
                        continue
                return pd.DataFrame(columns=['model', 'precision', 'in', 'out', 'execution', daily_date] if need_column else [daily_date], data=table)


def get_excel_data(dataframe, info_map) -> str:
    if dataframe.size == 0:
        return ''

    table_str = '\n\n'
    commit_line = ''
    ww_line = ''
    date_line = ''
    for name in dataframe.columns:
        commit_line += f'{info_map[name]["commit"]}\t'
        ww_line += f'{info_map[name]["workweek"]}\t'
        date_line += f'{name}\t'
    table_str += commit_line[:-1] + '\n'
    table_str += ww_line[:-1] + '\n\n'
    table_str += date_line[:-1] + '\n'

    for item in dataframe.itertuples(index=False):
        for i in range(0, len(item)):
            table_str += f'{item[i]}\t'
        table_str = table_str[:-1] + '\n'
    return table_str

def get_weekday(str):
    cal = datetime.strptime(str, "%Y%m%d_%H%M").isocalendar()
    return f'WW{cal.week}.{cal.weekday}'

def settings():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

def main():
    settings()

    parser = argparse.ArgumentParser(description="daily report viewer" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--report_dir', help=f'daily reports stored directory (for debugging. default path: {ROOT_DAILY_REPORT})', type=str, default=None)
    args = parser.parse_args()

    config_column_1, config_column_2, config_column_3 = st.columns(3)
    if args.report_dir == None:
        daily_server_only = st.checkbox('daily server only', value=True)

        with config_column_1:
            DAILY_SERVER_LIST = ['DUT4016PTLH', 'DUT6047BMGFRD', 'ARL1', 'LNL-02', 'MTL-01', 'dg2alderlake', 'BMG-01']
            server_list = DAILY_SERVER_LIST if daily_server_only else sorted(os.listdir(args.report_dir))
            report_dir = os.path.join(ROOT_DAILY_REPORT, st.selectbox("Select Server", server_list))
    else:
        report_dir = args.report_dir
    report_list = get_daily_report_list(report_dir)

    # filter report list by number + purpose
    with config_column_2:
        number = st.number_input("Insert a number to display reports", min_value=1, max_value=len(report_list))
    with config_column_3:
        filter_str = st.text_input(label='Filter:', value='daily')
    list, info_map = get_daily_report_data(report_list, number, filter_str)

    # parse reports
    df_all = pd.DataFrame()
    for item in list:
        df = get_dataframe_ccg_table(item, df_all.empty)
        df_all = df if df_all.empty else df_all.join(df, how='left')

    # tab interface
    view_tab_1, view_tab_2 = st.tabs(["excel paste", "summary"])

    with view_tab_1:
        st.dataframe(df_all)

        # generate excel data
        # input: removed model_name/in_token/out_token columns
        excel_str = ''
        if len(df_all.columns) == 4:
            excel_str = get_excel_data(df_all.iloc[:, 3:], info_map)
        elif len(df_all.columns) == 6:
            excel_str = get_excel_data(df_all.iloc[:, 5:], info_map)
        st.text_area('Text for Excel', value=excel_str, label_visibility="visible")

    # with view_tab_2:
    #     model_map = {}
    #     index = 0
    #     for items in df_all.iloc[:, :3].itertuples(index=False):
    #         if model_map.get(items[0], None):
    #             model_map[items[0]]['conf'].append([f'{items[1]}_{items[2]}'])
    #         else:
    #             model_map[items[0]] = {'first': index, 'conf': [f'{items[1]}_{items[2]}'] }
    #         index += 1
    #     model_name = st.selectbox("Select Model", [*model_map.keys()][:-3])


    #     new_columns = []
    #     for items in df_all.itertuples():
    #         new_columns.append(f'{items[0]}_{items[1]}_{items[2]}_{items[3]}')
    #     # df_all.columns = new_columns
    #     print(f'new_columns: {new_columns}')


    #     # st.dataframe(df_view)
    #     first_index = model_map[model_name]['first']
    #     last_index = first_index + len(model_map[model_name]['conf'])
    #     df_view = df_all.iloc[first_index:last_index, 3:].transpose()
    #     st.dataframe(df_view)
    #     st.line_chart(df_view)

if __name__ == "__main__":
    main()
