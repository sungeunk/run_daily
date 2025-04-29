
import argparse
import os
import pandas as pd
import re
import streamlit as st

from common_utils import *
from report import *
from datetime import datetime
from glob import glob
from pathlib import Path

ROOT_DAILY_REPORT = '/var/www/html/daily'

def get_report_info(filepath):
    purpose = ''
    commit_id = ''
    with open(filepath, 'rt', encoding='utf8') as fis:
        for line in fis.readlines():
            match_obj = re.search(r'\| +Purpose +\|([a-zA-Z0-9-_# .&\\/\(\)]+)\|', line)
            if match_obj:
                purpose = match_obj.groups()[0].strip()
                continue
            match_obj = re.search(r'\| +OpenVINO +\| +\d+.\d+.\d+-(\d+)-([\d\w]+)', line)
            if match_obj:
                values = match_obj.groups()
                commit_id = f'{values[0]}-{values[1]}'
                continue
    return purpose, commit_id

def get_str(rex, text):
    match_obj = re.search(rex, text)
    if match_obj:
        return str(match_obj.groups()[0])
    return ''

def get_float(rex, text):
    match_obj = re.search(rex, text)
    if match_obj:
        value = match_obj.groups()[0]
        if is_float(value):
            return float(value)
    return 0

def get_weekday(str):
    cal = datetime.strptime(str, "%Y%m%d").isocalendar()
    return f'WW{cal.week}.{cal.weekday}'

def get_weekday_str(filename):
    match_obj = re.search(r'\.([0-9]+)\_(\d+)\.', filename)
    if match_obj:
        values = match_obj.groups()
        return get_weekday(values[0]), f'{values[0]}_{values[1]}'
    return '', ''

def get_daily_report_dataframe(directory):
    # ['filename', 'purpose', 'commit', 'workweek', 'date]
    ret_table = []
    for report_file in glob(os.path.join(directory, '*.report')):
        purpose, commit_id = get_report_info(report_file)
        ww, datetime = get_weekday_str(report_file)
        ret_table.append([os.path.basename(report_file), purpose, commit_id, ww, datetime])
    df = pd.DataFrame(data=ret_table, columns=['filename', 'purpose', 'commit_id', 'workweek', 'datetime'])
    df.set_index(['datetime'], inplace=True)
    df.sort_index(inplace=True, ascending=False)
    return df

def get_dataframe_ccg_table_from_pickle(filename, need_column=False):
    match_obj = re.search(f'([\_0-9]+)\.(\d+\.\d\.\d)-(\d+-[a-z0-9]+)', filename)
    daily_date = match_obj.groups()[0]

    result_root = load_result_file(replace_ext(filename, "pickle"))
    csv_table = generate_csv_table(result_root, False)
    for item in csv_table:
        if len(item) == 6:
            if item[0] == 'qwen_usage' and item[4] == 'memory percent':
                item[5] = f'{float(item[5]):.2f}'
            elif item[0] == 'qwen_usage' and item[4] == 'memory size':
                item[5] = f'{float(item[5]) / (1024*1024*1024):.2f}'

    dataframe = pd.DataFrame(columns=['model', 'precision', 'in', 'out', 'execution', daily_date], data=csv_table)
    if not need_column:
        dataframe = dataframe.iloc[:, 5:]

    return dataframe

def get_excel_data(dataframe, report_df):
    if dataframe.size == 0:
        return None, ''
    df = dataframe.iloc[:, 5:]

    table_str = '\n\n'
    commit_line = ''
    ww_line = ''
    date_line = ''

    for name in df.columns:
        row_item = report_df.loc[name]
        commit_line += f'{row_item["commit_id"]}\t'
        ww_line += f'{row_item["workweek"]}\t'
        date_line += f'{name}\t'
    table_str += commit_line[:-1] + '\n'
    table_str += ww_line[:-1] + '\n\n'
    table_str += date_line[:-1] + '\n'

    for item in df.itertuples(index=False):
        for i in range(0, len(item)):
            table_str += f'{item[i]}\t'
        table_str = table_str[:-1] + '\n'
    return df, table_str

def settings():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    st.set_page_config(layout='wide')

def main():
    settings()

    parser = argparse.ArgumentParser(description="daily report viewer" , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--report_dir', help=f'daily reports stored directory (for debugging. default path: {ROOT_DAILY_REPORT})', type=str, default=ROOT_DAILY_REPORT)
    args = parser.parse_args()

    config_column_1, config_column_2, config_column_3 = st.columns(spec=[0.1, 0.2, 0.7], vertical_alignment="bottom")
    with config_column_1:
        daily_list_on = st.checkbox("daily", value=True)

    with config_column_2:
        server_selection = st.selectbox("Select Server", ['DUT4005PTLH', 'MTL-01', 'DUT6047BMGFRD', 'DUT133ARLH', 'LNL-02', 'dg2alderlake'] if daily_list_on else sorted(os.listdir(args.report_dir)))
        report_dir = os.path.join(args.report_dir, server_selection)
    report_df = get_daily_report_dataframe(report_dir)

    # filter report list by purpose
    with config_column_3:
        filter_str = st.text_input(label='Filter:', value='daily_CB')
        if len(filter_str):
            report_filtered_df = report_df[report_df['purpose'].str.contains(filter_str)]
        else:
            report_filtered_df = report_df

    report_selection_list = st.dataframe(data=report_filtered_df, key='choosed_report_df', on_select='rerun', selection_mode='multi-row', width=4000)
    report_filtered_selection_df = report_filtered_df.iloc[report_selection_list['selection']['rows']]

    # tab interface
    view_tab_1, view_tab_2 = st.tabs(["excel paste", "summary"])
    with view_tab_1:
        ccg_table_df_all = pd.DataFrame()
        for index, row in report_filtered_selection_df.iterrows():
            ccg_table_df = get_dataframe_ccg_table_from_pickle(os.path.join(*[report_dir, row['filename']]), ccg_table_df_all.empty)
            ccg_table_df_all = ccg_table_df if ccg_table_df_all.empty else ccg_table_df_all.join(ccg_table_df, how='left')

        # generate excel data
        # input: removed model_name/in_token/out_token columns
        excel_df, excel_str = get_excel_data(ccg_table_df_all, report_df)
        st.text_area('Text for Excel', value=excel_str, label_visibility="visible")
        st.write(excel_df)


if __name__ == "__main__":
    main()
