import argparse
import pandas as pd
import re
import streamlit as st
from common_utils import * # Assuming this contains load_result_file, is_float, etc.
from report import * # Assuming this contains generate_csv_table
from datetime import datetime
from pathlib import Path
from st_copy_to_clipboard import st_copy_to_clipboard
from typing import List, Dict, Tuple, Optional

# --- Constants ---
ROOT_DAILY_REPORT = Path('/var/www/html/daily')

# --- Data Loading and Parsing Functions ---

def parse_report_file(filepath: Path) -> Tuple[str, str]:
    """Extracts the purpose and commit ID from a .report file."""
    purpose = "N/A"
    commit_id = "N/A"
    try:
        with filepath.open('r', encoding='utf8') as f:
            content = f.read()
            purpose_match = re.search(r'\| +Purpose +\|([^|]+)\|', content)
            if purpose_match:
                purpose = purpose_match.group(1).strip()
            commit_match = re.search(r'\| +OpenVINO +\|.*?-(\d+)-([\da-fA-F]+)', content)
            if commit_match:
                commit_id = f"{commit_match.group(1)}-{commit_match.group(2)}"
    except IOError as e:
        st.error(f"Error reading {filepath}: {e}")
    return purpose, commit_id

def get_report_metadata(report_path: Path) -> Dict[str, str]:
    """Extracts all metadata from a report file path."""
    purpose, commit_id = parse_report_file(report_path)
    
    match = re.search(r'\.(\d{8})_(\d+)\.', report_path.name)
    if match:
        date_str, time_str = match.groups()
        datetime_obj = datetime.strptime(date_str, "%Y%m%d")
        cal = datetime_obj.isocalendar()
        workweek = f'WW{cal.week}.{cal.weekday}'
        datetime_key = f'{date_str}_{time_str}'
    else:
        workweek, datetime_key = "N/A", "N/A"
        
    return {
        'filename': report_path.name,
        'purpose': purpose,
        'commit_id': commit_id,
        'workweek': workweek,
        'datetime': datetime_key
    }

def load_all_reports(directory: Path) -> pd.DataFrame:
    """Scans a directory for .report files and loads their metadata into a DataFrame."""
    report_files = list(directory.glob('*.report'))
    if not report_files:
        return pd.DataFrame()
    
    data = [get_report_metadata(f) for f in report_files]
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    df.sort_index(ascending=False, inplace=True)
    return df

def load_perf_df_from_pickle(pickle_path: Path, first_df: bool) -> pd.DataFrame:
    """Loads and processes performance data from a pickle file."""
    match = re.search(r'(\d{8}_\d+)', pickle_path.name)
    date_key = match.group(1) if match else "performance"

    result_root = load_result_file(str(pickle_path))
    csv_table = generate_csv_table(result_root, False)

    for item in csv_table:
        if len(item) == 6 and item[0] == 'qwen_usage':
            metric, value = item[4], item[5]
            if metric == 'memory percent' and is_float(value):
                item[5] = f'{float(value):.2f}'
            elif metric == 'memory size' and is_float(value):
                item[5] = f'{float(value) / (1024**3):.2f}'

    columns = ['model', 'precision', 'in', 'out', 'execution', date_key]
    df = pd.DataFrame(data=csv_table, columns=columns)

    numeric_cols = ['in', 'out', date_key]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if not first_df:
        df = df.iloc[:, 5:]
        
    return df

def generate_excel_paste_data(df: pd.DataFrame, report_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """Formats the performance DataFrame for easy pasting into Excel."""
    if df.empty:
        return None, ""

    METADATA_COLS = ['model', 'precision', 'in', 'out', 'execution']
    perf_cols = [col for col in df.columns if col not in METADATA_COLS]
    if not perf_cols:
        return df, "No performance data columns found to generate Excel string."

    # This removes the leftmost metadata columns from the string output.
    perf_df = df[perf_cols]
    data_str = perf_df.to_csv(sep='\t', index=False, header=False, float_format='%.2f')

    header_data = report_df.loc[perf_cols]
    commit_line = "\t".join(header_data["commit_id"])
    ww_line = "\t".join(header_data["workweek"])
    date_line = "\t".join(header_data.index)

    full_paste_string = "\n\n" + "\n".join([commit_line, ww_line, '', date_line, data_str])
    
    # Return the original merged DataFrame for the preview table in the UI.
    return df, full_paste_string

# --- Streamlit UI ---

def setup_page():
    """Configures Streamlit page settings."""
    st.set_page_config(layout='wide')
    pd.set_option('display.float_format', '{:.3f}'.format)

def main():
    """Main function to run the Streamlit application."""
    setup_page()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--report_dir', type=Path, default=ROOT_DAILY_REPORT)
    args = parser.parse_args()

    st.title("Daily Performance Report Viewer")

    st.sidebar.header("Configuration")
    is_daily_list = st.sidebar.checkbox("Filter by Standard Servers", value=True)
    
    if is_daily_list:
        server_options = ['DUT4015PTLH', 'ARLH-01', 'LNL-03', 'MTL-01', 'BMG-02', 'dg2alderlake']
    else:
        server_options = sorted([d.name for d in args.report_dir.iterdir() if d.is_dir()])
    
    server_selection = st.sidebar.radio("Select Server", server_options)
    report_dir = args.report_dir / server_selection

    all_reports_df = load_all_reports(report_dir)
    if all_reports_df.empty:
        st.info(f"No report file in {server_selection}")
        return

    filter_str = st.text_input('Filter reports by purpose:', value='daily_CB')
    filtered_reports_df = all_reports_df
    if filter_str:
        filtered_reports_df = all_reports_df[all_reports_df['purpose'].str.contains(filter_str, na=False)]

    st.subheader("Select Reports to Compare")
    selection = st.dataframe(
        filtered_reports_df, 
        on_select='rerun', 
        selection_mode='multi-row',
        width='stretch'
    )
    
    selected_rows = selection['selection']['rows']
    if not selected_rows:
        st.info("Select one or more reports from the table above to see details.")
        return
        
    selected_reports_df = filtered_reports_df.iloc[selected_rows]

    excel_tab, summary_tab = st.tabs(["Excel Paste", "Summary"])

    with excel_tab:
        merged_df = pd.DataFrame()
        for datetime_key, row in selected_reports_df.iterrows():
            pickle_path = (report_dir / row['filename']).with_suffix('.pickle')
            if pickle_path.exists():
                perf_df = load_perf_df_from_pickle(pickle_path, merged_df.empty)
                merged_df = perf_df if merged_df.empty else merged_df.join(perf_df, how='left')
            else:
                st.warning(f"Pickle file not found for {row['filename']}")

        excel_df, excel_paste_str = generate_excel_paste_data(merged_df, all_reports_df)
        
        st.subheader("Data for Excel")
        st.text_area('Copy the text below and paste it into Excel:', value=excel_paste_str, height=100)
        
        st.subheader("Preview")
        if excel_df is not None:
            st.dataframe(excel_df, width='stretch')

    with summary_tab:
        st.subheader("Selected Reports Summary")
        st.dataframe(selected_reports_df, width='stretch')


if __name__ == "__main__":
    main()
