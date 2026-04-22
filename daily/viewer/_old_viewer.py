import argparse
import pandas as pd
import streamlit as st
import duckdb
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# --- Constants ---
METADATA_COLS = ['model', 'precision', 'in', 'out', 'execution']
DEFAULT_DB_PATH = Path(__file__).with_name('bench.duckdb')

# Token size classification
SHORT_INPUT_TOKEN = 'short'  # 32
LONG_INPUT_TOKEN = 'long'    # 1024
SHORT_OUTPUT_TOKEN = 'short'  # ~57
LONG_OUTPUT_TOKEN = 'long'    # ~156

FIXED_ROW_ORDER = [
    ('baichuan2-7b-chat', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('baichuan2-7b-chat', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('baichuan2-7b-chat', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('baichuan2-7b-chat', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('chatglm3-6b', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('chatglm3-6b', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('chatglm3-6b', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('chatglm3-6b', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('glm-4-9b-chat-hf', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('glm-4-9b-chat-hf', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('glm-4-9b-chat-hf', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('glm-4-9b-chat-hf', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('gemma-7b-it', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('gemma-7b-it', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('gemma-7b-it', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('gemma-7b-it', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('llama-2-7b-chat-hf', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('llama-2-7b-chat-hf', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('llama-2-7b-chat-hf', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('llama-2-7b-chat-hf', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('llama-3.1-8b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('llama-3.1-8b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('llama-3.1-8b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('llama-3.1-8b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('minicpm-1b-sft', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('minicpm-1b-sft', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('minicpm-1b-sft', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('minicpm-1b-sft', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('mistral-7b-instruct-v0.2', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('mistral-7b-instruct-v0.2', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('mistral-7b-instruct-v0.2', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('mistral-7b-instruct-v0.2', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('phi-3-mini-4k-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('phi-3-mini-4k-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('phi-3-mini-4k-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('phi-3-mini-4k-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('phi-3.5-mini-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('phi-3.5-mini-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('phi-3.5-mini-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('phi-3.5-mini-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('phi-3.5-vision-instruct', 'OV_FP16-4BIT_DEFAULT', 802, 256, '1st'),
    ('phi-3.5-vision-instruct', 'OV_FP16-4BIT_DEFAULT', 802, 256, '2nd'),
    ('phi-3.5-vision-instruct', 'OV_FP16-4BIT_DEFAULT', 1032, 256, '1st'),
    ('phi-3.5-vision-instruct', 'OV_FP16-4BIT_DEFAULT', 1032, 256, '2nd'),
    ('qwen2-7b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('qwen2-7b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('qwen2-7b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('qwen2-7b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('qwen2.5-7b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '1st'),
    ('qwen2.5-7b-instruct', 'OV_FP16-4BIT_DEFAULT', SHORT_INPUT_TOKEN, 256, '2nd'),
    ('qwen2.5-7b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('qwen2.5-7b-instruct', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('minicpm-v-2_6', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '1st'),
    ('minicpm-v-2_6', 'OV_FP16-4BIT_DEFAULT', LONG_INPUT_TOKEN, 256, '2nd'),
    ('whisper-large-v3', 'OV_FP16-4BIT_DEFAULT', 0, SHORT_OUTPUT_TOKEN, 'pipeline'),
    ('whisper-large-v3', 'OV_FP16-4BIT_DEFAULT', 0, LONG_OUTPUT_TOKEN, 'pipeline'),
    ('Resnet50', 'INT8', 0, 0, 'batch:1'),
    ('Resnet50', 'INT8', 0, 0, 'batch:64'),
    ('stable-diffusion-v1-5', 'FP16', 32, 0, 'pipeline'),
    ('stable-diffusion-v2-1', 'FP16', 32, 0, 'pipeline'),
    ('stable-diffusion-v3.0', 'FP16', 0, 0, 'pipeline'),
    ('stable-diffusion-xl', 'FP16', 0, 0, 'pipeline'),
    ('lcm-dreamshaper-v7', 'FP16', 32, 0, 'pipeline'),
]

# --- Data Loading and Parsing Functions ---

def classify_token_size(token_value: float, threshold: float = 100) -> str:
    """
    Classify token size as 'short' or 'long'.
    For token_value 0, returns '0' (special case).
    Tokens < threshold are 'short', >= threshold are 'long'.
    """
    if token_value == 0:
        return '0'
    return 'short' if token_value < threshold else 'long'

def find_matching_row(merged_df: pd.DataFrame, target_row: Tuple, verbose: bool = False) -> Optional[tuple]:
    """
    Find a row in merged_df that matches target_row.
    Matches model, precision, execution exactly.
    For in_token/out_token: if it's a string ('short'/'long'), classifies actual data; if numeric, matches exactly.

    Special handling: for stable-diffusion and lcm models, only match by model, precision, execution (ignore tokens).

    Args:
        merged_df: DataFrame with MultiIndex
        target_row: (model, precision, in_token, out_token, execution)
                    where in_token/out_token can be 'short', 'long', '0', or numeric
        verbose: Print debug info for non-matches

    Returns:
        Index of matching row or None
    """
    model, precision, in_token_target, out_token_target, execution = target_row

    # Models that should match only by model, precision, execution (ignore tokens)
    token_agnostic_models = {'stable-diffusion-v1-5', 'stable-diffusion-v2-1', 'stable-diffusion-v3.0',
                             'stable-diffusion-xl', 'lcm-dreamshaper-v7', 'flux.1-schnell'}

    # Filter by exact matches for model, precision, execution
    mask = (
        (merged_df['model'] == model) &
        (merged_df['precision'] == precision) &
        (merged_df['execution'] == execution)
    )

    candidates = merged_df[mask]

    if candidates.empty:
        if verbose:
            # Check which part failed
            model_match = (merged_df['model'] == model).sum()
            prec_match = (merged_df['precision'] == precision).sum()
            exec_match = (merged_df['execution'] == execution).sum()
            print(f"  Model '{model}' found: {model_match}, Precision '{precision}' found: {prec_match}, Execution '{execution}' found: {exec_match}")
        return None

    # For token-agnostic models, return first candidate (no token matching needed)
    if model in token_agnostic_models:
        return candidates.index[0]

    # For other models, find best match based on tokens
    best_match = None
    best_score = float('inf')
    is_in_category = isinstance(in_token_target, str) and in_token_target in ('short', 'long', '0')
    is_out_category = isinstance(out_token_target, str) and out_token_target in ('short', 'long', '0')

    for idx, row in candidates.iterrows():
        # Handle in_token matching
        if is_in_category:
            actual_in_category = classify_token_size(row['in'])
            in_matches = actual_in_category == in_token_target
        else:
            # Exact numeric match for in_token
            in_diff = abs(row['in'] - in_token_target)
            in_matches = in_diff <= max(in_token_target * 0.2, 1) if in_token_target > 0 else (in_diff == 0)

        # Handle out_token matching
        if is_out_category:
            actual_out_category = classify_token_size(row['out'])
            out_matches = actual_out_category == out_token_target
        else:
            # Exact numeric match for out_token
            out_diff = abs(row['out'] - out_token_target)
            out_matches = out_diff <= max(out_token_target * 0.2, 10) or out_token_target == 0

        if verbose:
            if is_in_category or is_out_category:
                in_category = classify_token_size(row['in']) if is_in_category else 'numeric'
                out_category = classify_token_size(row['out']) if is_out_category else 'numeric'
                print(f"  Candidate: in={row['in']:.0f} (category={in_category}, target={in_token_target}), out={row['out']:.0f} (category={out_category}, target={out_token_target})")
            else:
                out_diff = abs(row['out'] - out_token_target)
                print(f"  Candidate: in={row['in']:.0f} (target={in_token_target}), out={row['out']:.0f} (target={out_token_target}, diff={out_diff:.0f})")

        # Check if both input and output match
        if in_matches and out_matches:
            score = abs(row['in'] - (in_token_target if not is_in_category else 0)) + abs(row['out'] - (out_token_target if not is_out_category else 0))
            if score < best_score:
                best_score = score
                best_match = idx

    return best_match

def reorder_by_fixed_order(merged_df: pd.DataFrame, fixed_order: List[Tuple]) -> pd.DataFrame:
    """
    Reorder merged_df according to fixed_order with tolerance for token counts.

    Args:
        merged_df: DataFrame to reorder (may have MultiIndex)
        fixed_order: List of tuples defining the desired order

    Returns:
        Reordered DataFrame with reset index
    """
    if merged_df.empty:
        return merged_df

    # Handle MultiIndex: reset and drop old index
    if isinstance(merged_df.index, pd.MultiIndex):
        merged_df = merged_df.reset_index(drop=True)

    used_indices = set()
    ordered_indices = []

    for i, target_row in enumerate(fixed_order):
        idx = find_matching_row(merged_df, target_row, verbose=False)
        if idx is not None:
            ordered_indices.append(idx)
            used_indices.add(idx)
            print(f"✓ {i+1:2d}. {target_row} -> found")
        else:
            print(f"✗ {i+1:2d}. {target_row} -> not found")

    # Add any remaining rows that weren't in FIXED_ROW_ORDER
    remaining = []
    for idx in merged_df.index:
        if idx not in used_indices:
            remaining.append(idx)
            row = merged_df.loc[idx]
            print(f"⚠ Extra: ({row['model']}, {row['precision']}, {row['in']}, {row['out']}, {row['execution']})")

    ordered_indices.extend(remaining)

    print(f"\nTotal matched: {len(used_indices)}/{len(fixed_order)}")
    print(f"Extra rows: {len(remaining)}\n")

    return merged_df.loc[ordered_indices].reset_index(drop=True)

def load_all_reports_from_db(db_path: Path, machine: str) -> pd.DataFrame:
    """Loads report metadata from DuckDB for a specific machine."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        """
        SELECT
          run_id,
          report_file AS filename,
          purpose,
          ov_commit AS commit_id,
          ww AS workweek,
          strftime(ts, '%Y%m%d_%H%M') AS datetime
        FROM runs
        WHERE machine = ?
        ORDER BY ts DESC
        """,
        [machine],
    ).fetchdf()
    con.close()

    if df.empty:
        return df

    df.set_index('datetime', inplace=True)
    return df

def load_perf_df_from_db(db_path: Path, run_id: str, date_key: str, first_df: bool) -> pd.DataFrame:
    """Loads performance data from DuckDB for a given run_id."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        """
        SELECT model, precision, in_token, out_token, exec_mode, value
        FROM perf
        WHERE run_id = ?
        ORDER BY model, precision, in_token, out_token, exec_mode
        """,
        [run_id],
    ).fetchdf()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df.rename(columns={
        'in_token': 'in',
        'out_token': 'out',
        'exec_mode': 'execution',
        'value': date_key,
    }, inplace=True)

    columns = ['model', 'precision', 'in', 'out', 'execution', date_key]
    df = df[columns]

    numeric_cols = ['in', 'out', date_key]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df.set_index(METADATA_COLS, drop=False, inplace=True)

    # Reorder by FIXED_ROW_ORDER
    df = reorder_by_fixed_order(df, FIXED_ROW_ORDER)

    # Re-establish MultiIndex after reorder (which does reset_index)
    # Use drop=True to keep only performance columns
    df.set_index(METADATA_COLS, drop=True, inplace=True)

    if not first_df:
        df = df.iloc[:, 0:]

    return df

def generate_excel_paste_data(df: pd.DataFrame, report_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """Formats the performance DataFrame for easy pasting into Excel."""
    if df.empty:
        return None, ""

    perf_cols = [col for col in df.columns if col not in METADATA_COLS]
    if not perf_cols:
        return df, "No performance data columns found to generate Excel string."

    perf_df = df[perf_cols]
    data_str = perf_df.to_csv(sep='\t', index=False, header=False, float_format='%.2f')

    valid_perf_cols = [pc for pc in perf_cols if pc in report_df.index]
    header_data = report_df.loc[valid_perf_cols]

    commit_line = "\t".join(header_data["commit_id"])
    ww_line = "\t".join(header_data["workweek"])
    date_line = "\t".join(header_data.index)

    full_paste_string = "\n\n" + "\n".join([commit_line, ww_line, '', date_line, data_str])

    return df, full_paste_string

# --- Streamlit UI ---

def setup_page():
    """Configures Streamlit page settings."""
    st.set_page_config(layout='wide')
    pd.set_option('display.float_format', '{:.2f}'.format)

def main():
    """Main function to run the Streamlit application."""
    setup_page()

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    st.title("Daily Performance Report Viewer")

    st.sidebar.header("Configuration")
    is_daily_list = st.sidebar.checkbox("Filter by Daily Servers", value=True)

    if not args.db.exists():
        st.error(f"DuckDB not found: {args.db}")
        return

    con = duckdb.connect(str(args.db), read_only=True)
    server_options = [r[0] for r in con.execute("SELECT DISTINCT machine FROM runs ORDER BY machine").fetchall()]
    con.close()

    server_selection = st.sidebar.radio("Select Server", server_options)
    all_reports_df = load_all_reports_from_db(args.db, server_selection)
    if all_reports_df.empty:
        st.info(f"No report file in {server_selection}")
        return

    filter_str = st.text_input('Filter reports by purpose:', value='daily_CB timer')
    filtered_reports_df = all_reports_df
    if filter_str:
        filtered_reports_df = all_reports_df[all_reports_df['purpose'].str.contains(filter_str, na=False)]

    st.subheader("Select Reports to Compare")
    selection = st.dataframe(
        filtered_reports_df,
        on_select='rerun',
        selection_mode='multi-row'
    )

    selected_rows = selection['selection']['rows']
    if not selected_rows:
        st.info("Select one or more reports from the table above to see details.")
        return

    selected_reports_df = filtered_reports_df.iloc[selected_rows]

    # --- Data Loading for Selected Reports (Moved before tabs) ---
    merged_df = pd.DataFrame()
    for datetime_key, row in selected_reports_df.iterrows():
        perf_df = load_perf_df_from_db(args.db, row['run_id'], datetime_key, merged_df.empty)
        merged_df = perf_df if merged_df.empty else merged_df.join(perf_df, how='outer')

    # Data is already sorted by FIXED_ROW_ORDER from load_perf_df_from_db
    # Reset index to have clean integer index for display
    if not merged_df.empty:
        merged_df.reset_index(inplace=True)

    # --- UI Tabs ---
    # excel_tab, summary_tab = st.tabs(["Excel Paste", "Summary & Chart"])
    excel_tab = st.tabs(["Excel Paste"])[0]

    with excel_tab:
        excel_df, excel_paste_str = generate_excel_paste_data(merged_df, all_reports_df)

        st.subheader("Data for Excel")
        st.text_area('Copy the text below and paste it into Excel:', value=excel_paste_str, height=200)

        st.subheader("Preview")
        if excel_df is not None:
            st.dataframe(excel_df)

if __name__ == "__main__":
    main()