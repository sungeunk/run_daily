# This script is cross-platform compatible and works on both Windows & Ubuntu.

import re
import pandas as pd
import argparse
import sys
import os

def parse_log_file(filepath):
    """
    Reads a log file from a given path and parses it.
    Returns a pandas DataFrame.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            log_contents = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file '{filepath}': {e}")
        sys.exit(1)
    
    # Split the log data by 'model: ' to process each model's log separately.
    model_logs = re.split(r'BEGIN::model: ', log_contents)
    
    results = []
    for log in model_logs:
        if not log.strip():
            continue

        try:
            path_line = log.splitlines()[0].strip()
            model_name = os.path.basename(path_line).replace('.xml', '')
            
            precision = "N/A"
            if r'FP16\INT8' in path_line or r'FP16/INT8' in path_line:
                precision = "FP16-INT8"
            elif 'FP16' in path_line:
                precision = "FP16"
            elif 'FP32' in path_line:
                precision = "FP32"

            platform = "N/A"
            if 'onnx/onnx' in path_line or r'onnx\onnx' in path_line:
                platform = "ONNX"
            elif 'paddle/paddle' in path_line or r'paddle\paddle' in path_line:
                platform = "Paddle"

            batch_size_match = re.search(r'\[ INFO \] Model batch size: (\d+)', log)
            batch_size = int(batch_size_match.group(1)) if batch_size_match else 0
            
            throughput_match = re.search(r'Throughput:\s+([\d\.]+) FPS', log)
            throughput = float(throughput_match.group(1)) if throughput_match else 0.0
            
            results.append({
                "Model Name": model_name,
                "Precision": precision,
                "Platform": platform,
                "Batch Size": batch_size,
                "Throughput (FPS)": throughput
            })
        except (IndexError, AttributeError) as e:
            print(f"Warning: Could not parse a log block in '{filepath}'. Error: {e}")
            continue
            
    return pd.DataFrame(results)

def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="A tool to parse or compare benchmark log files.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    # --- Arguments for single file parsing mode ---
    parser.add_argument(
        "logfile", 
        nargs='?', # Makes the argument optional
        default=None,
        help="Path to a single log file to parse."
    )
    # --- Arguments for comparison mode ---
    parser.add_argument(
        "-t", "--target",
        help="Path to the target log file for comparison."
    )
    parser.add_argument(
        "-r", "--reference",
        help="Path to the reference log file for comparison."
    )
    args = parser.parse_args()

    # --- Mode 1: Comparison (--target and --reference are provided) ---
    if args.target and args.reference:
        if args.logfile:
            parser.error("Cannot specify a single logfile when using --target and --reference for comparison.")

        print("Running in Comparison Mode...")
        target_df = parse_log_file(args.target)
        reference_df = parse_log_file(args.reference)

        if target_df.empty or reference_df.empty:
            print("Error: One or both log files could not be parsed or contain no data.")
            sys.exit(1)

        merge_keys = ['Model Name', 'Precision', 'Platform', 'Batch Size']
        comparison_df = pd.merge(
            target_df, reference_df, on=merge_keys, suffixes=('_target', '_reference')
        )

        if comparison_df.empty:
            print("\nNo common models found between the target and reference files.")
            sys.exit(0)

        comparison_df.rename(columns={
            'Throughput (FPS)_target': 'Target FPS',
            'Throughput (FPS)_reference': 'Reference FPS'
        }, inplace=True)
        
        ref_fps = comparison_df['Reference FPS']
        tar_fps = comparison_df['Target FPS']
        
        # Calculate percentage change, handling division by zero
        comparison_df['Change (%)'] = 100 * (tar_fps - ref_fps) / ref_fps
        comparison_df['Change (%)'].replace([float('inf'), -float('inf')], 'N/A (inf)', inplace=True)

        output_df = comparison_df[['Model Name', 'Precision', 'Reference FPS', 'Target FPS', 'Change (%)']]
        output_df['Change (%)'] = output_df['Change (%)'].apply(
            lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else x
        )

        print("\n" + "="*50)
        print("Performance Comparison Report")
        print(f"Target: {args.target}")
        print(f"Reference: {args.reference}")
        print("="*50)
        print(output_df.to_markdown(index=False))

    # --- Mode 2: Single File Parsing (only logfile is provided) ---
    elif args.logfile:
        print("Running in Single File Parsing Mode...")
        result_df = parse_log_file(args.logfile)

        if result_df.empty:
            print(f"No valid model performance data was found in '{args.logfile}'.")
        else:
            print(f"\nResults for: {args.logfile}")
            print("-" * (13 + len(args.logfile)))
            print(result_df.to_markdown(index=False))
            
    # --- Invalid Usage ---
    else:
        print("Error: Invalid arguments provided.")
        print("Please use one of the following formats:\n")
        print("1. For single file parsing:")
        print(f"   python {sys.argv[0]} <path_to_logfile.txt>\n")
        print("2. For comparison:")
        print(f"   python {sys.argv[0]} --target <target.txt> --reference <reference.txt>\n")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()