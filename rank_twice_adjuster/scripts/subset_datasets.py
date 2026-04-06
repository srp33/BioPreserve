#!/usr/bin/env python3

import os
import argparse
import warnings
import pandas as pd
import numpy as np

# Load and concatenate datasets
def load_combined_data(input_csv):
    combined = pd.read_csv(input_csv, low_memory=False)
    return combined

# Create subset
def create_subset(df, data_list):
    selected_studies = set(data_list)
    print("Unique Meta Source: ")
    print(df["meta_source"].unique())
    all_studies = set(df["meta_source"].dropna().astype(str).unique())

    missing = selected_studies - all_studies
    if missing:
        raise ValueError(
            f"Requested datasets not found in meta_source: {missing}\n"
            f"Available datasets: {sorted(all_studies)}"
        )
    subset_df = df[df["meta_source"].isin(selected_studies)].copy()

    return subset_df

def write_subset(df, output_path, test, train):
    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f"{train}-{test}-subset.csv")
    df.to_csv(output_filename, index=False)
    print(f">>> Subset written to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Create study subset with preprocessing (Python version)"
    )
    parser.add_argument("--input", required=True,
                        help="All combined file")
    parser.add_argument("--test", required=True,
                        help="Test dataset meta_source")
    parser.add_argument("--train", required=True,
                        help="Train dataset meta_source")
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    df = load_combined_data(args.input)

    data_list = [args.train, args.test]

    subset = create_subset(df, data_list)
    write_subset(subset, args.out_dir, args.test, args.train)

if __name__ == "__main__":
    main()