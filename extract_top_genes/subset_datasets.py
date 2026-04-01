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

# Log transform per dataset
def log_transform_per_dataset(df, test_source):
    train_df = df[df["meta_source"] != test_source].copy()
    test_df = df[df["meta_source"] == test_source].copy()

    def log_transform_helper(sub_df):
        meta_cols = sub_df.filter(regex="^meta_")
        num_cols = sub_df.drop(columns=meta_cols.columns)

        num_mat = num_cols.to_numpy(dtype=float)

        for ds in sub_df["meta_source"].unique():
            idx = sub_df["meta_source"] == ds
            mat_ds = num_mat[idx, :]

            min_val = np.nanmin(mat_ds)
            print(f">>> Applying log1p to dataset: {ds} (min={min_val})")
            mat_ds = np.log1p(mat_ds - min_val)

            num_mat[idx, :] = mat_ds

        transformed = pd.concat(
            [meta_cols.reset_index(drop=True),
            pd.DataFrame(num_mat, columns=num_cols.columns)],
            axis=1
        )

        return transformed

    train_t = log_transform_helper(train_df)
    test_t = log_transform_helper(test_df)

    return pd.concat([train_t, test_t], axis=0, ignore_index=True)

def write_subset(df, output_path, n_studies, test):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_filename = os.path.join(output_path, f"test_{test}-subset.csv")
    df.to_csv(output_filename, index=False)
    print(f">>> Subset written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create study subset with preprocessing (Python version)"
    )
    parser.add_argument("--input", required=True,
                        help="All combined file")
    parser.add_argument("--test", required=True,
                        help="Test dataset meta_source")
    parser.add_argument("--dataset_list", nargs="+", required=True,
                        help="List of GSE ids for datasets to include")
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    df = load_combined_data(args.input)

    data_list = args.dataset_list

    subset = create_subset(df, data_list)
    processed = log_transform_per_dataset(subset, args.test)
    write_subset(processed, args.out_dir, len(data_list), args.test)

if __name__ == "__main__":
    main()