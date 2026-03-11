#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.stats import skew

# Root directory to start scanning
root_dir = "/grphome/grp_batch_effects/data/gold" 

# Output file to store the summaries
output_csv = os.path.join(root_dir, "csv_value_summaries.csv")

summary_records = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname == "unadjusted.csv":
            file_path = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                numeric_cols = df.select_dtypes(include=np.number)
                if numeric_cols.empty:
                    print(f"[WARN] No numeric columns in {file_path}")
                    continue

                num_cols = numeric_cols.loc[:, ~numeric_cols.columns.str.startswith("meta_")]

                values = num_cols.values.flatten()
                values = values[~pd.isna(values)]  # remove NaNs

                # Calculate quartiles
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)

                # Check skewness
                data_skew = skew(values)

                # Suggest log-transform if data is very right-skewed or max > threshold
                max_val = np.max(values)
                log_transform_flag = max_val > 10 or data_skew > 1.0

                record = {
                    "file": file_path,
                    "n_rows": df.shape[0],
                    "n_cols": df.shape[1],
                    "min": np.min(values),
                    "q1": q1,
                    "median": np.median(values),
                    "mean": np.mean(values),
                    "q3": q3,
                    "max": max_val,
                    "skew": data_skew,
                    "suggest_log_transform": log_transform_flag
                }
                summary_records.append(record)

                print(f"[INFO] {file_path} | min={record['min']:.3f}, Q1={q1:.3f}, median={record['median']:.3f}, mean={record['mean']:.3f}, Q3={q3:.3f}, max={max_val:.3f}, skew={data_skew:.3f}, log_transform={log_transform_flag}")

            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")

# Save summaries to CSV
if summary_records:
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved summary statistics to: {output_csv}")
else:
    print("No files processed.")
