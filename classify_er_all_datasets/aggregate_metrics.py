#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path

def main(metrics_dir, output_file):
    dfs = []
    metrics_dir = Path(metrics_dir)

    for adj_dir in metrics_dir.iterdir():
        adj_path = Path(adj_dir)
        if not adj_path.exists() or not adj_path.is_dir():
            print(f"⚠️  Adjuster directory {adj_dir} does not exist, skipping.")
            continue

        for csv_file in adj_path.glob("*_metrics.csv"):
            try:
                df = pd.read_csv(csv_file)
                # Add a column indicating the adjuster
                df["adjuster_folder"] = adj_path.name
                dfs.append(df)
            except Exception as e:
                print(f"ERROR reading {csv_file}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"✅ Aggregated metrics saved to {output_file}")
    else:
        print("⚠️  No metrics files found to aggregate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate classifier metrics across adjusters.")
    parser.add_argument("--metrics-dir", required=True, help="Directory for folders containing *_metrics.csv")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()
    main(args.metrics_dir, args.output)
