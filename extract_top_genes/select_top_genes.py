#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import numpy as np

def load_ranked(path):
    """Load a gene × adjuster CSV for a single metadata label, with Gene as index."""
    df = pd.read_csv(path, index_col=0)  # <-- Gene is already the index
    return df

def aggregate_scores(dfs, method="max"):
    """
    Aggregate multiple adjusters into a single score per gene.
    dfs: list of DataFrames, all with same genes as index.
    method: aggregation method ('max', 'mean', 'median').
    """
    combined = pd.concat(dfs, axis=1)
    
    if method == "max":
        agg_scores = combined.max(axis=1)
    elif method == "mean":
        agg_scores = combined.mean(axis=1)
    elif method == "median":
        agg_scores = combined.median(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return agg_scores

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load CSVs (already with Gene as index)
    dfs = [load_ranked(path) for path in args.ranked_lists]

    # Aggregate across adjusters
    agg_scores = aggregate_scores(dfs, method=args.method)

    # Sort by score descending
    agg_scores = agg_scores.sort_values(ascending=False)

    # Apply threshold and top_n
    if args.threshold is not None:
        agg_scores = agg_scores[agg_scores > args.threshold]

    if args.top_n:
        agg_scores = agg_scores.head(args.top_n)

    # Save output
    out_df = pd.DataFrame({
        "gene": agg_scores.index,
        "score": agg_scores.values
    })
    out_df.to_csv(args.output, index=False)
    print(f"Saved aggregated gene list: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked_lists", nargs="+", required=True, help="Input CSVs (gene × adjuster)")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--method", choices=["max", "mean", "median"], default="max", help="Aggregation method across adjusters")
    parser.add_argument("--top_n", type=int, default=None, help="Keep only top N genes")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum score threshold")
    args = parser.parse_args()
    main(args)