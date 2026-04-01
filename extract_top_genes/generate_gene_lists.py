#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os

# -------------------------
# Load CSVs
# -------------------------
def load_csvs(paths):
    """
    Load t-stat CSVs (Gene × metadata) for multiple adjusters.
    Gene column becomes index. Metadata columns starting with 'meta_' are kept as-is.
    """
    dfs = {}
    for p in paths:
        adjuster = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p, index_col=0)  # Gene column as index
        # Convert only numeric columns (exclude metadata)
        gene_cols = [c for c in df.columns if not c.startswith("meta_")]
        df[gene_cols] = df[gene_cols].apply(pd.to_numeric, errors="coerce")
        dfs[adjuster] = df
    return dfs

# -------------------------
# Transform scores (optional)
# -------------------------
def transform_scores(df, mode="tstat"):
    if mode == "tstat" or mode == "slope":
        return df
    elif mode == "abs_slope":
        return df.abs()
    elif mode == "pval":
        return -np.log10(df)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# -------------------------
# Combine adjusters per metadata target
# -------------------------
def combine_per_metadata(dfs, mode="tstat", target=None):
    """
    Combine multiple adjusters into gene × adjuster matrix for each metadata target.
    Missing genes per adjuster are filled with NaN.
    """
    # Determine metadata targets
    targets = [target] if target else [c for c in dfs[next(iter(dfs))].columns if c.startswith("meta_")]
    combined = {}

    for tgt in targets:
        # Collect all genes across adjusters
        all_genes = set()
        for df in dfs.values():
            all_genes.update(df.index)
        all_genes = sorted(all_genes)

        # Build empty DataFrame with genes as rows
        combined_df = pd.DataFrame(index=all_genes)

        for adj, df in dfs.items():
            # Take the column for this metadata target
            vals = transform_scores(df[[tgt]], mode=mode).squeeze()  # Series: index=genes
            # Align with all_genes, fill missing with NaN
            vals = vals.reindex(all_genes)
            combined_df[adj] = vals

        combined[tgt] = combined_df

    return combined

# -------------------------
# Save matrices to CSV
# -------------------------
def save_top_genes(dfs_per_target, outdir, threshold=None):
    os.makedirs(outdir, exist_ok=True)
    for target, df in dfs_per_target.items():
        if threshold is not None:
            df = df.applymap(lambda x: x if x > threshold else np.nan)
        out_path = os.path.join(outdir, f"{target}_gene_matrix.csv")
        df.to_csv(out_path)
        print(f"Saved gene matrix for '{target}': {out_path}")

# -------------------------
# Main
# -------------------------
def main(args):
    dfs = load_csvs(args.csvs)
    combined_per_target = combine_per_metadata(dfs, mode=args.mode, target=args.target)
    save_top_genes(combined_per_target, args.outdir, threshold=args.threshold)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gene × adjuster matrices from t-stat CSVs.")
    parser.add_argument("--csvs", nargs="+", required=True, help="Input t-stat CSV files")
    parser.add_argument("--outdir", required=True, help="Output directory for gene matrices")
    parser.add_argument("--target", required=False, help="Optional metadata target to process")
    parser.add_argument(
        "--mode",
        choices=["tstat", "slope", "abs_slope", "pval"],
        default="tstat",
        help="Transform mode for scores"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional: keep only scores above this threshold"
    )
    args = parser.parse_args()
    main(args)