#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import traceback
import sys
from glob import glob

def log(msg):
    print(msg, flush=True)

def load_matrix(path):
    log(f"Loading gene × adjuster matrix: {path}")
    try:
        df = pd.read_csv(path, index_col=0)
        log(f"Matrix shape: {df.shape}")
        log(f"Columns (adjusters): {list(df.columns)}")
        log(f"First 5 rows:\n{df.head()}")
        return df
    except Exception as e:
        log(f"ERROR loading {path}: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

def load_pvalues(paths):
    """Load target-specific p-value files → gene × adjuster matrix"""
    dfs = []

    for path in paths:
        # extract adjuster name from filename
        fname = os.path.basename(path)
        adjuster = fname.split("-")[0]

        df = pd.read_csv(path, index_col=0)

        if df.shape[1] != 1:
            raise ValueError(f"Expected 1 column in {path}, got {df.shape[1]}")

        df.columns = [adjuster]
        dfs.append(df)

    combined = pd.concat(dfs, axis=1)

    log(f"P-value matrix shape: {combined.shape}")
    log(f"Adjusters: {list(combined.columns)}")

    return combined

def filter_significant(matrix, pval_df, alpha=0.05):
    """Keep genes significant in at least one adjuster"""

    # Align genes
    common_genes = matrix.index.intersection(pval_df.index)
    matrix = matrix.loc[common_genes]
    pval_df = pval_df.loc[common_genes]

    # Keep genes with ANY significant p-value
    sig_mask = (pval_df < alpha).any(axis=1)

    filtered = matrix.loc[sig_mask]

    log(f"Genes before: {len(matrix)}")
    log(f"Genes after filtering (p<{alpha}): {len(filtered)}")

    return filtered

def plot_heatmap(matrix, outfile, target=None):
    if matrix.empty:
        raise ValueError("Matrix is empty. Nothing to plot.")

    plt.figure(figsize=(10, max(6, 0.25 * matrix.shape[0])))
    try:
        sns.heatmap(
            matrix,
            cmap="viridis",
            cbar_kws={"label": target or "score"},
            mask=matrix.isna(),
        )
        title = f"Heatmap for {target}" if target else "Heatmap"
        plt.title(title)
        plt.xlabel("Adjuster")
        plt.ylabel("Genes")
        plt.tight_layout()

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300)
        plt.close()
        log(f"Saved heatmap to {outfile}")
    except Exception as e:
        log(f"ERROR plotting heatmap: {e}")
        traceback.print_exc(file=sys.stdout)
        raise

def main():
    parser = argparse.ArgumentParser(description="Plot heatmap for top genes per adjuster with optional p-value filtering.")
    parser.add_argument("--input", required=True, help="Input CSV file (gene × adjuster)")
    parser.add_argument("--pvals", nargs="+", required=False, help="Optional p-value CSV files (one per adjuster)")
    parser.add_argument("--outfile", required=True, help="Output heatmap file")
    parser.add_argument("--target", required=False, help="Optional target label for colorbar/title")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for p-values")
    parser.add_argument("--top_pct", type=float, default=None, help="Optional: plot only top N percentile of gene scores")
    parser.add_argument("--top_n", type=int, default=None, help="Optional: number of top genes to plot based on p-value")
    args = parser.parse_args()

    try:
        # Load gene × adjuster matrix
        matrix = load_matrix(args.input)

        # Load p-values if provided
        pval_df = None
        if args.pvals:
            pval_df = load_pvalues(args.pvals)
            matrix = filter_significant(matrix, pval_df, alpha=args.alpha)

        #  Top-N filtering based on p-values
        if args.top_n and pval_df is not None:
            min_pvals = pval_df.min(axis=1)
            top_genes = min_pvals.nsmallest(args.top_n).index
            matrix = matrix.loc[top_genes]

        # Top-percent filtering based on gene scores
        if args.top_pct:
            gene_scores = matrix.abs().sum(axis=1)
            threshold = np.percentile(gene_scores, 100 - args.top_pct)
            top_genes = gene_scores[gene_scores >= threshold].index
            matrix = matrix.loc[top_genes]

        # Plot
        if not matrix.empty:
            plot_heatmap(matrix, args.outfile, target=args.target)
        else:
            log("No genes to plot after filtering.")

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

if __name__ == "__main__":
    main()