#!/usr/bin/env python3
"""
Metadata Exploration Script

For a given metadata column (or all columns), identifies the top N most
ANOVA-associated genes using the full combined dataset, then produces
per-gene images with one subplot per dataset:
  - Density plots (categorical metadata): gene expression colored by category
  - Scatter plots (continuous metadata): gene expression vs metadata value

Output structure:
    output_dir/
      meta_er_status/
        ESR1.png        <- subplots for each dataset
        GATA3.png
        ...

Usage:
    # Single column (for Snakemake parallelism):
    python explore_metadata.py --input data.parquet --output-dir out --column meta_er_status

    # All columns (standalone):
    python explore_metadata.py --input data.parquet --output-dir out
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, gaussian_kde

warnings.filterwarnings("ignore", category=RuntimeWarning)

MAX_CATEGORICAL_UNIQUE = 20


def print_now(*args, **kwargs):
    print(*args, flush=True, **kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path):
    """Load Parquet or CSV and split into gene expression + metadata DataFrames."""
    print_now(f"Loading {path} ...")
    if path.suffix == ".parquet":
        df_pl = pl.read_parquet(path)
    else:
        df_pl = pl.read_csv(path, infer_schema_length=10000)

    meta_cols = [c for c in df_pl.columns if c.startswith("meta_")]
    gene_cols = [c for c in df_pl.columns if not c.startswith("meta_")]

    meta_df = df_pl.select(meta_cols).to_pandas()
    gene_df = df_pl.select(gene_cols).to_pandas().apply(pd.to_numeric, errors="coerce")
    print_now(f"Loaded {len(gene_df)} samples, {len(gene_cols)} genes, {len(meta_cols)} meta cols")
    return gene_df, meta_df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_column(meta_df: pd.DataFrame, col: str) -> str:
    """Classify a single metadata column as 'categorical', 'continuous', or 'skip'."""
    if col in {"meta_source", "meta_Sample_ID"}:
        return "skip"
    series = meta_df[col].dropna()
    if series.empty:
        return "skip"
    numeric = pd.to_numeric(series, errors="coerce")
    frac_numeric = numeric.notna().sum() / len(series)
    n_unique = series.nunique()

    if frac_numeric > 0.9 and n_unique > MAX_CATEGORICAL_UNIQUE:
        return "continuous"
    if n_unique < 2:
        return "skip"
    return "categorical"


# ---------------------------------------------------------------------------
# Vectorized gene ranking
# ---------------------------------------------------------------------------

def top_genes_anova(gene_df: pd.DataFrame, groups: pd.Series, top_n: int = 6):
    """Top genes by one-way ANOVA F-statistic (vectorized numpy)."""
    mask = groups.notna()
    groups = groups[mask]
    genes = gene_df.loc[mask]

    unique_groups = groups.unique()
    if len(unique_groups) < 2:
        return []

    group_codes = pd.Categorical(groups).codes
    n_groups = len(unique_groups)
    X = genes.values.astype(np.float64)
    n_samples, n_genes = X.shape
    grand_mean = np.nanmean(X, axis=0)

    ss_between = np.zeros(n_genes)
    ss_within = np.zeros(n_genes)
    valid_groups = 0
    total_n = 0

    for g in range(n_groups):
        mask_g = group_codes == g
        n_g = mask_g.sum()
        if n_g < 2:
            continue
        group_data = X[mask_g]
        group_mean = np.nanmean(group_data, axis=0)
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += np.nansum((group_data - group_mean) ** 2, axis=0)
        valid_groups += 1
        total_n += n_g

    df_between = valid_groups - 1
    df_within = total_n - valid_groups
    if df_between < 1 or df_within < 1:
        return []

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    with np.errstate(divide="ignore", invalid="ignore"):
        f_stats = np.where(ms_within > 0, ms_between / ms_within, 0.0)

    top_idx = np.argsort(f_stats)[::-1][:top_n]
    return [genes.columns[i] for i in top_idx if np.isfinite(f_stats[i])]


def top_genes_correlation(gene_df: pd.DataFrame, values: pd.Series, top_n: int = 6):
    """Top genes by absolute Pearson correlation (vectorized numpy)."""
    mask = values.notna()
    values = values[mask].astype(float)
    genes = gene_df.loc[mask]

    X = genes.values.astype(np.float64)
    y = values.values
    row_valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[row_valid]
    y = y[row_valid]

    if len(y) < 10:
        return []

    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)
    cov_xy = (X_centered * y_centered[:, np.newaxis]).mean(axis=0)
    std_x = X_centered.std(axis=0)
    std_y = y_centered.std()

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(std_x * std_y > 0, cov_xy / (std_x * std_y), 0.0)

    abs_r = np.abs(r)
    top_idx = np.argsort(abs_r)[::-1][:top_n]
    return [genes.columns[i] for i in top_idx if np.isfinite(abs_r[i])]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
]


def _safe_kde(values, grid):
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.std(values) < 1e-10:
        return np.zeros_like(grid)
    try:
        return gaussian_kde(values, bw_method=0.3)(grid)
    except Exception:
        return np.zeros_like(grid)


def _global_color_map(all_unique_vals):
    vals = sorted(all_unique_vals, key=str)
    if "NA" in vals:
        vals = [v for v in vals if v != "NA"] + ["NA"]
    colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(vals)}
    if "NA" in colors:
        colors["NA"] = "#9ca3af"
    return vals, colors


def plot_density_per_gene(gene_name, gene_df, groups_by_ds, meta_col,
                          ordered_vals, colors, output_path):
    """One image per gene: subplots = datasets, density colored by category."""
    ds_names = list(groups_by_ds.keys())
    n_ds = len(ds_names)
    n_cols = min(n_ds, 3)
    n_rows = (n_ds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for i in range(n_ds, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for di, ds in enumerate(ds_names):
        ax = axes_flat[di]
        ds_groups = groups_by_ds[ds]["groups"]
        ds_gene_vals = groups_by_ds[ds]["genes"][gene_name]
        total_count = len(ds_groups)
        group_counts = {v: int((ds_groups == v).sum()) for v in ordered_vals}

        all_valid = ds_gene_vals.dropna()
        if all_valid.empty:
            ax.set_title(ds, fontsize=11, fontweight="bold")
            continue
        p1, p99 = np.nanpercentile(all_valid, [1, 99])
        margin = (p99 - p1) * 0.1
        grid = np.linspace(p1 - margin, p99 + margin, 300)

        for gval in ordered_vals:
            n = group_counts[gval]
            if n == 0:
                continue
            mask = ds_groups == gval
            subset = ds_gene_vals[mask].dropna().values
            density = _safe_kde(subset, grid)
            weight = n / total_count
            scaled = density * weight
            ax.fill_between(grid, scaled, alpha=0.15, color=colors[gval])
            ax.plot(grid, scaled, color=colors[gval], linewidth=1.5, alpha=0.85)

        ax.set_xlabel(f"{gene_name} expression", fontsize=10)
        ax.set_ylabel("Weighted density", fontsize=10)
        ax.set_title(f"{ds}  (n={total_count})", fontsize=11, fontweight="bold")

    # Shared legend across all subplots
    # Collect counts across all datasets for legend
    all_counts = {v: 0 for v in ordered_vals}
    for ds in ds_names:
        ds_groups = groups_by_ds[ds]["groups"]
        for v in ordered_vals:
            all_counts[v] += int((ds_groups == v).sum())

    handles = [
        Line2D([0], [0], color=colors[v], lw=3,
               label=f"{v}  (n={all_counts[v]})")
        for v in ordered_vals if all_counts[v] > 0
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 6), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{gene_name} — {meta_col}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_per_gene(gene_name, gene_df, values_by_ds, meta_col,
                          output_path):
    """One image per gene: subplots = datasets, scatter of gene vs continuous meta."""
    ds_names = list(values_by_ds.keys())
    n_ds = len(ds_names)
    n_cols = min(n_ds, 3)
    n_rows = (n_ds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for i in range(n_ds, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for di, ds in enumerate(ds_names):
        ax = axes_flat[di]
        ds_values = values_by_ds[ds]["values"]
        ds_gene_vals = values_by_ds[ds]["genes"][gene_name]
        common = ds_gene_vals.dropna().index.intersection(ds_values.index)
        if len(common) < 10:
            ax.set_title(ds, fontsize=11, fontweight="bold")
            continue
        x = ds_values[common].values.astype(float)
        y = ds_gene_vals[common].values.astype(float)

        ax.scatter(x, y, s=6, alpha=0.4, color="#3b82f6", edgecolors="none")
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, p(xs), color="#ef4444", linewidth=1.5, linestyle="--")
            r, _ = pearsonr(x, y)
            ax.set_title(f"{ds}  (r={r:.3f}, n={len(common)})",
                         fontsize=11, fontweight="bold")
        except Exception:
            ax.set_title(f"{ds}  (n={len(common)})", fontsize=11, fontweight="bold")

        ax.set_xlabel(meta_col, fontsize=10)
        ax.set_ylabel(f"{gene_name} expression", fontsize=10)

    fig.suptitle(f"{gene_name} — {meta_col}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Process a single metadata column
# ---------------------------------------------------------------------------

def process_column(col: str, gene_df, meta_df, datasets, out: Path, top_n: int):
    """Run ANOVA/correlation + per-gene images for one metadata column."""
    col_type = classify_column(meta_df, col)
    if col_type == "skip":
        print_now(f"  [{col}] Skipped (constant, empty, or identifier)")
        return 0

    safe_col = col.replace("/", "_")
    count = 0

    if col_type == "categorical":
        global_groups = meta_df[col].astype(str).replace(
            {"nan": "NA", "None": "NA", "": "NA"}
        )
        groups_no_na = global_groups.replace("NA", np.nan)
        top = top_genes_anova(gene_df, groups_no_na, top_n=top_n)
        if not top:
            print_now(f"  [{col}] Skipped (no valid ANOVA results)")
            return 0
        print_now(f"  [{col}] categorical — top genes: {top}")

        all_vals = set(global_groups.unique())
        ordered_vals, colors = _global_color_map(all_vals)

        # Build per-dataset data for datasets that have >=2 categories
        groups_by_ds = {}
        for ds in datasets:
            ds_mask = meta_df["meta_source"] == ds
            ds_groups = global_groups[ds_mask].reset_index(drop=True)
            ds_genes = gene_df.loc[ds_mask].reset_index(drop=True)
            if ds_groups.nunique() < 2:
                continue
            groups_by_ds[ds] = {"groups": ds_groups, "genes": ds_genes}

        if not groups_by_ds:
            return 0

        # One image per gene
        for gene in top:
            safe_gene = gene.replace("/", "_")
            out_path = out / safe_col / f"{safe_gene}.png"
            plot_density_per_gene(gene, gene_df, groups_by_ds, col,
                                 ordered_vals, colors, out_path)
            count += 1

    else:  # continuous
        global_values = pd.to_numeric(meta_df[col], errors="coerce")
        top = top_genes_correlation(gene_df, global_values, top_n=top_n)
        if not top:
            print_now(f"  [{col}] Skipped (no valid correlation results)")
            return 0
        print_now(f"  [{col}] continuous — top genes: {top}")

        # Build per-dataset data for datasets that have >=10 non-NA values
        values_by_ds = {}
        for ds in datasets:
            ds_mask = meta_df["meta_source"] == ds
            ds_values = global_values[ds_mask].dropna()
            ds_genes = gene_df.loc[ds_mask]
            if len(ds_values) < 10:
                continue
            values_by_ds[ds] = {"values": ds_values, "genes": ds_genes}

        if not values_by_ds:
            return 0

        # One image per gene
        for gene in top:
            safe_gene = gene.replace("/", "_")
            out_path = out / safe_col / f"{safe_gene}.png"
            plot_scatter_per_gene(gene, gene_df, values_by_ds, col, out_path)
            count += 1

    print_now(f"  [{col}] Generated {count} images")
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explore metadata–gene associations")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to .parquet or .csv combined data")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for output plots")
    parser.add_argument("--top-n", type=int, default=6,
                        help="Number of top genes per metadata variable")
    parser.add_argument("--column", type=str, default=None,
                        help="Process a single metadata column (for parallel Snakemake jobs)")
    args = parser.parse_args()

    gene_df, meta_df = load_data(args.input)

    if "meta_source" not in meta_df.columns:
        print_now("ERROR: meta_source column not found")
        return 1

    datasets = sorted(meta_df["meta_source"].dropna().unique())

    if args.column:
        # Single-column mode (called by Snakemake per-column jobs)
        if args.column not in meta_df.columns:
            print_now(f"ERROR: column {args.column} not found in data")
            return 1
        n = process_column(args.column, gene_df, meta_df, datasets,
                           args.output_dir, args.top_n)
        print_now(f"Done. {n} images for {args.column}")
    else:
        # All-columns mode (standalone)
        meta_cols = [c for c in meta_df.columns
                     if c.startswith("meta_") and c not in {"meta_source", "meta_Sample_ID"}]
        total = 0
        for col in meta_cols:
            total += process_column(col, gene_df, meta_df, datasets,
                                    args.output_dir, args.top_n)
        print_now(f"\nDone. {total} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
