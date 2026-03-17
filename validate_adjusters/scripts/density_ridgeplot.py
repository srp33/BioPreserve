#!/usr/bin/env python3
"""
Two-facet ridge/density plot showing ESR1 by ER status and ERBB2 by HER2 status
across all adjustment methods, with train (background) and test (foreground) overlaid.
"""

import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path
from collections import OrderedDict


# Adjusters: display_name -> (train_csv, test_csv)
# Paths are relative to the validate_adjusters directory.
ADJUSTERS = OrderedDict([
    ("Unadjusted", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv",
    )),
    ("Log-ComBat", (
        "adjusters/adjusted_log_combat_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/adjusted_log_combat_2studies_test_metabric_test_selected_genes.csv",
    )),
    ("Bayesian (Shift+Scale)", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv",
    )),
    ("Bayesian (Eff. Shift-Only)", (
        "adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv",
    )),
    ("DBA: Logistic (Shift+Scale)", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "outputs/dba_classifier_comparison_shift_scale/test_genes_dba_logistic.csv",
    )),
    ("DBA: Elasticnet (Shift+Scale)", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "outputs/dba_classifier_comparison_shift_scale/test_genes_dba_elasticnet.csv",
    )),
    ("DBA: Histgradient (Shift+Scale)", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "outputs/dba_classifier_comparison_shift_scale/test_genes_dba_histgradient.csv",
    )),
    ("DBA: Logistic (Shift-Only)", (
        "adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv",
        "outputs/dba_classifier_comparison_shift_only/test_genes_dba_logistic.csv",
    )),
    ("Min-Mean", (
        "adjusters/adjusted_min_mean_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/adjusted_min_mean_2studies_test_metabric_test_selected_genes.csv",
    )),
    ("GMM", (
        "adjusters/adjusted_gmm_2studies_test_metabric_train_selected_genes.csv",
        "adjusters/adjusted_gmm_2studies_test_metabric_test_selected_genes.csv",
    )),
])


def compute_density(values, grid):
    """Compute KDE density on a grid, handling edge cases."""
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.std(values) < 1e-10:
        return np.zeros_like(grid)
    try:
        kde = gaussian_kde(values, bw_method=0.3)
        return kde(grid)
    except Exception:
        return np.zeros_like(grid)


def draw_density_row(ax, gene_values_train, gene_values_test,
                     status_train, status_test, grid, row_y,
                     height=0.8, colors=None):
    """
    Draw overlaid train (background, muted) and test (foreground, vivid)
    density curves for a single adjuster row, split by status (0/1).
    """
    if colors is None:
        colors = {0: "#3b82f6", 1: "#ef4444"}  # blue / red

    for status_val, color in colors.items():
        # Train: background, muted
        mask_tr = status_train == status_val
        vals_tr = gene_values_train[mask_tr]
        d_tr = compute_density(vals_tr, grid)
        if d_tr.max() > 0:
            d_tr = d_tr / d_tr.max() * height * 0.45
        ax.fill_between(grid, row_y, row_y + d_tr, alpha=0.20, color=color, linewidth=0)
        ax.plot(grid, row_y + d_tr, color=color, alpha=0.35, linewidth=0.6)

        # Test: foreground, vivid
        mask_te = status_test == status_val
        vals_te = gene_values_test[mask_te]
        d_te = compute_density(vals_te, grid)
        if d_te.max() > 0:
            d_te = d_te / d_te.max() * height * 0.45
        ax.fill_between(grid, row_y, row_y + d_te, alpha=0.55, color=color, linewidth=0)
        ax.plot(grid, row_y + d_te, color=color, alpha=0.9, linewidth=0.9)


def main():
    parser = argparse.ArgumentParser(description="Density ridge plot for ESR1/ERBB2")
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--genes", nargs="+",
                        default=["ESR1", "ERBB2", "GREB1", "PGAP3", "KRT19", "CDH1"],
                        help="Genes to plot")
    parser.add_argument("--color-by", nargs="+",
                        default=["meta_er_status", "meta_her2_status",
                                 "meta_er_status", "meta_her2_status",
                                 "meta_er_status", "meta_er_status"],
                        help="Metadata column to color each gene facet by")
    args = parser.parse_args()

    meta_train = pl.read_csv(args.train_metadata)
    meta_test = pl.read_csv(args.test_metadata)

    genes = args.genes
    color_by = args.color_by
    # Pad color_by to match genes if needed
    while len(color_by) < len(genes):
        color_by.append("meta_er_status")

    # Load status arrays per metadata column
    status_arrays = {}
    for col in set(color_by):
        status_arrays[col] = (
            meta_train[col].to_numpy().astype(float),
            meta_test[col].to_numpy().astype(float),
        )

    # Collect all values per gene to set shared x-axis range
    all_vals = {g: [] for g in genes}

    adjuster_data = OrderedDict()
    for name, (train_path, test_path) in ADJUSTERS.items():
        try:
            df_tr = pl.read_csv(train_path)
            df_te = pl.read_csv(test_path)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        gene_data = {}
        skip = False
        for g in genes:
            if g not in df_tr.columns or g not in df_te.columns:
                print(f"  Skipping {name}: missing {g}")
                skip = True
                break
            tr = df_tr[g].to_numpy().astype(float)
            te = df_te[g].to_numpy().astype(float)
            gene_data[g] = (tr, te)
            all_vals[g].extend([tr, te])
        if skip:
            continue
        adjuster_data[name] = gene_data

    n_adj = len(adjuster_data)
    n_genes = len(genes)
    if n_adj == 0:
        print("No adjusters loaded.")
        return

    # Grids per gene
    grids = {}
    for g in genes:
        combined = np.concatenate(all_vals[g])
        p1, p99 = np.nanpercentile(combined, [1, 99])
        grids[g] = np.linspace(p1 - 0.5, p99 + 0.5, 300)

    # Plot
    fig, axes = plt.subplots(1, n_genes, figsize=(5.5 * n_genes, n_adj * 0.9 + 2),
                              sharey=True)
    if n_genes == 1:
        axes = [axes]

    names = list(adjuster_data.keys())
    row_height = 1.0

    # Nice label for metadata column
    meta_label = {
        "meta_er_status": "ER status",
        "meta_her2_status": "HER2 status",
        "meta_menopause_status": "Menopause",
        "meta_chemotherapy": "Chemo",
    }

    for gi, gene in enumerate(genes):
        ax = axes[gi]
        col = color_by[gi]
        st_train, st_test = status_arrays[col]

        for i, name in enumerate(names):
            tr, te = adjuster_data[name][gene]
            row_y = (n_adj - 1 - i) * row_height
            draw_density_row(ax, tr, te, st_train, st_test,
                             grids[gene], row_y, height=row_height,
                             colors={0: "#3b82f6", 1: "#ef4444"})

        ax.set_xlabel(f"{gene} expression", fontsize=10)
        ax.set_title(f"{gene} by {meta_label.get(col, col)}", fontsize=12, fontweight="bold")

    # Y-axis labels on leftmost
    yticks = [(n_adj - 1 - i) * row_height + row_height * 0.25 for i in range(n_adj)]
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_ylim(-0.1, n_adj * row_height + 0.1)
        for i in range(n_adj):
            y = (n_adj - 1 - i) * row_height
            ax.axhline(y, color="#e0e0e0", linewidth=0.4, zorder=0)
    axes[0].set_yticklabels(names, fontsize=9)

    # Legend
    legend_elements = [
        Patch(facecolor="#ef4444", alpha=0.55, label="Positive (test)"),
        Patch(facecolor="#ef4444", alpha=0.20, label="Positive (train)"),
        Patch(facecolor="#3b82f6", alpha=0.55, label="Negative (test)"),
        Patch(facecolor="#3b82f6", alpha=0.20, label="Negative (train)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Gene Expression Density by Adjustment Method",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
