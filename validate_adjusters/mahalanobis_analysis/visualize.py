#!/usr/bin/env python3
"""
Compare multiple adjusters simultaneously against Bayesian Shift-Scale.

Automatically selects visualization style:
  - ≤4 comparators : three-panel grouped bar chart
      Panel 1 — Mahalanobis distances
      Panel 2 — Scale (α) per gene
      Panel 3 — Shift (β) per gene
  - >4 comparators : three-panel heatmap (distances, scales, shifts)

Genes are sorted by mean Mahalanobis distance across comparators (descending).
A summary CSV is also written alongside the figure.

Usage:
    python visualize_multi.py \\
        --combined results/all_adjusters_gene_ranking.csv \\
        --output figures/multi_comparison.png

    # Restrict to a subset of adjusters:
    python visualize_multi.py \\
        --combined results/all_adjusters_gene_ranking.csv \\
        --adjusters gmm combat \\
        --output figures/multi_comparison.png

    # Override the grouped-vs-heatmap threshold (default: 4):
    python visualize_multi.py \\
        --combined results/all_adjusters_gene_ranking.csv \\
        --grouped-threshold 6 \\
        --output figures/multi_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BAYESIAN_KEY   = "bayesian"
BAYESIAN_LABEL = "Bayesian Shift-Scale"
GROUPED_THRESHOLD = 4   # use grouped bars when n_comparators <= this
BAR_WIDTH_BASE    = 0.8  # total width allocated per gene position


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(combined_path: str, adjusters: list[str] | None) -> pl.DataFrame:
    """Load long-form CSV, optionally filtering to a subset of comparators.

    The Bayesian reference is always kept.  Raises if a requested adjuster is
    absent from the file.
    """
    df = pl.read_csv(combined_path)
    present = set(df["adjuster"].unique().to_list())

    if adjusters:
        missing = [a for a in adjusters if a not in present]
        if missing:
            raise ValueError(
                f"Adjuster(s) not found in {combined_path}: {missing}. "
                f"Available: {sorted(present)}"
            )
        df = df.filter(pl.col("adjuster").is_in(set(adjusters) | {BAYESIAN_KEY}))

    return df


def pivot_metric(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Pivot long-form df → gene × adjuster for a single metric."""
    return df.pivot(
        values=metric, index="gene", on="adjuster", aggregate_function="first"
    )


def sort_genes_by_mean_distance(
    df: pl.DataFrame, comparators: list[str], top_n: int | None = None
) -> list[str]:
    """Return genes sorted by mean Mahalanobis distance across comparators."""
    dist_wide = pivot_metric(df, "mahalanobis_distance")
    comp_cols = [c for c in comparators if c in dist_wide.columns]
    genes = (
        dist_wide
        .with_columns(
            pl.mean_horizontal([pl.col(c) for c in comp_cols]).alias("_mean_dist")
        )
        .sort("_mean_dist", descending=True)
        ["gene"]
        .to_list()
    )
    if top_n is not None:
        genes = genes[:top_n]
    return genes


# ---------------------------------------------------------------------------
# Grouped bar chart (≤4 comparators)
# ---------------------------------------------------------------------------

def make_figure_grouped(df: pl.DataFrame, comparators: list[str], top_n: int | None = None) -> plt.Figure:
    """Three-panel grouped bar chart."""
    genes = sort_genes_by_mean_distance(df, comparators, top_n)
    n     = len(genes)
    x     = np.arange(n)

    all_adjusters  = [BAYESIAN_KEY] + comparators
    n_bars_params  = len(all_adjusters)          # bayesian + comparators
    n_bars_dist    = len(comparators)            # distance panel skips bayesian
    bar_width      = BAR_WIDTH_BASE / n_bars_params  # consistent width for all panels

    # Color map: bayesian = blue, comparators cycle through tab10 (skip index 0)
    palette = plt.cm.tab10.colors
    colors  = {BAYESIAN_KEY: "#1f77b4"}
    for i, name in enumerate(comparators):
        colors[name] = palette[(i + 1) % len(palette)]

    dist_wide  = pivot_metric(df, "mahalanobis_distance")
    scale_wide = pivot_metric(df, "scale")
    shift_wide = pivot_metric(df, "shift")

    def _get_vals(wide: pl.DataFrame, adjuster: str) -> np.ndarray:
        gene_idx = {g: i for i, g in enumerate(wide["gene"].to_list())}
        indices  = [gene_idx[g] for g in genes]
        if adjuster not in wide.columns:
            return np.full(n, np.nan)
        return wide[adjuster].to_numpy()[indices]

    fig_width = max(14, n * 0.45)
    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_width, 12),
        sharex=True,
        height_ratios=[1.25, 1.0, 1.0],
    )
    fig.suptitle(
        f"Adjuster Comparison — Mahalanobis Distance to {BAYESIAN_LABEL}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # -- Panel 1: Mahalanobis distances (comparators only) ------------------
    ax0 = axes[0]
    for i, name in enumerate(comparators):
        offset = (i - (n_bars_dist - 1) / 2) * bar_width
        ax0.bar(
            x + offset, _get_vals(dist_wide, name), bar_width,
            label=name, color=colors[name], zorder=3,
        )
    ax0.set_ylabel("Distance", fontsize=10)
    ax0.set_title("Mahalanobis Distance to Bayesian Shift-Scale", fontsize=11)
    ax0.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)
    ax0.legend(fontsize=8, loc="upper right", ncols=n_bars_dist)
    ax0.set_ylim(bottom=0)

    # -- Panels 2 & 3: scale and shift (all adjusters incl. bayesian) -------
    def _param_panel(ax, wide, ylabel, title, ref_line, baseline):
        for i, name in enumerate(all_adjusters):
            offset  = (i - (n_bars_params - 1) / 2) * bar_width
            heights = _get_vals(wide, name) - baseline
            ax.bar(
                x + offset, heights, bar_width,
                bottom=baseline, label=name, color=colors[name], zorder=3,
            )
        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)
        ax.legend(fontsize=8, loc="upper right", ncols=n_bars_params)

    _param_panel(axes[1], scale_wide,
                 ylabel="Scale (α)", title="Relative Scale per Gene",
                 ref_line=1.0, baseline=1.0)
    _param_panel(axes[2], shift_wide,
                 ylabel="Shift (β)", title="Relative Shift per Gene",
                 ref_line=0.0, baseline=0.0)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(genes, rotation=90, fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)
    axes[2].set_xlabel(
        "Genes (sorted by mean Mahalanobis distance, descending)", fontsize=10
    )
    axes[0].set_xlim(-0.75, n - 0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap (>4 comparators)
# ---------------------------------------------------------------------------

def make_figure_heatmap(df: pl.DataFrame, comparators: list[str], top_n: int | None = None) -> plt.Figure:
    """Three-panel heatmap: distances (comparators only), scale, shift."""
    genes   = sort_genes_by_mean_distance(df, comparators, top_n)
    n_genes = len(genes)

    dist_wide  = pivot_metric(df, "mahalanobis_distance")
    scale_wide = pivot_metric(df, "scale")
    shift_wide = pivot_metric(df, "shift")

    def _build_matrix(wide: pl.DataFrame, adj_list: list[str]) -> np.ndarray:
        gene_idx = {g: i for i, g in enumerate(wide["gene"].to_list())}
        col_idx  = [gene_idx[g] for g in genes]
        mat      = np.full((len(adj_list), n_genes), np.nan)
        for row_i, name in enumerate(adj_list):
            if name in wide.columns:
                mat[row_i] = wide[name].to_numpy()[col_idx]
        return mat

    all_adjusters = [BAYESIAN_KEY] + comparators
    dist_mat  = _build_matrix(dist_wide,  comparators)
    scale_mat = _build_matrix(scale_wide, all_adjusters)
    shift_mat = _build_matrix(shift_wide, all_adjusters)

    comp_labels    = comparators
    all_adj_labels = [BAYESIAN_LABEL if a == BAYESIAN_KEY else a
                      for a in all_adjusters]

    show_gene_labels = n_genes <= 60

    fig_width  = max(14, n_genes * 0.22)
    n_rows_dist   = len(comparators)
    n_rows_params = len(all_adjusters)
    fig_height = max(8, (n_rows_dist + 2 * n_rows_params) * 0.55 + 2)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [n_rows_dist, n_rows_params, n_rows_params]},
    )
    fig.suptitle(
        f"Adjuster Comparison — Mahalanobis Distance to {BAYESIAN_LABEL}",
        fontsize=13, fontweight="bold",
    )

    def _heatmap(ax, mat, row_labels, cmap, title, show_x=False):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        if show_x and show_gene_labels:
            ax.set_xticks(range(n_genes))
            ax.set_xticklabels(genes, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    _heatmap(axes[0], dist_mat, comp_labels, "YlOrRd",
             "Mahalanobis Distance to Bayesian Shift-Scale")
    _heatmap(axes[1], scale_mat, all_adj_labels, "RdBu_r",
             "Relative Scale (α) per Gene")
    _heatmap(axes[2], shift_mat, all_adj_labels, "RdBu_r",
             "Relative Shift (β) per Gene", show_x=True)

    axes[2].set_xlabel(
        "Genes (sorted by mean Mahalanobis distance, descending)", fontsize=10
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple adjusters simultaneously.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--combined", required=True,
        help="Path to all_adjusters_gene_ranking.csv",
    )
    parser.add_argument(
        "--adjusters", nargs="+", default=None,
        help=(
            "Subset of non-Bayesian adjusters to include. "
            "If omitted, all adjusters in the file are used."
        ),
    )
    parser.add_argument(
        "--output", required=True,
        help="Output figure path (PNG recommended).",
    )
    parser.add_argument(
        "--grouped-threshold", type=int, default=GROUPED_THRESHOLD,
        metavar="N",
        help=(
            f"Use grouped bars when n_comparators ≤ N, heatmap otherwise "
            f"(default: {GROUPED_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        metavar="N",
        help="Show only the top N genes by mean Mahalanobis distance. Defaults to all genes.",
    )
    args = parser.parse_args()

    df = load_data(args.combined, args.adjusters)

    all_present = sorted(df["adjuster"].unique().to_list())
    comparators = [a for a in all_present if a != BAYESIAN_KEY]
    if not comparators:
        raise SystemExit("No non-Bayesian adjusters found in the data. Nothing to compare.")

    print(f"Adjusters:  {comparators}")
    print(f"Genes:      {df['gene'].n_unique()}")
    if args.top_n is not None:
        print(f"Top-N:      {args.top_n}")

    use_heatmap = len(comparators) > args.grouped_threshold
    print(f"Style:      {'heatmap' if use_heatmap else 'grouped bars'}")

    fig = (
        make_figure_heatmap(df, comparators, args.top_n)
        if use_heatmap
        else make_figure_grouped(df, comparators, args.top_n)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


if __name__ == "__main__":
    main()
