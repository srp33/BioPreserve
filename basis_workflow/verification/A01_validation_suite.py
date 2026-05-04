#!/usr/bin/env python3
"""
Rule 7 — Validation Suite

Generate 4 diagnostic plots assessing cross-dataset biological consistency
of the gene community sets.

Outputs:
  plot1_subtypes.png  — ER-axis vs HER2-axis scatter, colored by subtype
  plot2_violins.png   — Top-5 axes violin plots by ER status × dataset
  plot3_purity.png    — A→B nearest-neighbor purity histograms (k=10)
  plot4_edges.png     — Stacked bar of intra-axis edge contribution
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# ssGSEA (inlined from 06_basis_align.py)
# ---------------------------------------------------------------------------

def ssgsea(expression_df, gene_sets, alpha=0.25):
    """Single-sample Gene Set Enrichment Analysis with hub weights.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Samples × genes expression matrix.
    gene_sets : dict
        {set_name: {gene: hub_weight, ...}}.
    alpha : float
        Weight exponent for rank positions (default 0.25).

    Returns
    -------
    pd.DataFrame
        Samples × pathway scores.
    """
    n_samples = len(expression_df)
    gene_names = expression_df.columns.tolist()
    n_genes = len(gene_names)

    scores = {}
    for set_name, gene_weights in gene_sets.items():
        idxs = []
        weights = []
        for gene, w in gene_weights.items():
            if gene in gene_names:
                idxs.append(gene_names.index(gene))
                weights.append(abs(w))

        if len(idxs) < 3:
            scores[set_name] = np.zeros(n_samples)
            continue

        idxs = np.array(idxs)
        weights = np.array(weights)

        set_scores = np.zeros(n_samples)
        for i in range(n_samples):
            expr = expression_df.iloc[i].values
            ranked_indices = np.argsort(-expr)

            in_set = np.zeros(n_genes, dtype=bool)
            in_set[idxs] = True
            in_set_ranked = in_set[ranked_indices]

            weight_map = np.zeros(n_genes)
            for idx, w in zip(idxs, weights):
                weight_map[idx] = w
            weight_ranked = weight_map[ranked_indices]

            n_in = in_set_ranked.sum()
            n_out = n_genes - n_in

            if n_in == 0 or n_out == 0:
                set_scores[i] = 0
                continue

            rank_positions = np.arange(n_genes, 0, -1)
            hits = np.where(
                in_set_ranked,
                np.abs(rank_positions) ** alpha * weight_ranked,
                0,
            )
            hits_sum = hits.sum()
            if hits_sum == 0:
                set_scores[i] = 0
                continue
            hits_norm = hits / hits_sum

            misses = np.where(~in_set_ranked, 1.0 / n_out, 0)

            running_sum = np.cumsum(hits_norm - misses)
            set_scores[i] = running_sum.max() - running_sum.min()

        scores[set_name] = set_scores

    return pd.DataFrame(scores, index=expression_df.index)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset_full(csv_path):
    """Load a CSV, return numeric gene columns and meta columns separately."""
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors='ignore')
    gene_df = gene_df.select_dtypes(include=[np.number])
    return gene_df, meta_df


def load_common_genes(path):
    """Load deduplicated common gene list from text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_dup_map(path):
    """Load duplicate gene map CSV. Returns set of duplicate gene names."""
    df = pd.read_csv(path)
    if len(df) == 0:
        return set()
    return set(df['duplicate'].values)


def prepare_dataset(csv_path, common_genes, dup_genes):
    """Load dataset, filter to common genes (dropping dups), log-transform.

    Returns gene expression DataFrame and meta DataFrame.
    """
    gene_df, meta_df = load_dataset_full(csv_path)
    # Filter to common genes that are not duplicates
    keep = [g for g in common_genes if g in gene_df.columns and g not in dup_genes]
    gene_df = gene_df[keep]
    # Log-transform
    log_df = np.log(gene_df - gene_df.min(axis=0) + 1.0)
    return log_df, meta_df


# ---------------------------------------------------------------------------
# Subtype helper
# ---------------------------------------------------------------------------

def get_subtype(er, her2):
    """Determine molecular subtype from ER and HER2 status."""
    if pd.isna(er) or pd.isna(her2):
        return 'Unknown'
    if er == 1 and her2 == 0:
        return 'ER+/HER2-'
    if er == 1 and her2 == 1:
        return 'ER+/HER2+'
    if er == 0 and her2 == 0:
        return 'ER-/HER2-'
    if er == 0 and her2 == 1:
        return 'ER-/HER2+'
    return 'Unknown'


# ---------------------------------------------------------------------------
# Axis identification helpers
# ---------------------------------------------------------------------------

def find_er_axis(gene_sets):
    """Find the axis containing ESR1, or fall back to first axis with 'ER' in name."""
    for name, genes in gene_sets.items():
        if 'ESR1' in genes:
            return name
    for name in gene_sets:
        if 'ER' in name.upper():
            return name
    # Fallback: first axis
    return list(gene_sets.keys())[0] if gene_sets else None


def find_her2_axis(gene_sets):
    """Find the axis containing ERBB2, or fall back to first axis with 'HER2' in name."""
    for name, genes in gene_sets.items():
        if 'ERBB2' in genes:
            return name
    for name in gene_sets:
        if 'HER2' in name.upper():
            return name
    # Fallback: second axis if available
    keys = list(gene_sets.keys())
    return keys[1] if len(keys) > 1 else keys[0] if keys else None


# ---------------------------------------------------------------------------
# Plot 1: Subtype scatter
# ---------------------------------------------------------------------------

def plot1_subtype_scatter(scores_a, meta_a, scores_b, meta_b,
                          er_axis, her2_axis, output_dir):
    """ER-axis vs HER2-axis scatter, colored by subtype, markers by dataset."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    color_map = {
        'ER+/HER2-': 'steelblue',
        'ER+/HER2+': 'darkorange',
        'ER-/HER2-': 'mediumseagreen',
        'ER-/HER2+': 'crimson',
        'Unknown': 'gray',
    }

    for scores, meta, marker, ds_name in [
        (scores_a, meta_a, 'o', 'Dataset A'),
        (scores_b, meta_b, 'x', 'Dataset B'),
    ]:
        er_col = 'meta_ER_status' if 'meta_ER_status' in meta.columns else 'meta_er_status'
        her2_col = 'meta_HER2_status' if 'meta_HER2_status' in meta.columns else 'meta_her2_status'
        er_vals = meta[er_col].values if er_col in meta.columns else [np.nan] * len(meta)
        her2_vals = meta[her2_col].values if her2_col in meta.columns else [np.nan] * len(meta)
        subtypes = [get_subtype(e, h) for e, h in zip(er_vals, her2_vals)]

        for subtype in ['ER+/HER2-', 'ER+/HER2+', 'ER-/HER2-', 'ER-/HER2+']:
            mask = [s == subtype for s in subtypes]
            if sum(mask) == 0:
                continue
            scatter_kw = dict(
                c=color_map[subtype], marker=marker, alpha=0.6, s=40,
                label=f'{ds_name} {subtype} (n={sum(mask)})',
            )
            # Only add edgecolors for filled markers (not 'x', '+', etc.)
            if marker not in ('x', '+', '*', '.', ','):
                scatter_kw['edgecolors'] = 'white'
                scatter_kw['linewidths'] = 0.3
            ax.scatter(
                scores[er_axis].values[mask],
                scores[her2_axis].values[mask],
                **scatter_kw,
            )

    ax.set_xlabel(f'{er_axis} (ER module)')
    ax.set_ylabel(f'{her2_axis} (HER2 amplicon)')
    ax.set_title('ssGSEA: ER vs HER2 axis — both datasets')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot1_subtypes.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Violin plots
# ---------------------------------------------------------------------------

def plot2_violins(scores_a, meta_a, scores_b, meta_b, top_axes, output_dir):
    """Top-5 axes violin plots by ER status × dataset."""
    n_axes = min(len(top_axes), 5)
    fig, axes_list = plt.subplots(1, n_axes, figsize=(4 * n_axes, 6))
    if n_axes == 1:
        axes_list = [axes_list]

    for idx, axis_name in enumerate(top_axes[:n_axes]):
        ax = axes_list[idx]
        data = []

        for scores, meta, ds in [(scores_a, meta_a, 'A'), (scores_b, meta_b, 'B')]:
            er_col = 'meta_ER_status' if 'meta_ER_status' in meta.columns else 'meta_er_status'
            if er_col not in meta.columns:
                continue
            for j in range(len(meta)):
                er = meta[er_col].iloc[j]
                if pd.isna(er):
                    continue
                label = f"{'ER+' if er == 1 else 'ER-'}\n{ds}"
                data.append({'group': label, 'score': scores[axis_name].iloc[j]})

        if not data:
            continue
        plot_df = pd.DataFrame(data)
        groups = sorted(plot_df['group'].unique())

        positions = list(range(len(groups)))
        for i, group in enumerate(groups):
            vals = plot_df[plot_df['group'] == group]['score'].values
            if len(vals) == 0:
                continue
            parts = ax.violinplot([vals], positions=[i], showmeans=True,
                                  showmedians=True)
            color = 'steelblue' if 'A' in group else 'coral'
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(axis_name, fontsize=10)
        ax.set_ylabel('ssGSEA score')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Pathway scores by ER status and dataset', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot2_violins.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Nearest-neighbor purity histograms
# ---------------------------------------------------------------------------

def plot3_neighbor_purity(scores_a, meta_a, scores_b, meta_b, k, output_dir):
    """A→B nearest-neighbor purity histograms for ER and HER2 status."""
    fig, axes_list = plt.subplots(1, 2, figsize=(14, 5))

    meta_pairs = [
        ('meta_ER_status', 'meta_er_status', 'ER status'),
        ('meta_HER2_status', 'meta_her2_status', 'HER2 status'),
    ]

    for col_idx, (col_upper, col_lower, meta_name) in enumerate(meta_pairs):
        ax = axes_list[col_idx]

        # Resolve column name (case-insensitive fallback)
        meta_col_a = col_upper if col_upper in meta_a.columns else col_lower
        meta_col_b = col_upper if col_upper in meta_b.columns else col_lower

        if meta_col_a not in meta_a.columns or meta_col_b not in meta_b.columns:
            ax.text(0.5, 0.5, f'{meta_name} not available',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        X_a = scores_a.values
        X_b = scores_b.values
        nn = NearestNeighbors(n_neighbors=k).fit(X_b)
        _, indices = nn.kneighbors(X_a)

        labels_a = meta_a[meta_col_a].values
        labels_b = meta_b[meta_col_b].values

        purities = []
        for i in range(len(X_a)):
            if pd.isna(labels_a[i]):
                continue
            neighbor_labels = labels_b[indices[i]]
            valid = ~pd.isna(neighbor_labels)
            if valid.sum() == 0:
                continue
            purity = (neighbor_labels[valid] == labels_a[i]).mean()
            purities.append(purity)

        # Random baseline: majority class fraction in B
        valid_b = labels_b[~pd.isna(labels_b)]
        if len(valid_b) > 0:
            majority_frac = max((valid_b == 0).mean(), (valid_b == 1).mean())
        else:
            majority_frac = 0.5

        ax.hist(purities, bins=20, alpha=0.7, color='steelblue', edgecolor='white',
                label=f'ssGSEA neighbors (mean={np.mean(purities):.2f})')
        ax.axvline(majority_frac, color='red', ls='--', alpha=0.7,
                   label=f'Random baseline ({majority_frac:.2f})')
        ax.set_xlabel(f'Neighbor purity ({meta_name})')
        ax.set_ylabel('Count')
        ax.set_title(f'A→B neighbor purity: {meta_name}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'Cross-dataset nearest neighbor purity (k={k})', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot3_purity.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Edge contribution stacked bar
# ---------------------------------------------------------------------------

def plot4_edge_contribution(edges_a_path, edges_b_path, gene_sets, output_dir):
    """Stacked bar of intra-axis edge contribution (both/A-only/B-only), top 20 axes."""
    try:
        ea = pd.read_csv(edges_a_path)
        eb = pd.read_csv(edges_b_path)
    except FileNotFoundError:
        print("Edge files not found, skipping plot 4")
        return

    edges_a_set = set(zip(ea['source'], ea['target']))
    edges_b_set = set(zip(eb['source'], eb['target']))

    records = []
    for axis_name, genes in gene_sets.items():
        gene_list = list(genes.keys())

        n_a_only = 0
        n_b_only = 0
        n_both = 0
        for g1 in gene_list:
            for g2 in gene_list:
                if g1 == g2:
                    continue
                in_a = (g1, g2) in edges_a_set
                in_b = (g1, g2) in edges_b_set
                if in_a and in_b:
                    n_both += 1
                elif in_a:
                    n_a_only += 1
                elif in_b:
                    n_b_only += 1

        total = n_a_only + n_b_only + n_both
        if total > 0:
            records.append({
                'axis': axis_name,
                'n_genes': len(gene_list),
                'frac_both': n_both / total,
                'frac_a_only': n_a_only / total,
                'frac_b_only': n_b_only / total,
                'total_edges': total,
            })

    if not records:
        print("No edge data for axes, skipping plot 4")
        return

    df = pd.DataFrame(records).sort_values('total_edges', ascending=False).head(20)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = range(len(df))
    ax.bar(x, df['frac_both'].values, label='Both datasets', color='steelblue')
    ax.bar(x, df['frac_a_only'].values, bottom=df['frac_both'].values,
           label='Dataset A only', color='coral')
    ax.bar(x, df['frac_b_only'].values,
           bottom=(df['frac_both'] + df['frac_a_only']).values,
           label='Dataset B only', color='mediumseagreen')
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{r['axis']}\n({r['n_genes']}g)" for _, r in df.iterrows()],
        fontsize=7, rotation=45, ha='right',
    )
    ax.set_ylabel('Fraction of intra-axis edges')
    ax.set_title('Edge contribution by dataset (top 20 axes by edge count)')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot4_edges.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 7 — Validation Suite: generate 4 diagnostic plots "
                    "assessing cross-dataset biological consistency.",
    )
    parser.add_argument(
        "--gene-sets", required=True,
        help="Path to gene_community_sets.json",
    )
    parser.add_argument(
        "--dataset-a", required=True,
        help="Path to Dataset_A CSV (reference)",
    )
    parser.add_argument(
        "--dataset-b", required=True,
        help="Path to Dataset_B CSV (target)",
    )
    parser.add_argument(
        "--edges-a", required=True,
        help="Path to edges_A.csv (for plot 4)",
    )
    parser.add_argument(
        "--edges-b", required=True,
        help="Path to edges_B.csv (for plot 4)",
    )
    parser.add_argument(
        "--common-genes", required=True,
        help="Path to common_genes.txt",
    )
    parser.add_argument(
        "--dup-map", required=True,
        help="Path to gene_dup_map.csv",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load gene community sets
    print("Loading gene community sets...")
    with open(args.gene_sets) as f:
        gene_sets = json.load(f)
    print(f"  {len(gene_sets)} axes loaded")

    # 2. Load common genes and dup map
    common_genes = load_common_genes(args.common_genes)
    dup_genes = load_dup_map(args.dup_map)
    print(f"  {len(common_genes)} common genes, {len(dup_genes)} duplicates")

    # 3. Load and prepare datasets
    print(f"Loading {args.dataset_a}...")
    log_a, meta_a = prepare_dataset(args.dataset_a, common_genes, dup_genes)
    print(f"  Expression: {log_a.shape}, Meta: {meta_a.shape}")

    print(f"Loading {args.dataset_b}...")
    log_b, meta_b = prepare_dataset(args.dataset_b, common_genes, dup_genes)
    print(f"  Expression: {log_b.shape}, Meta: {meta_b.shape}")

    # 4. Compute ssGSEA
    print("Computing ssGSEA for Dataset A...")
    scores_a = ssgsea(log_a, gene_sets)
    print(f"  Scores: {scores_a.shape}")

    print("Computing ssGSEA for Dataset B...")
    scores_b = ssgsea(log_b, gene_sets)
    print(f"  Scores: {scores_b.shape}")

    # 5. Identify ER and HER2 axes
    er_axis = find_er_axis(gene_sets)
    her2_axis = find_her2_axis(gene_sets)
    print(f"  ER axis: {er_axis}, HER2 axis: {her2_axis}")

    # 6. Plot 1: Subtype scatter
    print("\nPlot 1: Subtype scatter...")
    if er_axis and her2_axis:
        plot1_subtype_scatter(scores_a, meta_a, scores_b, meta_b,
                              er_axis, her2_axis, args.output_dir)
    else:
        print("  Skipped (ER or HER2 axis not found)")

    # 7. Plot 2: Violin plots — top 5 axes by score variance
    print("Plot 2: Violin plots...")
    combined = pd.concat([scores_a, scores_b])
    top_axes = combined.var().nlargest(5).index.tolist()
    plot2_violins(scores_a, meta_a, scores_b, meta_b, top_axes, args.output_dir)

    # 8. Plot 3: Neighbor purity
    print("Plot 3: Neighbor purity...")
    plot3_neighbor_purity(scores_a, meta_a, scores_b, meta_b, k=10,
                          output_dir=args.output_dir)

    # 9. Plot 4: Edge contribution
    print("Plot 4: Edge contribution...")
    plot4_edge_contribution(args.edges_a, args.edges_b, gene_sets,
                            args.output_dir)

    print("\nDone. All plots saved to", args.output_dir)


if __name__ == "__main__":
    main()
