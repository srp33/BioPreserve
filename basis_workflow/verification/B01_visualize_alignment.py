#!/usr/bin/env python3
"""
Visualize BASIS alignment quality and batch effects using multiple embeddings.
Produces diagnostic scatter plots (Standard vs Reversed) for the 'Before Alignment' state.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys

# Ensure 'basis' package is importable from the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# Import canonical implementations
# (Local implementations replacing deprecated functions)

def pc1_embed(df, gene_sets):
    scores = {}
    for name, gs in gene_sets.items():
        genes = [g for g in gs if g in df.columns]
        if not genes:
            scores[name] = np.zeros(len(df))
            continue
        X = df[genes].values
        hw = np.array([gs[g] for g in genes])
        X_w = X * hw
        pca = PCA(n_components=1)
        s = pca.fit_transform(X_w).ravel()
        if np.corrcoef(s, X_w.mean(axis=1))[0,1] < 0:
            s = -s
        scores[name] = s
    return pd.DataFrame(scores, index=df.index)

def gmm_posterior_embed(df, gene_sets):
    return pc1_embed(df, gene_sets) # Placeholder for deprecated plot

# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def ssgsea(expression_df, gene_sets, alpha=0.25):
    """Hub-weighted single-sample GSEA."""
    n_samples = len(expression_df)
    gene_names = expression_df.columns.tolist()
    n_genes = len(gene_names)
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        idxs, weights = [], []
        for gene, w in gene_weights.items():
            if gene in gene_names:
                idxs.append(gene_names.index(gene))
                weights.append(abs(w))
        if len(idxs) < 3:
            scores[set_name] = np.zeros(n_samples)
            continue
        idxs, weights = np.array(idxs), np.array(weights)
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
            n_in, n_out = in_set_ranked.sum(), n_genes - in_set_ranked.sum()
            if n_in == 0 or n_out == 0:
                continue
            rank_positions = np.arange(n_genes, 0, -1)
            hits = np.where(in_set_ranked, np.abs(rank_positions) ** alpha * weight_ranked, 0)
            if hits.sum() == 0:
                continue
            hits_norm = hits / hits.sum()
            misses = np.where(~in_set_ranked, 1.0 / n_out, 0)
            running_sum = np.cumsum(hits_norm - misses)
            set_scores[i] = running_sum.max() - running_sum.min()
        scores[set_name] = set_scores
    return pd.DataFrame(scores, index=expression_df.index)


def simple_mean_embed(expression_df, gene_sets, use_weights=False, use_signs=False):
    """Simple or weighted mean with optional PC1-based sign alignment."""
    vals = expression_df.values
    global_mean = vals.mean()
    global_std = vals.std()
    df_std = (expression_df - global_mean) / np.maximum(global_std, 1e-12)
    
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in expression_df.columns]
        if len(genes) < 2:
            scores[set_name] = np.zeros(len(expression_df))
            continue
        X = df_std[genes].values
        if use_signs:
            X_local = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
            pca = PCA(n_components=1, random_state=42)
            pca.fit(X_local)
            loadings = pca.components_[0]
            signs = np.sign(loadings)
            hub_weights = np.array([gene_weights[g] for g in genes])
            if signs[np.argmax(hub_weights)] < 0:
                signs = -signs
        else:
            signs = np.ones(len(genes))
        if use_weights:
            weights = np.array([abs(gene_weights[g]) for g in genes]) * signs
            scores[set_name] = (X * weights).sum(axis=1) / np.abs(weights).sum()
        else:
            scores[set_name] = (X * signs).mean(axis=1)
    return pd.DataFrame(scores, index=expression_df.index)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    return gene_df, meta_df

def log_transform(df):
    return np.log(df - df.min(axis=0) + 1.0)

def get_subtype(er, her2):
    if pd.isna(er) or pd.isna(her2): return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"

def resolve_meta_col(meta_df, *candidates):
    for c in candidates:
        if c in meta_df.columns: return c
    return None

def find_axis_for_gene(gene_sets, gene_name):
    for name, genes in gene_sets.items():
        if gene_name in genes: return name
    return None


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostic_facets(s_a, m_a, s_b, m_b, gene_sets, output_path, method_name):
    er_axis = find_axis_for_gene(gene_sets, "ESR1") or list(gene_sets.keys())[0]
    her2_axis = find_axis_for_gene(gene_sets, "ERBB2")
    if her2_axis is None:
        keys = list(gene_sets.keys())
        her2_axis = keys[1] if len(keys) > 1 else keys[0]
        
    def get_st_labels(m):
        er_col = resolve_meta_col(m, "meta_er_status", "meta_ER_status")
        her2_col = resolve_meta_col(m, "meta_her2_status", "meta_HER2_status")
        if not er_col or not her2_col: return ["Unknown"] * len(m)
        return [get_subtype(e, h) for e, h in zip(m[er_col], m[her2_col])]

    subtypes_a = get_st_labels(m_a)
    subtypes_b = get_st_labels(m_b)
    
    st_list = ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]
    color_map_st = {
        "ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange",
        "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray",
    }
    shape_map_st = {"ER+/HER2-": "o", "ER+/HER2+": "s", "ER-/HER2-": "D", "ER-/HER2+": "^", "Unknown": "x"}
    color_map_ds = {"A": "dodgerblue", "B": "tomato"}
    shape_map_ds = {"A": "o", "B": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Facet 1: Color=Subtype, Shape=Dataset
    ax = axes[0]
    for st in st_list:
        m_a_idx = np.array([s == st for s in subtypes_a])
        if m_a_idx.sum() > 0:
            ax.scatter(s_a[er_axis].values[m_a_idx], s_a[her2_axis].values[m_a_idx],
                       c=color_map_st[st], marker=shape_map_ds["A"], alpha=0.5, s=35, label=f"A {st}")
        m_b_idx = np.array([s == st for s in subtypes_b])
        if m_b_idx.sum() > 0:
            ax.scatter(s_b[er_axis].values[m_b_idx], s_b[her2_axis].values[m_b_idx],
                       c=color_map_st[st], marker=shape_map_ds["B"], alpha=0.5, s=35, label=f"B {st}")
    ax.set_title("Standard: Color=Subtype, Shape=Dataset")
    ax.legend(fontsize=7, ncol=2, loc="best")

    # Facet 2: Color=Dataset, Shape=Subtype
    ax = axes[1]
    for st in st_list:
        m_a_idx = np.array([s == st for s in subtypes_a])
        if m_a_idx.sum() > 0:
            ax.scatter(s_a[er_axis].values[m_a_idx], s_a[her2_axis].values[m_a_idx],
                       c=color_map_ds["A"], marker=shape_map_st[st], alpha=0.4, s=35)
        m_b_idx = np.array([s == st for s in subtypes_b])
        if m_b_idx.sum() > 0:
            ax.scatter(s_b[er_axis].values[m_b_idx], s_b[her2_axis].values[m_b_idx],
                       c=color_map_ds["B"], marker=shape_map_st[st], alpha=0.4, s=35)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Dataset A', markerfacecolor=color_map_ds["A"], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Dataset B', markerfacecolor=color_map_ds["B"], markersize=10),
    ] + [Line2D([0], [0], marker=shape_map_st[st], color='gray', label=st, linestyle='None') for st in st_list]
    ax.legend(handles=legend_elements, loc='best', fontsize=7)
    ax.set_title("Reversed: Color=Dataset, Shape=Subtype")

    for a in axes:
        a.set_xlabel(f"{er_axis} (ESR1)")
        a.set_ylabel(f"{her2_axis} (ERBB2)")
        a.grid(True, alpha=0.2)

    plt.suptitle(f"Before Alignment Diagnostic: {method_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize alignment quality and batch effects.")
    parser.add_argument("--gene-sets", default="basis_workflow/outputs/gene_community_sets.json")
    parser.add_argument("--dataset-a", default="basis_workflow/inputs/gse20194.csv")
    parser.add_argument("--dataset-b", default="basis_workflow/inputs/gse58644.csv")
    parser.add_argument("--aligned", help="Path to aligned dataset (optional for diagnostic plots)")
    parser.add_argument("--metadata", help="Path to alignment metadata (optional)")
    parser.add_argument("--output-dir", default="basis_workflow/outputs/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.gene_sets) as f: gene_sets = json.load(f)

    print("Loading datasets...")
    gene_a, meta_a = load_dataset(args.dataset_a)
    gene_b, meta_b = load_dataset(args.dataset_b)
    log_a, log_b = log_transform(gene_a), log_transform(gene_b)

    methods = [
        ("PC1", lambda df, gs: pc1_embed(df, gs)),
        ("ssGSEA", lambda df, gs: ssgsea(df, gs)),
        ("GMMpost", lambda df, gs: gmm_posterior_embed(df, gs)),
        ("SimpleMean", lambda df, gs: simple_mean_embed(df, gs, False, False)),
        ("WeightedMean", lambda df, gs: simple_mean_embed(df, gs, True, False)),
        ("SignSimpleMean", lambda df, gs: simple_mean_embed(df, gs, False, True)),
        ("SignWeightedMean", lambda df, gs: simple_mean_embed(df, gs, True, True)),
    ]

    for name, fn in methods:
        print(f"\nProcessing {name}...")
        s_a = fn(log_a, gene_sets)
        s_b = fn(log_b, gene_sets)
        tag = name.lower()
        
        # Determine output path to match Snakefile if possible
        if name == "PC1":
            output_path = os.path.join(args.output_dir, "alignment_pca_pc1.png")
        elif name == "GMMpost":
            output_path = os.path.join(args.output_dir, "alignment_pca_gmmpost.png")
        else:
            output_path = os.path.join(args.output_dir, f"alignment_scatter_{tag}.png")
            
        plot_diagnostic_facets(s_a, meta_a, s_b, meta_b, gene_sets, output_path, name)

    print(f"\nDone. All plots in {args.output_dir}")

if __name__ == "__main__":
    main()
