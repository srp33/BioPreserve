#!/usr/bin/env python3
"""
Diagnostic visualization for Before Alignment state.
Facet 1: Color=Subtype, Shape=Dataset (Standard)
Facet 2: Color=Dataset, Shape=Subtype (Reversed)
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Reusing logic from 15_sign_aware_embedding.py
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

def sign_aware_weighted_mean(expression_df, gene_sets):
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
        X_local = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pca = PCA(n_components=1, random_state=42)
        pca.fit(X_local)
        loadings = pca.components_[0]
        signs = np.sign(loadings)
        
        hub_weights = np.array([gene_weights[g] for g in genes])
        if signs[np.argmax(hub_weights)] < 0:
            signs = -signs
            
        weights = np.array([abs(gene_weights[g]) for g in genes]) * signs
        scores[set_name] = (X * weights).sum(axis=1) / np.abs(weights).sum()
            
    return pd.DataFrame(scores, index=expression_df.index)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnostic_facets(s_a, m_a, s_b, m_b, gene_sets, output_path):
    er_axis = find_axis_for_gene(gene_sets, "ESR1") or list(gene_sets.keys())[0]
    her2_axis = find_axis_for_gene(gene_sets, "ERBB2")
    
    # Subtype mapping
    subtypes_a = [get_subtype(m_a[resolve_meta_col(m_a, "meta_er_status", "meta_ER_status")].iloc[i],
                             m_a[resolve_meta_col(m_a, "meta_her2_status", "meta_HER2_status")].iloc[i]) 
                  for i in range(len(m_a))]
    subtypes_b = [get_subtype(m_b[resolve_meta_col(m_b, "meta_er_status", "meta_ER_status")].iloc[i],
                             m_b[resolve_meta_col(m_b, "meta_her2_status", "meta_HER2_status")].iloc[i]) 
                  for i in range(len(m_b))]
    
    st_list = ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]
    color_map_st = {
        "ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange",
        "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray",
    }
    shape_map_st = {"ER+/HER2-": "o", "ER+/HER2+": "s", "ER-/HER2-": "D", "ER-/HER2+": "^", "Unknown": "x"}
    
    # Dataset mapping
    color_map_ds = {"A": "dodgerblue", "B": "tomato"}
    shape_map_ds = {"A": "o", "B": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Facet 1: Color=Subtype, Shape=Dataset (Standard)
    ax = axes[0]
    for st in st_list:
        # Dataset A
        mask_a = np.array([s == st for s in subtypes_a])
        if mask_a.sum() > 0:
            ax.scatter(s_a[er_axis].values[mask_a], s_a[her2_axis].values[mask_a],
                       c=color_map_st[st], marker=shape_map_ds["A"], alpha=0.5, s=40, 
                       label=f"A {st}", edgecolors="white", linewidths=0.3)
        # Dataset B
        mask_b = np.array([s == st for s in subtypes_b])
        if mask_b.sum() > 0:
            ax.scatter(s_b[er_axis].values[mask_b], s_b[her2_axis].values[mask_b],
                       c=color_map_st[st], marker=shape_map_ds["B"], alpha=0.5, s=40, 
                       label=f"B {st}")
    ax.set_title("Standard: Color=Subtype, Shape=Dataset")
    ax.legend(fontsize=7, ncol=2)

    # Facet 2: Color=Dataset, Shape=Subtype (Reversed)
    ax = axes[1]
    # Plot Dataset A
    for st in st_list:
        mask_a = np.array([s == st for s in subtypes_a])
        if mask_a.sum() > 0:
            ax.scatter(s_a[er_axis].values[mask_a], s_a[her2_axis].values[mask_a],
                       c=color_map_ds["A"], marker=shape_map_st[st], alpha=0.4, s=40, 
                       label=f"A {st}" if st == st_list[0] else None)
    # Plot Dataset B
    for st in st_list:
        mask_b = np.array([s == st for s in subtypes_b])
        if mask_b.sum() > 0:
            ax.scatter(s_b[er_axis].values[mask_b], s_b[her2_axis].values[mask_b],
                       c=color_map_ds["B"], marker=shape_map_st[st], alpha=0.4, s=40, 
                       label=f"B {st}" if st == st_list[0] else None)
    
    # Custom legend for Facet 2 to explain colors and shapes separately
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Dataset A', markerfacecolor=color_map_ds["A"], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Dataset B', markerfacecolor=color_map_ds["B"], markersize=10),
        plt.Line2D([0], [0], marker=shape_map_st["ER+/HER2-"], color='gray', label='ER+/HER2-', linestyle='None'),
        plt.Line2D([0], [0], marker=shape_map_st["ER+/HER2+"], color='gray', label='ER+/HER2+', linestyle='None'),
        plt.Line2D([0], [0], marker=shape_map_st["ER-/HER2-"], color='gray', label='ER-/HER2-', linestyle='None'),
        plt.Line2D([0], [0], marker=shape_map_st["ER-/HER2+"], color='gray', label='ER-/HER2+', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=7)
    ax.set_title("Reversed: Color=Dataset, Shape=Subtype")

    for ax in axes:
        ax.set_xlabel(f"{er_axis} (ESR1)")
        ax.set_ylabel(f"{her2_axis} (ERBB2)")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Before Alignment Diagnostic: Sign-Aware Weighted Mean Embedding", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Before Alignment Diagnostic Facets")
    parser.add_argument("--dataset-a", default="basis_workflow/inputs/gse20194.csv")
    parser.add_argument("--dataset-b", default="basis_workflow/inputs/gse58644.csv")
    parser.add_argument("--gene-sets", default="basis_workflow/outputs/gene_community_sets.json")
    parser.add_argument("--output-dir", default="basis_workflow/outputs/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gene_sets) as f:
        gene_sets = json.load(f)

    print("Loading datasets...")
    gene_a, meta_a = load_dataset(args.dataset_a)
    gene_b, meta_b = load_dataset(args.dataset_b)
    
    log_a = log_transform(gene_a)
    log_b = log_transform(gene_b)

    print("Computing Sign-Aware Weighted Mean embeddings...")
    s_a = sign_aware_weighted_mean(log_a, gene_sets)
    s_b = sign_aware_weighted_mean(log_b, gene_sets)

    print("Generating diagnostic facets plot...")
    output_path = os.path.join(args.output_dir, "alignment_diagnostic_before.png")
    plot_diagnostic_facets(s_a, meta_a, s_b, meta_b, gene_sets, output_path)

    print(f"Done. Result saved to {output_path}")

if __name__ == "__main__":
    main()
