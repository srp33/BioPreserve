#!/usr/bin/env python3
"""
Rule 20 — Anchor eCDF Diagnostic Plot

Visualizes the cumulative distribution of anchor genes across both platforms.
Exposes the differences in background noise, dynamic range, and saturation limits.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

def main():
    # 1. Load Invariant Anchors
    anchor_path = "outputs/invariant_anchors.txt"
    if not os.path.exists(anchor_path):
        print(f"Error: {anchor_path} not found. Run the pipeline first.")
        return
        
    with open(anchor_path) as f:
        anchors = set([line.strip() for line in f if line.strip()])

    # 2. Load Datasets
    print("Loading datasets...")
    ref_gene = pd.read_csv("../gse20194.csv", index_col=0, low_memory=False) # Microarray
    tgt_gene = pd.read_csv("../gse58644.csv", index_col=0, low_memory=False) # RNA-seq

    # Drop metadata
    ref_expr = ref_gene.drop(columns=[c for c in ref_gene.columns if c.startswith("meta_")], errors='ignore').select_dtypes(include=[np.number])
    tgt_expr = tgt_gene.drop(columns=[c for c in tgt_gene.columns if c.startswith("meta_")], errors='ignore').select_dtypes(include=[np.number])

    # Ensure we only use anchors present in both datasets
    valid_anchors = list(anchors.intersection(ref_expr.columns).intersection(tgt_expr.columns))
    print(f"Plotting eCDF for {len(valid_anchors)} shared anchor genes.")

    fig, ax = plt.subplots(figsize=(12, 7))

    def plot_ecdf(dataset, label, color):
        n_samples = min(100, len(dataset))
        print(f"  Processing {label} ({n_samples} samples)...")
        
        for i in range(n_samples):
            patient = dataset.iloc[i]
            
            # 1. Rank ALL genes in this specific patient (Global Percentile)
            # This converts raw values into [0, 1] relative to the whole transcriptome
            global_ranks = patient.rank(pct=True)
            
            # 2. Extract the global ranks of just the anchor genes
            anchor_global_ranks = global_ranks[valid_anchors].values
            
            # 3. Sort the anchor ranks (The "Ladder")
            sorted_ranks = np.sort(anchor_global_ranks)
            
            # 4. Create the cumulative fraction (The "Ladder Rungs")
            y_vals = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
            
            # Plot as a faint step curve
            ax.step(sorted_ranks, y_vals, where='post', color=color, alpha=0.1, linewidth=0.8)
            
        ax.plot([], [], color=color, label=label, linewidth=2.5)

    plot_ecdf(ref_expr, "Reference (Microarray - GSE20194)", "steelblue")
    plot_ecdf(tgt_expr, "Target (RNA-seq - GSE58644)", "darkorange")

    ax.set_title("Relative Anchor Position: Platform Discrepancy", fontsize=16, fontweight='bold')
    ax.set_xlabel("Transcriptome Percentile (Global Rank of Gene)", fontsize=13)
    ax.set_ylabel("Cumulative Fraction of Anchor Genes", fontsize=13)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = "outputs/anchor_ecdf_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
