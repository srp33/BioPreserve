#!/usr/bin/env python3
"""
Verification Script: Find Lamé Shape Parameter (Unsupervised)

Finds the optimal curvature constant (γ) using all hub genes in the 
dictionary. This aligns platform sensitivities without needing any 
biological labels or metadata. Crucial for explaining the unsupervised 
'Digital Gain' correction in the paper.
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

def lame_warp(x, gamma):
    """Calculate the Lamé curve segment for a given gamma."""
    x_cl = np.clip(x, 0, 1 - 1e-12)
    return 1.0 - (1.0 - x_cl**gamma)**(1.0/gamma)

def load_numeric(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith('meta_')]
    return df.drop(columns=meta_cols, errors='ignore').select_dtypes(include=[np.number])

def main():
    with open("outputs/gene_community_sets.json") as f:
        gene_sets = json.load(f)
    with open("outputs/invariant_anchors.txt") as f:
        anchors = [line.strip() for line in f if line.strip()]

    # Identify all hub genes (unsupervised)
    all_hubs = set()
    for gs in gene_sets.values():
        top = sorted(gs.items(), key=lambda x: x[1], reverse=True)[:50]
        for g, w in top: all_hubs.add(g)
    
    print("Loading datasets...")
    ref_gene = load_numeric("../gse20194.csv")
    tgt_gene = load_numeric("../gse58644.csv")

    common = sorted(list(set(ref_gene.columns) & set(tgt_gene.columns)))
    hub_list = [g for g in all_hubs if g in common]
    print(f"Tracking {len(hub_list)} hub genes for purely unsupervised mapping.")

    # Apply baseline SMART (γ=1.0)
    print("Computing baseline SMART ranks (γ=1.0)...")
    atlas = BasisAtlas(gene_sets, anchors)
    atlas.fit(ref_gene)
    
    # Translate entire cohorts (Discovery mode)
    # We take the median rank of every hub across the entire unlabeled cohort
    ref_ranks = ref_gene.rank(axis=1, pct=True)[hub_list].median(axis=0)
    tgt_ranks = tgt_gene.rank(axis=1, pct=True)[hub_list].median(axis=0)

    # Lamé Q-Q Analysis to find Deflation Gamma
    x_data = tgt_ranks.values
    y_data = ref_ranks.values
    
    # Weights based on hub strength (unsupervised)
    hub_weights = {}
    for gs in gene_sets.values():
        for g, w in gs.items():
            if g in hub_list: hub_weights[g] = max(hub_weights.get(g, 0), abs(w))
    w_data = np.array([hub_weights[g] for g in hub_list])

    mask = (x_data > 0.05) & (x_data < 0.95) & (y_data > 0.05) & (y_data < 0.95)
    x_fit, y_fit, w_fit = x_data[mask], y_data[mask], w_data[mask]

    def objective(x, g):
        return lame_warp(x, g)

    # Fit the Lamé curve parameter
    popt, _ = curve_fit(objective, x_fit, y_fit, p0=[1.0], sigma=1.0/w_fit, bounds=(0.1, 10.0))
    gamma_best = popt[0]
    
    print(f"\nUnsupervised Lamé Shape Parameter (γ): {gamma_best:.4f}")

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(x_data, y_data, alpha=0.3, s=w_data*50, label="Hub Genes (Median Cohort Rank)")
    
    grid = np.linspace(0.01, 0.99, 100)
    plt.plot(grid, grid, 'k--', alpha=0.3, label="Identity (No Warp)")
    plt.plot(grid, lame_warp(grid, gamma_best), 'r-', linewidth=2.5, label=f"Unsupervised Lamé Fit (γ={gamma_best:.2f})")
    
    plt.xlabel("Target Ranks (RNA-seq / GSE58644)")
    plt.ylabel("Reference Ranks (Microarray / GSE20194)")
    plt.title("Purely Unsupervised Lamé Discovery: Label-Free Alignment", fontweight='bold', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/gamma_lame_fit_unsupervised.png")
    print("Fit plot saved to outputs/gamma_lame_fit_unsupervised.png")

if __name__ == "__main__":
    main()
