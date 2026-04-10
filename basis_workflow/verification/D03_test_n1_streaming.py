#!/usr/bin/env python3
"""
Rule 18 — Streaming N=1 Validation (Soft GMM Edition)

Visualizes:
1. Unadjusted Raw Log.
2. Calibrated Continuous Embedding (N=1).
3. Soft GMM Latent Interpolation (Global Gene Alignment).
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

def load_data(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors='ignore').select_dtypes(include=[np.number])
    return gene_df, meta_df

def find_primary_axes(gene_sets):
    er_axis, her2_axis = None, None
    for name, gs in gene_sets.items():
        if "PGR" in gs and gs["PGR"] > 0.8: er_axis = name
        if "ERBB2" in gs and gs["ERBB2"] > 0.8: her2_axis = name
    return er_axis, her2_axis

def get_subtype(meta_row):
    er, her2 = meta_row.get("meta_er_status"), meta_row.get("meta_her2_status")
    if pd.isna(er) or pd.isna(her2): return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"

def main():
    # 1. Load artifacts
    with open("outputs/gene_community_sets.json") as f: gene_sets = json.load(f)
    with open("outputs/invariant_anchors.txt") as f: anchors = [line.strip() for line in f if line.strip()]
    er_axis, her2_axis = find_primary_axes(gene_sets)

    # 2. Load Datasets
    ref_gene, ref_meta = load_data("../gse20194.csv")
    tgt_gene, tgt_meta = load_data("../gse58644.csv")
    common = sorted(list(set(ref_gene.columns) & set(tgt_gene.columns)))
    ref_gene, tgt_gene = ref_gene[common], tgt_gene[common]

    # 3. Discovery: Fit Soft GMM Atlas
    print("Fitting Reference Atlas (Soft GMM)...")
    atlas = BasisAtlas(gene_sets, anchors)
    atlas.fit(ref_gene)
    print("Learning state-specific translation factors...")
    atlas.learn_translation(tgt_gene, ref_gene)

    # 4. Diagnosis: Independent Transformation (N=1)
    print("Generating Calibrated Embeddings (Panel 2)...")
    ref_embed = atlas.transform(ref_gene).rename(columns={er_axis: "ER", her2_axis: "HER2"})
    tgt_embed = atlas.transform(tgt_gene).rename(columns={er_axis: "ER", her2_axis: "HER2"})

    print("Applying Soft GMM Gene Alignment (Panel 3)...")
    # This aligns the RAW genes to Reference scale
    tgt_aligned_genes = atlas.correct_genes(tgt_gene)
    # Re-embed the ALIGNED genes using the Reference Atlas (Self-Projection)
    tgt_aligned_embed = atlas.transform(np.exp(tgt_aligned_genes) + atlas.gene_mins.values - 1.0).rename(columns={er_axis: "ER", her2_axis: "HER2"})

    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    color_map = {"ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange", "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray"}
    ref_sub, tgt_sub = ref_meta.apply(get_subtype, axis=1), tgt_meta.apply(get_subtype, axis=1)

    # Panel 1: Unadjusted
    def process_raw(df):
        er_h = [g for g in gene_sets[er_axis].keys() if g in df.columns]
        he_h = [g for g in gene_sets[her2_axis].keys() if g in df.columns]
        return pd.DataFrame({"ER": df[er_h].mean(axis=1), "HER2": df[he_h].mean(axis=1)}, index=df.index)
    ref_raw = process_raw(ref_gene)
    tgt_raw = process_raw(tgt_gene)

    titles = ["Panel 1: Unadjusted (Raw Log)", "Panel 2: Continuous Embedding (N=1)", "Panel 3: Soft GMM Corrected (Global)"]
    data_ref = [ref_raw, ref_embed, ref_embed]
    data_tgt = [tgt_raw, tgt_embed, tgt_aligned_embed]

    for i in range(3):
        ax = axes[i]
        rp, tp = data_ref[i], data_tgt[i]
        for st in ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]:
            rm, tm = (ref_sub == st), (tgt_sub == st)
            if rm.any(): ax.scatter(rp.loc[rm, "ER"], rp.loc[rm, "HER2"], c=color_map[st], alpha=0.4, s=15)
            if tm.any(): ax.scatter(tp.loc[tm, "ER"], tp.loc[tm, "HER2"], c=color_map[st], marker="^", s=70, label=f"Tgt: {st}", edgecolors="black", linewidths=0.5)
        ax.set_title(titles[i], fontweight="bold")
        ax.grid(True, alpha=0.1)
        if i == 2: ax.legend(loc="upper right", fontsize="x-small")

    plt.suptitle("Clinical N=1 Soft GMM Validation: Latent State Interpolation", fontsize=16)
    plt.savefig("outputs/n1_streaming_validation_faceted.png", dpi=150, bbox_inches="tight")
    print(f"Validation plot saved.")

if __name__ == "__main__":
    main()
