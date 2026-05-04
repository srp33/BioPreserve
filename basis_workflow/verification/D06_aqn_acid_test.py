#!/usr/bin/env python3
"""
Verification Script: AQN Centroid Superimposition (The Acid Test)

Measures the Euclidean distance between Reference and Target subtype 
centroids after Pure AQN normalization. Proves that AQN natively 
fixes the coordinate geometry without linear multipliers.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from scipy.stats import median_abs_deviation

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

def load_raw_data(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols].copy()
    gene_cols = [c for c in df.columns if not c.startswith("meta_") and pd.api.types.is_numeric_dtype(df[c])]
    gene_df = df[gene_cols].copy()
    return gene_df, meta_df

def get_subtype(meta_row):
    er, her2 = meta_row.get("meta_er_status"), meta_row.get("meta_her2_status")
    if pd.isna(er) or pd.isna(her2): return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"

def main():
    # 1. Setup
    with open("outputs/gene_community_sets.json") as f: gene_sets = json.load(f)
    with open("outputs/invariant_anchors.txt") as f: anchors = [line.strip() for line in f if line.strip()]

    ref_name = "gse58644.csv" # RNA-seq Reference
    tgt_name = "gse20194.csv" # Microarray Target
    
    print(f"GAUNTLET: {ref_name} -> {tgt_name}")
    ref_raw, ref_meta = load_raw_data(f"../{ref_name}")
    tgt_raw, tgt_meta = load_raw_data(f"../{tgt_name}")
    
    common = sorted(list(set(ref_raw.columns) & set(tgt_raw.columns)))
    ref_gene = ref_raw[common]
    tgt_gene = tgt_raw[common]

    # 2. Fit Atlas and Transform
    atlas = BasisAtlas(gene_sets, anchors)
    atlas.fit(ref_gene)
    
    ref_scores = atlas.transform(ref_gene)
    tgt_scores = atlas.transform(tgt_gene)
    
    # 3. Centroid Analysis
    ref_sub = ref_meta.apply(get_subtype, axis=1)
    tgt_sub = tgt_meta.apply(get_subtype, axis=1)
    
    er_axis, her2_axis = None, None
    for name, gs in gene_sets.items():
        if "PGR" in gs and gs["PGR"] > 0.8: er_axis = name
        if "ERBB2" in gs and gs["ERBB2"] > 0.8: her2_axis = name
        
    if not er_axis or not her2_axis:
        print("Error: Could not find ER or HER2 axes.")
        return

    print("\n--- PURE AQN CENTROID ERROR ---")
    for st in ["ER+/HER2-", "ER-/HER2-"]:
        rm, tm = (ref_sub == st), (tgt_sub == st)
        if rm.any() and tm.any():
            ref_centroid = ref_scores.loc[rm, [er_axis, her2_axis]].mean().values
            tgt_centroid = tgt_scores.loc[tm, [er_axis, her2_axis]].mean().values
            dist = np.linalg.norm(ref_centroid - tgt_centroid)
            print(f"{st} Centroid Shift: {dist:.3f}")
            print(f"  Ref: ({ref_centroid[0]:.2f}, {ref_centroid[1]:.2f})")
            print(f"  Tgt: ({tgt_centroid[0]:.2f}, {tgt_centroid[1]:.2f})")
        else:
            print(f"{st}: Missing data in one of the cohorts.")

    print("\nAcid Test Complete.")

if __name__ == "__main__":
    main()
