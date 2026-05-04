#!/usr/bin/env python3
"""
Verification Script: Comprehensive Pairwise Ablation Study

Evaluates normalization layers across dataset pairs to prove the
superiority of Anchor-Quantile Normalization (AQN).
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

DATASETS = ["gse20194.csv", "gse20271.csv", "gse58644.csv"]

def load_raw_data(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols].copy()
    gene_cols = [c for c in df.columns if not c.startswith("meta_") and pd.api.types.is_numeric_dtype(df[c])]
    gene_df = df[gene_cols].copy()
    return gene_df, meta_df

def main():
    with open("outputs/gene_community_sets.json") as f: gene_sets = json.load(f)
    with open("outputs/invariant_anchors.txt") as f: anchors = [line.strip() for line in f if line.strip()]

    print("Finding common genes across gauntlet...")
    common_set = None
    for d in DATASETS:
        df_genes, _ = load_raw_data(f"../{d}")
        genes = set(df_genes.columns)
        common_set = genes if common_set is None else common_set.intersection(genes)
    
    common_list = sorted(list(common_set))
    print(f"Common genes: {len(common_list)}")

    ablation_modes = [
        {"name": "1. Global Rank (Base)"},
        {"name": "2. AQN (Champion)"},
    ]

    all_results = []

    for ref_name in DATASETS:
        for tgt_name in DATASETS:
            if ref_name == tgt_name: continue
            print(f"\nGAUNTLET: {ref_name} -> {tgt_name}")
            
            ref_raw, ref_meta = load_raw_data(f"../{ref_name}")
            tgt_raw, tgt_meta = load_raw_data(f"../{tgt_name}")
            
            ref_gene = ref_raw[common_list]
            tgt_gene = tgt_raw[common_list]

            for mode in ablation_modes:
                # Setup Atlas
                atlas = BasisAtlas(gene_sets, anchors)
                atlas.fit(ref_gene)

                if mode["name"] == "1. Global Rank (Base)":
                    # Manually disable AQN by zeroing the ref_anchor_rank_profile
                    # This falls back to pure Global Rank mapping.
                    atlas.ref_anchor_rank_profile = pd.Series(
                        np.linspace(0, 1, len(atlas.ref_anchor_rank_profile)), 
                        index=atlas.ref_anchor_rank_profile.index
                    )

                r_s = atlas.transform(ref_gene)
                t_s = atlas.transform(tgt_gene)

                def get_auc(train_X, train_y, test_X, test_y):
                    mask_tr, mask_te = ~train_y.isna(), ~test_y.isna()
                    if mask_tr.sum() < 10 or mask_te.sum() < 5: return 0.5
                    if len(np.unique(train_y[mask_tr])) < 2 or len(np.unique(test_y[mask_te])) < 2: return 0.5
                    clf = HistGradientBoostingClassifier(random_state=42).fit(train_X.loc[mask_tr], train_y[mask_tr])
                    return roc_auc_score(test_y[mask_te], clf.predict_proba(test_X.loc[mask_te])[:, 1])

                er_auc = get_auc(r_s, ref_meta["meta_er_status"], t_s, tgt_meta["meta_er_status"])
                h2_auc = get_auc(r_s, ref_meta["meta_her2_status"], t_s, tgt_meta["meta_her2_status"])
                
                all_results.append({
                    "Pair": f"{ref_name[:8]}->{tgt_name[:8]}",
                    "Ablation": mode["name"],
                    "ER_AUC": er_auc,
                    "HER2_AUC": h2_auc,
                    "Avg_AUC": (er_auc + h2_auc) / 2.0
                })

    res_df = pd.DataFrame(all_results)
    
    print("\n--- ABLATION RANKINGS BY DATASET PAIR ---")
    for pair, group in res_df.groupby("Pair"):
        print(f"\nGAUNTLET PAIR: {pair}")
        sorted_group = group.sort_values("Avg_AUC", ascending=False)
        print(sorted_group[["Ablation", "ER_AUC", "HER2_AUC", "Avg_AUC"]].to_string(index=False))
        print("-" * 50)

    summary = res_df.groupby("Ablation")[["ER_AUC", "HER2_AUC", "Avg_AUC"]].mean().reset_index()
    summary = summary.sort_values("Avg_AUC", ascending=False)
    
    report = "# Final Pipeline Ablation Analysis (6-Pair Gauntlet)\n\n"
    report += summary.to_markdown(index=False)
    report += "\n\n## Per-Pair Rankings\n\n"
    for pair, group in res_df.groupby("Pair"):
        report += f"### {pair}\n"
        report += group.sort_values("Avg_AUC", ascending=False).to_markdown(index=False)
        report += "\n\n"

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ablation_full_gauntlet.md", "w") as f: f.write(report)

if __name__ == "__main__":
    main()
