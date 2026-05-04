#!/usr/bin/env python3
"""
Rule 25 — Systematic Pairwise N=1 Validation (Streaming Clinical Edition)

Performs a cross-platform gauntlet while comparing algorithms side-by-side.
Processes patients STICTLY one-at-a-time to prove N=1 isolation.
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

DATASETS = ["gse20194.csv", "gse20271.csv", "gse25065.csv", "gse58644.csv"]

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

    output_dir = "outputs/pairwise_n1"
    os.makedirs(output_dir, exist_ok=True)

    print("Finding common genes across all datasets...")
    common_set = None
    for d in DATASETS:
        df_genes, _ = load_raw_data(f"../{d}")
        genes = set(df_genes.columns)
        common_set = genes if common_set is None else common_set.intersection(genes)
    common_list = sorted(list(common_set))
    print(f"  {len(common_list)} common genes identified.")

    results = []
    er_axis, her2_axis = None, None
    for name, gs in gene_sets.items():
        if "PGR" in gs and gs["PGR"] > 0.8: er_axis = name
        if "ERBB2" in gs and gs["ERBB2"] > 0.8: her2_axis = name

    for ref_name in DATASETS:
        for tgt_name in DATASETS:
            if ref_name == tgt_name: continue
            
            pair_id = f"{ref_name.replace('.csv', '')}_to_{tgt_name.replace('.csv', '')}"
            print(f"\nGAUNTLET: {pair_id}")
            
            ref_raw, ref_meta = load_raw_data(f"../{ref_name}")
            tgt_raw, tgt_meta = load_raw_data(f"../{tgt_name}")
            ref_gene, tgt_gene = ref_raw[common_list], tgt_raw[common_list]
            ref_sub, tgt_sub = ref_meta.apply(get_subtype, axis=1), tgt_meta.apply(get_subtype, axis=1)

            # Fit Master Atlas
            atlas_master = BasisAtlas(gene_sets, anchors)
            atlas_master.fit(ref_gene)
            # learn_translation is N=Many Discovery
            atlas_master.learn_translation(tgt_gene, ref_gene)

            fig, axes = plt.subplots(4, 3, figsize=(24, 28))
            methods = [
                {"name": "Row 1: Global Rank (Base)", "mode": "latent_aqn", "disable_aqn": True},
                {"name": "Row 2: Latent AQN",         "mode": "latent_aqn", "disable_aqn": False},
                {"name": "Row 3: Physical APS",       "mode": "physical_aps", "disable_aqn": False},
                {"name": "Row 4: Unified USB",        "mode": "unified_usb",  "disable_aqn": False},
            ]
            
            for row_idx, m in enumerate(methods):
                print(f"  Evaluating {m['name']} (Streaming N=1)...")
                
                # Setup specific atlas
                row_atlas = BasisAtlas(gene_sets, anchors)
                row_atlas.fit(ref_gene)
                if m["disable_aqn"]:
                    row_atlas.ref_anchor_rank_profile = pd.Series(np.linspace(0, 1, len(row_atlas.ref_anchor_rank_profile)), index=row_atlas.ref_anchor_rank_profile.index)
                
                # Discovery logic (learn_translation) only for Latent/USB rows
                if m["name"] != "Row 3: Physical APS":
                    row_atlas.learn_translation(tgt_gene, ref_gene)

                # --- STRICT N=1 STREAMING LOOP ---
                ref_scores_list = []
                tgt_scores_list = []
                tgt_aligned_scores_list = []
                
                # Process Reference cohort (as N=1 for symmetry)
                for i in range(len(ref_gene)):
                    sample = ref_gene.iloc[[i]]
                    ref_scores_list.append(row_atlas.transform(sample, mode=m["mode"]))
                
                # Process Target cohort (strictly isolated)
                for i in range(len(tgt_gene)):
                    sample = tgt_gene.iloc[[i]]
                    
                    # 1. Calibrated Scores
                    t_s = row_atlas.transform(sample, mode=m["mode"])
                    tgt_scores_list.append(t_s)
                    
                    # 2. Reconstructed Scores (Panel 3)
                    t_aligned_phys = row_atlas.correct_genes(sample, mode=m["mode"])
                    # Re-embed
                    t_aligned_s = row_atlas.transform(t_aligned_phys, mode=m["mode"])
                    tgt_aligned_scores_list.append(t_aligned_s)

                ref_scores = pd.concat(ref_scores_list)
                tgt_scores = pd.concat(tgt_scores_list)
                tgt_aligned_scores = pd.concat(tgt_aligned_scores_list)

                # Metrics
                def get_auc(X_tr, y_tr, X_te, y_te):
                    m_tr, m_te = ~y_tr.isna(), ~y_te.isna()
                    if m_tr.sum() < 10 or m_te.sum() < 5: return np.nan
                    clf = HistGradientBoostingClassifier(random_state=42).fit(X_tr.loc[m_tr], y_tr[m_tr])
                    return roc_auc_score(y_te[m_te], clf.predict_proba(X_te.loc[m_te])[:, 1])

                er_auc = get_auc(ref_scores, ref_meta["meta_er_status"], tgt_scores, tgt_meta["meta_er_status"])
                her2_auc = get_auc(ref_scores, ref_meta["meta_her2_status"], tgt_scores, tgt_meta["meta_her2_status"])

                def get_shift(ref_s, tgt_s, sub_ref, sub_tgt):
                    shifts = []
                    mad_er = row_atlas.axis_params[er_axis]["ref_mad"]
                    mad_her2 = row_atlas.axis_params[her2_axis]["ref_mad"]
                    for st in ["ER+/HER2-", "ER-/HER2-"]:
                        rm, tm = (sub_ref == st), (sub_tgt == st)
                        if rm.any() and tm.any():
                            rc = ref_s.loc[rm, [er_axis, her2_axis]].mean().values
                            tc = tgt_s.loc[tm, [er_axis, her2_axis]].mean().values
                            dz_er = (rc[0] - tc[0]) / max(mad_er, 1e-6)
                            dz_her2 = (rc[1] - tc[1]) / max(mad_her2, 1e-6)
                            shifts.append(np.sqrt(dz_er**2 + dz_her2**2))
                    return np.mean(shifts) if shifts else np.nan

                avg_shift = get_shift(ref_scores, tgt_scores, ref_sub, tgt_sub)
                results.append({"Pair": pair_id, "Method": m['name'], "ER_AUC": er_auc, "HER2_AUC": her2_auc, "Avg_AUC": (er_auc+her2_auc)/2.0, "Centroid_Shift": avg_shift})

                # Plotting
                plot_ref = [ref_scores, ref_scores, ref_scores]
                plot_tgt = [tgt_scores, tgt_scores, tgt_aligned_scores]
                titles = ["1. Pure Space", "2. Calibrated", "3. Corrected"]
                color_map = {"ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange", "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray"}

                for col_idx in range(3):
                    ax = axes[row_idx, col_idx]
                    rp = plot_ref[col_idx][[er_axis, her2_axis]].rename(columns={er_axis: "ER", her2_axis: "HER2"})
                    tp = plot_tgt[col_idx][[er_axis, her2_axis]].rename(columns={er_axis: "ER", her2_axis: "HER2"})
                    for st in ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]:
                        rm, tm = (ref_sub == st), (tgt_sub == st)
                        if rm.any(): ax.scatter(rp.loc[rm, "ER"], rp.loc[rm, "HER2"], c=color_map[st], alpha=0.3, s=10)
                        if tm.any(): ax.scatter(tp.loc[tm, "ER"], tp.loc[tm, "HER2"], c=color_map[st], marker="^", s=50, edgecolors="black", linewidths=0.3)
                    if row_idx == 0: ax.set_title(titles[col_idx], fontweight="bold")
                    if col_idx == 0: ax.set_ylabel(f"{m['name']}\n\nHER2 Axis Score", fontweight="bold")
                    if col_idx == 1:
                        ax.text(0.05, 0.95, f"Shift: {avg_shift:.3f}", transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    ax.grid(True, alpha=0.1)

            plt.suptitle(f"Hardened N=1 Gauntlet (Streaming Mode): {ref_name} -> {tgt_name}", fontsize=16)
            plt.savefig(f"{output_dir}/{pair_id}.png", dpi=100, bbox_inches="tight")
            plt.close()

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{output_dir}/comparative_results.csv", index=False)
    print("\n--- COMPARATIVE RANKINGS (Streaming N=1) ---")
    summary = res_df.groupby("Method")[["ER_AUC", "HER2_AUC", "Avg_AUC", "Centroid_Shift"]].mean().sort_values("Avg_AUC", ascending=False)
    print(summary.to_string())

if __name__ == "__main__":
    main()
