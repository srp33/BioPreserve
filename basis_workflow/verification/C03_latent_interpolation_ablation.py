#!/usr/bin/env python3
"""
Verification Script: Latent State Interpolation Ablation Study

Tests 4 different methods of Embedding-to-Gene batch correction:
1. Cohort OT (N=Many)
2. Global BGN (N=1 Baseline)
3. Hard GMM Assignment (N=1 Negative Control)
4. Soft GMM Interpolation (N=1 Champion)

Crucial for proving that Soft GMM provides the highest combination 
of Clinical Accuracy and Visual Superimposition.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
import scipy.spatial.distance as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

def load_data(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols].copy()
    gene_cols = [c for c in df.columns if not c.startswith("meta_") and pd.api.types.is_numeric_dtype(df[c])]
    gene_df = df[gene_cols].copy()
    return gene_df, meta_df

def main():
    with open("outputs/gene_community_sets.json") as f: gene_sets = json.load(f)
    with open("outputs/invariant_anchors.txt") as f: anchors = [line.strip() for line in f if line.strip()]

    ref_name, tgt_name = "gse20194.csv", "gse58644.csv"
    ref_raw, ref_meta = load_data(f"../{ref_name}")
    tgt_raw, tgt_meta = load_data(f"../{tgt_name}")
    common = sorted(list(set(ref_raw.columns) & set(tgt_raw.columns)))
    ref_raw, tgt_raw = ref_raw[common], tgt_raw[common]

    er_axis = "axis_1"
    for name, gs in gene_sets.items():
        if "PGR" in gs and gs["PGR"] > 0.8: er_axis = name
    print(f"Study Focus: ER Axis ({er_axis})")

    # Discovery Phase: Fit Atlas
    atlas = BasisAtlas(gene_sets, anchors)
    atlas.fit(ref_raw)
    atlas.learn_translation(tgt_raw, ref_raw)

    # Calculate base embeddings
    ref_scores = atlas.transform(ref_raw)
    tgt_scores = atlas.transform(tgt_raw)
    
    er_ref = ref_scores[er_axis].values
    er_tgt = tgt_scores[er_axis].values

    # Get GMM states
    gmm = atlas.global_gmm
    means = gmm.means_[:, list(ref_scores.columns).index(er_axis)]
    off_idx = np.argmin(means)
    on_idx = np.argmax(means)
    
    ref_probs = gmm.predict_proba(ref_scores.values)
    ref_on_mask = ref_probs[:, on_idx] > 0.8
    ref_off_mask = ref_probs[:, off_idx] > 0.8
    
    tgt_probs = gmm.predict_proba(tgt_scores.values)
    
    print("Calculating state-specific correction factors...")
    log_ref = np.log(np.maximum(ref_raw - ref_raw.min(axis=0), 0) + 1.0)
    log_tgt = np.log(np.maximum(tgt_raw - tgt_raw.min(axis=0), 0) + 1.0)
    
    ref_mu_on = log_ref[ref_on_mask].mean(axis=0)
    ref_mu_off = log_ref[ref_off_mask].mean(axis=0)
    ref_std_on = log_ref[ref_on_mask].std(axis=0)
    ref_std_off = log_ref[ref_off_mask].std(axis=0)
    
    tgt_on_mask = tgt_probs[:, on_idx] > 0.8
    tgt_off_mask = tgt_probs[:, off_idx] > 0.8
    tgt_mu_on = log_tgt[tgt_on_mask].mean(axis=0)
    tgt_mu_off = log_tgt[tgt_off_mask].mean(axis=0)
    tgt_std_on = log_tgt[tgt_on_mask].std(axis=0)
    tgt_std_off = log_tgt[tgt_off_mask].std(axis=0)

    # Reference Centroids for Superimposition Check
    ref_hubs = [g for g in gene_sets[er_axis] if g in log_ref.columns]
    ref_mu_on_hubs = log_ref.loc[ref_on_mask, ref_hubs].mean().values
    ref_mu_off_hubs = log_ref.loc[ref_off_mask, ref_hubs].mean().values

    def evaluate_arm(corrected_tgt_log, name):
        er_genes = [g for g in gene_sets[er_axis] if g in corrected_tgt_log.columns]
        w = np.array([gene_sets[er_axis][g] for g in er_genes])
        scores = (corrected_tgt_log[er_genes].values * w).sum(axis=1)
        auc = roc_auc_score(tgt_meta["meta_er_status"], scores)
        
        t_on_m = (tgt_meta["meta_er_status"] == 1)
        t_off_m = (tgt_meta["meta_er_status"] == 0)
        
        tgt_mu_on_hubs = corrected_tgt_log.loc[t_on_m, ref_hubs].mean().values
        tgt_mu_off_hubs = corrected_tgt_log.loc[t_off_m, ref_hubs].mean().values
        
        err_on = np.linalg.norm(tgt_mu_on_hubs - ref_mu_on_hubs)
        err_off = np.linalg.norm(tgt_mu_off_hubs - ref_mu_off_hubs)
        avg_err = (err_on + err_off) / 2.0
        
        return auc, avg_err

    results = []

    # ARM 1: Cohort OT
    print("Arm 1: Cohort OT...")
    X, Y = ref_scores.values, tgt_scores.values
    C = dist.cdist(X, Y, metric="sqeuclidean")
    C = C / (np.max(C) + 1e-8)
    P = np.exp(-C / 0.01)
    P /= P.sum()
    P_norm = P / np.maximum(np.sum(P, axis=0), 1e-12)
    tgt_ot = pd.DataFrame((log_ref.values.T @ P_norm).T, columns=common, index=tgt_raw.index)
    auc, err = evaluate_arm(tgt_ot, "Arm 1")
    results.append({"Arm": "1. Cohort OT (N=Many)", "ER_AUC": auc, "Centroid_Err": err})

    # ARM 2: Global BGN Reversal
    print("Arm 2: Global BGN...")
    global_scale = log_ref.values.std() / log_tgt.values.std()
    global_shift = log_ref.values.mean() - (log_tgt.values.mean() * global_scale)
    tgt_bgn = (log_tgt * global_scale) + global_shift
    auc, err = evaluate_arm(tgt_bgn, "Arm 2")
    results.append({"Arm": "2. Global BGN (Baseline N=1)", "ER_AUC": auc, "Centroid_Err": err})

    # ARM 3: Hard GMM Assignment
    print("Arm 3: Hard GMM...")
    tgt_hard = log_tgt.copy()
    for i in range(len(tgt_hard)):
        state = on_idx if tgt_probs[i, on_idx] > 0.5 else off_idx
        if state == on_idx:
            scale = (ref_std_on / np.maximum(tgt_std_on, 1e-6)).fillna(1.0)
            shift = (ref_mu_on - (tgt_mu_on * scale)).fillna(0.0)
        else:
            scale = (ref_std_off / np.maximum(tgt_std_off, 1e-6)).fillna(1.0)
            shift = (ref_mu_off - (tgt_mu_off * scale)).fillna(0.0)
        tgt_hard.iloc[i] = (tgt_hard.iloc[i] * scale) + shift
    auc, err = evaluate_arm(tgt_hard, "Arm 3")
    results.append({"Arm": "3. Hard GMM (N=1 Control)", "ER_AUC": auc, "Centroid_Err": err})

    # ARM 4: Soft GMM Interpolation
    print("Arm 4: Soft GMM...")
    p_on, p_off = tgt_probs[:, on_idx], tgt_probs[:, off_idx]
    scale_on = (ref_std_on / np.maximum(tgt_std_on, 1e-6)).fillna(1.0)
    shift_on = (ref_mu_on - (tgt_mu_on * scale_on)).fillna(0.0)
    scale_off = (ref_std_off / np.maximum(tgt_std_off, 1e-6)).fillna(1.0)
    shift_off = (ref_mu_off - (tgt_mu_off * scale_off)).fillna(0.0)
    
    term_on = p_on[:, np.newaxis] * (log_tgt.values * scale_on.values + shift_on.values)
    term_off = p_off[:, np.newaxis] * (log_tgt.values * scale_off.values + shift_off.values)
    tgt_soft = pd.DataFrame(term_on + term_off, columns=common, index=tgt_raw.index)
    auc, err = evaluate_arm(tgt_soft, "Arm 4")
    results.append({"Arm": "4. Soft GMM (N=1 Champion)", "ER_AUC": auc, "Centroid_Err": err})

    # Summary
    res_df = pd.DataFrame(results)
    print("\n--- ABLATION RESULTS: EMBEDDING-TO-GENE ---")
    print(res_df.to_string(index=False))
    
    os.makedirs("outputs", exist_ok=True)
    report = "# Embedding-to-Gene Ablation Study\n\n"
    report += res_df.to_markdown(index=False)
    with open("outputs/ablation_gene_correction.md", "w") as f: f.write(report)

if __name__ == "__main__":
    main()
