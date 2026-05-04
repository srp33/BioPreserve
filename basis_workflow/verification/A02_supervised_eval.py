#!/usr/bin/env python3
"""
Rule 14 — Supervised Evaluation

Calculate AUC for predicting meta_er_status and meta_her2_status
from unsupervised community scores. This provides a gold-standard
metric to tune the unsupervised clustering algorithm.
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def load_dataset_full(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors='ignore')
    gene_df = gene_df.select_dtypes(include=[np.number])
    return gene_df, meta_df

def compute_weighted_means(log_expr, gene_sets):
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in log_expr.columns]
        if len(genes) < 3:
            scores[set_name] = np.zeros(len(log_expr))
            continue
        X = log_expr[genes].values
        # Standardize
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        # Weights
        w = np.array([gene_weights[g] for g in genes])
        # Weighted mean
        scores[set_name] = (X_std * w).sum(axis=1) / np.maximum(w.sum(), 1e-12)
    return pd.DataFrame(scores, index=log_expr.index)

def main():
    parser = argparse.ArgumentParser(description="Supervised evaluation of unsupervised axes")
    parser.add_argument("--gene-sets", required=True)
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    args = parser.parse_args()

    with open(args.gene_sets) as f:
        gene_sets = json.load(f)

    results = []
    for name, csv in [("A", args.dataset_a), ("B", args.dataset_b)]:
        gene_df, meta_df = load_dataset_full(csv)
        # Filter meta for status columns
        er_col = [c for c in meta_df.columns if "er_status" in c.lower()]
        her2_col = [c for c in meta_df.columns if "her2_status" in c.lower()]
        
        if not er_col or not her2_col:
            print(f"Skipping {name}: status columns not found")
            continue
            
        er_labels = meta_df[er_col[0]]
        her2_labels = meta_df[her2_col[0]]
        
        # Log transform
        log_df = np.log(gene_df - gene_df.min(axis=0) + 1.0)
        scores_df = compute_weighted_means(log_df, gene_sets)
        
        # Calculate AUC for each axis
        er_aucs = []
        her2_aucs = []
        
        valid_er = ~er_labels.isna()
        valid_her2 = ~her2_labels.isna()
        
        axis_names = list(scores_df.columns)
        for axis in axis_names:
            s = scores_df[axis].values
            
            e_auc = 0.5
            if valid_er.sum() > 0 and len(np.unique(er_labels[valid_er])) > 1:
                e_auc = max(roc_auc_score(er_labels[valid_er], s[valid_er]), 
                            1 - roc_auc_score(er_labels[valid_er], s[valid_er]))
            er_aucs.append(e_auc)
            
            h_auc = 0.5
            if valid_her2.sum() > 0 and len(np.unique(her2_labels[valid_her2])) > 1:
                h_auc = max(roc_auc_score(her2_labels[valid_her2], s[valid_her2]), 
                            1 - roc_auc_score(her2_labels[valid_her2], s[valid_her2]))
            her2_aucs.append(h_auc)
        
        er_aucs = np.array(er_aucs)
        her2_aucs = np.array(her2_aucs)
        
        # Identification of winners
        best_er_idx = np.argsort(-er_aucs)
        best_her2_idx = np.argsort(-her2_aucs)
        
        best_er_axis = axis_names[best_er_idx[0]]
        best_her2_axis = axis_names[best_her2_idx[0]]
        
        # Purity and Redundancy
        purity = np.abs(er_aucs - her2_aucs) / (er_aucs + her2_aucs - 1.0 + 1e-12)
        strong_er = (er_aucs > 0.8).sum()
        strong_her2 = (her2_aucs > 0.8).sum()
        
        r_best = np.abs(np.corrcoef(scores_df[best_er_axis], scores_df[best_her2_axis])[0, 1])
        
        print(f"\n--- Dataset {name} Details ---")
        print(f"  Best ER Axis:   {best_er_axis} (AUC={er_aucs[best_er_idx[0]]:.3f})")
        if len(er_aucs) > 1:
            print(f"  2nd ER Axis:    {axis_names[best_er_idx[1]]} (AUC={er_aucs[best_er_idx[1]]:.3f})")
        
        print(f"  Best HER2 Axis: {best_her2_axis} (AUC={her2_aucs[best_her2_idx[0]]:.3f})")
        if len(her2_aucs) > 1:
            print(f"  2nd HER2 Axis:  {axis_names[best_her2_idx[1]]} (AUC={her2_aucs[best_her2_idx[1]]:.3f})")
        
        # Top genes for best axes
        def get_top_genes(axis_id, n=5):
            genes = gene_sets.get(axis_id, {})
            sorted_genes = sorted(genes.items(), key=lambda x: x[1], reverse=True)
            return ", ".join([f"{g}({w:.2f})" for g, w in sorted_genes[:n]])

        print(f"  ER Axis Genes:   {get_top_genes(best_er_axis)}")
        print(f"  HER2 Axis Genes: {get_top_genes(best_her2_axis)}")

        results.append({
            "Dataset": name,
            "Max_ER_AUC": er_aucs[best_er_idx[0]],
            "Max_HER2_AUC": her2_aucs[best_her2_idx[0]],
            "ER_Redundancy": strong_er,
            "HER2_Redundancy": strong_her2,
            "Best_Axes_Corr": r_best,
            "Avg_Purity": np.mean(purity[ (er_aucs > 0.7) | (her2_aucs > 0.7) ]),
            "N_Axes": len(axis_names)
        })

    print("\n--- Supervised Separation Summary ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Global summary score: Higher is better
    # Rewards AUC, penalizes Correlation and Redundancy
    if not res_df.empty:
        score = (res_df["Max_ER_AUC"].mean() + res_df["Max_HER2_AUC"].mean()) / 2.0
        score -= 0.2 * res_df["Best_Axes_Corr"].mean()
        # Small penalty for too much redundancy (fragmentation)
        score -= 0.05 * (res_df["ER_Redundancy"].mean() + res_df["HER2_Redundancy"].mean())
        print(f"\nOverall Separation Score: {score:.4f}")

if __name__ == "__main__":
    main()
