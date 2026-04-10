#!/usr/bin/env python3
"""
Rule 17 — Find Invariant Anchors (Strict N=1 Ref-Only)

Identify ~500 anchors using ONLY Dataset A (Reference).
1. Stable expression (Lowest CV in Ref).
2. Low biological signal (Lowest correlation with Ref PCA components).
Target data is only used to ensure gene name overlap.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description="Find invariant anchors (Reference-Only)")
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading Reference dataset for anchor selection...")
    df_a = pd.read_csv(args.dataset_a, index_col=0, low_memory=False)
    meta_cols = [c for c in df_a.columns if c.startswith('meta_')]
    df_a = df_a.drop(columns=meta_cols, errors='ignore').select_dtypes(include=[np.number])
    
    # Load headers of B only to ensure overlap
    df_b_cols = pd.read_csv(args.dataset_b, index_col=0, nrows=0).columns
    
    common = sorted(list(set(df_a.columns) & set(df_b_cols)))
    log_a = np.log(df_a[common] - df_a[common].min(axis=0) + 1.0)

    # 1. Intra-batch stability: Low Coefficient of Variation (CV)
    cv_a = log_a.std(axis=0) / np.maximum(log_a.mean(axis=0), 1e-12)

    # 2. Low Biological Signal: Correlation with top 5 Reference PCs
    pca = PCA(n_components=5, random_state=42)
    pc_scores = pca.fit_transform(log_a.values)
    
    max_pc_corr = []
    for i, gene in enumerate(common):
        corrs = [abs(np.corrcoef(log_a[gene], pc_scores[:, k])[0,1]) for k in range(5)]
        max_pc_corr.append(max(corrs))
    max_pc_corr = np.array(max_pc_corr)

    # Filter to genes with pc_corr < 0.2 (non-hubs)
    mask = max_pc_corr < 0.2
    candidates = np.array(common)[mask]
    
    # Rank by lowest CV
    final_anchors = candidates[np.argsort(cv_a[mask].values)[:500]]

    print(f"Selected {len(final_anchors)} technical anchors using REF-ONLY criteria.")
    with open(args.output, 'w') as f:
        for g in final_anchors:
            f.write(f"{g}\n")

if __name__ == "__main__":
    main()
