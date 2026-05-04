#!/usr/bin/env python3
"""
Rule 5 — Dictionary Construction

Merge Leiden communities into independent biological axes using greedy
merge with Hub-Weighted Mean correlation and per-dataset redundancy checks.
Enforces Push-Pull logic (signed merging) and unsupervised GMM thresholding.

Outputs:
  gene_community_sets.json — {axis_name: {gene: signed_hub_weight, ...}}
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if meta_cols:
        df = df.drop(columns=meta_cols)
    return df.select_dtypes(include=[np.number])


def log_transform(df):
    return np.log(df - df.min(axis=0) + 1.0)


def load_common_genes(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# PCA Ranking
# ---------------------------------------------------------------------------

def rank_communities_by_eigenvalue(communities, hits_weights, log_expr, min_size=10):
    """Rank communities by their Dataset_A PC1 eigenvalue (variance explained)."""
    ranked = []
    for comm_id, genes in communities.items():
        if len(genes) < min_size:
            continue
        valid = [g for g in genes if g in log_expr.columns]
        if len(valid) < min_size:
            continue

        X = log_expr[valid].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pca = PCA(n_components=1)
        pca.fit(X_std)
        ranked.append({
            "community": comm_id,
            "genes": valid,
            "eigenvalue": float(pca.explained_variance_[0]),
        })
    return sorted(ranked, key=lambda x: x["eigenvalue"], reverse=True)


# ---------------------------------------------------------------------------
# Greedy merge with per-dataset correlation (Push-Pull)
# ---------------------------------------------------------------------------

def compute_weighted_mean(log_expr, genes, weights_dict, sample_mask=None):
    """Compute hub-weighted mean for a gene set, optionally on a sample subset."""
    X = log_expr[genes].values
    if sample_mask is not None:
        X = X[sample_mask]
    X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    w = np.array([weights_dict.get(g, 0.01) for g in genes])
    return (X_std * w).sum(axis=1) / np.maximum(w.sum(), 1e-12)


def get_unsupervised_merge_threshold(ranked_communities, log_expr, mask_a, mask_b, hits_weights):
    """Find a conservative, tail-focused merge threshold using a 3-component GMM."""
    import itertools
    
    # Consider top communities to find typical correlation distributions
    top_comms = ranked_communities[:100]
    corrs = []
    
    means_a = {}
    means_b = {}
    for rec in top_comms:
        c_id = rec["community"]
        g = list(rec["genes"])
        w = hits_weights.get(str(c_id), {x: 0.01 for x in g})
        means_a[c_id] = compute_weighted_mean(log_expr, g, w, mask_a)
        means_b[c_id] = compute_weighted_mean(log_expr, g, w, mask_b)
        
    for c1, c2 in itertools.combinations([r["community"] for r in top_comms], 2):
        r_a = abs(np.corrcoef(means_a[c1], means_a[c2])[0, 1])
        r_b = abs(np.corrcoef(means_b[c1], means_b[c2])[0, 1])
        corrs.append(max(r_a, r_b))
        
    corrs = np.array(corrs)
    if len(corrs) < 10: return 0.85
    
    # Use 3 components to better isolate the extreme 'redundancy' tail
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(corrs.reshape(-1, 1))
    
    # The highest-mean component represents the truly redundant axes
    high_idx = np.argmax(gmm.means_.ravel())
    preds = gmm.predict(corrs.reshape(-1, 1))
    high_corrs = corrs[preds == high_idx]
    
    if len(high_corrs) == 0: return 0.85
    threshold = np.min(high_corrs)
    
    # For redundancy, we want to be very strict. Floor at 0.75.
    return float(np.clip(threshold, 0.75, 0.95))


def greedy_merge(ranked_communities, log_expr, n_samples_a, hits_weights):
    """Greedy merge communities into biological axes, allowing Push-Pull anti-correlation merges."""
    mask_a = np.zeros(len(log_expr), dtype=bool)
    mask_a[:n_samples_a] = True
    mask_b = ~mask_a

    print("    Finding unsupervised merge threshold via GMM...")
    merge_threshold = get_unsupervised_merge_threshold(ranked_communities, log_expr, mask_a, mask_b, hits_weights)
    print(f"    GMM selected merge threshold: {merge_threshold:.3f}")

    axes = []

    for rec in ranked_communities:
        comm_id = rec["community"]
        genes = list(rec["genes"])
        w_cand = hits_weights.get(str(comm_id), {g: 0.01 for g in genes})

        merged = False
        for axis in axes:
            axis_genes = list(axis["genes"])
            w_axis = axis["weights"]

            score_cand_a = compute_weighted_mean(log_expr, genes, w_cand, mask_a)
            score_cand_b = compute_weighted_mean(log_expr, genes, w_cand, mask_b)
            score_axis_a = compute_weighted_mean(log_expr, axis_genes, w_axis, mask_a)
            score_axis_b = compute_weighted_mean(log_expr, axis_genes, w_axis, mask_b)

            r_a_val = np.corrcoef(score_cand_a, score_axis_a)[0, 1]
            r_b_val = np.corrcoef(score_cand_b, score_axis_b)[0, 1]
            r_a = abs(r_a_val)
            r_b = abs(r_b_val)
            max_r = max(r_a, r_b)

            # --- Push-Pull Merging ---
            # If max|r| is extremely high, they are redundant (either correlated or anti-correlated)
            if max_r > merge_threshold:
                axis["genes"].update(genes)
                axis["source_ids"].append(comm_id)
                
                # Determine sign alignment: if anti-correlated, flip candidate signs
                flip = (r_a_val < 0) if r_a > r_b else (r_b_val < 0)
                
                for g, w in w_cand.items():
                    w_signed = -w if flip else w
                    if g not in axis["weights"] or abs(w_signed) > abs(axis["weights"][g]):
                        axis["weights"][g] = w_signed
                    
                merged = True
                print(
                    f"    Community {comm_id} ({len(genes)} genes) \u2192 merged axis {axes.index(axis)} "
                    f"(max|r|={max_r:.2f}, flipped={flip})"
                )
                break

        if not merged:
            axes.append({
                "genes": set(genes),
                "source_ids": [comm_id],
                "weights": dict(w_cand)
            })
            print(f"    Community {comm_id} ({len(genes)} genes) \u2192 new axis {len(axes) - 1}")

    return axes


# ---------------------------------------------------------------------------
# HITS weight merging and ghost gene pruning
# ---------------------------------------------------------------------------

def merge_hits_weights(axes):
    weighted_sets = {}
    for i, axis in enumerate(axes):
        weighted_sets[f"axis_{i}"] = axis["weights"]
    return weighted_sets


def prune_ghost_genes(weighted_sets, min_effective_size=5.0):
    """Prune axes based on Effective Size (sum of diffusion mass)."""
    kept = []
    
    for axis_name in sorted(weighted_sets.keys()):
        gw = weighted_sets[axis_name]
        raw_weights = np.array(list(gw.values()))
        abs_weights = np.abs(raw_weights)
        
        # Calculate Effective Size
        eff_size = np.sum(abs_weights)
        if eff_size < min_effective_size:
            continue
            
        # Per-axis GMM to prune internal noise
        weights_2d = abs_weights.reshape(-1, 1)
        log_w = np.log1p(weights_2d * 100)
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        try:
            gmm.fit(log_w)
            high_idx = np.argmax(gmm.means_.ravel())
            probs = gmm.predict_proba(log_w)
            mask = (probs[:, high_idx] > 0.5) & (abs_weights >= 0.01)
        except Exception:
            mask = abs_weights >= 0.05
            
        pruned_genes = {
            g: w for i, (g, w) in enumerate(gw.items()) if mask[i]
        }
        
        # Check final effective size after pruning
        final_eff_size = np.sum(np.abs(list(pruned_genes.values())))
        if final_eff_size >= 3.0: # Need at least 3 effective genes
            kept.append(pruned_genes)

    return {f"axis_{i}": gw for i, gw in enumerate(kept)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule 5: Dictionary Construction")
    parser.add_argument("--communities", required=True)
    parser.add_argument("--hits-weights", required=True)
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--common-genes", required=True)
    parser.add_argument("--gmm-fuzzy-threshold", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading communities...")
    comm_df = pd.read_csv(args.communities)
    communities = defaultdict(list)
    for _, row in comm_df.iterrows():
        communities[row["community"]].append(row["gene"])

    print("Loading HITS weights...")
    with open(args.hits_weights) as f:
        hits_weights = json.load(f)

    print("Loading datasets...")
    common_genes = load_common_genes(args.common_genes)
    df_a = load_dataset(args.dataset_a)
    df_b = load_dataset(args.dataset_b)

    shared = sorted(set(df_a.columns) & set(df_b.columns) & set(common_genes))
    df_a, df_b = df_a[shared], df_b[shared]
    n_samples_a = len(df_a)
    pooled = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    log_expr = log_transform(pooled)

    print("\nRanking communities by PC1 eigenvalue...")
    ranked = rank_communities_by_eigenvalue(communities, hits_weights, log_expr)

    print("\nGreedy merge using unsupervised GMM thresholding (Push-Pull)...")
    axes = greedy_merge(ranked, log_expr, n_samples_a, hits_weights)

    print("\nFormatting weights...")
    weighted_sets = merge_hits_weights(axes)

    print("\nPruning ghost genes using Effective Size threshold...")
    final_sets = prune_ghost_genes(weighted_sets, min_effective_size=5.0)
    
    print(f"  {len(final_sets)} axes retained after pruning")
    for axis_name, gw in sorted(final_sets.items()):
        eff_size = sum(np.abs(list(gw.values())))
        print(f"    {axis_name}: {len(gw)} genes (Effective Size: {eff_size:.2f})")

    with open(args.output, "w") as f:
        json.dump(final_sets, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
