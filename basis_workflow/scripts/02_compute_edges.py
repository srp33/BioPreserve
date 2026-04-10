#!/usr/bin/env python3
"""
Rule 1 — Per-Dataset Edge Computation

Load a single expression dataset, filter to deduplicated common genes,
log-transform, fit 2-component GMMs, then build a directed edge list
using Cohen's d × responsibility correlation as edge weight.

Outputs:
  edges_{label}.csv  — columns: source, target, weight
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path to import basis package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis import gmm_adjust


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    """Load a CSV, keep only numeric non-meta_* columns."""
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df.select_dtypes(include=[np.number])
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if meta_cols:
        df = df.drop(columns=meta_cols)
    print(f"  {csv_path}: {df.shape[0]} samples × {df.shape[1]} genes")
    return df


def load_common_genes(path):
    """Read common_genes.txt (one gene per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_dup_map(path):
    """Read gene_dup_map.csv → dict {duplicate: representative}."""
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return dict(zip(df["duplicate"], df["representative"]))


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------

def compute_edges(log_df, resp_lower, resp_upper,
                  top_k, corr_ceiling):
    """
    Build directed edge list using Cohen's d × responsibility correlation,
    filtered by a 2-component Gaussian Mixture Model (unsupervised thresholding).

    For each anchor gene i, uses its GMM responsibilities as soft group
    weights to compute Cohen's d of every target gene j, then multiplies
    by the responsibility correlation R_ij.

    Parameters
    ----------
    log_df : DataFrame  (samples × genes), log-transformed
    resp_lower, resp_upper : arrays [n_genes × n_samples]
    top_k : int — max edges per anchor gene
    corr_ceiling : float — exclude pairs with |expression r| above this

    Returns
    -------
    edges : list of (source, target, weight) tuples
    """
    gene_names = log_df.columns.tolist()
    n_genes = len(gene_names)
    values = log_df.values  # (n_samples, n_genes)
    values_sq = values ** 2  # precompute for variance

    # --- Precompute per-anchor scalars ---
    sum_w0_all = resp_lower.sum(axis=1)  # (n_genes,)
    sum_w1_all = resp_upper.sum(axis=1)
    valid_anchor = (sum_w0_all >= 2) & (sum_w1_all >= 2)

    # --- Precompute normalised upper-component responsibilities for correlation ---
    resp_centered = resp_upper - resp_upper.mean(axis=1, keepdims=True)
    resp_norms = np.linalg.norm(resp_centered, axis=1, keepdims=True)
    resp_norms = np.maximum(resp_norms, 1e-12)
    resp_normed = resp_centered / resp_norms  # (n_genes, n_samples)

    # --- Expression correlation mask (for corr_ceiling filter) ---
    high_corr_mask = None
    if corr_ceiling is not None:
        print(f"  Precomputing expression correlation mask (|r| > {corr_ceiling})...")
        centered = values - values.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        standardized = centered / norms
        high_corr_mask = np.zeros((n_genes, n_genes), dtype=bool)
        chunk = 500
        for start in range(0, n_genes, chunk):
            end = min(start + chunk, n_genes)
            block = standardized[:, start:end].T @ standardized
            high_corr_mask[start:end, :] = np.abs(block) > corr_ceiling
        np.fill_diagonal(high_corr_mask, False)

    # --- Batched edge construction ---
    print(f"  Computing edges (top-{top_k}/anchor)...")
    candidate_edges = []
    candidate_weights = []
    candidate_signs = []
    BATCH = 500

    for b_start in range(0, n_genes, BATCH):
        b_end = min(b_start + BATCH, n_genes)
        batch_valid = valid_anchor[b_start:b_end]
        if not batch_valid.any():
            continue

        w0_batch = resp_lower[b_start:b_end]
        w1_batch = resp_upper[b_start:b_end]
        sw0 = sum_w0_all[b_start:b_end, None]
        sw1 = sum_w1_all[b_start:b_end, None]

        mean0 = (w0_batch @ values) / np.maximum(sw0, 1e-12)
        mean1 = (w1_batch @ values) / np.maximum(sw1, 1e-12)

        mean_sq0 = (w0_batch @ values_sq) / np.maximum(sw0, 1e-12)
        mean_sq1 = (w1_batch @ values_sq) / np.maximum(sw1, 1e-12)

        var0 = np.maximum(mean_sq0 - mean0 ** 2, 0)
        var1 = np.maximum(mean_sq1 - mean1 ** 2, 0)

        pooled_std = np.sqrt(np.maximum(0.5 * (var0 + var1), 1e-12))
        effect_size = np.abs(mean1 - mean0) / pooled_std

        del mean0, mean1, mean_sq0, mean_sq1, var0, var1

        resp_corr = resp_normed[b_start:b_end] @ resp_normed.T

        for bi in range(b_end - b_start):
            i = b_start + bi
            if not batch_valid[bi]:
                continue

            d_i = effect_size[bi]
            rc_i = resp_corr[bi]

            # Candidate mask: any non-zero Cohen's d and non-zero resp correlation, no self-edges
            sig_mask = (d_i > 0) & (rc_i != 0)
            sig_mask[i] = False

            if high_corr_mask is not None:
                sig_mask &= ~high_corr_mask[i]

            # Use absolute responsibility correlation for weight
            geo_w = d_i * np.abs(rc_i)
            
            # For candidates, we don't apply a floor yet, but we drop non-positive
            kept_idx = np.where(sig_mask & (geo_w > 0))[0]
            if len(kept_idx) == 0:
                continue
            kept_w = geo_w[kept_idx]
            kept_signs = np.sign(rc_i[kept_idx])

            # Top-K per anchor before GMM to save memory
            if len(kept_w) > top_k:
                top_order = np.argpartition(-kept_w, top_k)[:top_k]
                kept_idx = kept_idx[top_order]
                kept_w = kept_w[top_order]
                kept_signs = kept_signs[top_order]

            src = gene_names[i]
            for j, w, s in zip(kept_idx, kept_w, kept_signs):
                candidate_edges.append((src, gene_names[j]))
                candidate_weights.append(float(w))
                candidate_signs.append(int(s))

    # --- Unsupervised Thresholding via GMM ---
    print(f"  Collected {len(candidate_weights)} candidate edges.")
    if len(candidate_weights) < 2:
        return [(candidate_edges[i][0], candidate_edges[i][1], candidate_weights[i], candidate_signs[i]) 
                for i in range(len(candidate_edges))]
        
    print("  Fitting GMM to edge weights to establish unsupervised floor...")
    from sklearn.mixture import GaussianMixture
    
    weights = np.array(candidate_weights).reshape(-1, 1)
    log_weights = np.log1p(weights * 100) 
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(log_weights)
    
    high_idx = np.argmax(gmm.means_.ravel())
    probs = gmm.predict_proba(log_weights)
    signal_mask = probs[:, high_idx] > 0.5
    signal_mask &= (weights.ravel() >= 0.05)
    
    final_edges = [
        (candidate_edges[i][0], candidate_edges[i][1], candidate_weights[i], candidate_signs[i])
        for i in range(len(candidate_edges)) if signal_mask[i]
    ]
    
    print(f"  Unsupervised GMM retained {len(final_edges)} signal edges (out of {len(candidate_edges)}).")
    return final_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 1: Per-dataset GMM-weighted edge computation"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--common-genes", required=True, help="Path to common_genes.txt")
    parser.add_argument("--dup-map", required=True, help="Path to gene_dup_map.csv")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Max edges per anchor gene (default: 200)")
    parser.add_argument("--corr-ceiling", type=float, default=0.99,
                        help="Exclude gene pairs with |expression r| > this (default: 0.99)")
    parser.add_argument("--output", required=True, help="Output path for edges CSV")
    args = parser.parse_args()

    # 1. Load dataset
    print("Loading dataset...")
    df = load_dataset(args.dataset)

    # 2. Filter to deduplicated common genes
    common_genes = load_common_genes(args.common_genes)
    dup_map = load_dup_map(args.dup_map)
    # common_genes already excludes duplicates; just intersect with dataset columns
    available = [g for g in common_genes if g in df.columns]
    print(f"  Filtering to {len(available)} common genes (of {len(common_genes)} requested)")
    df = df[available]

    # 3. Log-transform: log(x - col_min + 1)
    min_vals = df.min(axis=0)
    log_df = np.log(df - min_vals + 1.0)

    # 4. Fit GMM via gmm_adjust
    print("Fitting 2-component GMM...")
    data_array = df.values.T  # (n_genes, n_samples) — raw values, gmm_adjust handles log
    gmm_result = gmm_adjust.get_gmm_responsibilities(
        data_array, genes_are_rows=True, log_transform=True
    )
    resp_lower, resp_upper = gmm_result["responsibilities"]
    # gmm_adjust returns responsibilities in (n_genes, n_samples) when genes_are_rows=True
    print(f"  Responsibilities shape: {resp_lower.shape}")

    # 5. Compute edges
    print("Computing edges...")
    edges = compute_edges(
        log_df, resp_lower, resp_upper,
        top_k=args.top_k,
        corr_ceiling=args.corr_ceiling,
    )

    # 6. Write output
    edge_df = pd.DataFrame(edges, columns=["source", "target", "weight", "sign"])
    edge_df.to_csv(args.output, index=False)
    print(f"Wrote {len(edges)} edges to {args.output}")


if __name__ == "__main__":
    main()
