#!/usr/bin/env python3
"""
Visualize the OT transport plan P as a heatmap.

Sorts samples by ER status within each dataset so biological structure
is visible. Also shows the cost matrix C for comparison.

Outputs:
  transport_plan.png — heatmap of P (ref × tgt), sorted by ER status
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.spatial.distance as dist
import scipy.special as sp
from sklearn.decomposition import PCA

EPS = 1e-8


# ---------------------------------------------------------------------------
# PC1 embedding (same as 06_basis_align.py)
# ---------------------------------------------------------------------------

def pc1_embed(expression_df, gene_sets):
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in expression_df.columns]
        if len(genes) < 3:
            scores[set_name] = np.zeros(len(expression_df))
            continue
        X = expression_df[genes].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pca = PCA(n_components=1, random_state=42)
        scores[set_name] = pca.fit_transform(X_std).ravel()
    return pd.DataFrame(scores, index=expression_df.index)


# ---------------------------------------------------------------------------
# Sinkhorn UOT — returns P matrix too
# ---------------------------------------------------------------------------

def sinkhorn_uot_full(X_embed, Y_embed, ot_epsilon=0.01, ot_tau=0.95):
    """Run Sinkhorn UOT, return (P, C, w_ref, w_tgt, mass)."""
    N_ref, N_tgt = X_embed.shape[0], Y_embed.shape[0]
    C = dist.cdist(X_embed, Y_embed, metric="sqeuclidean")
    C = C / (np.max(C) + EPS)

    log_a = np.log(np.ones(N_ref) / N_ref)
    log_b = np.log(np.ones(N_tgt) / N_tgt)
    f, g = np.zeros(N_ref), np.zeros(N_tgt)
    fi = ot_tau / (ot_tau + ot_epsilon)

    for iteration in range(1000):
        f_prev = f.copy()
        g = fi * (log_b - sp.logsumexp((-C / ot_epsilon) + f[:, None], axis=0))
        f = fi * (log_a - sp.logsumexp((-C / ot_epsilon) + g[None, :], axis=1))
        if np.max(np.abs(f - f_prev)) < 1e-5:
            break

    P = np.exp((-C / ot_epsilon) + f[:, None] + g[None, :])
    w_ref = np.sum(P, axis=1) * N_ref
    w_tgt = np.sum(P, axis=0) * N_tgt
    return P, C, w_ref, w_tgt, float(np.sum(P))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset_with_meta(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    return gene_df, meta_df


def log_transform(df):
    return np.log(df - df.min(axis=0) + 1.0)


def get_er_status(meta):
    col = "meta_er_status" if "meta_er_status" in meta.columns else "meta_ER_status"
    if col not in meta.columns:
        return np.zeros(len(meta))
    return meta[col].fillna(-1).values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize OT transport plan P.")
    parser.add_argument("--gene-sets", required=True)
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--ot-epsilon", type=float, default=0.01)
    parser.add_argument("--ot-tau", type=float, default=0.95)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gene_sets) as f:
        gene_sets = json.load(f)

    gene_a, meta_a = load_dataset_with_meta(args.dataset_a)
    gene_b, meta_b = load_dataset_with_meta(args.dataset_b)
    log_a = log_transform(gene_a)
    log_b = log_transform(gene_b)

    print("Computing hub-weighted rank embeddings...")
    from scipy.stats import rankdata

    all_common = sorted(set(log_a.columns) & set(log_b.columns))
    n_all = len(all_common)

    def global_rank_norm(df, cols):
        vals = df[cols].values
        ranked = np.zeros_like(vals)
        for i in range(vals.shape[0]):
            r = rankdata(vals[i], method='average')
            ranked[i] = 2.0 * (r - 1) / max(n_all - 1, 1) - 1.0
        return pd.DataFrame(ranked, index=df.index, columns=cols)

    R_ref_all = global_rank_norm(log_a, all_common)
    R_tgt_all = global_rank_norm(log_b, all_common)

    ref_scores, tgt_scores = {}, {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in all_common]
        if len(genes) < 3:
            ref_scores[set_name] = np.zeros(len(log_a))
            tgt_scores[set_name] = np.zeros(len(log_b))
            continue
        hw = np.array([gene_weights[g] for g in genes])
        R_ref = R_ref_all[genes].values * hw[np.newaxis, :]
        R_tgt = R_tgt_all[genes].values * hw[np.newaxis, :]
        pca = PCA(n_components=1, random_state=42)
        pca.fit(np.vstack([R_ref, R_tgt]))
        fve = pca.explained_variance_ratio_[0]
        ref_scores[set_name] = pca.transform(R_ref).ravel() * fve
        tgt_scores[set_name] = pca.transform(R_tgt).ravel() * fve

    scores_a = pd.DataFrame(ref_scores, index=log_a.index)
    scores_b = pd.DataFrame(tgt_scores, index=log_b.index)

    # Feed directly to Sinkhorn — no per-axis standardization (FVE scaling preserved)
    X_embed = scores_a.values
    Y_embed = scores_b.values

    print(f"Running Sinkhorn (epsilon={args.ot_epsilon})...")
    P, C, w_ref, w_tgt, mass = sinkhorn_uot_full(X_embed, Y_embed, ot_epsilon=args.ot_epsilon, ot_tau=args.ot_tau)
    print(f"  Mass={mass:.4f}, P shape={P.shape}")
    print(f"  P min={P.min():.2e}, max={P.max():.2e}, mean={P.mean():.2e}")
    print(f"  w_ref — min={w_ref.min():.4f} mean={w_ref.mean():.4f} max={w_ref.max():.4f} std={w_ref.std():.4f}")
    print(f"  w_tgt — min={w_tgt.min():.4f} mean={w_tgt.mean():.4f} max={w_tgt.max():.4f} std={w_tgt.std():.4f}")

    # Print weights by ER status
    er_a_vals = get_er_status(meta_a)
    er_b_vals = get_er_status(meta_b)
    for label, val in [("ER-", 0), ("ER+", 1)]:
        mask_a = er_a_vals == val
        mask_b = er_b_vals == val
        if mask_a.sum() > 0:
            print(f"  w_ref [{label}, n={mask_a.sum()}] mean={w_ref[mask_a].mean():.4f} std={w_ref[mask_a].std():.4f}")
        if mask_b.sum() > 0:
            print(f"  w_tgt [{label}, n={mask_b.sum()}] mean={w_tgt[mask_b].mean():.4f} std={w_tgt[mask_b].std():.4f}")

    # Sort by ER status for visual structure
    er_a = get_er_status(meta_a)
    er_b = get_er_status(meta_b)
    sort_a = np.argsort(er_a)
    sort_b = np.argsort(er_b)

    P_sorted = P[sort_a][:, sort_b]
    C_sorted = C[sort_a][:, sort_b]

    # ER status boundaries for annotation
    n_er_neg_a = (er_a[sort_a] == 0).sum()
    n_er_neg_b = (er_b[sort_b] == 0).sum()

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Transport plan P
    ax = axes[0]
    im = ax.imshow(P_sorted, aspect="auto", cmap="hot", interpolation="nearest")
    ax.axhline(n_er_neg_a - 0.5, color="cyan", lw=1, ls="--", alpha=0.7)
    ax.axvline(n_er_neg_b - 0.5, color="cyan", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Target (B) samples")
    ax.set_ylabel("Reference (A) samples")
    ax.set_title(f"Transport plan P\n(ε={args.ot_epsilon}, mass={mass:.3f})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.text(n_er_neg_b / 2, -8, "ER-", ha="center", color="cyan", fontsize=9)
    ax.text(n_er_neg_b + (len(er_b) - n_er_neg_b) / 2, -8, "ER+", ha="center", color="cyan", fontsize=9)

    # Cost matrix C
    ax = axes[1]
    im = ax.imshow(C_sorted, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.axhline(n_er_neg_a - 0.5, color="white", lw=1, ls="--", alpha=0.7)
    ax.axvline(n_er_neg_b - 0.5, color="white", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Target (B) samples")
    ax.set_ylabel("Reference (A) samples")
    ax.set_title("Cost matrix C (sqeuclidean, normalized)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # P row-normalized (per ref sample: where does its mass go?)
    ax = axes[2]
    P_row_norm = P_sorted / (P_sorted.sum(axis=1, keepdims=True) + EPS)
    im = ax.imshow(P_row_norm, aspect="auto", cmap="hot", interpolation="nearest")
    ax.axhline(n_er_neg_a - 0.5, color="cyan", lw=1, ls="--", alpha=0.7)
    ax.axvline(n_er_neg_b - 0.5, color="cyan", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Target (B) samples")
    ax.set_ylabel("Reference (A) samples")
    ax.set_title("P row-normalized\n(per-ref-sample transport distribution)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("OT Transport Plan — sorted by ER status (cyan lines = ER-/ER+ boundary)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "transport_plan.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
