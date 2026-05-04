#!/usr/bin/env python3
"""
Compare ssGSEA vs PC1 as biological embeddings for OT alignment.

For each gene community axis, computes:
  - ssGSEA enrichment score (rank-based, hub-weighted)
  - PC1 projection (linear, variance-maximizing)

Then runs Sinkhorn UOT on each embedding and compares:
  - Bimodality of each axis (histogram overlay)
  - OT weight distributions (how sharp vs flat)
  - PCA of the embedding space colored by dataset

Outputs (in --output-dir):
  embedding_bimodality.png  — per-axis histograms: ssGSEA vs PC1
  embedding_ot_weights.png  — OT weight distributions for each method
  embedding_pca.png         — PCA of each embedding, colored by dataset
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.spatial.distance as dist
import scipy.special as sp

EPS = 1e-8

# ---------------------------------------------------------------------------
# ssGSEA (inlined)
# ---------------------------------------------------------------------------

def ssgsea(expression_df, gene_sets, alpha=0.25):
    """Hub-weighted single-sample GSEA."""
    n_samples = len(expression_df)
    gene_names = expression_df.columns.tolist()
    n_genes = len(gene_names)
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        idxs, weights = [], []
        for gene, w in gene_weights.items():
            if gene in gene_names:
                idxs.append(gene_names.index(gene))
                weights.append(abs(w))
        if len(idxs) < 3:
            scores[set_name] = np.zeros(n_samples)
            continue
        idxs = np.array(idxs)
        weights = np.array(weights)
        set_scores = np.zeros(n_samples)
        for i in range(n_samples):
            expr = expression_df.iloc[i].values
            ranked_indices = np.argsort(-expr)
            in_set = np.zeros(n_genes, dtype=bool)
            in_set[idxs] = True
            in_set_ranked = in_set[ranked_indices]
            weight_map = np.zeros(n_genes)
            for idx, w in zip(idxs, weights):
                weight_map[idx] = w
            weight_ranked = weight_map[ranked_indices]
            n_in = in_set_ranked.sum()
            n_out = n_genes - n_in
            if n_in == 0 or n_out == 0:
                continue
            rank_positions = np.arange(n_genes, 0, -1)
            hits = np.where(in_set_ranked, np.abs(rank_positions) ** alpha * weight_ranked, 0)
            hits_sum = hits.sum()
            if hits_sum == 0:
                continue
            hits_norm = hits / hits_sum
            misses = np.where(~in_set_ranked, 1.0 / n_out, 0)
            running_sum = np.cumsum(hits_norm - misses)
            set_scores[i] = running_sum.max() - running_sum.min()
        scores[set_name] = set_scores
    return pd.DataFrame(scores, index=expression_df.index)


# ---------------------------------------------------------------------------
# PC1 projection embedding
# ---------------------------------------------------------------------------

def pc1_embed(expression_df, gene_sets):
    """Project each sample onto PC1 of each gene set's genes.

    PCA is fit on the REFERENCE data only (first call), then applied to
    both. But for this comparison script we fit on the pooled data to
    keep it simple — the key question is bimodality shape, not
    cross-dataset generalization.

    Returns DataFrame (samples × axes) like ssGSEA output.
    """
    n_samples = len(expression_df)
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in expression_df.columns]
        if len(genes) < 3:
            scores[set_name] = np.zeros(n_samples)
            continue
        X = expression_df[genes].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pca = PCA(n_components=1, random_state=42)
        pc1 = pca.fit_transform(X_std).ravel()
        scores[set_name] = pc1
    return pd.DataFrame(scores, index=expression_df.index)


# ---------------------------------------------------------------------------
# Sinkhorn UOT (copied from 06_basis_align.py)
# ---------------------------------------------------------------------------

def sinkhorn_uot(X_pca, Y_pca, ot_epsilon=0.01, ot_tau=0.95):
    """Run Sinkhorn UOT, return (w_ref, w_tgt, intersection_mass)."""
    N_ref, N_tgt = X_pca.shape[0], Y_pca.shape[0]
    C = dist.cdist(X_pca, Y_pca, metric="sqeuclidean")
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
    return w_ref, w_tgt, float(np.sum(P)), iteration + 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df.select_dtypes(include=[np.number])
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if meta_cols:
        df = df.drop(columns=meta_cols)
    return df


def load_dataset_with_meta(csv_path):
    """Load CSV, return (gene_df, meta_df) separately."""
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    return gene_df, meta_df


def log_transform(df):
    return np.log(df - df.min(axis=0) + 1.0)


def run_ot_on_embedding(scores_ref, scores_tgt, pca_var=0.85, ot_epsilon=0.01):
    """Scale, PCA-reduce, run Sinkhorn. Returns (w_ref, w_tgt, mass, iters, pca_obj)."""
    X_act = scores_ref.values.T
    Y_act = scores_tgt.values.T
    X_s = (X_act - X_act.mean(axis=1, keepdims=True)) / (X_act.std(axis=1, keepdims=True) + EPS)
    Y_s = (Y_act - Y_act.mean(axis=1, keepdims=True)) / (Y_act.std(axis=1, keepdims=True) + EPS)
    pca = PCA(n_components=pca_var, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_s.T)
    Y_pca = pca.transform(Y_s.T)
    w_ref, w_tgt, mass, iters = sinkhorn_uot(X_pca, Y_pca, ot_epsilon=ot_epsilon)
    return w_ref, w_tgt, mass, iters, pca, X_pca, Y_pca


# ---------------------------------------------------------------------------
# Plot 1: Per-axis bimodality comparison
# ---------------------------------------------------------------------------

def plot_bimodality(ssgsea_ref, ssgsea_tgt, pc1_ref, pc1_tgt, output_dir, top_n=8):
    """Side-by-side histograms: ssGSEA vs PC1 for top axes."""
    combined_ss = pd.concat([ssgsea_ref, ssgsea_tgt])
    top_axes = combined_ss.var().nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(top_n, 2, figsize=(12, 2.5 * top_n))
    if top_n == 1:
        axes = axes.reshape(1, 2)

    for i, ax_name in enumerate(top_axes):
        # ssGSEA
        ax = axes[i, 0]
        ax.hist(ssgsea_ref[ax_name], bins=30, alpha=0.5, color="steelblue",
                label="Ref", density=True)
        ax.hist(ssgsea_tgt[ax_name], bins=30, alpha=0.5, color="coral",
                label="Tgt", density=True)
        ax.set_title(f"{ax_name} — ssGSEA", fontsize=9)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7)

        # PC1
        ax = axes[i, 1]
        ax.hist(pc1_ref[ax_name], bins=30, alpha=0.5, color="steelblue",
                label="Ref", density=True)
        ax.hist(pc1_tgt[ax_name], bins=30, alpha=0.5, color="coral",
                label="Tgt", density=True)
        ax.set_title(f"{ax_name} — PC1", fontsize=9)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Bimodality comparison: ssGSEA (left) vs PC1 (right)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "embedding_bimodality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: OT weight distributions
# ---------------------------------------------------------------------------

def plot_ot_weights(results, output_dir):
    """Compare OT weight distributions for each embedding method."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, (method, res) in enumerate(results.items()):
        w_ref, w_tgt = res["w_ref"], res["w_tgt"]

        ax = axes[0, col]
        ax.hist(w_ref, bins=40, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_title(f"{method} — w_ref (std={w_ref.std():.4f})", fontsize=10)
        ax.axvline(w_ref.mean(), color="red", ls="--", alpha=0.7)
        ax.set_xlabel("Weight")

        ax = axes[1, col]
        ax.hist(w_tgt, bins=40, alpha=0.7, color="coral", edgecolor="white")
        ax.set_title(f"{method} — w_tgt (std={w_tgt.std():.4f})", fontsize=10)
        ax.axvline(w_tgt.mean(), color="red", ls="--", alpha=0.7)
        ax.set_xlabel("Weight")

    plt.suptitle("OT weight distributions: ssGSEA vs PC1", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "embedding_ot_weights.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: PCA of embedding space
# ---------------------------------------------------------------------------

def plot_embedding_pca(results, output_dir):
    """PCA scatter of each embedding, colored by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for col, (method, res) in enumerate(results.items()):
        ax = axes[col]
        X_pca, Y_pca = res["X_pca"], res["Y_pca"]
        combined = np.vstack([X_pca, Y_pca])
        # Use first 2 components of the already-PCA'd space
        if combined.shape[1] >= 2:
            x, y = combined[:, 0], combined[:, 1]
        else:
            x = combined[:, 0]
            y = np.zeros_like(x)
        n_ref = X_pca.shape[0]
        ax.scatter(x[:n_ref], y[:n_ref], c="steelblue", alpha=0.5, s=25, label="Reference")
        ax.scatter(x[n_ref:], y[n_ref:], c="coral", alpha=0.5, s=25, label="Target")
        ax.set_title(f"{method} embedding (PCA space)", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Embedding PCA: ssGSEA vs PC1", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "embedding_pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: ER vs HER2 scatter — ssGSEA vs PC1, colored by subtype
# ---------------------------------------------------------------------------

def _find_axis(gene_sets, gene_name):
    for name, genes in gene_sets.items():
        if gene_name in genes:
            return name
    return None


def _get_subtype(er, her2):
    if pd.isna(er) or pd.isna(her2):
        return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"


def _resolve_col(meta, *candidates):
    for c in candidates:
        if c in meta.columns:
            return c
    return None


def plot_scatter_comparison(ss_ref, ss_tgt, pc1_ref, pc1_tgt,
                            meta_ref, meta_tgt, gene_sets, output_dir):
    """ER vs HER2 scatter in both embedding spaces, colored by subtype."""
    er_axis = _find_axis(gene_sets, "ESR1") or list(gene_sets.keys())[0]
    her2_axis = _find_axis(gene_sets, "ERBB2")
    if her2_axis is None:
        keys = list(gene_sets.keys())
        her2_axis = keys[1] if len(keys) > 1 else keys[0]

    color_map = {
        "ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange",
        "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for col, (method, scores_ref, scores_tgt) in enumerate([
        ("ssGSEA", ss_ref, ss_tgt),
        ("PC1", pc1_ref, pc1_tgt),
    ]):
        ax = axes[col]
        for scores, meta, marker, ds in [
            (scores_ref, meta_ref, "o", "Ref"),
            (scores_tgt, meta_tgt, "^", "Tgt"),
        ]:
            er_col = _resolve_col(meta, "meta_er_status", "meta_ER_status")
            her2_col = _resolve_col(meta, "meta_her2_status", "meta_HER2_status")
            if not er_col or not her2_col:
                continue
            subtypes = [_get_subtype(e, h) for e, h in
                        zip(meta[er_col], meta[her2_col])]
            for st in ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]:
                mask = np.array([s == st for s in subtypes])
                if mask.sum() == 0:
                    continue
                kw = dict(c=color_map[st], marker=marker, alpha=0.5, s=30,
                          label=f"{ds} {st}")
                if marker == "o":
                    kw["edgecolors"] = "white"
                    kw["linewidths"] = 0.3
                ax.scatter(scores[er_axis].values[mask],
                           scores[her2_axis].values[mask], **kw)
        ax.set_xlabel(f"{er_axis} (ESR1)")
        ax.set_ylabel(f"{her2_axis} (ERBB2)")
        ax.set_title(f"{method}: ER vs HER2 axis")
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(True, alpha=0.2)

    plt.suptitle("ER vs HER2 scatter: ssGSEA (left) vs PC1 (right)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "embedding_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ssGSEA vs PC1 embeddings for OT alignment.",
    )
    parser.add_argument("--gene-sets", required=True, help="gene_community_sets.json")
    parser.add_argument("--dataset-a", required=True, help="Reference dataset CSV")
    parser.add_argument("--dataset-b", required=True, help="Target dataset CSV")
    parser.add_argument("--ot-epsilon", type=float, default=0.01,
                        help="Sinkhorn entropy regularization (default: 0.01)")
    parser.add_argument("--output-dir", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    with open(args.gene_sets) as f:
        gene_sets = json.load(f)
    print(f"{len(gene_sets)} axes loaded")

    print("Loading datasets...")
    gene_a, meta_a = load_dataset_with_meta(args.dataset_a)
    gene_b, meta_b = load_dataset_with_meta(args.dataset_b)
    df_a = log_transform(gene_a)
    df_b = log_transform(gene_b)
    print(f"  Ref: {df_a.shape}, Tgt: {df_b.shape}")

    # Compute both embeddings
    print("\nComputing ssGSEA embedding...")
    ss_ref = ssgsea(df_a, gene_sets)
    ss_tgt = ssgsea(df_b, gene_sets)
    print(f"  ssGSEA shapes: ref={ss_ref.shape}, tgt={ss_tgt.shape}")

    print("Computing PC1 embedding...")
    pc1_ref = pc1_embed(df_a, gene_sets)
    pc1_tgt = pc1_embed(df_b, gene_sets)
    print(f"  PC1 shapes: ref={pc1_ref.shape}, tgt={pc1_tgt.shape}")

    # Run OT on each
    print(f"\nRunning Sinkhorn UOT (epsilon={args.ot_epsilon}) on ssGSEA embedding...")
    w_ref_ss, w_tgt_ss, mass_ss, iters_ss, pca_ss, Xp_ss, Yp_ss = \
        run_ot_on_embedding(ss_ref, ss_tgt, ot_epsilon=args.ot_epsilon)
    print(f"  Converged in {iters_ss} iters, mass={mass_ss:.4f}")
    print(f"  w_ref std={w_ref_ss.std():.4f}, w_tgt std={w_tgt_ss.std():.4f}")

    print(f"Running Sinkhorn UOT (epsilon={args.ot_epsilon}) on PC1 embedding...")
    w_ref_pc, w_tgt_pc, mass_pc, iters_pc, pca_pc, Xp_pc, Yp_pc = \
        run_ot_on_embedding(pc1_ref, pc1_tgt, ot_epsilon=args.ot_epsilon)
    print(f"  Converged in {iters_pc} iters, mass={mass_pc:.4f}")
    print(f"  w_ref std={w_ref_pc.std():.4f}, w_tgt std={w_tgt_pc.std():.4f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'ssGSEA':>12} {'PC1':>12}")
    print("-" * 60)
    print(f"{'Intersection mass':<30} {mass_ss:>12.4f} {mass_pc:>12.4f}")
    print(f"{'Sinkhorn iterations':<30} {iters_ss:>12d} {iters_pc:>12d}")
    print(f"{'w_ref std':<30} {w_ref_ss.std():>12.4f} {w_ref_pc.std():>12.4f}")
    print(f"{'w_tgt std':<30} {w_tgt_ss.std():>12.4f} {w_tgt_pc.std():>12.4f}")
    print(f"{'w_ref min/max':<30} {w_ref_ss.min():>5.3f}/{w_ref_ss.max():<5.3f}  {w_ref_pc.min():>5.3f}/{w_ref_pc.max():<5.3f}")
    print(f"{'w_tgt min/max':<30} {w_tgt_ss.min():>5.3f}/{w_tgt_ss.max():<5.3f}  {w_tgt_pc.min():>5.3f}/{w_tgt_pc.max():<5.3f}")
    print("=" * 60)

    results = {
        "ssGSEA": {"w_ref": w_ref_ss, "w_tgt": w_tgt_ss, "X_pca": Xp_ss, "Y_pca": Yp_ss},
        "PC1":    {"w_ref": w_ref_pc, "w_tgt": w_tgt_pc, "X_pca": Xp_pc, "Y_pca": Yp_pc},
    }

    # Plots
    print("\nPlot 1: Bimodality comparison...")
    plot_bimodality(ss_ref, ss_tgt, pc1_ref, pc1_tgt, args.output_dir)

    print("Plot 2: OT weight distributions...")
    plot_ot_weights(results, args.output_dir)

    print("Plot 3: Embedding PCA...")
    plot_embedding_pca(results, args.output_dir)

    print("Plot 4: ER vs HER2 scatter...")
    plot_scatter_comparison(ss_ref, ss_tgt, pc1_ref, pc1_tgt,
                            meta_a, meta_b, gene_sets, args.output_dir)

    print(f"\nDone. All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
