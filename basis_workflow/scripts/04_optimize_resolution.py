#!/usr/bin/env python3
"""
Rule 3 — Resolution Optimization

GP-optimize the Leiden resolution parameter using a histogram-based
bimodality proxy (sum_dip) as the objective.  Each evaluation runs a
single Leiden partition and scores it; the GP noise model handles
stochasticity from Leiden's randomness.

Outputs:
  gp_optimization_history.csv — columns: resolution, sum_dip, n_scored, n_communities
  optimal_resolution.txt      — single float (GP-predicted optimum)
"""

import argparse
from collections import defaultdict

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import expected_minimum


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


# ---------------------------------------------------------------------------
# Bimodality scoring
# ---------------------------------------------------------------------------

def _dip_proxy(pc1):
    """Histogram-based bimodality proxy: 1 - (median_density / mode_density)."""
    hist, bin_edges = np.histogram(pc1, bins=20)
    median_bin = min(np.searchsorted(bin_edges[1:], np.median(pc1)), len(hist) - 1)
    mode_density = hist.max()
    median_density = hist[median_bin]
    return 1.0 - (median_density / max(mode_density, 1))


def _gini(weights):
    """Gini coefficient of a weight vector."""
    if len(weights) == 0: return 0
    w = np.sort(np.abs(weights))
    n = len(w)
    if n < 2 or w.sum() == 0: return 0
    cum = np.cumsum(w)
    return 1.0 - 2.0 * np.sum(cum) / (n * cum[-1]) + 1.0 / n


def score_partition_comprehensive(partition, log_genes_a, log_genes_b, g_ig, min_size=10):
    """
    Score a partition using multiple metrics:
    1. Bimodality (Dip proxy)
    2. Hierarchy (Gini of hub weights)
    3. Coherence (PC1 variance ratio)
    """
    communities = defaultdict(list)
    for gene, comm_id in partition.items():
        if gene in log_genes_a.columns:
            communities[comm_id].append(gene)

    total_composite = 0.0
    n_scored = 0
    
    node_names = g_ig.vs["name"]
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    all_sizes = []
    for genes in communities.values():
        n_g = len(genes)
        if n_g < min_size:
            continue
        n_scored += 1
        all_sizes.append(n_g)
        
        # 1. Bimodality & Coherence (per dataset)
        comm_dip = 0
        comm_fve = 0
        for log_df in [log_genes_a, log_genes_b]:
            valid = [g for g in genes if g in log_df.columns]
            if len(valid) < 5: 
                continue
            X = log_df[valid].values
            X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
            pca = PCA(n_components=2)
            pca.fit(X_std)
            pc1 = pca.transform(X_std)[:, 0]
            comm_dip += _dip_proxy(pc1)
            if pca.explained_variance_[1] > 1e-12:
                comm_fve += pca.explained_variance_[0] / pca.explained_variance_[1]
            else:
                comm_fve += 10.0
        
        avg_comm_dip = comm_dip / 2.0
        avg_comm_fve = np.log1p(comm_fve / 2.0)

        # 2. Hierarchy (Gini)
        sub_idxs = [name_to_idx[g] for g in genes if g in name_to_idx]
        comm_gini = 0
        if len(sub_idxs) > 2:
            subgraph = g_ig.subgraph(sub_idxs)
            hub_weights = subgraph.hub_score(weights="weight")
            comm_gini = _gini(hub_weights)

        # Quality = (Bimodality AND Hierarchy AND Coherence)
        # Squaring rewards "Exceptional" axes over many mediocre ones
        total_composite += (avg_comm_dip * comm_gini * avg_comm_fve) ** 2

    # 3. Balance (Normalized Entropy)
    # Rewards partitions where genes are spread across many communities rather than one massive one.
    balance_penalty = 1.0
    if n_scored > 1:
        s = np.array(all_sizes)
        p = s / s.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        max_entropy = np.log(len(p))
        balance_penalty = entropy / max_entropy if max_entropy > 0 else 1.0

    # 4. Orthogonality (Diversity)
    # We want axes to be as uncorrelated as possible.
    diversity_penalty = 1.0
    if n_scored > 1:
        all_means = []
        for genes in communities.values():
            if len(genes) < min_size: continue
            valid = [g for g in genes if g in log_genes_a.columns]
            if len(valid) < 5: continue
            X = log_genes_a[valid].values
            X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
            all_means.append(X_std.mean(axis=1))
        
        if len(all_means) > 1:
            means_mat = np.array(all_means)
            corr_mat = np.abs(np.corrcoef(means_mat))
            n = corr_mat.shape[0]
            avg_corr = (corr_mat.sum() - n) / (n * (n - 1))
            diversity_penalty = 1.0 - avg_corr

    if n_scored == 0:
        return 0, 0
        
    # Final Score combines quality, balance, and diversity.
    final_score = total_composite * balance_penalty * diversity_penalty
    return final_score, n_scored


# ---------------------------------------------------------------------------
# GP optimization
# ---------------------------------------------------------------------------

def optimize_resolution(g_ig, all_nodes, log_a, log_b, n_calls=25,
                        res_range=(1.0, 200.0)):
    """GP-optimize Leiden resolution using per-dataset sum_dip as the objective.

    Parameters
    ----------
    g_ig : ig.Graph
        Undirected igraph with edge attribute ``weight``.
    all_nodes : list[str]
        Node names corresponding to igraph vertex indices.
    log_a : pd.DataFrame
        Log-transformed expression matrix for Dataset A.
    log_b : pd.DataFrame
        Log-transformed expression matrix for Dataset B.
    n_calls : int
        Number of GP evaluations.
    res_range : tuple[float, float]
        (min, max) resolution search range.

    Returns
    -------
    best_resolution : float
        GP-predicted optimal resolution via ``expected_minimum()``.
    history : pd.DataFrame
        Columns: resolution, sum_dip, n_scored, n_communities.
    """
    print(f"\n── GP optimization of resolution ({n_calls} evaluations) ──")
    print(f"  Search range: [{res_range[0]}, {res_range[1]}] (log-uniform)")
    print(f"  Bimodality scored per-dataset (A: {len(log_a)} samples, B: {len(log_b)} samples)")
    history = []

    def objective(params):
        resolution = params[0]
        part = leidenalg.find_partition(
            g_ig.copy(),
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=np.random.randint(0, 10000),
        )
        partition = {
            all_nodes[i]: part.membership[i] for i in range(len(all_nodes))
        }
        score, n_scored = score_partition_comprehensive(partition, log_a, log_b, g_ig)
        n_comms = len(set(partition.values()))
        history.append({
            "resolution": resolution,
            "score": score,
            "n_scored": n_scored,
            "n_communities": n_comms,
        })
        print(
            f"  res={resolution:>7.2f} → score={score:.4f}, "
            f"{n_scored} scored, {n_comms} communities"
        )
        return -score  # minimize negative = maximize score

    result = gp_minimize(
        objective,
        [Real(res_range[0], res_range[1], prior="log-uniform")],
        n_calls=n_calls,
        n_initial_points=min(10, n_calls // 2),
        noise="gaussian",
        random_state=42,
    )

    best_x, best_y = expected_minimum(result)
    best_resolution = best_x[0]
    print(
        f"\n  GP-predicted optimum: resolution={best_resolution:.2f} "
        f"(predicted score={-best_y:.4f})"
    )

    return best_resolution, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 3 — GP-optimize Leiden resolution using bimodality "
                    "proxy (sum_dip) as the objective.",
    )
    parser.add_argument(
        "--graphml", required=True,
        help="Path to undirected igraph GraphML file",
    )
    parser.add_argument(
        "--dataset-a", required=True, help="Path to Dataset A CSV",
    )
    parser.add_argument(
        "--dataset-b", required=True, help="Path to Dataset B CSV",
    )
    parser.add_argument(
        "--common-genes", required=True, help="Path to common_genes.txt",
    )
    parser.add_argument(
        "--gp-n-calls", type=int, default=25,
        help="Number of GP evaluations (default: 25)",
    )
    parser.add_argument(
        "--res-range-min", type=float, default=1.0,
        help="Minimum resolution for search range (default: 1.0)",
    )
    parser.add_argument(
        "--res-range-max", type=float, default=200.0,
        help="Maximum resolution for search range (default: 200.0)",
    )
    parser.add_argument(
        "--output-history", required=True,
        help="Output path for gp_optimization_history.csv",
    )
    parser.add_argument(
        "--output-resolution", required=True,
        help="Output path for optimal_resolution.txt",
    )
    args = parser.parse_args()

    # 1. Load igraph from GraphML and reconstruct node list
    print("Loading graph from GraphML...")
    g = ig.Graph.Read_GraphML(args.graphml)
    all_nodes = g.vs["name"]
    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

    # 2. Load common genes
    common_genes = load_common_genes(args.common_genes)
    print(f"  Common genes: {len(common_genes)}")

    # 3. Load both datasets, filter to common genes, log-transform, concatenate
    print("Loading and preparing expression data...")
    df_a = load_dataset(args.dataset_a)
    df_b = load_dataset(args.dataset_b)

    # Filter to common genes (intersect with available columns)
    genes_a = [g for g in common_genes if g in df_a.columns]
    genes_b = [g for g in common_genes if g in df_b.columns]
    shared = sorted(set(genes_a) & set(genes_b))
    df_a = df_a[shared]
    df_b = df_b[shared]

    # Log-transform each dataset independently
    min_a = df_a.min(axis=0)
    log_a = np.log(df_a - min_a + 1.0)
    min_b = df_b.min(axis=0)
    log_b = np.log(df_b - min_b + 1.0)
    print(f"  Dataset A log-transformed: {log_a.shape}")
    print(f"  Dataset B log-transformed: {log_b.shape}")

    # 4. Run GP optimization
    best_resolution, history_df = optimize_resolution(
        g, all_nodes, log_a, log_b,
        n_calls=args.gp_n_calls,
        res_range=(args.res_range_min, args.res_range_max),
    )

    # 5. Write outputs
    history_df.to_csv(args.output_history, index=False)
    print(f"  Saved optimization history to {args.output_history}")

    with open(args.output_resolution, "w") as f:
        f.write(f"{best_resolution}\n")
    print(f"  Saved optimal resolution ({best_resolution:.4f}) to {args.output_resolution}")


if __name__ == "__main__":
    main()
