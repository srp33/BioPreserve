"""
Gene community dictionary construction.

Combines: dedup → per-dataset edges → graph merge → resolution optimization →
consensus Leiden + HITS → greedy merge + ghost gene pruning.
"""

import logging
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path, meta_prefix="meta_"):
    """Load CSV, keep only numeric non-meta_* columns."""
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df.select_dtypes(include=[np.number])
    meta_cols = [c for c in df.columns if c.startswith(meta_prefix)]
    if meta_cols:
        df = df.drop(columns=meta_cols)
    return df


def log_transform(df):
    """Log-transform: log(x - col_min + 1)."""
    return np.log(df - df.min(axis=0) + 1.0)


# ---------------------------------------------------------------------------
# Step 0: Pooled deduplication
# ---------------------------------------------------------------------------

def deduplicate_genes(log_df, threshold=0.999):
    """Collapse genes with |r| > threshold via connected components."""
    values = log_df.values
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    standardized = centered / norms

    gene_names = log_df.columns.tolist()
    n_genes = len(gene_names)
    variances = log_df.var(axis=0).values

    adj = lil_matrix((n_genes, n_genes), dtype=bool)
    chunk = 500
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        block = standardized[:, start:end].T @ standardized
        mask = np.abs(block) > threshold
        np.fill_diagonal(mask[:, start:end], False)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            adj[start + r, c] = True

    n_components, labels = connected_components(adj, directed=False)
    keep = set()
    dup_pairs = []
    for comp_id in range(n_components):
        members = [i for i in range(n_genes) if labels[i] == comp_id]
        if len(members) == 1:
            keep.add(members[0])
            continue
        rep = max(members, key=lambda i: variances[i])
        keep.add(rep)
        for m in members:
            if m != rep:
                dup_pairs.append((gene_names[m], gene_names[rep]))

    keep_names = [gene_names[i] for i in sorted(keep)]
    logger.info(f"Dedup (|r|>{threshold}): {n_genes} → {len(keep_names)} genes ({len(dup_pairs)} dups)")
    return keep_names, dup_pairs


def pooled_dedup(datasets, dedup_threshold=0.999):
    """Find common genes across all datasets, pool, deduplicate.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        Log-transformed expression DataFrames.
    dedup_threshold : float

    Returns
    -------
    common_genes : list[str]
    dup_map : dict
    """
    common_genes = set(datasets[0].columns)
    for df in datasets[1:]:
        common_genes &= set(df.columns)
    common_genes = sorted(common_genes)
    logger.info(f"Common genes across {len(datasets)} datasets: {len(common_genes)}")

    log_pooled = pd.concat([df[common_genes] for df in datasets], axis=0)
    keep_names, dup_pairs = deduplicate_genes(log_pooled, threshold=dedup_threshold)
    dup_map = dict(dup_pairs)
    return keep_names, dup_map


# ---------------------------------------------------------------------------
# Step 1: Per-dataset edge computation
# ---------------------------------------------------------------------------

def compute_edges(log_df, common_genes, d_threshold=0.5, w_floor=0.25,
                  top_k=200, corr_ceiling=0.99):
    """Compute GMM-weighted Cohen's d edges for one dataset."""
    try:
        import gmm_adjust
    except ImportError:
        # Fallback: look in parent directory (adjust/)
        import sys
        parent_dir = str(Path(__file__).resolve().parent.parent)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        import gmm_adjust

    df = log_df[common_genes]
    data_array = df.values.T  # (n_genes, n_samples)

    # Fit GMM (gmm_adjust handles its own log transform, pass raw)
    # Actually we already log-transformed, so pass with log_transform=False
    gmm_result = gmm_adjust.get_gmm_responsibilities(
        data_array, genes_are_rows=True, log_transform=False
    )
    resp_lower, resp_upper = gmm_result["responsibilities"]

    gene_names = df.columns.tolist()
    n_genes = len(gene_names)
    values = df.values  # (n_samples, n_genes)
    values_sq = values ** 2

    sum_w0 = resp_lower.sum(axis=1)
    sum_w1 = resp_upper.sum(axis=1)
    valid_anchor = (sum_w0 >= 2) & (sum_w1 >= 2)

    resp_centered = resp_upper - resp_upper.mean(axis=1, keepdims=True)
    resp_norms = np.linalg.norm(resp_centered, axis=1, keepdims=True)
    resp_norms = np.maximum(resp_norms, 1e-12)
    resp_normed = resp_centered / resp_norms

    edges = []
    BATCH = 500
    for b_start in range(0, n_genes, BATCH):
        b_end = min(b_start + BATCH, n_genes)
        batch_valid = valid_anchor[b_start:b_end]
        if not batch_valid.any():
            continue

        w0 = resp_lower[b_start:b_end]
        w1 = resp_upper[b_start:b_end]
        sw0 = sum_w0[b_start:b_end, None]
        sw1 = sum_w1[b_start:b_end, None]

        mean0 = (w0 @ values) / np.maximum(sw0, 1e-12)
        mean1 = (w1 @ values) / np.maximum(sw1, 1e-12)
        mean_sq0 = (w0 @ values_sq) / np.maximum(sw0, 1e-12)
        mean_sq1 = (w1 @ values_sq) / np.maximum(sw1, 1e-12)
        var0 = np.maximum(mean_sq0 - mean0 ** 2, 0)
        var1 = np.maximum(mean_sq1 - mean1 ** 2, 0)

        pooled_std = np.sqrt(np.maximum(0.5 * (var0 + var1), 1e-12))
        effect_size = np.abs(mean1 - mean0) / pooled_std
        resp_corr = resp_normed[b_start:b_end] @ resp_normed.T

        for bi in range(b_end - b_start):
            i = b_start + bi
            if not batch_valid[bi]:
                continue
            d_i = effect_size[bi]
            rc_i = resp_corr[bi]
            sig_mask = (d_i >= d_threshold) & (rc_i > 0)
            sig_mask[i] = False
            geo_w = d_i * rc_i
            sig_mask &= geo_w >= w_floor
            kept_idx = np.where(sig_mask)[0]
            if len(kept_idx) == 0:
                continue
            kept_w = geo_w[kept_idx]
            if len(kept_w) > top_k:
                top_order = np.argpartition(-kept_w, top_k)[:top_k]
                kept_idx = kept_idx[top_order]
                kept_w = kept_w[top_order]
            src = gene_names[i]
            for j, w in zip(kept_idx, kept_w):
                edges.append((src, gene_names[j], float(w)))

    logger.info(f"  Computed {len(edges)} edges")
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


# ---------------------------------------------------------------------------
# Step 2: Graph merge
# ---------------------------------------------------------------------------

def merge_graphs(edge_list):
    """Merge N edge DataFrames using max weight, build undirected igraph.

    Parameters
    ----------
    edge_list : list of pd.DataFrame
        Each with columns (source, target, weight).
    """
    import igraph as ig

    combined = pd.concat(edge_list, ignore_index=True)
    merged = combined.groupby(["source", "target"], as_index=False)["weight"].max()

    edge_dict = {}
    for src, tgt, w in merged.itertuples(index=False):
        key = (src, tgt) if src < tgt else (tgt, src)
        if key not in edge_dict or w > edge_dict[key]:
            edge_dict[key] = w

    all_nodes = sorted({n for edge in merged.itertuples(index=False) for n in (edge.source, edge.target)})
    node_map = {name: idx for idx, name in enumerate(all_nodes)}
    ig_edges = [(node_map[s], node_map[t]) for s, t in edge_dict]
    ig_weights = list(edge_dict.values())

    g = ig.Graph(n=len(all_nodes), edges=ig_edges, directed=False)
    g.vs["name"] = all_nodes
    g.es["weight"] = ig_weights

    logger.info(f"Merged graph: {g.vcount()} nodes, {g.ecount()} edges")
    return g, merged


# ---------------------------------------------------------------------------
# Step 3: Resolution optimization
# ---------------------------------------------------------------------------

def _dip_proxy(pc1):
    hist, bin_edges = np.histogram(pc1, bins=20)
    median_bin = min(np.searchsorted(bin_edges[1:], np.median(pc1)), len(hist) - 1)
    return 1.0 - (hist[median_bin] / max(hist.max(), 1))


def score_partition_dip(partition, datasets, min_size=10):
    """Score partition bimodality per-dataset, then sum.

    Parameters
    ----------
    partition : dict
    datasets : list of pd.DataFrame
        Log-transformed expression DataFrames.
    """
    communities = defaultdict(list)
    ref_cols = datasets[0].columns
    for gene, comm_id in partition.items():
        if gene in ref_cols:
            communities[comm_id].append(gene)

    total_dip = 0.0
    n_scored = 0
    for genes in communities.values():
        if len(genes) < min_size:
            continue
        n_scored += 1
        for log_df in datasets:
            valid = [g for g in genes if g in log_df.columns]
            if len(valid) < min_size:
                continue
            X = log_df[valid].values
            X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
            pc1 = PCA(n_components=1).fit_transform(X_std).ravel()
            total_dip += _dip_proxy(pc1)
    return total_dip, n_scored


def optimize_resolution(g_ig, all_nodes, datasets, n_calls=25,
                        res_range=(1.0, 200.0)):
    """GP-optimize Leiden resolution using per-dataset bimodality.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        Log-transformed expression DataFrames.
    """
    import leidenalg
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import expected_minimum

    history = []

    def objective(params):
        resolution = params[0]
        part = leidenalg.find_partition(
            g_ig.copy(), leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution,
            seed=np.random.randint(0, 10000),
        )
        partition = {all_nodes[i]: part.membership[i] for i in range(len(all_nodes))}
        sum_dip, n_scored = score_partition_dip(partition, datasets)
        history.append({"resolution": resolution, "sum_dip": sum_dip, "n_scored": n_scored})
        return -sum_dip

    result = gp_minimize(
        objective, [Real(res_range[0], res_range[1], prior="log-uniform")],
        n_calls=n_calls, n_initial_points=min(10, n_calls // 2),
        noise="gaussian", random_state=42,
    )
    best_x, best_y = expected_minimum(result)
    best_res = best_x[0]
    logger.info(f"GP optimum: resolution={best_res:.2f} (sum_dip={-best_y:.2f})")
    return best_res, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Step 4: Consensus Leiden + HITS
# ---------------------------------------------------------------------------

def _leiden_worker(args):
    """Module-level worker for ProcessPoolExecutor."""
    import igraph as ig
    import leidenalg
    edges, weights, n_nodes, resolution, seed = args
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    return np.array(part.membership)


def consensus_leiden_hits(g_ig, edges_df, all_nodes, dup_map, resolution,
                          n_runs=20, consensus_threshold=0.7):
    """Run consensus Leiden and compute HITS hub scores."""
    import igraph as ig
    import leidenalg
    from concurrent.futures import ProcessPoolExecutor

    n_nodes = g_ig.vcount()

    if n_runs <= 1:
        part = leidenalg.find_partition(
            g_ig, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution, seed=42,
        )
        membership = np.array(part.membership)
    else:
        edges = list(g_ig.get_edgelist())
        weights = list(g_ig.es["weight"])
        args_list = [(edges, weights, n_nodes, resolution, i) for i in range(n_runs)]
        max_workers = min(n_runs, 10)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            all_labels = list(pool.map(_leiden_worker, args_list))

        coassign = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for labels in all_labels:
            for comm_id in np.unique(labels):
                members = np.where(labels == comm_id)[0]
                if len(members) > 1:
                    coassign[np.ix_(members, members)] += 1
        coassign /= n_runs
        np.fill_diagonal(coassign, 1.0)

        rows, cols = np.where(np.triu(coassign, k=1) > consensus_threshold)
        cons_edges = list(zip(rows.tolist(), cols.tolist()))
        cons_weights = [float(coassign[i, j]) for i, j in cons_edges]

        if len(cons_edges) == 0:
            membership = np.arange(n_nodes)
        else:
            g_cons = ig.Graph(n=n_nodes, edges=cons_edges, directed=False)
            g_cons.es["weight"] = cons_weights
            final_part = leidenalg.find_partition(
                g_cons, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=1.0, seed=42,
            )
            membership = np.array(final_part.membership)

    # Build partition dict and expand with dup_map
    partition = {all_nodes[i]: int(membership[i]) for i in range(len(all_nodes))}
    for dup_gene, rep_gene in dup_map.items():
        if rep_gene in partition:
            partition[dup_gene] = partition[rep_gene]

    # HITS hub scores per community
    communities = defaultdict(list)
    for gene, comm_id in partition.items():
        communities[comm_id].append(gene)

    hits_weights = {}
    for comm_id, genes in sorted(communities.items()):
        if len(genes) < 10:
            continue
        gene_set = set(genes)
        mask = edges_df["source"].isin(gene_set) & edges_df["target"].isin(gene_set)
        sub_edges = edges_df[mask]
        if len(sub_edges) == 0:
            hub_scores = {g: 1.0 / len(genes) for g in genes}
        else:
            sub_genes = sorted(gene_set)
            gene_to_idx = {g: i for i, g in enumerate(sub_genes)}
            ig_edges = [(gene_to_idx[r.source], gene_to_idx[r.target])
                        for r in sub_edges.itertuples() if r.source in gene_to_idx and r.target in gene_to_idx]
            ig_w = [r.weight for r in sub_edges.itertuples() if r.source in gene_to_idx and r.target in gene_to_idx]
            g_dir = ig.Graph(n=len(sub_genes), edges=ig_edges, directed=True)
            g_dir.es["weight"] = ig_w
            try:
                hits = g_dir.hub_score(weights="weight")
                hub_scores = {sub_genes[i]: hits[i] for i in range(len(sub_genes))}
            except Exception:
                hub_scores = {g: 1.0 / len(genes) for g in genes}
            max_s = max(hub_scores.values()) if hub_scores else 1.0
            if max_s > 0:
                hub_scores = {g: s / max_s for g, s in hub_scores.items()}
        for g in genes:
            if g not in hub_scores:
                hub_scores[g] = 0.01
        max_s = max(hub_scores.values())
        if max_s > 0:
            hub_scores = {g: s / max_s for g, s in hub_scores.items()}
        hits_weights[str(comm_id)] = hub_scores

    logger.info(f"Partition: {len(partition)} genes, {len(communities)} communities, "
                f"{len(hits_weights)} with HITS (≥10 genes)")
    return partition, hits_weights


# ---------------------------------------------------------------------------
# Step 5: Dictionary construction (greedy merge + ghost gene pruning)
# ---------------------------------------------------------------------------

def build_gene_community_sets(partition, hits_weights, datasets, common_genes,
                               merge_threshold=0.7, ghost_gene_floor=0.05):
    """Greedy merge communities into axes, prune ghost genes.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        Log-transformed expression DataFrames.
    """
    communities = defaultdict(list)
    for gene, comm_id in partition.items():
        communities[comm_id].append(gene)

    log_expr = pd.concat(datasets, axis=0, ignore_index=True)
    # Track dataset boundaries for per-dataset correlation checks
    boundaries = [0]
    for df in datasets:
        boundaries.append(boundaries[-1] + len(df))

    # Rank by PC1 eigenvalue
    records = []
    for comm_id, genes in communities.items():
        if str(comm_id) not in hits_weights:
            continue
        valid = [g for g in genes if g in log_expr.columns]
        if len(valid) < 10:
            continue
        X = log_expr[valid].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pca = PCA(n_components=1)
        pca.fit(X_std)
        records.append({"community": comm_id, "genes": valid, "eigenvalue": pca.explained_variance_[0]})
    records.sort(key=lambda r: r["eigenvalue"], reverse=True)

    # Greedy merge with per-dataset correlation checks
    dataset_masks = []
    for i in range(len(datasets)):
        mask = np.zeros(len(log_expr), dtype=bool)
        mask[boundaries[i]:boundaries[i+1]] = True
        dataset_masks.append(mask)

    def compute_pc1(genes, sample_mask):
        X = log_expr[genes].values[sample_mask]
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        return PCA(n_components=1, random_state=42).fit_transform(X_std).ravel()

    axes = []
    for rec in records:
        genes = rec["genes"]
        merged = False
        for axis in axes:
            axis_genes = list(axis["genes"])
            # Check correlation in each dataset, merge if max > threshold
            max_r = 0.0
            for mask in dataset_masks:
                if mask.sum() < 10:
                    continue
                r = abs(np.corrcoef(compute_pc1(genes, mask), compute_pc1(axis_genes, mask))[0, 1])
                max_r = max(max_r, r)
            if max_r > merge_threshold:
                axis["genes"].update(genes)
                axis["source_ids"].append(rec["community"])
                merged = True
                break
        if not merged:
            axes.append({"genes": set(genes), "source_ids": [rec["community"]]})

    # Combine HITS weights and prune
    result = {}
    axis_idx = 0
    for axis in axes:
        merged_w = {}
        for comm_id in axis["source_ids"]:
            if str(comm_id) in hits_weights:
                for gene, w in hits_weights[str(comm_id)].items():
                    if gene not in merged_w or w > merged_w[gene]:
                        merged_w[gene] = w
        for gene in axis["genes"]:
            if gene not in merged_w:
                merged_w[gene] = 0.01
        pruned = {g: w for g, w in merged_w.items() if w >= ghost_gene_floor}
        if len(pruned) >= 15:
            result[f"axis_{axis_idx}"] = pruned
            axis_idx += 1

    logger.info(f"Dictionary: {len(result)} axes after merge+prune")
    return result
