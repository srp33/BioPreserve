"""
Main pipeline entry points.

    from basis import preprocess, build_dictionary, align, run_pipeline
"""

import logging
import json

import numpy as np
import pandas as pd

from basis.combat import combat_correct
from basis.ot import sinkhorn_uot
from basis import dictionary as dict_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing (single dataset)
# ---------------------------------------------------------------------------

def preprocess(csv_path, log_transform=False):
    """Load a dataset, keep numeric non-meta columns, optionally log-transform.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    log_transform : bool
        If True, apply log(x - col_min + 1) transform.

    Returns (gene_df, meta_df, gene_df_raw).
    """
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    gene_raw = gene_df.copy()
    if log_transform:
        gene_df = np.log(gene_df - gene_df.min(axis=0) + 1.0)
    logger.info(f"Loaded {csv_path}: {gene_df.shape[0]} samples × {gene_df.shape[1]} genes"
                f"{' (log-transformed)' if log_transform else ''}")
    return gene_df, meta_df, gene_raw


# ---------------------------------------------------------------------------
# Dictionary building (two datasets)
# ---------------------------------------------------------------------------

def build_dictionary(datasets, config=None):
    """Build gene community dictionary from N log-transformed datasets.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        Log-transformed expression DataFrames. First is reference.
    config : dict or None
        Override default parameters.

    Returns
    -------
    gene_sets : dict
        {axis_name: {gene: hub_weight, ...}}.
    """
    cfg = {
        "dedup_threshold": 0.999,
        "d_threshold": 0.5, "w_floor": 0.25, "top_k_edges": 200, "corr_ceiling": 0.99,
        "gp_n_calls": 25, "res_range_min": 1.0, "res_range_max": 200.0,
        "n_runs": 20, "consensus_threshold": 0.7,
        "greedy_merge_threshold": 0.7, "ghost_gene_floor": 0.05,
    }
    if config:
        cfg.update(config)

    # Step 0: Dedup across all datasets
    logger.info(f"Step 0: Pooled deduplication ({len(datasets)} datasets)...")
    common_genes, dup_map = dict_mod.pooled_dedup(datasets, cfg["dedup_threshold"])

    # Step 1: Per-dataset edges
    logger.info("Step 1: Computing edges per dataset...")
    all_edges = []
    for i, df in enumerate(datasets):
        dedup_df = df[[g for g in common_genes if g in df.columns]]
        edges = dict_mod.compute_edges(dedup_df, common_genes,
                                        cfg["d_threshold"], cfg["w_floor"],
                                        cfg["top_k_edges"], cfg["corr_ceiling"])
        logger.info(f"  Dataset {i}: {len(edges)} edges")
        all_edges.append(edges)

    # Step 2: Merge all edge lists
    logger.info("Step 2: Merging graphs...")
    g_ig, merged_edges = dict_mod.merge_graphs(all_edges)
    all_nodes = g_ig.vs["name"]

    # Step 3: Optimize resolution
    logger.info("Step 3: Optimizing resolution...")
    dedup_datasets = [df[[g for g in common_genes if g in df.columns]] for df in datasets]
    best_res, _ = dict_mod.optimize_resolution(
        g_ig, all_nodes, dedup_datasets,
        n_calls=cfg["gp_n_calls"],
        res_range=(cfg["res_range_min"], cfg["res_range_max"]),
    )

    # Step 4: Consensus Leiden + HITS
    logger.info("Step 4: Consensus Leiden + HITS...")
    partition, hits_weights = dict_mod.consensus_leiden_hits(
        g_ig, merged_edges, all_nodes, dup_map, best_res,
        n_runs=cfg["n_runs"], consensus_threshold=cfg["consensus_threshold"],
    )

    # Step 5: Build dictionary
    logger.info("Step 5: Building dictionary...")
    gene_sets = dict_mod.build_gene_community_sets(
        partition, hits_weights, dedup_datasets, common_genes,
        merge_threshold=cfg["greedy_merge_threshold"],
        ghost_gene_floor=cfg["ghost_gene_floor"],
    )

    return gene_sets


# ---------------------------------------------------------------------------
# Alignment (two datasets + dictionary)
# ---------------------------------------------------------------------------

def align(ref_log, tgt_log, gene_sets, ot_epsilon=0.01, ot_tau=0.1):
    """Align one target to reference using GMM posterior embedding + OT + ComBat.

    For multiple targets, call this once per target.

    Parameters
    ----------
    ref_log, tgt_log : pd.DataFrame
        Log-transformed expression (samples × genes).
    gene_sets : dict
        {axis_name: {gene: hub_weight, ...}}.
    ot_epsilon, ot_tau : float

    Returns
    -------
    aligned : pd.DataFrame
        Corrected target expression (log-space).
    metadata : dict
    """
    common_genes = sorted(set(ref_log.columns) & set(tgt_log.columns))
    X_df = ref_log[common_genes]
    Y_df = tgt_log[common_genes]

    # Step 1: GMM posterior embedding
    logger.info("Embedding: GMM posterior per axis...")
    X_scores = gmm_posterior_embed(X_df, gene_sets)
    Y_scores = gmm_posterior_embed(Y_df, gene_sets)

    # Feed directly to OT (no per-axis standardization — gini² scaling preserved)
    X_embed = X_scores.values
    Y_embed = Y_scores.values

    # Step 2: OT weights
    logger.info(f"OT: Sinkhorn UOT (epsilon={ot_epsilon}, tau={ot_tau})...")
    w_ref, w_tgt, mass = sinkhorn_uot(X_embed, Y_embed, ot_epsilon, ot_tau)
    logger.info(f"  Mass={mass:.4f}, w_ref std={w_ref.std():.4f}, w_tgt std={w_tgt.std():.4f}")

    # Step 3: ComBat
    logger.info("ComBat: R-faithful batch correction...")
    X_mat = X_df.values.T
    Y_mat = Y_df.values.T
    combined = np.hstack([X_mat, Y_mat])
    batch_labels = np.array([0] * X_mat.shape[1] + [1] * Y_mat.shape[1])
    corrected = combat_correct(combined, batch_labels, ref_batch=0)
    Y_final = corrected[:, X_mat.shape[1]:]

    logger.info(f"  Ref mean: {np.mean(X_mat):.4f}, Tgt before: {np.mean(Y_mat):.4f}, "
                f"Tgt after: {np.mean(Y_final):.4f}")

    # Build output
    aligned = tgt_log.copy()
    for i, gene in enumerate(common_genes):
        if gene in aligned.columns:
            aligned[gene] = Y_final[i, :]

    metadata = {
        "ot_epsilon": ot_epsilon, "ot_tau": ot_tau,
        "intersection_mass": mass,
        "n_common_genes": len(common_genes),
        "n_axes": len(gene_sets),
    }
    return aligned, metadata


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(ref_path=None, tgt_path=None, output_dir=None, config=None,
                 ot_epsilon=0.01, ot_tau=0.1, log_transform=False, viz=True,
                 cfg=None):
    """Run the complete BASIS pipeline.

    Supports N datasets via cfg.datasets (first = reference, rest = targets).
    For backward compat, ref_path/tgt_path work for the 2-dataset case.

    Returns
    -------
    results : dict
        {label: (aligned_df, metadata)} for each target.
    gene_sets : dict
    """
    import json, os
    from basis.config import BASISConfig, DatasetConfig

    if cfg is None:
        cfg = BASISConfig(
            datasets=[
                DatasetConfig(path=ref_path, log_transform=log_transform, label="ref"),
                DatasetConfig(path=tgt_path, log_transform=log_transform, label="target"),
            ],
            output_dir=output_dir or "",
            viz=viz,
            ot_epsilon=ot_epsilon,
            ot_tau=ot_tau,
        )
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    for i, ds in enumerate(cfg.datasets):
        if not ds.label:
            ds.label = "ref" if i == 0 else f"target_{i}"

    logger.info(f"=== BASIS Pipeline ({len(cfg.datasets)} datasets) ===")

    all_data = []
    for ds in cfg.datasets:
        df, meta, _ = preprocess(ds.path, log_transform=ds.log_transform)
        all_data.append((df, meta, ds.label))

    gene_sets = build_dictionary([d[0] for d in all_data], cfg.dict_config())

    ref_df, ref_meta, ref_label = all_data[0]
    results = {}
    for df, meta, label in all_data[1:]:
        logger.info(f"Aligning {label} to {ref_label}...")
        aligned, metadata = align(ref_df, df, gene_sets, cfg.ot_epsilon, cfg.ot_tau)
        results[label] = (aligned, metadata)

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "gene_community_sets.json"), "w") as f:
            json.dump(gene_sets, f, indent=2)
        for label, (aligned, metadata) in results.items():
            aligned.to_csv(os.path.join(cfg.output_dir, f"aligned_{label}.csv"))
            with open(os.path.join(cfg.output_dir, f"metadata_{label}.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        logger.info(f"Outputs saved to {cfg.output_dir}")

        if cfg.viz:
            from basis.viz.pca_plots import full_pca
            for label, (aligned, _) in results.items():
                tgt_df = [d[0] for d in all_data if d[2] == label][0]
                tgt_meta = [d[1] for d in all_data if d[2] == label][0]
                full_pca(ref_df, tgt_df, aligned, ref_meta, tgt_meta,
                         os.path.join(cfg.output_dir, f"full_pca_{label}.png"))
            logger.info("Visualizations saved")

    logger.info("=== Pipeline complete ===")
    return results, gene_sets
