"""
Main pipeline entry points.

    from basis import preprocess, load_combined, build_dictionary, align, run_pipeline
"""

import logging
import json
import os

import numpy as np
import pandas as pd

from basis.combat import combat_correct
from basis.ot import sinkhorn_uot
from basis.embedding import gmm_posterior_embed
from basis import dictionary as dict_mod
from basis.config import BASISConfig, DatasetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing & Data Loading
# ---------------------------------------------------------------------------

def preprocess(csv_path, log_transform=False, meta_prefix="meta_"):
    """Load a dataset, keep numeric non-meta columns, optionally log-transform.

    Returns (gene_df, meta_df, gene_df_raw).
    """
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith(meta_prefix)]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    gene_raw = gene_df.copy()
    if log_transform:
        gene_df = np.log(gene_df - gene_df.min(axis=0) + 1.0)
    logger.info(f"Loaded {csv_path}: {gene_df.shape[0]} samples × {gene_df.shape[1]} genes"
                f"{' (log-transformed)' if log_transform else ''}")
    return gene_df, meta_df, gene_raw


def load_combined(csv_path, test_source=None, ref_source=None, log_transform=False, meta_prefix="meta_"):
    """Load a combined CSV, split into reference and target components.

    Can operate in two modes:
    1. Target-Centric (test_source provided): Reference is 'everything else'.
    2. Reference-Centric (ref_source provided): Targets are 'everything else' (split by study).

    Returns (ref_log, ref_meta, targets) where targets is list of (tgt_log, tgt_meta, label).
    """
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    source_col = f"{meta_prefix}source"
    if source_col not in df.columns:
        raise ValueError(f"Combined CSV missing '{source_col}' column.")

    meta_cols = [c for c in df.columns if c.startswith(meta_prefix)]

    def _split_data(sub_df):
        m = sub_df[meta_cols]
        g = sub_df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
        g = g.dropna(axis=1, how='all')
        if log_transform:
            g = np.log(g - g.min(axis=0) + 1.0)
        return g, m

    targets = []

    if test_source:
        # Mode 1: LOO / Target-Centric
        # Support single string or list of strings
        if isinstance(test_source, str): test_source = [test_source]
        test_source_lower = [s.lower() for s in test_source]
        
        is_test = df[source_col].astype(str).str.lower().isin(test_source_lower)
        ref_df = df[~is_test]
        
        if len(ref_df) == 0:
            raise ValueError(f"No reference samples found after excluding targets: {test_source}")
        
        ref_log, ref_meta = _split_data(ref_df)
        
        for src_id in test_source:
            tgt_df = df[df[source_col].astype(str).str.lower() == src_id.lower()]
            if len(tgt_df) > 0:
                t_log, t_meta = _split_data(tgt_df)
                targets.append((t_log, t_meta, src_id))

    elif ref_source:
        # Mode 2: Anchor / Reference-Centric
        is_ref = df[source_col].astype(str).str.lower() == ref_source.lower()
        ref_df = df[is_ref]
        
        if len(ref_df) == 0:
            raise ValueError(f"Reference source '{ref_source}' not found in file.")
        
        ref_log, ref_meta = _split_data(ref_df)
        
        other_ids = [s for s in df[source_col].unique() if str(s).lower() != ref_source.lower()]
        for src_id in other_ids:
            tgt_df = df[df[source_col] == src_id]
            t_log, t_meta = _split_data(tgt_df)
            targets.append((t_log, t_meta, str(src_id)))
    else:
        raise ValueError("Must provide either test_source or ref_source.")

    # Harmonize genes across all
    common = set(ref_log.columns)
    for t_log, _, _ in targets:
        common &= set(t_log.columns)
    common = sorted(list(common))
    
    ref_log = ref_log[common]
    final_targets = []
    for t_log, t_meta, label in targets:
        final_targets.append((t_log[common], t_meta, label))

    logger.info(f"Loaded combined CSV: Ref={len(ref_log)} samples, {len(final_targets)} target studies. Shared genes: {len(common)}")
    return ref_log, ref_meta, final_targets


# ---------------------------------------------------------------------------
# Dictionary building (two datasets)
# ---------------------------------------------------------------------------

def build_dictionary(datasets, config=None):
    """Build gene community dictionary from N log-transformed datasets."""
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

def align(ref_log, tgt_log, gene_sets, ot_epsilon=0.01, ot_tau=0.1, keep_shared_only=True):
    """Align one target to reference using GMM posterior embedding + OT + ComBat."""
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
    logger.info("ComBat: R-faithful batch correction (Weighted)...")
    X_mat = X_df.values.T
    Y_mat = Y_df.values.T
    combined = np.hstack([X_mat, Y_mat])
    batch_labels = np.array([0] * X_mat.shape[1] + [1] * Y_mat.shape[1])
    combat_weights = np.concatenate([w_ref, w_tgt])
    
    corrected = combat_correct(combined, batch_labels, ref_batch=0, weights=combat_weights)
    Y_final = corrected[:, X_mat.shape[1]:]

    logger.info(f"  Ref mean: {np.mean(X_mat):.4f}, Tgt before: {np.mean(Y_mat):.4f}, "
                f"Tgt after: {np.mean(Y_final):.4f}")

    # Build output
    if keep_shared_only:
        aligned = pd.DataFrame(Y_final.T, index=tgt_log.index, columns=common_genes)
    else:
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


def combine_results(ref_df, results_dict, keep_shared_only=True, meta_prefix="meta_"):
    """Combine reference and corrected targets into one DataFrame."""
    if not results_dict:
        return ref_df

    # Identify all datasets
    all_targets = [v[0] for v in results_dict.values()]
    all_dfs = [ref_df] + all_targets

    if keep_shared_only:
        def _get_genes(df):
            return [c for c in df.columns if not c.startswith(meta_prefix)]
        
        common_genes = set(_get_genes(ref_df))
        all_meta_cols = set()
        
        for df in all_dfs:
            common_genes &= set(_get_genes(df))
            all_meta_cols.update([c for c in df.columns if c.startswith(meta_prefix)])
            
        common_genes = sorted(list(common_genes))
        all_meta_cols = sorted(list(all_meta_cols))
        
        final_dfs = []
        for df in all_dfs:
            sub_m = pd.DataFrame(index=df.index)
            for c in all_meta_cols:
                sub_m[c] = df[c] if c in df.columns else np.nan
            sub_g = df[common_genes]
            final_dfs.append(pd.concat([sub_m, sub_g], axis=1))
        all_dfs = final_dfs

    return pd.concat(all_dfs, axis=0)


# ---------------------------------------------------------------------------
# High-level Pipeline Execution
# ---------------------------------------------------------------------------

def auto_order_targets(ref_log, targets, gene_sets, cfg):
    """Sort targets by biological intersection mass (OT similarity to reference)."""
    masses = []
    logger.info("Auto-merge: Calculating biological overlap for all targets...")
    for tgt_log, _, label in targets:
        # Step 1: Embed
        X_df = ref_log[tgt_log.columns]
        X_scores = gmm_posterior_embed(X_df, gene_sets)
        Y_scores = gmm_posterior_embed(tgt_log, gene_sets)
        
        # Step 2: Estimate mass
        _, _, mass = sinkhorn_uot(X_scores.values, Y_scores.values, cfg.ot_epsilon, cfg.ot_tau)
        masses.append((mass, label))
        logger.info(f"  {label}: {mass:.4f}")
    
    # Sort descending by mass
    masses.sort(key=lambda x: x[0], reverse=True)
    ordered_labels = [m[1] for m in masses]
    
    label_to_target = {t[2]: t for t in targets}
    return [label_to_target[lbl] for lbl in ordered_labels]


def joint_align(ref_log, targets, gene_sets, ot_epsilon=0.01, ot_tau=0.1, keep_shared_only=True):
    """Simultaneously align multiple targets to the reference using joint ComBat.

    Returns
    -------
    results : dict
        {label: (aligned_df, metadata)}
    """
    logger.info(f"Joint Alignment: Aligning {len(targets)} targets simultaneously...")
    
    # 1. Find common genes across ref and ALL targets
    common_genes = set(ref_log.columns)
    for t_log, _, _ in targets:
        common_genes &= set(t_log.columns)
    common_genes = sorted(list(common_genes))
    
    X_df = ref_log[common_genes]
    X_scores = gmm_posterior_embed(X_df, gene_sets)
    X_embed = X_scores.values
    
    # 2. Compute individual OT masses for logging/metadata
    masses = {}
    Y_dfs = []
    tgt_weights_list = []
    
    for t_log, t_meta, label in targets:
        Y_df = t_log[common_genes]
        Y_scores = gmm_posterior_embed(Y_df, gene_sets)
        _, w_tgt, mass = sinkhorn_uot(X_embed, Y_scores.values, ot_epsilon, ot_tau)
        masses[label] = mass
        Y_dfs.append(Y_df.values.T)
        tgt_weights_list.append(w_tgt)
        logger.info(f"  {label} intersection mass: {mass:.4f}")

    # 3. Concatenate all datasets for joint ComBat
    X_mat = X_df.values.T
    combined = np.hstack([X_mat] + Y_dfs)
    
    # 4. Create batch labels (0=ref, 1=tgt1, 2=tgt2, ...) and weights
    batch_labels = [0] * X_mat.shape[1]
    for i, y_mat in enumerate(Y_dfs):
        batch_labels.extend([i + 1] * y_mat.shape[1])
    batch_labels = np.array(batch_labels)
    
    # Reference is the anchor, give it uniform weights or mean of w_ref
    # In standard BASIS, reference gets w_ref, but for joint, a uniform 1.0 is a stable anchor
    ref_weights = np.ones(X_mat.shape[1])
    combat_weights = np.concatenate([ref_weights] + tgt_weights_list)
    
    # 5. Joint ComBat Correction
    logger.info("ComBat: Joint R-faithful batch correction (Weighted)...")
    corrected = combat_correct(combined, batch_labels, ref_batch=0, weights=combat_weights)
    
    # 6. Split back into individual targets
    results = {}
    current_idx = X_mat.shape[1]
    for i, (t_log, t_meta, label) in enumerate(targets):
        n_samples = Y_dfs[i].shape[1]
        Y_final = corrected[:, current_idx:current_idx + n_samples]
        current_idx += n_samples
        
        logger.info(f"  {label} mean before: {np.mean(Y_dfs[i]):.4f}, after: {np.mean(Y_final):.4f}")
        
        if keep_shared_only:
            aligned = pd.DataFrame(Y_final.T, index=t_log.index, columns=common_genes)
        else:
            aligned = t_log.copy()
            for j, gene in enumerate(common_genes):
                if gene in aligned.columns:
                    aligned[gene] = Y_final[j, :]
                    
        metadata = {
            "ot_epsilon": ot_epsilon, "ot_tau": ot_tau,
            "intersection_mass": masses[label],
            "n_common_genes": len(common_genes),
            "n_axes": len(gene_sets),
            "alignment_mode": "joint"
        }
        results[label] = (aligned, metadata)
        
    return results


def execute_pipeline(ref_log, ref_meta, targets, cfg, gene_sets=None, save_combined_path=None):
    """Core execution logic: dictionary -> alignment -> saving -> visualization."""
    # 1. Dictionary phase (always using all datasets for the shared basis)
    if gene_sets is None:
        gene_sets = build_dictionary([ref_log] + [t[0] for t in targets], cfg.dict_config())
    else:
        logger.info("Using precomputed gene community dictionary.")

    # 2. Output handling
    results = {}
    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "gene_community_sets.json"), "w") as f:
            json.dump(gene_sets, f, indent=2)

    # 3. Alignment phase
    if cfg.joint:
        logger.info("Executing Joint Alignment Strategy...")
        aligned_results = joint_align(ref_log, targets, gene_sets, 
                                      ot_epsilon=cfg.ot_epsilon, ot_tau=cfg.ot_tau, 
                                      keep_shared_only=cfg.keep_shared_only)
        for label, (aligned, metadata) in aligned_results.items():
            results[label] = (aligned, metadata)
            if cfg.output_dir:
                tgt_meta = [t[1] for t in targets if t[2] == label][0]
                final_df = pd.concat([tgt_meta, aligned], axis=1)
                final_df.to_csv(os.path.join(cfg.output_dir, f"aligned_{label}.csv"))
                with open(os.path.join(cfg.output_dir, f"metadata_{label}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

    elif cfg.merge_order and isinstance(cfg.merge_order, list) and any(isinstance(i, list) for i in cfg.merge_order):
        # Hierarchical Tree Mode
        logger.info("Executing Hierarchical Merge Tree...")
        data_map = {t[2]: t for t in targets}
        data_map["ref"] = (ref_log, ref_meta, "ref")
        
        def _resolve_node(node):
            if isinstance(node, (str, int)):
                return data_map[str(node)]
            
            # Recursive merge: treat the first element as the anchor for the rest of the sub-tree
            children = [_resolve_node(child) for child in node]
            anchor_log, anchor_meta, anchor_label = children[0]
            
            resolved_results = {}
            for child_log, child_meta, child_label in children[1:]:
                aligned, meta = align(anchor_log, child_log, gene_sets, 
                                      cfg.ot_epsilon, cfg.ot_tau, 
                                      keep_shared_only=cfg.keep_shared_only)
                resolved_results[child_label] = (pd.concat([child_meta, aligned], axis=1), meta)
                
                # Merge into anchor for the next child in this sub-tree (progressive within node)
                anchor_log = pd.concat([anchor_log, aligned])
                anchor_meta = pd.concat([anchor_meta, child_meta])
            
            # Sub-tree label
            new_label = f"merge({anchor_label},...)"
            return anchor_log, anchor_meta, new_label

        final_log, final_meta, _ = _resolve_node(cfg.merge_order)
        # For simplicity, hierarchical mode currently returns the unified pool as the result
        if save_combined_path:
            final_df = pd.concat([final_meta, final_log], axis=1)
            final_df.to_csv(save_combined_path)
            logger.info(f"Saved hierarchical merge result to {save_combined_path}")
        return {}, gene_sets

    else:
        # Standard / Progressive Loop
        if cfg.auto_merge:
            targets = auto_order_targets(ref_log, targets, gene_sets, cfg)
        elif cfg.merge_order:
            label_to_target = {t[2]: t for t in targets}
            targets = [label_to_target[str(lbl)] for lbl in cfg.merge_order if str(lbl) in label_to_target]

        current_ref_log = ref_log
        current_ref_meta = ref_meta

        for tgt_log, tgt_meta, label in targets:
            logger.info(f"Aligning {label} to current reference (Ref Size={len(current_ref_log)})...")
            aligned, metadata = align(current_ref_log, tgt_log, gene_sets, 
                                      cfg.ot_epsilon, cfg.ot_tau, 
                                      keep_shared_only=cfg.keep_shared_only)
            results[label] = (aligned, metadata)

            # Progressive Reference Expansion
            if cfg.progressive:
                logger.info(f"  [Progressive] Adding {label} to reference pool.")
                current_ref_log = pd.concat([current_ref_log, aligned])
                current_ref_meta = pd.concat([current_ref_meta, tgt_meta])

            if cfg.output_dir:
                final_df = pd.concat([tgt_meta, aligned], axis=1)
                final_df.to_csv(os.path.join(cfg.output_dir, f"aligned_{label}.csv"))
                with open(os.path.join(cfg.output_dir, f"metadata_{label}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

    # 4. Save combined 'source' file
    if cfg.output_dir and save_combined_path and targets:
        # Prepare targets for combine_results
        tgt_dict = {}
        for label, (aligned, meta) in results.items():
            t_meta = [t[1] for t in targets if t[2] == label][0]
            tgt_dict[label] = (pd.concat([t_meta, aligned], axis=1), meta)
            
        common = next(iter(results.values()))[0].columns
        ref_to_combine = pd.concat([ref_meta, ref_log[common]], axis=1)
        combined_df = combine_results(ref_to_combine, tgt_dict, 
                                      keep_shared_only=cfg.keep_shared_only,
                                      meta_prefix=cfg.meta_prefix)
        combined_df.to_csv(save_combined_path)
        logger.info(f"Saved combined source file to {save_combined_path}")

    # 5. Visualization
    if cfg.viz and cfg.output_dir:
        from basis.viz.pca_plots import full_pca
        for label, (aligned, _) in results.items():
            t_log = [t[0] for t in targets if t[2] == label][0]
            t_meta = [t[1] for t in targets if t[2] == label][0]
            full_pca(ref_log, t_log, aligned, ref_meta, t_meta,
                     os.path.join(cfg.output_dir, f"full_pca_{label}.png"))
        logger.info(f"Outputs and visualizations saved to {cfg.output_dir}")

    return results, gene_sets


def run_pipeline(ref_path=None, tgt_path=None, output_dir=None, config=None,
                 ot_epsilon=0.01, ot_tau=0.1, log_transform=False, viz=True,
                 cfg=None, gene_sets=None, save_combined_path=None):
    """Run the complete BASIS pipeline from CSV paths."""
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
            keep_shared_only=True
        )
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    logger.info(f"=== BASIS Pipeline ({len(cfg.datasets)} datasets) ===")

    all_data = []
    for ds in cfg.datasets:
        df, meta, _ = preprocess(ds.path, log_transform=ds.log_transform, meta_prefix=cfg.meta_prefix)
        all_data.append((df, meta, ds.label))

    ref_log, ref_meta, ref_label = all_data[0]
    targets = all_data[1:]

    results, gene_sets = execute_pipeline(ref_log, ref_meta, targets, cfg, 
                                          gene_sets=gene_sets, 
                                          save_combined_path=save_combined_path)

    logger.info("=== Pipeline complete ===")
    return results, gene_sets
