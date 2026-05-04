#!/usr/bin/env python3
"""
Rule 6 — BASIS Alignment with Soft GMM Latent Interpolation

1. Fits a BasisAtlas on the Reference Dataset (Identify states).
2. Learns translation factors between Ref and Tgt states (Discovery).
3. Applies N=1 Soft GMM Correction to every gene in the Target dataset.
"""

import argparse
import json
import logging
import sys
import os

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from basis.embedding import BasisAtlas

EPS = 1e-8
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def align(ref_df, target_df, gene_sets, anchors):
    common_genes = sorted(set(ref_df.columns) & set(target_df.columns))
    X_df, Y_df = ref_df[common_genes], target_df[common_genes]

    # 1. Discovery Phase (Fit Reference Atlas and States)
    atlas = BasisAtlas(gene_sets, anchors)
    atlas.fit(X_df)

    # 2. State Mapping Phase (Discovery of Platform Transition)
    logger.info("Step 1: Learning State-Specific Translation Factors...")
    atlas.learn_translation(Y_df, X_df)

    # 3. Batch Correction (Diagnosis Style: N=1 capable)
    logger.info("Step 2: Applying Soft GMM Latent Interpolation...")
    aligned_target_log = atlas.correct_genes(Y_df)
    
    # Inverse log transform using reference mins
    Y_final = np.exp(aligned_target_log) + atlas.gene_mins.reindex(common_genes).values - 1.0

    aligned_target = target_df.copy()
    for i, gene in enumerate(common_genes):
        aligned_target[gene] = Y_final[gene].values

    return aligned_target, {"version": "26.0-BASIS-Soft-GMM-Ideal", "n_states": atlas.global_gmm.n_components}

def main():
    parser = argparse.ArgumentParser(description="Rule 6 — BASIS Soft GMM")
    parser.add_argument("--gene-sets", required=True)
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output-aligned", required=True)
    parser.add_argument("--output-metadata", required=True)
    args = parser.parse_args()

    with open(args.gene_sets) as f: gene_sets = json.load(f)
    with open(args.anchors) as f: anchors = [l.strip() for l in f if l.strip()]
    
    def load_raw(path):
        df = pd.read_csv(path, index_col=0, low_memory=False)
        gene_cols = [c for c in df.columns if not c.startswith("meta_") and pd.api.types.is_numeric_dtype(df[c])]
        return df[gene_cols].copy()

    df_a, df_b = load_raw(args.dataset_a), load_raw(args.dataset_b)
    aligned_target, metadata = align(df_a, df_b, gene_sets, anchors)
    aligned_target.to_csv(args.output_aligned)
    with open(args.output_metadata, "w") as f: json.dump(metadata, f, indent=2)
    print(f"Done. Aligned target saved to {args.output_aligned}")

if __name__ == "__main__":
    main()
