#!/usr/bin/env python3
"""
Rule 0 — Pooled Deduplication

Load two expression datasets, intersect their gene columns, pool and
log-transform, then deduplicate near-identical genes via chunked
correlation and connected components (keeping the highest-variance
representative per group).

Outputs:
  common_genes.txt   — one gene per line (deduplicated)
  gene_dup_map.csv   — columns: duplicate, representative
"""

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_genes(log_df, threshold=0.999):
    """
    Collapse genes with |r| > threshold into connected-component groups,
    keeping the highest-variance representative per group.

    Returns
    -------
    keep_names : list[str]
        Gene names to keep (deduplicated).
    dup_map : list[tuple[str, str]]
        (duplicate, representative) pairs for every dropped gene.
    """
    values = log_df.values
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    standardized = centered / norms

    gene_names = log_df.columns.tolist()
    n_genes = len(gene_names)
    variances = log_df.var(axis=0).values

    # Build adjacency via chunked correlation
    adj = lil_matrix((n_genes, n_genes), dtype=bool)
    chunk = 500
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        block = standardized[:, start:end].T @ standardized  # (chunk, n_genes)
        mask = np.abs(block) > threshold
        np.fill_diagonal(mask[:, start:end], False)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            adj[start + r, c] = True

    # Connected components → groups of near-duplicates
    n_components, labels = connected_components(adj, directed=False)

    keep = set()
    dup_pairs = []
    n_groups = 0
    for comp_id in range(n_components):
        members = [i for i in range(n_genes) if labels[i] == comp_id]
        if len(members) == 1:
            keep.add(members[0])
            continue
        # Highest-variance representative
        rep = max(members, key=lambda i: variances[i])
        keep.add(rep)
        n_groups += 1
        for m in members:
            if m != rep:
                dup_pairs.append((gene_names[m], gene_names[rep]))

    keep_names = [gene_names[i] for i in sorted(keep)]
    print(
        f"Deduplication (|r| > {threshold}): {n_genes} → {len(keep_names)} genes "
        f"({len(dup_pairs)} duplicates in {n_groups} groups)"
    )
    return keep_names, dup_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 0: Pooled deduplication of two expression datasets"
    )
    parser.add_argument("--dataset-a", required=True, help="Path to Dataset A CSV")
    parser.add_argument("--dataset-b", required=True, help="Path to Dataset B CSV")
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.999,
        help="Correlation threshold for deduplication (default: 0.999)",
    )
    parser.add_argument(
        "--output-genes", required=True, help="Output path for common_genes.txt"
    )
    parser.add_argument(
        "--output-dup-map", required=True, help="Output path for gene_dup_map.csv"
    )
    args = parser.parse_args()

    # 1. Load both datasets
    print("Loading datasets...")
    df_a = load_dataset(args.dataset_a)
    df_b = load_dataset(args.dataset_b)

    # 2. Intersect gene columns
    common_genes = sorted(set(df_a.columns) & set(df_b.columns))
    print(f"Common genes: {len(common_genes)}")
    df_a = df_a[common_genes]
    df_b = df_b[common_genes]

    # 3. Pool rows and log-transform: log(x - col_min + 1)
    pooled = pd.concat([df_a, df_b], axis=0)
    min_vals = pooled.min(axis=0)
    log_pooled = np.log(pooled - min_vals + 1.0)

    # 4. Deduplicate
    keep_names, dup_pairs = deduplicate_genes(log_pooled, threshold=args.dedup_threshold)

    # 5. Write common_genes.txt
    with open(args.output_genes, "w") as f:
        for gene in keep_names:
            f.write(gene + "\n")
    print(f"Wrote {len(keep_names)} genes to {args.output_genes}")

    # 6. Write gene_dup_map.csv (header-only if no duplicates)
    dup_df = pd.DataFrame(dup_pairs, columns=["duplicate", "representative"])
    dup_df.to_csv(args.output_dup_map, index=False)
    print(f"Wrote {len(dup_pairs)} duplicate mappings to {args.output_dup_map}")


if __name__ == "__main__":
    main()
