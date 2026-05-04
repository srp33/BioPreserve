#!/usr/bin/env python3
"""
Rule 2 — Graph Merge

Load two per-dataset directed edge CSVs, merge using max(W_A, W_B) per
directed (source, target) pair, build an undirected igraph collapsing
(A→B, B→A) into a single edge with max weight, and serialize to GraphML.

Outputs:
  ttest_edges.csv        — merged directed edge list (source, target, weight)
  igraph_structure.graphml — undirected igraph with edge attribute 'weight'
"""

import argparse

import igraph as ig
import pandas as pd


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def merge_directed_edges(edges_a: pd.DataFrame, edges_b: pd.DataFrame) -> pd.DataFrame:
    """Merge two directed edge DataFrames using max(W_A, W_B) per (source, target).
    Preserves the sign from the heavier edge.
    """
    combined = pd.concat([edges_a, edges_b], ignore_index=True)
    # Sort by weight descending so drop_duplicates keeps the max weight row (including its sign)
    merged = combined.sort_values("weight", ascending=False).drop_duplicates(
        subset=["source", "target"], keep="first"
    )
    return merged.sort_values(["source", "target"])


def apply_mknn_filter(merged: pd.DataFrame) -> pd.DataFrame:
    """Keep only mutual edges (where A->B and B->A both exist in the merged list)."""
    edge_set = set(zip(merged["source"], merged["target"]))
    mask = [(t, s) in edge_set for s, t in zip(merged["source"], merged["target"])]
    return merged[mask].copy()


def build_graph(directed_edges: pd.DataFrame, directed: bool = False) -> ig.Graph:
    """Build an igraph from directed edges.

    If directed=False, collapse directed edges (A→B, B→A) into
    a single undirected edge with the maximum weight.
    """
    all_nodes = sorted({n for edge in directed_edges.itertuples(index=False)
                        for n in (edge[0], edge[1])})
    node_map = {name: idx for idx, name in enumerate(all_nodes)}

    if not directed:
        # Collapse (A→B, B→A) by sorting each pair and taking max weight
        edge_dict: dict[tuple[str, str], float] = {}
        for src, tgt, w, s in directed_edges.itertuples(index=False):
            key = (src, tgt) if src < tgt else (tgt, src)
            if key not in edge_dict or w > edge_dict[key]:
                edge_dict[key] = w
        ig_edges = [(node_map[s], node_map[t]) for s, t in edge_dict]
        ig_weights = list(edge_dict.values())
    else:
        ig_edges = [(node_map[row.source], node_map[row.target]) for row in directed_edges.itertuples(index=False)]
        ig_weights = directed_edges["weight"].tolist()

    g = ig.Graph(n=len(all_nodes), edges=ig_edges, directed=directed)
    g.vs["name"] = all_nodes
    g.es["weight"] = ig_weights
    return g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule 2 — Merge two per-dataset edge lists into a single "
                    "igraph structure and serialize to GraphML.",
    )
    parser.add_argument("--edges-a", required=True, help="Path to edges CSV for dataset A")
    parser.add_argument("--edges-b", required=True, help="Path to edges CSV for dataset B")
    parser.add_argument("--output-edges", required=True, help="Path for merged directed edge CSV")
    parser.add_argument("--output-graphml", required=True, help="Path for GraphML")
    parser.add_argument("--use-mknn", action="store_true", help="Apply Mutual K-NN filter")
    parser.add_argument("--directed", action="store_true", help="Build a directed graph")
    args = parser.parse_args()

    # Load both edge CSVs
    print("Loading edge lists...")
    edges_a = pd.read_csv(args.edges_a)
    edges_b = pd.read_csv(args.edges_b)
    print(f"  Dataset A: {len(edges_a)} directed edges")
    print(f"  Dataset B: {len(edges_b)} directed edges")

    # Merge directed edges using max(W_A, W_B)
    print("Merging directed edges (max weight per pair)...")
    merged = merge_directed_edges(edges_a, edges_b)
    print(f"  Merged: {len(merged)} directed edges")
    
    if args.use_mknn:
        print("Applying Mutual K-NN filter...")
        merged = apply_mknn_filter(merged)
        print(f"  After mKNN: {len(merged)} directed edges")

    # Write merged directed edge CSV
    merged.to_csv(args.output_edges, index=False)
    print(f"  Saved {args.output_edges}")

    # Build igraph and serialize to GraphML
    print(f"Building {'directed' if args.directed else 'undirected'} igraph...")
    g = build_graph(merged, directed=args.directed)
    print(f"  igraph: {g.vcount()} nodes, {g.ecount()} edges")

    g.write_graphml(args.output_graphml)
    print(f"  Saved {args.output_graphml}")


if __name__ == "__main__":
    main()
