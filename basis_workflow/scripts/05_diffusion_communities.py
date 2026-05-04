#!/usr/bin/env python3
"""
Rule 4 — Disjoint Leiden Core + Network Diffusion (Soft Satellites)

1. The Hard Core: Runs standard Leiden to assign every gene to exactly one 
   primary community (preventing 'epidemic' collapses).
2. Hub Identification: Computes HITS hub scores on these strictly disjoint 
   subgraphs. Any gene with HITS > 0.5 is a 'Core Hub' and is locked to its axis.
3. Soft Satellites via Network Diffusion: Uses a Random Walk with Restart (RWR)
   where Core Hubs act as the restart 'anchors'. Satellite genes (HITS <= 0.5) 
   adopt a continuous probability distribution across communities based on graph 
   topology. If a satellite's probability for a secondary community exceeds a 
   threshold, it is shared into that community with a low weight.

Outputs:
  gene_communities.csv         — columns: gene, community
  pre_merge_hits_weights.json  — {community_id: {gene: normalized_hub_score}}
"""

import argparse
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import igraph as ig
import scipy.sparse as sp
import leidenalg

# ---------------------------------------------------------------------------
# Leiden partitioning
# ---------------------------------------------------------------------------

def run_single_leiden(g_ig, resolution):
    """Single Leiden partition at *resolution*, seed=42."""
    print(f"  Running Leiden partition (resolution={resolution:.4f}, seed=42)...")
    part = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    return np.array(part.membership)

# ---------------------------------------------------------------------------
# Dup-map expansion
# ---------------------------------------------------------------------------

def expand_partition_with_dupmap(partition_list, dup_map):
    """Assign each duplicate gene to all communities of its representative."""
    gene_to_comms = defaultdict(list)
    for gene, comm in partition_list:
        gene_to_comms[gene].append(comm)
        
    expanded = list(partition_list)
    for dup_gene, rep_gene in dup_map.items():
        if rep_gene in gene_to_comms:
            for comm in gene_to_comms[rep_gene]:
                expanded.append((dup_gene, comm))
                
    return expanded

# ---------------------------------------------------------------------------
# HITS hub scoring
# ---------------------------------------------------------------------------

def compute_community_signs(edges_df, genes, primary_hub):
    """Propagate signs from the primary hub through the community graph."""
    gene_set = set(genes)
    # Build a local adjacency dict with signs
    adj = defaultdict(list)
    mask = edges_df["source"].isin(gene_set) & edges_df["target"].isin(gene_set)
    # itertuples yields (Index, source, target, weight, sign)
    for row in edges_df[mask].itertuples(index=False):
        u, v, w, s = row
        adj[u].append((v, s))
        adj[v].append((u, s)) # Treat as undirected for sign propagation
        
    signs = {primary_hub: 1}
    queue = [primary_hub]
    visited = {primary_hub}
    
    while queue:
        u = queue.pop(0)
        for v, s in adj[u]:
            if v not in visited:
                signs[v] = signs[u] * s
                visited.add(v)
                queue.append(v)
                
    # Default to 1 for any disconnected genes
    for g in genes:
        if g not in signs:
            signs[g] = 1
    return signs


def compute_hits_weights(edges_df, partition_list, min_size=10):
    """Compute HITS hub scores on directed subgraph per community."""
    communities = defaultdict(list)
    for gene, comm_id in partition_list:
        communities[comm_id].append(gene)

    all_weights = {}
    for comm_id, genes in sorted(communities.items()):
        if len(genes) < min_size:
            continue

        gene_set = set(genes)
        mask = edges_df["source"].isin(gene_set) & edges_df["target"].isin(gene_set)
        sub_edges = edges_df[mask]

        if len(sub_edges) == 0:
            hub_scores = {g: 1.0 / len(genes) for g in genes}
        else:
            sub_genes = sorted(gene_set)
            gene_to_idx = {g: i for i, g in enumerate(sub_genes)}
            ig_edges = [
                (gene_to_idx[row.source], gene_to_idx[row.target])
                for row in sub_edges.itertuples()
                if row.source in gene_to_idx and row.target in gene_to_idx
            ]
            ig_weights = [
                row.weight
                for row in sub_edges.itertuples()
                if row.source in gene_to_idx and row.target in gene_to_idx
            ]

            g_dir = ig.Graph(n=len(sub_genes), edges=ig_edges, directed=True)
            g_dir.es["weight"] = ig_weights
            g_dir.vs["name"] = sub_genes

            try:
                hits = g_dir.hub_score(weights="weight")
                hub_scores = {sub_genes[i]: hits[i] for i in range(len(sub_genes))}
            except Exception:
                hub_scores = {g: 1.0 / len(genes) for g in genes}

        # Normalize and ensure all genes have a score
        max_s = max(hub_scores.values()) if hub_scores else 1.0
        if max_s > 0:
            hub_scores = {g: s / max_s for g, s in hub_scores.items()}
        for g in genes:
            if g not in hub_scores: hub_scores[g] = 0.01

        # Identify primary hub and propagate signs
        primary_hub = max(hub_scores, key=hub_scores.get)
        signs = compute_community_signs(edges_df, genes, primary_hub)
        
        # Store signed weights
        signed_hub_weights = {g: hub_scores[g] * signs[g] for g in genes}
        all_weights[str(comm_id)] = signed_hub_weights

    return all_weights

# ---------------------------------------------------------------------------
# Network Diffusion (Random Walk with Restart)
# ---------------------------------------------------------------------------

def run_diffusion(edges_df, all_nodes, memberships, hits_weights, hub_threshold, alpha, max_iter=50):
    """
    Run DIRECTED Network Diffusion to find overlapping satellite probabilities.
    Flows from Hub -> Satellite based on GMM switch direction.
    """
    print(f"  Preparing Directed Network Diffusion (alpha={alpha}, hub_threshold={hub_threshold})...")
    N = len(all_nodes)
    node_to_idx = {name: i for i, name in enumerate(all_nodes)}
    
    unique_comms = sorted(list(set(c for _, c in memberships)))
    C = len(unique_comms)
    comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
    
    node_to_comm = {n: c for n, c in memberships}
    
    # Construct Y0 (HITS-Weighted Anchor Matrix)
    Y0 = np.zeros((N, C))
    n_anchors = 0
    for u in range(N):
        gene = all_nodes[u]
        comm = node_to_comm[gene]
        c_idx = comm_to_idx[comm]
        
        w = hits_weights.get(str(comm), {}).get(gene, 0.0)
        # Use continuous HITS weight as the anchor strength
        if abs(w) > hub_threshold:
            Y0[u, c_idx] = abs(w)
            n_anchors += 1
            
    print(f"  Anchored {n_anchors} Core Hubs with HITS-weighted strengths")
    
    # Build DIRECTED row-stochastic Transition Matrix P
    # We follow the edges as they were built: Anchor -> Follower
    valid_mask = edges_df["source"].isin(node_to_idx) & edges_df["target"].isin(node_to_idx)
    sub_edges = edges_df[valid_mask]
    
    row = [node_to_idx[s] for s in sub_edges["source"]]
    col = [node_to_idx[t] for t in sub_edges["target"]]
    data = sub_edges["weight"].values
    
    W = sp.csr_matrix((data, (row, col)), shape=(N, N))
    
    # Row normalization for Directed Flow
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv.dot(W)
    
    # RWR Iteration
    Y = np.copy(Y0)
    for _ in range(max_iter):
        Y = alpha * P.dot(Y) + (1 - alpha) * Y0
        
    # Row normalize final Y to get probabilities
    row_sums_Y = Y.sum(axis=1)
    row_sums_Y[row_sums_Y == 0] = 1.0
    Y_norm = Y / row_sums_Y[:, np.newaxis]
    
    return Y_norm, unique_comms

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_common_genes(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def load_dup_map(path):
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return dict(zip(df["duplicate"], df["representative"]))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule 4 — Disjoint Core + Diffusion Satellites")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--edges-csv", required=True)
    parser.add_argument("--common-genes", required=True)
    parser.add_argument("--dup-map", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--hub-threshold", type=float, default=0.5)
    parser.add_argument("--diffusion-alpha", type=float, default=0.8)
    parser.add_argument("--diffusion-threshold", type=float, default=0.15)
    parser.add_argument("--output-communities", required=True)
    parser.add_argument("--output-hits", required=True)
    args = parser.parse_args()

    print("Loading graph from GraphML...")
    g_ig = ig.Graph.Read_GraphML(args.graphml)
    all_nodes = g_ig.vs["name"]
    print(f"  Graph: {g_ig.vcount()} nodes, {g_ig.ecount()} edges")

    # 1. The Hard Core (Leiden)
    membership = run_single_leiden(g_ig, args.resolution)
    hard_memberships = [(all_nodes[u], membership[u]) for u in range(g_ig.vcount())]
    
    # Compute HITS on strictly disjoint subgraphs
    print("\nComputing HITS hub scores on strictly disjoint subgraphs...")
    edges_df = pd.read_csv(args.edges_csv)
    hits_weights = compute_hits_weights(edges_df, hard_memberships, min_size=10)
    print(f"  Computed HITS for {len(hits_weights)} core communities (>= 10 genes)")

    # 2. Soft Satellites (Directed Network Diffusion)
    Y_norm, unique_comms = run_diffusion(
        edges_df, all_nodes, hard_memberships, hits_weights, 
        hub_threshold=args.hub_threshold, 
        alpha=args.diffusion_alpha
    )
    
    print(f"\nEvaluating soft satellite overlaps (Threshold = {args.diffusion_threshold})...")
    gene_to_primary = {g: c for g, c in hard_memberships}
    
    final_partition = []
    import copy
    final_weights = copy.deepcopy(hits_weights)
    
    dup_map = load_dup_map(args.dup_map)
    rep_to_dups = defaultdict(list)
    if dup_map:
        for dup, rep in dup_map.items():
            rep_to_dups[rep].append(dup)
            
    shared_count = 0
    for u in range(g_ig.vcount()):
        rep_gene = all_nodes[u]
        primary_comm = gene_to_primary[rep_gene]
        
        associated_genes = [rep_gene] + rep_to_dups[rep_gene]
        
        for gene in associated_genes:
            w_orig = final_weights.get(str(primary_comm), {}).get(gene, 0.01)
            
            if w_orig > args.hub_threshold: 
                # Core Hub: stays only in primary community, full weight
                final_partition.append((gene, primary_comm))
            else:
                # Satellite: shared based on diffusion probabilities
                probs = Y_norm[u]
                max_prob = max(probs)
                assigned_any = False
                
                for c_idx, prob in enumerate(probs):
                    label = unique_comms[c_idx]
                    if prob > 0 and (prob / max_prob) >= args.diffusion_threshold:
                        final_partition.append((gene, label))
                        if str(label) not in final_weights:
                            final_weights[str(label)] = {}
                            
                        # Scale original hub weight by the diffusion probability
                        # This mathematically lowers the contribution of genes that are split
                        new_weight = w_orig * prob
                        
                        if label != primary_comm:
                            shared_count += 1
                            # For secondary communities, assign the scaled weight
                            final_weights[str(label)][gene] = max(final_weights[str(label)].get(gene, 0.0), new_weight)
                        else:
                            # Update primary community weight to the scaled weight
                            final_weights[str(label)][gene] = new_weight
                            
                        assigned_any = True
                
                # Fallback: if no prob exceeds threshold, keep in primary
                if not assigned_any:
                    final_partition.append((gene, primary_comm))
                        
    print(f"  Added {shared_count} secondary satellite assignments")
    
    n_unique_genes = len(set(g for g, c in final_partition))
    n_unique_comms = len(set(c for g, c in final_partition))
    print(f"  Final Partition: {n_unique_genes} unique genes across {n_unique_comms} communities")
    print(f"  Total assignments (including overlaps): {len(final_partition)}")
    
    # Write gene_communities.csv
    comm_df = pd.DataFrame(final_partition, columns=["gene", "community"]).sort_values(["community", "gene"])
    comm_df = comm_df.drop_duplicates()
    comm_df.to_csv(args.output_communities, index=False)
    print(f"  Saved {args.output_communities} ({len(comm_df)} rows)")

    # Reporting Effective Size
    total_mass = 0
    for cid, gweights in final_weights.items():
        mass = sum(gweights.values())
        total_mass += mass
    
    print(f"  Total biological mass across all communities: {total_mass:.2f} (Effective Genes)")

    with open(args.output_hits, "w") as f:
        json.dump(final_weights, f, indent=2)
    print(f"  Saved {args.output_hits}")

if __name__ == "__main__":
    main()
