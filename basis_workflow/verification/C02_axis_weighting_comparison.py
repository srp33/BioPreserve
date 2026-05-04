#!/usr/bin/env python3
"""Compare 5 unsupervised axis weighting methods across all axes."""

import json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

with open('outputs/gene_community_sets.json') as f:
    gene_sets = json.load(f)

df_a = pd.read_csv('inputs/gse20194.csv', index_col=0, low_memory=False)
df_b = pd.read_csv('inputs/gse58644.csv', index_col=0, low_memory=False)

meta_a_cols = [c for c in df_a.columns if c.startswith('meta_')]
meta_b_cols = [c for c in df_b.columns if c.startswith('meta_')]

gene_a = df_a.drop(columns=meta_a_cols).select_dtypes(include=[np.number])
gene_b = df_b.drop(columns=meta_b_cols).select_dtypes(include=[np.number])
gene_a = np.log(gene_a - gene_a.min(axis=0) + 1.0)
gene_b = np.log(gene_b - gene_b.min(axis=0) + 1.0)

common = sorted(set(gene_a.columns) & set(gene_b.columns))
gene_a = gene_a[common]
gene_b = gene_b[common]
n_all = len(common)

# Load edges for method 4
edges_a = pd.read_csv('outputs/edges_A.csv')
edges_b = pd.read_csv('outputs/edges_B.csv')
edges_a_set = set(zip(edges_a['source'], edges_a['target']))
edges_b_set = set(zip(edges_b['source'], edges_b['target']))

def global_rank_norm(df):
    vals = df.values
    ranked = np.zeros_like(vals)
    for i in range(vals.shape[0]):
        r = rankdata(vals[i], method='average')
        ranked[i] = 2.0 * (r - 1) / max(n_all - 1, 1) - 1.0
    return pd.DataFrame(ranked, index=df.index, columns=df.columns)

R_a = global_rank_norm(gene_a)
R_b = global_rank_norm(gene_b)

def dip_proxy(vals):
    hist, bin_edges = np.histogram(vals, bins=20)
    median_bin = min(np.searchsorted(bin_edges[1:], np.median(vals)), len(hist) - 1)
    return 1.0 - (hist[median_bin] / max(hist.max(), 1))

results = []

for axis_name, gene_weights in sorted(gene_sets.items()):
    genes = [g for g in gene_weights if g in common]
    if len(genes) < 3:
        continue
    hw = np.array([gene_weights[g] for g in genes])

    # Hub-weighted rank vectors
    HRa = R_a[genes].values * hw[np.newaxis, :]
    HRb = R_b[genes].values * hw[np.newaxis, :]

    # --- Method 1: Per-dataset bimodality (min dip) ---
    pca_a = PCA(n_components=1, random_state=42)
    pc1_a = pca_a.fit_transform(HRa).ravel()
    pca_b = PCA(n_components=1, random_state=42)
    pc1_b = pca_b.fit_transform(HRb).ravel()
    dip_a = dip_proxy(pc1_a)
    dip_b = dip_proxy(pc1_b)
    min_dip = min(dip_a, dip_b)

    # --- Method 2: Cross-dataset rank correlation ---
    n_pairs = 30
    idx_a = np.random.RandomState(42).choice(len(gene_a), n_pairs, replace=False)
    idx_b = np.random.RandomState(42).choice(len(gene_b), n_pairs, replace=False)
    corrs = []
    for i in range(n_pairs):
        va = HRa[idx_a[i]]
        vb = HRb[idx_b[i]]
        corrs.append(np.corrcoef(va, vb)[0, 1])
    cross_corr = np.mean(corrs)

    # --- Method 3: Hub concentration (Gini) ---
    hw_sorted = np.sort(hw)
    n_g = len(hw_sorted)
    cum = np.cumsum(hw_sorted)
    gini = 1.0 - 2.0 * np.sum(cum) / (n_g * cum[-1]) + 1.0 / n_g

    # --- Method 4: Edge reproducibility ---
    gene_set = set(genes)
    n_both = 0
    n_any = 0
    for g1 in genes:
        for g2 in genes:
            if g1 == g2:
                continue
            in_a = (g1, g2) in edges_a_set
            in_b = (g1, g2) in edges_b_set
            if in_a or in_b:
                n_any += 1
                if in_a and in_b:
                    n_both += 1
    edge_repro = n_both / max(n_any, 1)

    # --- Method 5: PC1 loading agreement ---
    loadings_a = pca_a.components_[0]
    loadings_b = pca_b.components_[0]
    loading_corr = abs(np.corrcoef(loadings_a, loadings_b)[0, 1])

    results.append({
        'axis': axis_name,
        'n_genes': len(genes),
        'min_dip': min_dip,
        'cross_corr': cross_corr,
        'hub_gini': gini,
        'edge_repro': edge_repro,
        'loading_agree': loading_corr,
    })

df = pd.DataFrame(results)

# Rank each method (higher = better for all)
for col in ['min_dip', 'cross_corr', 'hub_gini', 'edge_repro', 'loading_agree']:
    df[f'{col}_rank'] = df[col].rank(ascending=False).astype(int)

print(f"{'Axis':<10} {'Genes':>5} {'MinDip':>7} {'Rk':>3} {'XCorr':>7} {'Rk':>3} {'Gini':>7} {'Rk':>3} {'EdgeR':>7} {'Rk':>3} {'LoadA':>7} {'Rk':>3}")
print('=' * 95)
for _, r in df.sort_values('axis').iterrows():
    marker = ' <--' if r['axis'] in ('axis_3', 'axis_5') else ''
    print(f"{r['axis']:<10} {r['n_genes']:>5} "
          f"{r['min_dip']:>7.3f} {r['min_dip_rank']:>3} "
          f"{r['cross_corr']:>7.3f} {r['cross_corr_rank']:>3} "
          f"{r['hub_gini']:>7.3f} {r['hub_gini_rank']:>3} "
          f"{r['edge_repro']:>7.3f} {r['edge_repro_rank']:>3} "
          f"{r['loading_agree']:>7.3f} {r['loading_agree_rank']:>3}{marker}")
