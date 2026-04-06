"""
Biological embedding methods for cross-platform sample comparison.

All methods produce a DataFrame (samples × axes) of scores.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _gini(weights):
    """Gini coefficient of a weight vector."""
    w = np.sort(weights)
    n = len(w)
    cum = np.cumsum(w)
    return 1.0 - 2.0 * np.sum(cum) / (n * cum[-1]) + 1.0 / n


def gmm_posterior_embed(expr_df, gene_sets):
    """Per-dataset PC1 → 2-component GMM → P(high | sample), min-max normalized, gini² scaled.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Log-transformed expression (samples × genes).
    gene_sets : dict
        {axis_name: {gene: hub_weight, ...}}.

    Returns
    -------
    pd.DataFrame
        Samples × axes scores in [0, gini²].
    """
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in expr_df.columns]
        if len(genes) < 3:
            scores[set_name] = np.zeros(len(expr_df))
            continue

        hw = np.array([gene_weights[g] for g in genes])
        X = expr_df[genes].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        pc1 = PCA(n_components=1, random_state=42).fit_transform(X_std).ravel()

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(pc1.reshape(-1, 1))
        proba = gmm.predict_proba(pc1.reshape(-1, 1))
        high_idx = np.argmax(gmm.means_.ravel())
        post = proba[:, high_idx]

        # Min-max normalize per dataset
        lo, hi = post.min(), post.max()
        if hi - lo > 1e-12:
            post = (post - lo) / (hi - lo)
        else:
            post = np.full_like(post, 0.5)

        gini = _gini(hw)
        scores[set_name] = post * (gini ** 2)

    return pd.DataFrame(scores, index=expr_df.index)


def pc1_embed(expr_df, gene_sets):
    """Per-dataset PC1 projection (fit independently per dataset).

    Parameters
    ----------
    expr_df : pd.DataFrame
        Log-transformed expression (samples × genes).
    gene_sets : dict
        {axis_name: {gene: hub_weight, ...}}.

    Returns
    -------
    pd.DataFrame
        Samples × axes PC1 scores.
    """
    scores = {}
    for set_name, gene_weights in gene_sets.items():
        genes = [g for g in gene_weights if g in expr_df.columns]
        if len(genes) < 3:
            scores[set_name] = np.zeros(len(expr_df))
            continue
        X = expr_df[genes].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        scores[set_name] = PCA(n_components=1, random_state=42).fit_transform(X_std).ravel()
    return pd.DataFrame(scores, index=expr_df.index)
