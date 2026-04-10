"""Module 3: The Projection Layer (Latent Embedding)"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def int_pca_projection(aligned_ranks_df, axis_name, axis_params, gene_sets):
    """
    Champion: Inverse Normal Transform + PCA.
    Pushes ranks into a Gaussian space before projection.
    """
    genes = axis_params["genes"]
    hw = np.array([gene_sets[axis_name][g] for g in genes])
    
    # INT Transformation
    X_int = norm.ppf(np.clip(aligned_ranks_df[genes].values, 1e-6, 1 - 1e-6))
    
    # PCA Projection
    scores = axis_params["pca"].transform(X_int * hw).ravel()
    return scores

def simple_weighted_average(aligned_ranks_df, axis_name, axis_params, gene_sets):
    """Alternative: Bypasses INT and PCA entirely."""
    genes = axis_params["genes"]
    hw = np.array([gene_sets[axis_name][g] for g in genes])
    scores = (aligned_ranks_df[genes].values * hw).sum(axis=1) / np.abs(hw).sum()
    return scores
