"""Module 1: The Input Layer (Rank Space Conversion)"""
import numpy as np
import pandas as pd

def global_ranking(expr_df, gene_mins=None, add_jitter=True):
    """
    Champion: Dense percentile across all genes.
    Maximized resolution for hub gene ordering.
    """
    df = expr_df.copy()
    if gene_mins is not None:
        mins = gene_mins.reindex(df.columns, fill_value=0)
        df = np.maximum(df - mins, 0)
    
    if add_jitter:
        # Prevent digital wall by spreading exact zeros into a uniform tail
        # This breaks ties and creates a continuous analog distribution for INT space
        vals = df.values.astype(float)
        for i in range(vals.shape[0]):
            row = vals[i, :]
            zero_mask = (row == 0)
            if zero_mask.any():
                # Spread zeros uniformly between 1e-12 and 1e-9
                row[zero_mask] = np.random.uniform(1e-12, 1e-9, size=zero_mask.sum())
            
            # Add general sub-float noise to all non-zeros to break ties
            non_zero_mask = ~zero_mask
            if non_zero_mask.any():
                row[non_zero_mask] += np.random.normal(0, 1e-11, size=non_zero_mask.sum())
        
        return pd.DataFrame(vals, index=df.index, columns=df.columns).rank(axis=1, pct=True)
    
    return df.rank(axis=1, pct=True)

def anchor_only_ranking(expr_df, anchors):
    """Alternative: Trims transcriptome padding at the cost of resolution."""
    valid = [g for g in anchors if g in expr_df.columns]
    anchor_data = expr_df[valid].values
    sorted_anchors = np.sort(anchor_data, axis=1)
    n_anchors = len(valid)
    
    ranks = np.zeros_like(expr_df.values, dtype=float)
    for i in range(expr_df.shape[0]):
        ranks[i, :] = np.searchsorted(sorted_anchors[i, :], expr_df.iloc[i].values, side='left') / n_anchors
    return pd.DataFrame(ranks, index=expr_df.index, columns=expr_df.columns)
