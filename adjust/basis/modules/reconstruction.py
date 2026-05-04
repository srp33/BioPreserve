"""Module 6: The Reconstruction Layer (Feature Translation)"""
import numpy as np
import pandas as pd

def gmm_semantic_blending(expr_df, gene_mins, posteriors, state_mappings):
    """
    Champion: Soft Blend Correction.
    Translates latent state back into high-dimensional gene space.
    """
    mins = gene_mins.reindex(expr_df.columns, fill_value=0)
    log_in = np.log(np.maximum(expr_df - mins, 0) + 1.0)
    log_out = np.zeros_like(log_in.values)
    
    n_states = posteriors.shape[1]
    for k in range(n_states):
        m = state_mappings[k]
        # Weighted linear translation
        term = posteriors[:, k][:, np.newaxis] * (log_in.values * m["scale"].values + m["shift"].values)
        log_out += term
        
    return pd.DataFrame(log_out, index=expr_df.index, columns=expr_df.columns)
