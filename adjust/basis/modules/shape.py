"""Module 2: The Shape Layer (Non-Linear Calibration)"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def anchor_quantile_normalization(ranks_df, ref_anchor_profile):
    """
    AQN Spline: Maps patient's anchor ranks to the Atlas Ideal Backbone.
    """
    valid_anchors = ref_anchor_profile.index
    aligned_ranks = np.zeros_like(ranks_df.values)
    for i in range(ranks_df.shape[0]):
        sample_ranks = ranks_df.values[i, :]
        tgt_anchor_ranks = ranks_df.iloc[i][valid_anchors].values
        x = np.concatenate([[0.0], tgt_anchor_ranks, [1.0]])
        y = np.concatenate([[0.0], ref_anchor_profile.values, [1.0]])
        x = np.maximum.accumulate(x)
        _, unique_idx = np.unique(x, return_index=True)
        spline = PchipInterpolator(x[unique_idx], y[unique_idx])
        aligned_ranks[i, :] = np.clip(spline(sample_ranks), 0.0, 1.0)
    return pd.DataFrame(aligned_ranks, index=ranks_df.index, columns=ranks_df.columns)

def batch_anchor_physical_spline(expr_df, ref_phys_profile, anchors):
    """
    Champion: Batch-APS (Multi-Sample Physical Spline).
    Denoises the physical spline using hardware consensus (Batch Median).
    Immune to Subpopulation Imbalance Paradox.
    """
    valid_anchors = [g for g in anchors if g in expr_df.columns]
    
    # 1. Target Hardware Consensus (Median across the batch)
    batch_anchor_medians = expr_df[valid_anchors].median(axis=0).sort_values()
    # Sort reference to match target anchor order
    ref_phys_profile_matched = ref_phys_profile.reindex(batch_anchor_medians.index)
    
    # 2. Hardware Bounds
    batch_min = expr_df.values.min()
    batch_max = expr_df.values.max()
    ref_min = ref_phys_profile.min()
    ref_max = ref_phys_profile.max() * 1.5 
    
    # 3. Build the single, Denoised Hardware Spline
    x = np.concatenate([[batch_min], batch_anchor_medians.values, [batch_max]])
    y = np.concatenate([[ref_min], ref_phys_profile_matched.values, [ref_max]])
    
    # Ensure strict monotonicity
    x = np.maximum.accumulate(x)
    _, unique_idx = np.unique(x, return_index=True)
    spline = PchipInterpolator(x[unique_idx], y[unique_idx])
    
    # 4. Universal Correction (Matrix Operation)
    corrected_matrix = np.clip(spline(expr_df.values), 0.0, None)
    
    return pd.DataFrame(corrected_matrix, index=expr_df.index, columns=expr_df.columns)

def individual_anchor_physical_spline(expr_df, ref_phys_profile, anchors):
    """
    N=1 APS: Bespoke physical spline for clinical streaming.
    """
    valid_anchors = [g for g in anchors if g in expr_df.columns]
    corrected_matrix = np.zeros_like(expr_df.values)
    ref_min = ref_phys_profile.min()
    ref_max = ref_phys_profile.max() * 1.5

    for i in range(expr_df.shape[0]):
        patient_vals = expr_df.values[i, :]
        patient_anchors = expr_df.iloc[i][valid_anchors].sort_values()
        ref_phys_profile_matched = ref_phys_profile.reindex(patient_anchors.index)
        
        tgt_min = patient_vals.min()
        tgt_max = patient_vals.max()
        
        x = np.concatenate([[tgt_min], patient_anchors.values, [tgt_max]])
        y = np.concatenate([[ref_min], ref_phys_profile_matched.values, [ref_max]])
        x = np.maximum.accumulate(x)
        _, u_aps = np.unique(x, return_index=True)
        spline = PchipInterpolator(x[u_aps], y[u_aps])
        corrected_matrix[i, :] = np.clip(spline(patient_vals), 0.0, None)
        
    return pd.DataFrame(corrected_matrix, index=expr_df.index, columns=expr_df.columns)

def lame_warp_calibration(ranks_df, gamma):
    """Alternative: Parametric bending."""
    x = np.clip(ranks_df.values, 0, 1 - 1e-12)
    warped = 1.0 - (1.0 - x**gamma)**(1.0/gamma)
    return pd.DataFrame(warped, index=ranks_df.index, columns=ranks_df.columns)

def identity_calibration(ranks_df):
    """Bypass."""
    return ranks_df
