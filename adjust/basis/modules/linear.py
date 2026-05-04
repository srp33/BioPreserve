"""Module 4: The Linear Correction Layer (Scale & Shift)"""
import numpy as np

def bypass_correction(raw_scores):
    """
    Champion: Do nothing. 
    AQN makes downstream linear corrections redundant and destructive.
    """
    return raw_scores

def sentinel_bgn_correction(raw_scores, axis_params, p_gain=1.0):
    """
    Alternative: Subtracts ghost load and scales by variance ratio.
    (Our previous pipeline iteration).
    """
    gain_ratio = axis_params["ref_gain"] / max(p_gain, 1e-12)
    # Basic BGN centering logic
    return (raw_scores - axis_params["ref_median"]) * np.clip(gain_ratio, 0.5, 1.15) + axis_params["ref_median"]
