"""Module 5: The Diagnostic Layer (State Estimation)"""
import numpy as np

def soft_gmm_estimation(calibrated_scores_df, gmm_model):
    """
    Champion: Soft GMM Interpolation.
    Reads the exact biological gradient without hard cutoffs.
    """
    return gmm_model.predict_proba(calibrated_scores_df.values)

def hard_threshold_estimation(calibrated_scores_df):
    """Alternative: Binary classification based on origin."""
    probs = (calibrated_scores_df.values > 0).astype(float)
    return probs / probs.sum(axis=1, keepdims=True)
