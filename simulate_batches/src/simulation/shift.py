from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd 

from .base import BaseBatchEffect, BatchEffectResult
from .split import BatchSplit
    
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(
            self, 
            scale: float = 1.0, 
            global_shift: bool = False,
            sparse_prob: float | None=None, 
            heteroskedastic: bool = False, 
            random_state=None
    ):
        super().__init__(random_state)
        self.scale = scale
        self.global_shift = global_shift
        self.sparse_prob = sparse_prob
        self.heteroskedastic = heteroskedastic
        
    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()
        X_batch = X.copy()
        
        shift = {}
        scale = {}

        n_features = X.shape[1]

        if self.heteroskedastic:
            feature_scale = X.mean().values
        else:
            feature_scale = np.ones(n_features)

        for batch_id in unique_batches:
            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            # --- base shift ---
            if self.global_shift:
                shift_val = self.rng.normal(0, self.scale)
                shift_effect = np.full(n_features, shift_val)
            else:
                shift_effect = self.rng.normal(0, self.scale, size=n_features)

            # --- heteroskedastic scaling ---
            shift_effect = shift_effect * feature_scale

            # --- sparse mask ---
            if self.sparse_prob is not None:
                mask_sparse = self.rng.binomial(1, self.sparse_prob, size=n_features)
                shift_effect = shift_effect * mask_sparse

            X_shifted = X_sub + shift_effect
            X_batch.loc[mask] = X_shifted

            shift[batch_id] = shift_effect
            scale[batch_id] = np.ones(n_features)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            batch_labels=split.batch_labels,
            batch_shift=shift,
            batch_scale=scale,
        )