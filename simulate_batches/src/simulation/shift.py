from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd 

from .base import BaseBatchEffect, BatchEffectResult
from .split import BatchSplit
    
# Think about how to add a global shift, would number of batches just = 1?
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(self, scale: float = 1.0, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        
    def apply(self, X: pd.DataFrame, split: BatchSplit,) -> BatchEffectResult:

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()
        X_batch = X.copy()
        
        shift = {}
        scale = {}

        n_features = X.shape[1]

        for batch_id in unique_batches:

            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            shift_effect = self.rng.normal(0, self.scale, size=n_features)

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
    

"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""