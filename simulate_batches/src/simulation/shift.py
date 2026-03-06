from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd 

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class AdditiveShiftDescription(BatchEffectDescription):
    """
    Stores true additive batch shifts
    """

    def __init__(self, shift: np.ndarray):
        self.shift = shift

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        return X_batch - self.shift
    
    def parameters(self) -> dict:
        return {
            "type": "additive_shift",
            "shift_vector": self.shift,
        }
    
    def extract_shift_scale(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract shift and scale for inverse transformation.
        
        Forward: Y = X + shift_amount
        Inverse: X = Y - shift_amount = Y * 1.0 + (-shift_amount)
        """
        n_features = len(self.shift)
        shift = -self.shift
        scale = np.ones(n_features)
        return shift, scale
    
# Think about how to add a global shift
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
        descriptions = {}

        n_features = X.shape[1]

        for batch_id in unique_batches:

            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            shift = self.rng.normal(0, self.scale, size=n_features)

            X_shifted = X_sub * shift

            X_batch.loc[mask] = X_shifted

            descriptions[batch_id] = AdditiveShiftDescription(shift=shift)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=descriptions,
        )


"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""