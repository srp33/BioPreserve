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
    
# Think about how to add a global shift
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(self, scale: float = 1.0, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        self.last_shift = None
        
    def apply(self, X: pd.DataFrame, split: BatchSplit,) -> BatchEffectResult:

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()
        X_batch = X.copy()
        self.last_shift = {}

        n_features = X.shape[1]

        for batch_id in unique_batches:

            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            shift = self.rng.normal(0, self.scale, size=n_features)

            X_shifted = X_sub + shift

            X_batch.loc[mask] = X_shifted

            self.last_shift[batch_id] = AdditiveShiftDescription(shift=shift)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=self.last_shift,
        )
    
    def extract_effect(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns per-batch shift and scale vectors for inversion.
        """
        shift_scale = {}
        for batch_id, desc in self.last_shift.items():
            n_features = len(desc.shift)
            shift_scale[batch_id] = (-desc.shift, np.ones(n_features))
        return shift_scale


"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""