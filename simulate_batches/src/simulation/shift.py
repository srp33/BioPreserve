from __future__ import annotations
import numpy as np
import pandas as pd 

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class AdditiveShiftDescription(BatchEffectDescription):
    """
    Stores true additive batch shifts
    """

    def __init__(self, shifts: dict[str, pd.Series], batch_labels: pd.Series):
        self.shifts = shifts
        self.batch_labels = batch_labels

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        if not set(self.shifts.keys()).issubset(set(self.batch_labels.unique())):
            raise ValueError("Shift description batches do not match batch labels.")
        X = X_batch.copy()

        for batch_id, shift in self.shifts.items():
            mask = self.batch_labels == batch_id
            X.loc[mask] = X.loc[mask] - shift

        return X
    
    def parameters(self) -> dict:
        return {
            "type": "additive_shift",
            "shift_vector": self.shifts,
        }
    
# Think about how to add a global shift
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(self, scale: float = 1.0, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        
class AdditiveShiftEffect:
    def __init__(self, scale: float = 1.0, random_state=None):
        self.scale = scale
        self.rng = np.random.default_rng(random_state)

    def apply(self, X: pd.DataFrame, batch_labels: pd.Series):
        batch_parameters = {}
        X_batch = X.copy()
        for batch_id in batch_labels.unique():
            mask = batch_labels == batch_id
            shift_vec = self.rng.normal(0, self.scale, size=X.shape[1])
            X_batch.loc[mask] += shift_vec
            batch_parameters[batch_id] = {"shift": shift_vec, "scale": np.ones(X.shape[1])}
        desc = BatchEffectDescription(batch_parameters, batch_labels)
        return X_batch, desc

"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""