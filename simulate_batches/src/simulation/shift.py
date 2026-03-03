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

    def __init__(self, shifts: dict[int, np.ndarray], batch_labels: pd.Series, feature_names: list[str]):
        self.shifts = shifts
        self.batch_labels: pd.Series = batch_labels
        self.feature_names = feature_names

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        if not X_batch.index.equals(self.batch_labels.index):
            raise ValueError("Index mismatch between X_batch and stored batch labels.")
        X = X_batch.copy()

        for batch_id, shift in self.shifts.items():
            idx = self.batch_labels == batch_id
            X.loc[idx] -= shift

        return X
    
    def parameters(self) -> dict:
        return {
            "type": "additive_shift",
            "n_batches": len(self.shifts),
        }
    
# Think about how to add a global shift
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(self, scale: float = 1.0, random_state=None):
        super().__init__(random_state)
        self.scale = scale
        self.rng = np.random.default_rng(random_state)
        
    def apply(self, X: pd.DataFrame, split: BatchSplit,) -> BatchEffectResult:
        rng = self.rng

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()

        n_features = X.shape[1]

        shifts = {
            b: rng.normal(0, self.scale, size=n_features)
            for b in unique_batches
        }

        X_batch = X.copy()

        # Apply shifts
        shift_matrix = np.vstack([shifts[b] for b in batch_labels])
        X_batch = X + shift_matrix

        description = AdditiveShiftDescription(
            shifts=shifts,
            batch_labels=batch_labels,
            feature_names=list(X.columns),
        )

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=description,
        )


"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""