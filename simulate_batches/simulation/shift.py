from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd 

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription


class AdditiveShiftDescription(BatchEffectDescription):
    """
    Stores true additive batch shifts
    """

    def __init__(self, shifts: dict[int, np.ndarray], batch_labels: np.ndarray, feature_names: list[str]):
        self.shifts = shifts
        self.batch_labels = batch_labels
        self.feature_names = feature_names

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        X = X_batch.copy()

        for i, batch_id in enumerate(self.batch_labels):
            X.iloc[i] -= self.shifts[batch_id]

        return X
    
    def parameters(self) -> dict:
        return {
            "type": "additive_shift",
            "n_batches": len(self.shifts),
        }
    
class AdditiveShiftEffect(BaseBatchEffect):
    """
    Simulates additive batch-specific mean shifts.
    """

    def __init__(self, n_batches: int, scale: float = 1.0, random_state=None):
        super().__init__(random_state)
        self.n_batches = n_batches
        self.scale = scale

    def apply(self, X: pd.DataFrame, metadata=None) -> BatchEffectResult:
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        batch_labels = rng.integers(0, self.n_batches, size=n_samples)

        shifts = {
            b: rng.normal(0, self.scale, size=n_features)
            for b in range(self.n_batches)
        }

        X_batch = X.copy()

        for i, batch_id in enumerate(batch_labels):
            X_batch.iloc[i] += shifts[batch_id]

        description = AdditiveShiftDescription(
            shifts=shifts,
            batch_labels=batch_labels,
            feature_names=list(X.columns),
        )

        batch_meta = pd.DataFrame(
            {"batch": batch_labels},
            index=X.index,
        )

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=batch_meta,
            description=description,
        )


"""
Could also choose Sparse shift (only 20% of genes affected), block shift (shift only specific gene modules), heteroskedastsic shift (shift magnitude proportional to gene mean)
mask = rng.choice([0, 1], size=n_features, p=[0.8, 0.2])
shift = rng.normal(0, scale, size=n_features) * mask
"""