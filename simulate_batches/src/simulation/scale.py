from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class MultiplicativeScaleDescription(BatchEffectDescription):
    """
    Stores true multiplicative batch scaling
    """

    def __init__(self, scaling: dict[int, np.ndarray], batch_labels: pd.Series, feature_names: list[str]):
        self.scaling = scaling
        self.batch_labels: pd.Series = batch_labels
        self.feature_names = feature_names

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        X = X_batch.copy()

        for i, batch_id in enumerate(self.batch_labels):
            X.iloc[i] /= self.scaling[batch_id]

        return X
    
    def parameters(self) -> dict:
        return {
            "type": "multiplicative_scale",
            "n_batches": len(self.scaling),
        }
    
class MultiplicativeScaleEffect(BaseBatchEffect):
    """
    Simulates multiplicative batch-specific scaling
    """

    def __init__(self, scale: float = 1.0, random_state = None):
        super().__init__(random_state)
        self.scale = scale

    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()
                
        n_features = X.shape[1]

        # Normal or log-normal scaling? 
        scaling = {
            b: self.rng.lognormal(mean=0.0, sigma=self.scale, size=n_features)
            for b in unique_batches
        }

        X_batch = X.copy()

        scale_matrix = np.vstack([scaling[b] for b in batch_labels])
        X_batch = X * scale_matrix

        description = MultiplicativeScaleDescription(
            scaling=scaling,
            batch_labels=batch_labels,
            feature_names=list(X.columns),
        )

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=description,
        )