from __future__ import annotations
import numpy as np
import pandas as pd

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class MultiplicativeScaleDescription(BatchEffectDescription):
    """
    Stores true multiplicative batch scaling
    """

    def __init__(self, scalings: dict[str, pd.Series], batch_labels: pd.Series):
        self.scalings = scalings
        self.batch_labels = batch_labels

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        X = X_batch.copy()

        for batch_id, scaling in self.scalings.items():
            mask = self.batch_labels == batch_id
            X.loc[mask] = X.loc[mask] / scaling 

        return X

    def parameters(self) -> dict:
        return {
            "type": "multiplicative_scale",
            "scale_vector": self.scalings,
        }
    
class MultiplicativeScaleEffect(BaseBatchEffect):
    def __init__(self, scale: float = 1.0, random_state=None):
        self.scale = scale
        self.rng = np.random.default_rng(random_state)

    def apply(self, X: pd.DataFrame, batch_labels: pd.Series):
        batch_parameters = {}
        X_batch = X.copy()
        for batch_id in batch_labels.unique():
            mask = batch_labels == batch_id
            scale_vec = self.rng.normal(1.0, self.scale, size=X.shape[1])
            X_batch.loc[mask] *= scale_vec
            batch_parameters[batch_id] = {"shift": np.zeros(X.shape[1]), "scale": scale_vec}
        desc = BatchEffectDescription(batch_parameters, batch_labels)
        return X_batch, desc