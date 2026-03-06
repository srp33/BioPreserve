from __future__ import annotations
import numpy as np
import pandas as pd

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class MultiplicativeScaleDescription(BatchEffectDescription):
    """
    Stores true multiplicative batch scaling
    """

    def __init__(self, scaling: np.ndarray):
        self.scaling = scaling

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        return X_batch / self.scaling
    
    def parameters(self) -> dict:
        return {
            "type": "multiplicative_scale",
            "scale_vector": self.scaling,
        }
    
    def extract_shift_scale(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract shift and scale for inverse transformation.
        
        Forward: Y = X * scale_amount
        Inverse: X = Y / scale_amount = Y * (1/scale_amount) + 0
        """
        n_features = len(self.scaling)
        shift = np.zeros(n_features)
        scale = 1.0 / self.scaling
        return shift, scale
    
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
        
        X_batch = X.copy()
        descriptions = {}

        n_features = X.shape[1]

        # Log-normal scaling? 
        for batch_id in unique_batches:
            
            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            scaling = self.rng.lognormal(mean=0.0, sigma=self.scale, size=n_features)
            
            X_scaled = X_sub * scaling

            X_batch.loc[mask] = X_scaled

            descriptions[batch_id] = MultiplicativeScaleDescription(scaling=scaling)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=descriptions,
        )