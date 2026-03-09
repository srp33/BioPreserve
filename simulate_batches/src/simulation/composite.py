from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class CompositeBatchEffect(BaseBatchEffect):
    """
    Stack of multiple batch effects, applied sequentially
    """

    def __init__(self, effects: list[BaseBatchEffect], random_state: int | None = None):
        self.effects = effects
        self.last_results: list[BaseBatchEffect] = []
        
    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:
        X_current = X.copy()
        self.last_results = [] # Could also make this a dictionary?

        for effect in self.effects:
            result = effect.apply(X_current, split)
            X_current = result.X_batch
            effect.last_description = result.description
            self.last_results.append(effect)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_current,
            metadata=split.metadata,
            description=[res.description for res in self.last_results],
        )
    
    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Inverts the composite effect by applying the inverses in reverse order.
        """
        shift, scale = self.extract_effect(X_batch)
        return (X_batch - shift) / scale
    
    def parameters(self) -> list[dict]:
        """
        Returns the parameters of all applied effects in order.
        """
        if not self.last_results:
            raise RuntimeError("No descriptions available; apply() must be called first.")
        
        param_list = []
        for desc in self.last_results:
            if isinstance(desc, dict):
                # merge all batch descriptions
                batch_params = {k: v.parameters() for k, v in desc.items()}
            else:
                param_list.append(desc.parameters())
        return param_list

    def extract_effect(self, X_batch: pd.DataFrame)  -> tuple[np.ndarray, np.ndarray]:
        """
        Extract shift and scale for inverse transformation.
        """
        n_features = X_batch.shape[1]
        total_shift = np.zeros(n_features)
        total_scale = np.ones(n_features)

        for effect in reversed(self.last_results):
            shift, scale = effect.extract_effect(X_batch)
            total_shift = total_shift * scale + shift
            total_scale = total_scale * scale 

        return total_shift, total_scale