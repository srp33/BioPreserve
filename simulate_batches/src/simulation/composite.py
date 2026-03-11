from __future__ import annotations
import pandas as pd
import numpy as np

from .base import BaseBatchEffect, BatchEffectResult
from .split import BatchSplit

class CompositeBatchEffect(BaseBatchEffect):
    """
    Composite batch effect: sequentially applies multiple batch effects.
    Provides inversion and parameter extraction using per-batch diagonal approximation.
    """

    def __init__(self, effects: list[BaseBatchEffect], random_state: int | None = None):
        super().__init__(random_state)
        self.effects = effects
        self.results: list[BatchEffectResult] = []

    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:
        """
        Apply all effects sequentially, storing result data per effect.
        """
        X_current = X.copy()
        self.results = []
        
        batch_labels = split.batch_labels
        batches = batch_labels.unique()

        n_features = X_current.shape[1]

        total_shift = {b: np.zeros(n_features) for b in batches}
        total_scale = {b: np.ones(n_features) for b in batches}

        for effect in self.effects:
            result = effect.apply(X_current, split)
            for b in batches:
                shift_new = result.batch_shift[b]
                scale_new = result.batch_scale[b]

                shift_old = total_shift[b]
                scale_old = total_scale[b]

                total_shift[b] = shift_old * scale_new + shift_new
                total_scale[b] = scale_old * scale_new

            X_current = result.X_batch
            self.results.append(result)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_current,
            batch_labels = split.batch_labels,
            batch_shift = total_shift,
            batch_scale = total_scale
        )
