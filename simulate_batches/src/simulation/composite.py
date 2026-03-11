from __future__ import annotations
import pandas as pd
import numpy as np
from typing import list, Tuple

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
        self.total_shift: dict[str, np.ndarray] = {}
        self.total_scale: dict[str, np.ndarray] = {}

    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:
        """
        Apply all effects sequentially, storing result data per effect.
        """
        X_current = X.copy()
        self.results = []
        n_features = X_current.shape[1]
        self.total_shift = np.zeros(n_features)
        self.total_scale = np.ones(n_features)

        for effect in self.effects:
            result = effect.apply(X_current, split)
            self.total_shift += result.batch_shift
            self.total_shift *= result.batch_scale
            self.total_scale *= result.batch_scale
            X_current = result.X_batch
            self.results.append(result)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_current,
            batch_labels = split.batch_labels,
            batch_shift = self.total_shift,
            batch_scale = self.total_scale
        )