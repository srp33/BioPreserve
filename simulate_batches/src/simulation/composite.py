from __future__ import annotations
import pandas as pd
import numpy as np
from typing import list, Tuple

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class CompositeBatchEffect(BaseBatchEffect):
    """
    Composite batch effect: sequentially applies multiple batch effects.
    Provides inversion and parameter extraction using per-batch diagonal approximation.
    """

    def __init__(self, effects: list[BaseBatchEffect], random_state: int | None = None):
        super().__init__(random_state)
        self.effects = effects
        self.last_results: list[BatchEffectResult] = []

    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:
        """
        Apply all effects sequentially, storing descriptions per effect.
        """
        X_current = X.copy()
        self.last_results = []

        for effect in self.effects:
            result = effect.apply(X_current, split)
            X_current = result.X_batch
            self.last_results.append(result)

        return BatchEffectResult(
            X_original=X,
            X_batch=X_current,
            metadata=split.metadata,
            description=[res.description for res in self.last_results],
        )

    def extract_shift_scale(self) -> dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Combine per-batch shift and scale from all applied effects (diagonal approximation).
        Returns:
            dict[batch_id] = (shift_vector, scale_vector)
        """
        if not self.last_results:
            raise RuntimeError("No effects applied yet; call apply() first.")

        # Collect all batch IDs from first effect
        batch_ids = set()
        for result in self.last_results:
            batch_ids.update(result.description.keys())

        combined: dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for batch_id in batch_ids:
            total_shift = None
            total_scale = None

            for result in self.last_results:
                desc = result.description[batch_id]

                # Use the effect's own extract_shift_scale method
                if hasattr(result, "effect"):  # optional link back to effect
                    effect_obj = result.effect
                else:
                    # fallback: assume the description itself has extract_shift_scale
                    effect_obj = desc

                s, sc = effect_obj.extract_shift_scale()[batch_id]

                if total_shift is None:
                    total_shift = s
                    total_scale = sc
                else:
                    # Combine shifts/scales: forward composition
                    total_shift = total_shift * sc + s
                    total_scale = total_scale * sc

            combined[batch_id] = (total_shift, total_scale)

        return combined

    def invert(self, X_batch: pd.DataFrame, split: BatchSplit) -> pd.DataFrame:
        """
        Invert the composite effect using diagonal approximation.
        """
        shift_scale = self.extract_shift_scale()
        X_hat = X_batch.copy()

        for batch_id, (shift, scale) in shift_scale.items():
            mask = split.batch_labels == batch_id
            X_hat.loc[mask] = (X_hat.loc[mask] - shift) / scale

        return X_hat

    def parameters(self) -> list[dict]:
        """
        Returns a list of parameter dictionaries for all applied effects.
        """
        params_list = []
        for result in self.last_results:
            for desc in result.description.values():
                params_list.append(desc.parameters())
        return params_list