from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

class CompositeEffectDescription(BatchEffectDescription):
    """
    Stores per-batch descriptions for multiple sequential effects.
    Inversion is done by applying each effect's invert method in reverse order. 
    """

    def __init__(self, descriptions: list[BatchEffectDescription], batch_labels: pd.Series):
        self.descriptions = descriptions
        self.batch_labels = batch_labels

    def invert(self, X_batch: pd.DataFrame):
        """
        Inverts the composite effect by applying the inverses in reverse order.
        """
        shift, scale = self.total_shift_scale
        return (X_batch - shift) / scale
    
    def parameters(self) -> list[dict]:
        """
        Returns the parameters of all applied effects in order.
        """
        return [desc.parameters() for desc in self.descriptions]

    @property
    def total_shift_scale(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes total diagonal shift and scale vectors for approximate inversion. 
        """
        if not self.descriptions: 
            raise RuntimeError("No descriptions available; apply() must be called first.")
        
        # Initialize per-feature vectors
        first_desc = self.descriptions[0]
        if hasattr(first_desc, "covariance"):
            n_features = next(iter(first_desc.covariance.values()))["D"].shape[0]
        elif hasattr(first_desc, "scalings"):
            n_features = next(iter(first_desc.scalings.values())).shape[0]
        elif hasattr(first_desc, "shifts"):
            n_features = next(iter(first_desc.shifts.values())).shape[0]
        else:
            n_features = 1  # fallback
            
        total_shift = np.zeros(n_features)
        total_scale = np.ones(n_features)

        # Make a copy of batch_labels from first description
        batch_labels = self.batch_labels
        X_current = None

        # Apply effects in reverse order
        for desc in reversed(self.descriptions):
            # Determine which dictionary exists in the description
            if hasattr(desc, "covariance"):
                # CovarianceDescription
                batch_dict = desc.covariance
                shift_vec = np.zeros(n_features)
                scale_vec = np.ones(n_features)
                for batch_id, params in batch_dict.items():
                    mask = batch_labels == batch_id
                    n_batch = mask.sum()
                    D = params["D"]
                    C = params["C"]
                    # Weighted contribution by batch size
                    shift_vec += C * n_batch / len(batch_labels)
                    scale_vec *= D ** (n_batch / len(batch_labels))

            elif hasattr(desc, "scalings"):
                # MultiplicativeScaleDescription
                batch_dict = desc.scalings
                shift_vec = np.zeros(n_features)
                scale_vec = np.ones(n_features)
                for batch_id, scaling in batch_dict.items():
                    mask = batch_labels == batch_id
                    n_batch = mask.sum()
                    scale_vec *= scaling.values ** (n_batch / len(batch_labels))

            elif hasattr(desc, "shifts"):
                # AdditiveShiftDescription
                batch_dict = desc.shifts
                shift_vec = np.zeros(n_features)
                scale_vec = np.ones(n_features)
                for batch_id, shift in batch_dict.items():
                    mask = batch_labels == batch_id
                    n_batch = mask.sum()
                    shift_vec += shift.values * (n_batch / len(batch_labels))

            else:
                # Fallback for effects without stored batch dictionaries
                shift_vec = np.zeros(n_features)
                scale_vec = np.ones(n_features)

            # Update total shift and scale
            total_shift = total_shift * scale_vec + shift_vec
            total_scale = total_scale * scale_vec
        return total_shift, total_scale

class CompositeBatchEffect(BaseBatchEffect):
    """
    Stack multiple effects sequentially.
    """
    def __init__(self, effects: list):
        self.effects = effects
        self.description = None

    def apply(self, X: pd.DataFrame, batch_labels: pd.Series):
        X_current = X.copy()
        batch_parameters = {b: {"shift": np.zeros(X.shape[1]),
                                "scale": np.ones(X.shape[1])} 
                            for b in batch_labels.unique()}

        # Apply each effect and compose shift/scale
        for effect in self.effects:
            X_current, desc = effect.apply(X_current, batch_labels)
            for batch_id, p in desc.batch_parameters.items():
                # compose: new_total = old_total * scale + shift
                batch_parameters[batch_id]["shift"] = batch_parameters[batch_id]["shift"] * p["scale"] + p["shift"]
                batch_parameters[batch_id]["scale"] *= p["scale"]

        self.description = BatchEffectDescription(batch_parameters, batch_labels)
        return X_current, self.description