from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .split import BatchSplit


@dataclass
class BatchEffectResult:
    X_original: pd.DataFrame
    X_batch: pd.DataFrame
    metadata: pd.DataFrame
    description: BatchEffectDescription

class BatchEffectDescription:
    """
    Base class: stores true per-batch shift and scale vectors.
    """
    def __init__(self, batch_parameters: dict[str, dict], batch_labels: pd.Series):
        """
        batch_parameters: dict mapping batch_id -> {"shift": np.ndarray, "scale": np.ndarray}
        """
        self.batch_parameters = batch_parameters
        self.batch_labels = batch_labels

    def invert(self, X_batch: pd.DataFrame):
        X_hat = X_batch.copy()
        for batch_id, params in self.batch_parameters.items():
            mask = self.batch_labels == batch_id
            X_hat.loc[mask] = (X_batch.loc[mask] - params["shift"]) / params["scale"]
        return X_hat

class BaseBatchEffect(ABC):
    """
    Abstract batch effect generator.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    @abstractmethod
    def apply(
        self, 
        X: pd.DataFrame,
        split: BatchSplit,
    ) -> BatchEffectResult:
        ...
        