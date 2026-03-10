from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .split import BatchSplit

class BatchEffectResult:
    def __init__(self, X_original: pd.DataFrame, X_batch: pd.DataFrame, batch_labels: pd.Series, batch_shift: dict[str, np.ndarray], batch_scale: dict[str, np.ndarray]):
        self.X_original = X_original
        self.X_batch = X_batch
        self.batch_labels = batch_labels
        self.batch_shift = batch_shift
        self.batch_scale = batch_scale

    def invert(self) -> pd.DataFrame:
        """
        Apply the ground-truth inverse transformation. 
        """
        X = self.X_batch.copy()
        batch_labels = self.batch_labels

        for batch_id in self.batch_shift:
            mask = batch_labels == batch_id
            shift = self.batch_shift[batch_id]
            scale = self.batch_scale[batch_id]

            X.loc[mask] = (X.loc[mask] - shift) / scale

        return X


class BatchEffectDescription(ABC):
    """
    Describes the true batch transformation applied.
    Must be able to invert or expose true parameters.
    """

    @abstractmethod
    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        ...
    
    @abstractmethod
    def parameters(self) -> dict:
        ...

    @abstractmethod
    def extract_effect(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns vectors for inverting the applied effect.
        """
        ...


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