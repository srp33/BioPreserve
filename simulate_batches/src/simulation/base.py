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
    description: "BatchEffectDescription"

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
    def extract_shift_scale(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract shift and scale parameters for the inverse transformation.
        
        Returns parameters in the form: X = Y * scale + shift
        where Y is the batch-affected data and X is the recovered original.
        
        Parameters
        ----------
        X_batch : pd.DataFrame
            The batch-affected data (may be needed for approximations).
        
        Returns
        -------
        shift : np.ndarray
            Shift vector (one value per feature).
        scale : np.ndarray
            Scale vector (one value per feature).
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
        
    def extract_effect(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns vectors for inverting the applied effect.
        """
        n_features = X_batch.shape[1]
        shift = np.zeros(n_features)
        scale = np.ones(n_features)
        return shift, scale