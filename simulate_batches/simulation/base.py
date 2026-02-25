from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd


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
    def invert(self, X_batch: pd.DataFrame) -> pd.Dataframe:
        ...
    
    @abstractmethod
    def parameters(self) -> dict:
        ...


class BaseBatchEffect(ABC):
    """
    Abstract batch effect generator.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    @abstractmethod
    def apply(
        self, 
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
    ) -> BatchEffectResult:
        ...
        