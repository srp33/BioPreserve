from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import bisect
import scipy.special as sps
from sklearn.metrics import mutual_info_score
import pandas as pd
import random

# Core Data Structure
@dataclass
class BatchSplit:
    batch_labels: pd.Series 
    metadata: pd.DataFrame | None = None
    info: dict | None = None


# Base Class
class BaseBatchSplit:
    def __init__(self, n_batches: int, random_state: int | None = None):
        self.n_batches = n_batches
        self.random_state = random_state
        # Initialize rng in the Base Class so that each apply generates a different random number controlled by the state
        self.rng = np.random.default_rng(self.random_state)
    
    def apply(
            self,
            X: pd.DataFrame,
            metadata: pd.DataFrame | None = None,
    ) -> BatchSplit:
        raise NotImplementedError
    

class RandomSplit(BaseBatchSplit):
    """
    Uniform Random Assignment.
    """
    def apply(self, X, metadata=None) -> BatchSplit:

        batch_labels = self.rng.integers(
            0, self.n_batches, size=len(X)
        )

        batch_series = pd.Series(
            batch_labels,
            index=X.index,
            name="batch",
        )

        return BatchSplit(
            batch_labels=batch_series,
            metadata=metadata,
            info={
                "type": "rando",
                "n_batches": self.n_batches,
            },
        )


class StratifiedSplit(BaseBatchSplit):
    """
    Balances distribution of a categorical metadata column across batches.
    """
    def __init__(
        self,
        n_batches: int, 
        column: str,
        random_state: int | None = None,
    ):
        super().__init__(n_batches, random_state)
        self.column = column

    def apply(self, X, metadata) -> BatchSplit:
        if metadata is None:
            raise ValueError("StratifiedSplit requires metadata.")
        
        batch_labels = np.zeros(len(X), dtype=int)

        for _, idx in metadata.groupby(self.column).groups.items():
            idx = list(idx)
            assignments = self.rng.integers(
                0, self.n_batches, size=len(idx)
            )
            batch_labels[[X.index.get_loc(i) for i in idx]] = assignments

        batch_series = pd.Series(
            batch_labels,
            index=X.index,
            name="batch",
        )
        
        return BatchSplit(
            batch_labels=batch_series,
            metadata=metadata,
            info={"type": "stratified",
                  "column": self.column,
            },
        )


class ConfoundedSplit(BaseBatchSplit):
    """
    Introduces controlled confounding between a binary metadata column and batch.
    
    Parameters
    ----------
    n_batches = Number of batches to create
    column = Binary metadata column to confound with batch.
    strength = Blend factor between independent (0) and perfectly confounded (1).
                Cannot be used with entropy.
    temperature = Energy parameter for confounded assignment. 0.01 is perfectly confounded, 100 is independent.
                Cannot be used with strength.
    random_state = RNG seed for reproducibility
    """
    
    def __init__(
            self,
            n_batches: int,
            column: str,
            strength: Optional[float] = None,
            temperature: Optional[float] = None,
            random_state: int | None = None,
    ):
        super().__init__(n_batches, random_state)
        self.column = column

        if (strength is not None) and (temperature is not None):
            raise ValueError("Specify either strength or temperature, not both.")
        if (strength is None) and (temperature is None):
            strength = 0.5

        if strength is not None and not (0 <= strength <= 1):
            raise ValueError("strength must be in [0,1]")

        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be positive")
        
        self.strength = strength
        self.temperature = temperature

    def _allocate_batches(self, samples_per_class) -> np.ndarray:
        """D'Hondt allocation when batches > classes."""
        n_classes = len(samples_per_class)
        perfectly_confounded = np.eye(n_classes, self.n_batches, dtype=float)

        # Distribute the remaining batches one by one
        for new_batch_idx in range(n_classes, self.n_batches):
            largest_class = np.argmax(samples_per_class / perfectly_confounded.sum(axis=1))
            perfectly_confounded[largest_class, new_batch_idx] = 1.0
            
        return perfectly_confounded

    def _distribute_classes(self, samples_per_class) -> np.ndarray:
        """Greedy bin packing when batches < classes."""
        n_classes = len(samples_per_class)
        batch_sizes = np.zeros(self.n_batches)
        perfectly_confounded = np.zeros((n_classes, self.n_batches), dtype=float)

        for c in np.argsort(-samples_per_class):
            b = int(np.argmin(batch_sizes))
            perfectly_confounded[c, b] = 1.0
            batch_sizes[b] += samples_per_class[c]

        return perfectly_confounded

    def apply(self, X, metadata, debug=False) -> BatchSplit:

        if metadata is None:
            raise ValueError("ConfoundedSplit requires metadata.")

        class_idx, _ = pd.factorize(metadata[self.column])
        samples_per_class = np.bincount(class_idx)
        n_classes = len(samples_per_class)

        if self.n_batches == n_classes:
            perfectly_confounded = np.eye(n_classes, self.n_batches, dtype=float)
        elif self.n_batches > n_classes:
            perfectly_confounded = self._allocate_batches(samples_per_class)
        else: # self.n_batches < n_classes:
            perfectly_confounded = self._distribute_classes(samples_per_class)

        n = len(X)

        if self.temperature is not None:
            # Apply temperature to affinities to compute logits
            # If temp -> inf, then logits -> uniform probability over all batches
            # If temp -> 0, then logits -> assignment only to designated batches for that class.
            logits = perfectly_confounded / self.temperature
            probs = sps.softmax(logits, axis=1)
        else:
            # Normalize rows to yield uniform probabilities across assigned batches
            confounded_probs = perfectly_confounded / perfectly_confounded.sum(axis=1, keepdims=True)
            equal_probs = np.full_like(confounded_probs, 1 / self.n_batches)
            probs = (self.strength * confounded_probs + (1 - self.strength) * equal_probs)
            
        # Probs is the probability of assigning samples from each class to each batch, normalized over each class.
        confounded = np.empty(n, dtype=int)
        for c in range(n_classes):
            mask = class_idx == c
            confounded[mask] = self.rng.choice(a=self.n_batches, p=probs[c], size=mask.sum())

        # Mutual information ranges from 0 (independent) to log(min(n_batches, n_classes)) nats.
        empirical_mi = mutual_info_score(class_idx, confounded)

        if debug:
            print(f"DEBUG: n_batches={self.n_batches}, classes={n_classes}, strength={self.strength}, temp={self.temperature}")
            print(f"DEBUG: Batch distribution: {np.bincount(confounded, minlength=self.n_batches)}")
            print(f"DEBUG: Empirical Mutual Information: {empirical_mi:.4f} nats")

        return BatchSplit(
            batch_labels=pd.Series(confounded, index=X.index, name="batch"),
            metadata=metadata,
            info=dict(
                type="confounded_general",
                column=self.column,
                strength=self.strength,
                temperature=self.temperature,
                classes=n_classes,
                batches=self.n_batches,
                mutual_information=empirical_mi,
            ),
        )
