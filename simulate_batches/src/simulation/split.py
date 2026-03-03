from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import bisect
import pandas as pd

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

    def _rng(self):
        return np.random.default_rng(self.random_state)
    
    def apply(
            self,
            X: pd.DataFrame,
            metadata: pd.DataFrame | None = None,
    ) -> BatchSplit:
        raise NotImplementedError
    
class RandomBatchSplit(BaseBatchSplit):
    """
    Uniform Random Assignment.
    """
    def apply(self, X, metadata=None) -> BatchSplit:
        rng = self._rng()

        batch_labels = rng.integers(
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
        
        rng = self.rng()
        batch_labels = np.zeros(len(X), dtype=int)

        for _, idx in metadata.groupby(self.column).groups.items():
            idx = list(idx)
            assignments = rng.integers(
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
    entropy = Desired conditional entropy of batch given metadata (in bits)
                Cannot be used with strength.
    random_state = RNG seed for reproducibility
    """
    
    def __init__(
            self,
            n_batches: int,
            column: str,
            strength: Optional[float] = None,
            entropy: Optional[float] = None,
            random_state: int | None = None,
    ):
        self.n_batches = n_batches
        self.column = column
        self.rng = np.random.default_rng(random_state)

        if (strength is not None) and (entropy is not None):
            raise ValueError("Specify either strength or entropy, not both.")
        if (strength is None) and (entropy is None):
            strength = 0.5

        if strength is not None:
            if not (0 <= strength <= 1):
                raise ValueError("strength must be in [0,1]")
            if entropy is not None:
                if not (0 <= entropy <= 1):
                    raise ValueError("entropy muts be in [0,1] bits for binary metadata")
        super().__init__(n_batches, random_state)

        if not (0 <= strength <= 1):
            raise ValueError("strength must be in [0, 1]")
        
        self.strength = strength
        self.entropy = entropy

    def apply(self, X, metadata) -> BatchSplit:
        if metadata is None:
            raise ValueError("ConfoundedSplit requires metadata.")

        y = metadata[self.column].values
        if len(np.unique(y)) != 2:
            raise ValueError("ConfoundedSplit currently supports binary variables.")
        
        # Map to 0/1
        y_binary = pd.factorize(y)[0]

        # Probability of batch 1 given condition
        base_prob = 1 / self.n_batches
        
        if self.entropy is not None:
            # Compute probability p that yields desired entropy
            # Solve H(p) = entropy --> p*log2(p) + (1-p)*log2(1-p) = entropy
            # For binary, p in [0.5, 1] since symmetric
            target_entropy = self.entropy
            if target_entropy == 0:
                p = 1.0
            elif target_entropy == 1.0:
                p = 0.5
            else:
                # Numeric solution using simple bisection
                def H(p):
                    p = np.clip(p, 1e-10, 1-1e-10)
                    return -(p*np.log2(p) + (1-p)*np.log2(1-p))
                p = bisect(lambda x: H(x)-target_entropy, 0.5, 1.0)
            probs = np.where(y_binary == 1, p, 1-p)
        else:
            # strength is specified
            probs = (1 - self.strength) * base_prob + self.strength * y_binary

        # Assign batches
        if self.n_batches == 2:
            batch_labels = self.rng.binomial(1, probs)
        else:
            batch_labels = self.rng.integers(0, self.n_batches, size=len(X))
            mask = self.rng.random(len(X)) < (self.entropy if self.entropy is not None else self.strength)
            batch_labels[mask] = y_binary[mask] % self.n_batches

        batch_series = pd.Series(
            batch_labels,
            index=X.index,
            name="batch",
        )

        empirical_corr = np.corrcoef(batch_labels, y_binary)[0,1]

        return BatchSplit(
            batch_labels=batch_series,
            metadata=metadata,
            info={
                "type": "confounded",
                "column": self.column,
                "strength": self.strength,
                "entropy": self.entropy,
                "empirical_correlation": empirical_corr,
            },
        )