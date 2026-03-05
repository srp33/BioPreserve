from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.linalg import sqrtm, inv
from dataclasses import dataclass

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

@dataclass
class CovarianceDescription(BatchEffectDescription):
    """
    Stores approximate inversion for covariance 
    """
    D: np.ndarray
    M: coo_matrix | None
    C: np.ndarray
    X_original: pd.DataFrame

    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate inversion using only per-gene shift and scale.
        Ignores off-diagonal covariance.
        """

        X = self.X_original.values
        Y = X_batch.values

        # Means
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)

        # Centered
        X_c = X - mean_X
        Y_c = Y - mean_Y

        # Full covariance of Y_c
        Sigma_Y = np.cov(Y_c, rowvar=False)

        # Regularize
        eps = 1e-12
        Sigma_Y += eps * np.eye(Sigma_Y.shape[0])

        # Whitening matrix (inverse square root)
        W = inv(sqrtm(Sigma_Y))

        # Apply whitening
        X_hat = Y_c @ W + X.mean(axis=0)

        return pd.DataFrame(
            X_hat,
            index=X_batch.index,
            columns=X_batch.columns,
        )
    
    def parameters(self) -> dict:
        # Gives true inversion ???
        return {
            "D": self.D,
            "M": self.M,
            "C": self.C
        }

class CovarianceEffect(BaseBatchEffect):
    """
    Simulates covariance effects between genes (columns)
    """

    def __init__(self, scale_std: float = 0.1, shift_std: float = 0.2, cov_sparsity: float = 0.01, cov_scale: float = 0.02, random_state: int = None):
        super().__init__(random_state)
        self.scale_std = scale_std
        self.shift_std = shift_std
        self.cov_sparsity = cov_sparsity
        self.cov_scale = cov_scale
    
    def _generate_sparse_M(self, g: int, sparsity: float, scale: float):
        # number of non-zero off-diagonal entries
        k = int(sparsity * g * g)

        rows = self.rng.integers(0, g, size=k)
        cols = self.rng.integers(0, g, size=k)

        # Remove diagonal entries
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]

        values = self.rng.normal(0, scale, size=len(rows))

        M = coo_matrix((values, (rows, cols)), shape=(g, g))

        return M.tocsr()
    
    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:

        X_batch = X.copy()
        descriptions = {}

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()

        for batch_id in unique_batches:
            mask = batch_labels == batch_id
            X_sub = X.loc[mask]

            n, g = X_sub.shape

            # --- Generate parameters ---
            D_vec = self.rng.normal(1.0, self.scale_std, size=g)
            M = self._generate_sparse_M(g, self.cov_sparsity, self.cov_scale)
            C = self.rng.normal(0, self.shift_std, size=g)
            
            # --- Apply transformation ---
            XD = X_sub.values * D_vec   # broadcast multiply
            XM = X_sub.values @ M 
            Y = XD + XM + C.reshape(1, -1)

            X_batch.loc[mask] = Y

            descriptions[batch_id] = CovarianceDescription(
                D=D_vec,
                M=M,
                C=C,
                X_original=X_sub.copy(),
            )
        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=descriptions,
        )
