from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.linalg import sqrtm, inv
from dataclasses import dataclass

from .base import BaseBatchEffect, BatchEffectResult
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

    def find_matrices(self, X_batch: pd.DataFrame):
        """
        Find optimal inverse shift and scale matrices.
        """

        # True inverse would be X = (Y - C)(D + M)-1
        X = self.X_original.values
        Y = X_batch.values

        # Means
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)

        # Centered
        X_c = X - mean_X
        Y_c = Y - mean_Y

        # Calculate optimal diagonal scale (covariance / variance)
        cov_xy = np.sum(Y_c * X_c, axis=0)
        var_y=np.sum(Y_c * Y_c, axis=0)

        # Handle zero variance to avoid division by zero
        D_inv = np.divide(cov_xy, var_y, out=np.zeros_like(cov_xy), where=var_y!=0)

        # Calculate optimal shift
        C_inv = mean_X - mean_Y * D_inv

        return D_inv, C_inv

    def invert(self, X_batch: pd.DataFrame):
        D_inv, C_inv = self.find_matrices(X_batch)
        X = self.X_original.values 
        Y = X_batch.values 
        X_hat = Y @ np.diag(D_inv) + C_inv.reshape(1, -1)
        X_delta = X_hat - X
        mse = np.mean(X_delta**2)

        # return X_delta.round(4) #, 
        return X_hat
    
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
            batch_labels=batch_labels,
            # FILL IN HERE
            #batch_shift=shift,
            #batch_scale=scale,
        )
    
    def extract_shift_scale(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Returns diagonal approximation of shift and scale per batch.
        Only suitable for combining with other diagonal effects.
        """
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for batch_id, desc in self.last_results.items():  # or last_description per batch
            # X_original = desc.X_original
            # D, C = desc.D, desc.C
            # Approximate inversion: shift = -C, scale = 1/D
            n_features = len(desc.D)
            shift = -desc.C
            scale = 1.0 / desc.D
            result[batch_id] = (shift, scale)

        return result
