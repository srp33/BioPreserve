from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.linalg import sqrtm, inv
from dataclasses import dataclass

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit

@dataclass
class CovarianceDescription(BatchEffectDescription):
    """
    Stores approximate inversion for covariance 
    """
    def __init__(
        self, 
        covariance: dict[str, dict], 
        batch_labels: pd.Series
    ):
        self.covariance = covariance
        self.batch_labels = batch_labels


    def find_matrices(self, X_batch: pd.DataFrame, batch_id: str):
        """
        Find optimal inverse shift and scale matrices.
        """
        # True inverse would be X = (Y - C)(D + M)-1
        params = self.covariance[batch_id]

        X = params["X_original"].values
        Y = X_batch.loc[self.batch_labels == batch_id].values

        # Mean and Center
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)
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
        X_hat = X_batch.copy()
        mse_total = 0.0

        for batch_id in self.covariance.keys():
            mask = self.batch_labels == batch_id
            D_inv, C_inv = self.find_matrices(X_batch, batch_id)
            Y = X_batch.loc[mask].values
            X_orig = self.covariance[batch_id]["X_original"].values

            # Compute shift and scale approximation
            X_approx = Y * D_inv + C_inv.reshape(1, -1)
            X_hat.loc[mask] = X_approx

            mse_total += np.mean((X_approx - X_orig) ** 2)

        return X_hat, mse_total / len(self.covariance)
    
    def parameters(self) -> dict:
        # Gives true inversion ???
        return {
            "type": "covariance_effect", 
            "covariance_matrices": self.covariance,
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

        batch_labels = split.batch_labels
        unique_batches = batch_labels.unique()

        X_batch = X.copy()
        covariance = {}

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

            covariance[batch_id] = {
                "D" : D_vec,
                "M" : M,
                "C" : C,
                "X_original" : X_sub.copy()
            }

        description = CovarianceDescription(
            covariance=covariance,
            batch_labels=batch_labels
        )
        return BatchEffectResult(
            X_original=X,
            X_batch=X_batch,
            metadata=split.metadata,
            description=description,
        )
