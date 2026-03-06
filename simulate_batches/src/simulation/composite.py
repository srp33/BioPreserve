from __future__ import annotations
import pandas as pd

from .base import BaseBatchEffect, BatchEffectResult, BatchEffectDescription
from .split import BatchSplit


class CompositeEffectDescription(BatchEffectDescription):
    """
    Stores descriptions from multiple effects in application order.
    Inverts by applying inverse transformations in reverse order.
    """
    
    def __init__(
        self, 
        descriptions: list[dict[int, BatchEffectDescription]],
        batch_labels: pd.Series
    ):
        self.descriptions = descriptions
        self.batch_labels = batch_labels
    
    def invert(self, X_batch: pd.DataFrame) -> pd.DataFrame:
        X_current = X_batch.copy()
        
        # Reverse order: last effect applied gets inverted first
        for effect_descriptions in reversed(self.descriptions):
            X_inverted = X_current.copy()
            
            # Invert per batch
            for batch_id, desc in effect_descriptions.items():
                mask = self.batch_labels == batch_id
                X_inverted.loc[mask] = desc.invert(X_current.loc[mask])
            
            X_current = X_inverted
        
        return X_current
    
    def parameters(self) -> dict:
        """Return nested parameters from all effects."""
        return {
            f"effect_{i}": {
                batch_id: desc.parameters() 
                for batch_id, desc in effect_desc.items()
            }
            for i, effect_desc in enumerate(self.descriptions)
        }
    
    def extract_shift_scale(self, X_batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract combined shift and scale for a single batch.
        
        Note: This method assumes X_batch contains data from only one batch.
        For multi-batch data, use get_combined_transformation() instead.
        
        Returns
        -------
        shift : np.ndarray
            Combined shift vector
        scale : np.ndarray
            Combined scale vector
        """
        # Get the batch ID for this data
        # Assume all rows belong to the same batch
        batch_id = self.batch_labels.iloc[0]
        
        # Use get_combined_transformation and extract for this batch
        combined = self.get_combined_transformation(X_batch)
        shift_series, scale_series = combined[batch_id]
        
        return shift_series.values, scale_series.values
    
    def get_combined_transformation(
        self, 
        X_batch: pd.DataFrame
    ) -> dict[int, tuple[pd.Series, pd.Series]]:
        """
        Extract combined shift and scale parameters for each batch.
        
        Composes all transformations in the chain into a single
        shift + scale representation per batch.
        
        Parameters
        ----------
        X_batch : pd.DataFrame
            The batch-affected data (used for covariance effects that
            need data to compute optimal shift/scale approximation).
        
        Returns
        -------
        dict
            Mapping batch_id -> (shift_series, scale_series)
            where shift and scale are pd.Series with feature names as index.
            
            The combined inverse transformation is:
            X = Y * scale + shift
            
            This is element-wise: X[i,j] = Y[i,j] * scale[j] + shift[j]
        
        Notes
        -----
        - For pure shift effects: scale = 1.0, shift = -shift_amount
        - For pure scale effects: shift = 0.0, scale = 1/scale_amount
        - For covariance effects: uses find_matrices() approximation
        - Transformations are composed in reverse order (for inversion)
        
        Examples
        --------
        If forward transformations are:
            Y1 = X * s1 + c1
            Y2 = Y1 * s2 + c2
        
        Then combined inverse is:
            X = Y2 * (1/(s1*s2)) + (-c2/(s1*s2) - c1/s1)
            X = Y2 * scale + shift
        where:
            scale = 1/(s1*s2)
            shift = -c2/(s1*s2) - c1/s1
        """
        # Get unique batch IDs from the first effect
        batch_ids = list(self.descriptions[0].keys())
        
        combined = {}
        
        for batch_id in batch_ids:
            mask = self.batch_labels == batch_id
            
            # Initialize with identity transformation: X = Y * 1.0 + 0.0
            net_scale = pd.Series(1.0, index=X_batch.columns)
            net_shift = pd.Series(0.0, index=X_batch.columns)
            
            # Compose transformations in reverse order (for inversion)
            # If forward is: X -> f1 -> f2 -> f3 -> Y
            # Then inverse is: Y -> inv_f3 -> inv_f2 -> inv_f1 -> X
            for effect_descriptions in reversed(self.descriptions):
                desc = effect_descriptions[batch_id]
                
                # Extract shift and scale from this effect's inverse
                # Each returns parameters for: X_prev = Y_curr * scale + shift
                shift, scale = self._extract_shift_scale(
                    desc, 
                    X_batch.loc[mask]
                )
                
                # Compose with existing transformation
                # Current layer: X_prev = Y_curr * scale + shift
                # Previous layers: X_final = X_prev * net_scale + net_shift
                # Combined: X_final = (Y_curr * scale + shift) * net_scale + net_shift
                #                   = Y_curr * (scale * net_scale) + (shift * net_scale + net_shift)
                
                net_shift = shift * net_scale + net_shift
                net_scale = scale * net_scale
            
            combined[batch_id] = (net_shift, net_scale)
        
        return combined
    
    def _extract_shift_scale(
        self, 
        desc: BatchEffectDescription,
        X_batch_subset: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """
        Extract shift and scale from a single effect's description.
        
        Returns parameters for the inverse transformation in the form:
        X = Y * scale + shift
        
        Returns
        -------
        shift : pd.Series
            Shift vector for inverse transformation
        scale : pd.Series  
            Scale vector for inverse transformation
        """
        # Use the abstract method implemented by each description class
        shift_array, scale_array = desc.extract_shift_scale(X_batch_subset)
        
        # Convert to Series with proper index
        shift = pd.Series(shift_array, index=X_batch_subset.columns)
        scale = pd.Series(scale_array, index=X_batch_subset.columns)
        
        return shift, scale


class CompositeEffect(BaseBatchEffect):
    
    def __init__(self, effects: list[BaseBatchEffect], random_state: int | None = None):
        super().__init__(random_state)
        self.effects = effects
    
    def apply(self, X: pd.DataFrame, split: BatchSplit) -> BatchEffectResult:
        X_current = X
        descriptions = []
        
        # Apply each effect in order
        for effect in self.effects:
            result = effect.apply(X_current, split)
            X_current = result.X_batch
            descriptions.append(result.description)
        
        # Create composite description with all transformations
        composite_desc = CompositeEffectDescription(
            descriptions=descriptions,
            batch_labels=split.batch_labels
        )
        
        return BatchEffectResult(
            X_original=X,
            X_batch=X_current,
            metadata=split.metadata,
            description=composite_desc
        )
