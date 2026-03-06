"""Test workflow with only scale effects (no buggy shift)."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation import (
    MultiplicativeScaleEffect,
    CompositeEffect,
    RandomBatchSplit,
)

np.random.seed(42)
X = pd.DataFrame(
    np.random.randn(50, 10) * 5 + 20,
    columns=[f"gene_{i}" for i in range(10)]
)

splitter = RandomBatchSplit(n_batches=2, random_state=42)
batch_split = splitter.apply(X, None)

# Composite with only scale effects
effects = [
    MultiplicativeScaleEffect(scale=0.2, random_state=42),
    MultiplicativeScaleEffect(scale=0.3, random_state=43),
]
composite = CompositeEffect(effects)
result = composite.apply(X, split=batch_split)

# Get combined params
combined_params = result.description.get_combined_transformation(result.X_batch)

# Sequential inversion
X_seq = result.description.invert(result.X_batch)

# Combined inversion
X_comb = result.X_batch.copy()
for batch_id, (shift, scale) in combined_params.items():
    mask = batch_split.batch_labels == batch_id
    X_comb.loc[mask] = result.X_batch.loc[mask] * scale + shift

print("Scale-only composite test:")
print(f"Sequential recovers original: {np.allclose(X_seq.values, X.values)}")
print(f"Combined recovers original: {np.allclose(X_comb.values, X.values)}")
print(f"Methods match: {np.allclose(X_seq.values, X_comb.values)}")
print(f"Max error (sequential): {np.abs(X_seq.values - X.values).max():.2e}")
print(f"Max error (combined): {np.abs(X_comb.values - X.values).max():.2e}")
