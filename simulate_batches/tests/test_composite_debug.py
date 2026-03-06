import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation.scale import MultiplicativeScaleEffect
from simulation.split import RandomBatchSplit
from simulation.composite import CompositeEffect

# Simple test data
np.random.seed(42)
X = pd.DataFrame(
    np.ones((10, 5)) * 5.0,  # All values = 5.0
    columns=[f"gene_{i}" for i in range(5)]
)

# Create batch split
splitter = RandomBatchSplit(n_batches=2, random_state=42)
batch_split = splitter.apply(X, None)

print("Original data (all 5.0):")
print(X.head())

# Test with just scale effect (which should work correctly)
print("\n" + "="*60)
print("Test: Single Scale Effect")
print("="*60)

scale_effect = MultiplicativeScaleEffect(scale=0.1, random_state=42)
result = scale_effect.apply(X, split=batch_split)

print("\nAfter scaling:")
print(result.X_batch.head())

# Manual inversion
X_manual = result.X_batch.copy()
for batch_id, desc in result.description.items():
    mask = batch_split.batch_labels == batch_id
    X_manual.loc[mask] = desc.invert(result.X_batch.loc[mask])

print("\nManually inverted:")
print(X_manual.head())
print(f"Close to original: {np.allclose(X_manual.values, X.values)}")

# Test composite with two scale effects
print("\n" + "="*60)
print("Test: Composite with Two Scale Effects")
print("="*60)

effects = [
    MultiplicativeScaleEffect(scale=0.1, random_state=42),
    MultiplicativeScaleEffect(scale=0.1, random_state=43),
]
composite = CompositeEffect(effects)
result_comp = composite.apply(X, split=batch_split)

print("\nAfter composite scaling:")
print(result_comp.X_batch.head())

X_inverted = result_comp.description.invert(result_comp.X_batch)
print("\nComposite inverted:")
print(X_inverted.head())
print(f"Close to original: {np.allclose(X_inverted.values, X.values)}")

# Show the actual differences
diff = np.abs(X_inverted.values - X.values)
print(f"\nMax difference: {diff.max():.2e}")
print(f"Mean difference: {diff.mean():.2e}")
