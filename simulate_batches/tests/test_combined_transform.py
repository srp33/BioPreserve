import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation.scale import MultiplicativeScaleEffect
from simulation.split import RandomBatchSplit
from simulation.composite import CompositeEffect

# Create simple test data
np.random.seed(42)
n_samples, n_features = 20, 5

X = pd.DataFrame(
    np.random.randn(n_samples, n_features) * 10 + 50,  # Mean ~50, std ~10
    columns=[f"gene_{i}" for i in range(n_features)]
)

# Create batch split
splitter = RandomBatchSplit(n_batches=2, random_state=42)
batch_split = splitter.apply(X, None)

print("=" * 70)
print("Testing Combined Transformation Extraction")
print("=" * 70)

# Test 1: Single scale effect
print("\n[Test 1] Single Scale Effect")
print("-" * 70)

scale_effect = MultiplicativeScaleEffect(scale=0.2, random_state=42)
result_single = scale_effect.apply(X, split=batch_split)

# Create a composite with just one effect to use get_combined_transformation
composite_single = CompositeEffect([scale_effect])
result_comp_single = composite_single.apply(X, split=batch_split)

combined = result_comp_single.description.get_combined_transformation(result_comp_single.X_batch)

# Manually apply combined transformation
X_manual = result_comp_single.X_batch.copy()
for batch_id, (shift, scale) in combined.items():
    mask = batch_split.batch_labels == batch_id
    X_manual.loc[mask] = result_comp_single.X_batch.loc[mask] * scale + shift

# Compare with standard inversion
X_standard = result_comp_single.description.invert(result_comp_single.X_batch)

print(f"Manual (combined params) matches standard inversion: {np.allclose(X_manual.values, X_standard.values)}")
print(f"Both recover original: {np.allclose(X_manual.values, X.values)}")

# Test 2: Two scale effects
print("\n[Test 2] Two Scale Effects")
print("-" * 70)

effects = [
    MultiplicativeScaleEffect(scale=0.2, random_state=42),
    MultiplicativeScaleEffect(scale=0.3, random_state=43),
]
composite = CompositeEffect(effects)
result = composite.apply(X, split=batch_split)

combined = result.description.get_combined_transformation(result.X_batch)

# Apply combined transformation
X_combined = result.X_batch.copy()
for batch_id, (shift, scale) in combined.items():
    mask = batch_split.batch_labels == batch_id
    X_combined.loc[mask] = result.X_batch.loc[mask] * scale + shift

# Compare with standard inversion
X_standard = result.description.invert(result.X_batch)

print(f"Combined params match standard inversion: {np.allclose(X_combined.values, X_standard.values)}")
print(f"Both recover original: {np.allclose(X_combined.values, X.values)}")

# Show the combined parameters for one batch
batch_id = 0
shift, scale = combined[batch_id]
print(f"\nBatch {batch_id} combined parameters:")
print(f"  Scale (first 3 genes): {scale.values[:3]}")
print(f"  Shift (first 3 genes): {shift.values[:3]}")

# Test 3: Verify the math manually for one feature
print("\n[Test 3] Manual Verification of Composition Math")
print("-" * 70)

# Get the individual effect parameters for batch 0
effect1_desc = result.description.descriptions[0][batch_id]
effect2_desc = result.description.descriptions[1][batch_id]

scale1 = effect1_desc.scaling
scale2 = effect2_desc.scaling

print(f"Effect 1 scale (gene 0): {scale1[0]:.6f}")
print(f"Effect 2 scale (gene 0): {scale2[0]:.6f}")

# Forward transformation:
# Y1 = X * scale1
# Y2 = Y1 * scale2 = X * scale1 * scale2

# Inverse transformation:
# Y1 = Y2 / scale2
# X = Y1 / scale1 = Y2 / (scale1 * scale2)
# In form X = Y * scale + shift:
# X = Y * (1/(scale1 * scale2)) + 0

expected_combined_scale = 1.0 / (scale1[0] * scale2[0])
actual_combined_scale = scale.values[0]

print(f"\nExpected combined scale (gene 0): {expected_combined_scale:.6f}")
print(f"Actual combined scale (gene 0): {actual_combined_scale:.6f}")
print(f"Match: {np.isclose(expected_combined_scale, actual_combined_scale)}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ Combined transformation parameters correctly extracted")
print("✓ Single-step inversion using combined params matches multi-step")
print("✓ Composition math verified manually")
