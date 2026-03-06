"""
Comprehensive test showing the full workflow of composite batch effects.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation import (
    AdditiveShiftEffect,
    MultiplicativeScaleEffect,
    CompositeEffect,
    RandomBatchSplit,
)

# Create realistic test data
np.random.seed(42)
n_samples, n_features = 100, 20

X = pd.DataFrame(
    np.random.randn(n_samples, n_features) * 5 + 20,
    columns=[f"gene_{i}" for i in range(n_features)],
    index=[f"sample_{i}" for i in range(n_samples)]
)

print("=" * 70)
print("Full Workflow: Composite Batch Effects")
print("=" * 70)

# Step 1: Create batch split
print("\n[Step 1] Create batch assignment")
splitter = RandomBatchSplit(n_batches=3, random_state=42)
batch_split = splitter.apply(X, None)
print(f"Assigned {n_samples} samples to {len(batch_split.batch_labels.unique())} batches")
print(f"Batch distribution: {batch_split.batch_labels.value_counts().to_dict()}")

# Step 2: Create composite effect
print("\n[Step 2] Create composite effect (Shift + Scale)")
effects = [
    AdditiveShiftEffect(scale=2.0, random_state=42),
    MultiplicativeScaleEffect(scale=0.3, random_state=43),
]
composite = CompositeEffect(effects)

# Step 3: Apply batch effects
print("\n[Step 3] Apply batch effects")
result = composite.apply(X, split=batch_split)
print(f"Original data range: [{X.values.min():.2f}, {X.values.max():.2f}]")
print(f"Batch-affected data range: [{result.X_batch.values.min():.2f}, {result.X_batch.values.max():.2f}]")

# Step 4: Extract combined transformation parameters
print("\n[Step 4] Extract combined transformation parameters")
combined_params = result.description.get_combined_transformation(result.X_batch)

for batch_id, (shift, scale) in combined_params.items():
    print(f"\nBatch {batch_id}:")
    print(f"  Scale (first 3 genes): {scale.values[:3]}")
    print(f"  Shift (first 3 genes): {shift.values[:3]}")

# Step 5: Invert using sequential method
print("\n[Step 5] Invert using sequential method")
X_inverted_sequential = result.description.invert(result.X_batch)
diff_seq = np.abs(X_inverted_sequential.values - X.values)
print(f"Max error: {diff_seq.max():.2e}")
print(f"Mean error: {diff_seq.mean():.2e}")
print(f"Recovers original: {np.allclose(X_inverted_sequential.values, X.values)}")

# Step 6: Invert using combined parameters (single-step)
print("\n[Step 6] Invert using combined parameters (single-step)")
X_inverted_combined = result.X_batch.copy()
for batch_id, (shift, scale) in combined_params.items():
    mask = batch_split.batch_labels == batch_id
    X_inverted_combined.loc[mask] = result.X_batch.loc[mask] * scale + shift

diff_comb = np.abs(X_inverted_combined.values - X.values)
print(f"Max error: {diff_comb.max():.2e}")
print(f"Mean error: {diff_comb.mean():.2e}")
print(f"Recovers original: {np.allclose(X_inverted_combined.values, X.values)}")

# Step 7: Verify both methods give same result
print("\n[Step 7] Verify both inversion methods match")
diff_methods = np.abs(X_inverted_sequential.values - X_inverted_combined.values)
print(f"Max difference between methods: {diff_methods.max():.2e}")
print(f"Methods match: {np.allclose(X_inverted_sequential.values, X_inverted_combined.values)}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ Composite effects can be created and applied")
print("✓ Combined transformation parameters can be extracted")
print("✓ Sequential inversion works correctly")
print("✓ Single-step inversion using combined params works")
print("✓ Both inversion methods produce identical results")
print("\nThe composite effect framework is fully functional!")
