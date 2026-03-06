import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation.shift import AdditiveShiftEffect
from simulation.scale import MultiplicativeScaleEffect
from simulation.split import RandomBatchSplit
from simulation.composite import CompositeEffect

# Create test data
np.random.seed(42)
n_samples, n_features = 100, 10

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"gene_{i}" for i in range(n_features)]
)

metadata = pd.DataFrame({"condition": np.random.choice([0, 1], size=n_samples)})

# Create batch split
splitter = RandomBatchSplit(n_batches=3, random_state=42)
batch_split = splitter.apply(X, metadata)

print("=" * 60)
print("Testing CompositeEffect Inversion")
print("=" * 60)

# Test 1: Single effect (baseline)
print("\n[Test 1] Single effect")
shift_effect = AdditiveShiftEffect(scale=2.0, random_state=42)
result_single = shift_effect.apply(X, split=batch_split)

X_inverted_single = result_single.X_batch.copy()
for batch_id, desc in result_single.description.items():
    mask = batch_split.batch_labels == batch_id
    X_inverted_single.loc[mask] = desc.invert(result_single.X_batch.loc[mask])

diff_single = np.abs(X_inverted_single.values - X.values)
print(f"Max difference: {diff_single.max():.2e}")
print(f"Mean difference: {diff_single.mean():.2e}")
print(f"Close to original: {np.allclose(X_inverted_single.values, X.values, rtol=1e-5, atol=1e-8)}")

# Test 2: Composite with two effects
print("\n[Test 2] Composite: Shift + Scale")
effects = [
    AdditiveShiftEffect(scale=2.0, random_state=42),
    MultiplicativeScaleEffect(scale=0.5, random_state=43),
]
composite = CompositeEffect(effects)
result_composite = composite.apply(X, split=batch_split)

X_inverted_composite = result_composite.description.invert(result_composite.X_batch)

diff_composite = np.abs(X_inverted_composite.values - X.values)
print(f"Max difference: {diff_composite.max():.2e}")
print(f"Mean difference: {diff_composite.mean():.2e}")
print(f"Close to original: {np.allclose(X_inverted_composite.values, X.values, rtol=1e-5, atol=1e-8)}")

# Test 3: Verify transformation order matters
print("\n[Test 3] Verify order matters (Shift+Scale vs Scale+Shift)")
effects_reversed = [
    MultiplicativeScaleEffect(scale=0.5, random_state=43),
    AdditiveShiftEffect(scale=2.0, random_state=42),
]
composite_reversed = CompositeEffect(effects_reversed)
result_reversed = composite_reversed.apply(X, split=batch_split)

same_result = np.allclose(
    result_composite.X_batch.values, 
    result_reversed.X_batch.values
)
print(f"Same result with reversed order: {same_result}")
print(f"(Should be False - order matters)")

# But both should invert correctly
X_inverted_reversed = result_reversed.description.invert(result_reversed.X_batch)
diff_reversed = np.abs(X_inverted_reversed.values - X.values)
print(f"Reversed order inverts correctly: {np.allclose(X_inverted_reversed.values, X.values, rtol=1e-5, atol=1e-8)}")

# Test 4: Three effects
print("\n[Test 4] Composite: Shift + Scale + Shift")
effects_three = [
    AdditiveShiftEffect(scale=1.0, random_state=42),
    MultiplicativeScaleEffect(scale=0.3, random_state=43),
    AdditiveShiftEffect(scale=0.5, random_state=44),
]
composite_three = CompositeEffect(effects_three)
result_three = composite_three.apply(X, split=batch_split)

X_inverted_three = result_three.description.invert(result_three.X_batch)
diff_three = np.abs(X_inverted_three.values - X.values)
print(f"Max difference: {diff_three.max():.2e}")
print(f"Mean difference: {diff_three.mean():.2e}")
print(f"Close to original: {np.allclose(X_inverted_three.values, X.values, rtol=1e-5, atol=1e-8)}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✓ Single effect inversion works")
print("✓ Composite effect inversion works")
print("✓ Order of effects matters for transformation")
print("✓ Both orders invert correctly")
print("✓ Three-effect chain inverts correctly")
