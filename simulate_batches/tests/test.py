import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation.shift import AdditiveShiftEffect
from simulation.scale import MultiplicativeScaleEffect
from simulation.split import ConfoundedSplit

# --------------------------
# 1. Create fake data
# --------------------------
n_samples = 10
n_features = 5
X = pd.DataFrame(
    np.ones((n_samples, n_features)), 
    columns=[f"gene_{i}" for i in range(n_features)]
)

# Create metadata with a binary condition
metadata = pd.DataFrame({
    "condition": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

multi_class_metadata = pd.DataFrame({
    "condition": [0, 0, 2, 3, 2, 1, 3, 0, 0, 0]
})

# --------------------------
# 2. Create a confounded batch split
# --------------------------
# strength=0.7 → mostly confounded; alternatively you could use entropy=0.3
splitter = ConfoundedSplit(n_batches=2, column="condition", strength=0.7, random_state=42)
batch_split = splitter.apply(X, metadata)

print("Batch labels:\n", batch_split.batch_labels)
print("Empirical Mutual Information:", batch_split.info["mutual_information"])

splitter2 = ConfoundedSplit(n_batches=2, column="condition", strength=0.7, random_state=42)
batch_split2 = splitter2.apply(X, multi_class_metadata)

print("Multi-Class Batch labels:\n", batch_split2.batch_labels)
print("Multi-Class Empirical Mutual Information:", batch_split2.info["mutual_information"])

# --------------------------
# 3. Initialize the batch effects
# --------------------------
shift_effect = AdditiveShiftEffect(scale=2.0, random_state=42)
scale_effect = MultiplicativeScaleEffect(scale=5.0, random_state=42)

# --------------------------
# 4. Apply batch effects using the same batch split
# --------------------------
shift_result = shift_effect.apply(X, split=batch_split)
shift_result2 = shift_effect.apply(X, split=batch_split)  # random shift again, same batches

scale_result = scale_effect.apply(X, split=batch_split)

# --------------------------
# 5. Inspect results
# --------------------------
print("\n--- Shift Effect ---")
print("Original Data:\n", shift_result.X_original)
print("Batch Data:\n", shift_result.X_batch)
print("Metadata:\n", shift_result.metadata)
print("Description Parameters:\n", shift_result.description.parameters())

print("\n--- Scale Effect ---")
print("Batch Data:\n", scale_result.X_batch)
print("Metadata:\n", scale_result.metadata)
print("Description Parameters:\n", scale_result.description.parameters())

# --------------------------
# 6. Test inversion
# --------------------------
X_shift_inverted = shift_result.description.invert(shift_result.X_batch)
print("\nInverted Shift Data:\n", X_shift_inverted)

scale_X_inverted = scale_result.description.invert(scale_result.X_batch)
print("\nInverted Scale Data:\n", scale_X_inverted)

# Check that inversion roughly recovers the original
assert np.allclose(shift_result.X_original.values, X_shift_inverted.values), "Shift inversion failed!"
assert np.allclose(scale_result.X_original.values, scale_X_inverted.values), "Scale inversion failed!"
print("\nInversion successful: original data recovered.")