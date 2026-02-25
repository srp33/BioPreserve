import pandas as pd
import numpy as np

from simulation.shift import AdditiveShiftEffect

# --------------------------
# 1. Create fake data
# --------------------------
n_samples = 10
n_features = 5
X = pd.DataFrame(
    np.ones((n_samples, n_features)), 
    columns=[f"gene_{i}" for i in range(n_features)]
)

# --------------------------
# 2. Initialize the effect
# --------------------------
shift_effect = AdditiveShiftEffect(n_batches=3, scale=2.0, random_state=42)

# --------------------------
# 3. Apply the batch effect
# --------------------------
result = shift_effect.apply(X)

print("Original Data:\n", result.X_original)
print("Batch Data:\n", result.X_batch)
print("Metadata:\n", result.metadata)
print("Description Parameters:\n", result.description.parameters())

# --------------------------
# 4. Test inversion
# --------------------------
X_inverted = result.description.invert(result.X_batch)
print("Inverted Data:\n", X_inverted)

# Check that inversion roughly recovers the original
assert np.allclose(result.X_original.values, X_inverted.values), "Inversion failed!"
print("Inversion successful: original data recovered.")