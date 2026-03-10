import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulation.shift import AdditiveShiftEffect
from simulation.scale import MultiplicativeScaleEffect
from simulation.split import ConfoundedSplit
from simulation.covariance import CovarianceEffect

# --------------------------
# 1. Create fake data
# --------------------------
n_samples, n_features = 10, 5

X = pd.DataFrame(
    np.ones((n_samples, n_features)), 
    columns=[f"gene_{i}" for i in range(n_features)]
)

X_rand = pd.DataFrame(
    np.random.rand(n_samples, n_features),
    columns=[f"gene_{i}" for i in range(n_features)]
)

rng = np.random.default_rng(seed=15)
X_variance = pd.DataFrame(
    rng.choice([-1, 1], size=(n_samples, n_features)),
    columns = [f"gene_{i}" for i in range(n_features)]
)

metadata = pd.DataFrame({"condition": [0, 1] * (n_samples // 2)})
multi_class_metadata = pd.DataFrame({"condition": [0, 0, 2, 3, 2, 1, 3, 0, 0, 0]})

# --------------------------
# 2. Create confounded batch splits
# --------------------------
splitter = ConfoundedSplit(n_batches=2, column="condition", strength=0.7, random_state=42)
batch_split = splitter.apply(X, metadata)

# print("Batch labels:\n", batch_split.batch_labels)
# print("Empirical Mutual Information:", batch_split.info["mutual_information"])

splitter2 = ConfoundedSplit(n_batches=2, column="condition", strength=0.7, random_state=42)
batch_split2 = splitter2.apply(X, multi_class_metadata)

# print("Multi-Class Batch labels:\n", batch_split2.batch_labels)
# print("Multi-Class Empirical Mutual Information:", batch_split2.info["mutual_information"])

# --------------------------
# 3. Initialize batch effects
# --------------------------
shift_effect = AdditiveShiftEffect(scale=2.0, random_state=42)
# scale_effect = MultiplicativeScaleEffect(scale=5.0, random_state=42)
# cov_effect = CovarianceEffect(scale_std=0.1, shift_std=0.2, cov_sparsity=0.01, cov_scale=0.02, random_state=42)

# --------------------------
# 4. Apply batch effects
# --------------------------
shift_result = shift_effect.apply(X, split=batch_split)
# scale_result = scale_effect.apply(X, split=batch_split)
# cov_result = cov_effect.apply(X_variance, split=batch_split)

# --------------------------
# 5. Test Inversion
# --------------------------

print("\n--- Shift Effect ---")
print("Original Data:\n", shift_result.X_original)
print("Batch Data:\n", shift_result.X_batch)
print("Inverted Data:\n", shift_result.invert())
