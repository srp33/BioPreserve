import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef


# Helper to run a single model

def run_classifier(X_train, y_train, X_test, y_test, random_state=35):
    model = HistGradientBoostingClassifier(max_iter=50, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_test, y_pred)
    try: 
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan
    # Convert probabilities → class predictions for MCC
    y_pred_mcc = (y_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred_mcc)

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "Accuracy": acc,
        "ROC_AUC": auc,
        "MCC": mcc,
        "Sensitivity": sens,
        "Specificity": spec
    }

# Load combined dataset
parser = argparse.ArgumentParser(
    description = "Cross-Validation for each test set."
)

parser.add_argument("--combined_csv", required=True, help="Input combined CSV file")
parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
parser.add_argument("--k_folds", type=int, required=True, help="Number of folds")

args = parser.parse_args()

combined_csv = args.combined_csv
output_dir = args.outdir
k_folds = args.k_folds

print("Loading combined csv...")
df = pd.read_csv(combined_csv)

meta_cols = [c for c in df.columns if c.startswith("meta_")]
feature_cols = [c for c in df.columns if c not in meta_cols]

results = []

for test_source in df['meta_source'].unique():
    print("Processing test source ", test_source)
    test_df = df[df['meta_source'] == test_source]

    n_missing = test_df['meta_er_status'].isna().sum()
    if n_missing > 0:
        print(f"[WARNING] Dropped {n_missing} rows with NaN labels for test_source '{test_source}'")
    test_df = test_df.dropna(subset=["meta_er_status"])

    X_test_full = test_df[feature_cols]
    y_test_full = test_df['meta_er_status']

    # Stratified K-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = []
    for train_idx, val_idx in skf.split(X_test_full, y_test_full):
        X_train_fold = X_test_full.iloc[train_idx]
        y_train_fold = y_test_full.iloc[train_idx]
        X_val_fold = X_test_full.iloc[val_idx]
        y_val_fold = y_test_full.iloc[val_idx]

        metrics = run_classifier(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        fold_metrics.append(metrics)

    avg_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) 
               for k in fold_metrics[0]}
    avg_metrics["test_source"] = test_source
    print("Appending results...")
    results.append(avg_metrics)


baseline_df = pd.DataFrame(results)
full_path = Path(output_dir) / "cv_metrics.csv"
baseline_df.to_csv(full_path, index=False)
print("Saved basline CV metrics: ", baseline_df, "to ", full_path)
