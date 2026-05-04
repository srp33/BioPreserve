import os
import re
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import functools

print = functools.partial(print, flush=True)

# -------------------------
# Filename parsing
# -------------------------
def parse_filename(filename):
    basename = os.path.basename(filename).replace(".csv", "")

    pattern = r"^(?P<adjuster>.+)-(?P<n>\d+)_studies-test_(?P<test>.+)$"
    match = re.match(pattern, basename)

    if not match:
        raise ValueError(f"Cannot parse filename: {basename}")

    return (
        match.group("adjuster"),
        int(match.group("n")),
        match.group("test")
    )

# -------------------------
# Classifier
# -------------------------
def run_classifier(X_train, y_train, X_test, y_test, random_state=42):
    model = HistGradientBoostingClassifier(max_iter=50, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    mcc = matthews_corrcoef(y_test, y_pred)

    return {
        "Accuracy": acc,
        "ROC_AUC": auc,
        "Sensitivity": sens,
        "Specificity": spec,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-run classifier evaluation (no bootstraps)")
    parser.add_argument("--csv", required=True, help="Input adjusted CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory")

    args = parser.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(args.csv)

    adjuster, n_studies, test_source = parse_filename(args.csv)
    print(f"Adjuster: {adjuster}, k(studies): {n_studies}, test_source: {test_source}")

    # -------------------------
    # Split train/test by study
    # -------------------------
    train_df = df[df["meta_source"].str.lower() != test_source.lower()]
    test_df = df[df["meta_source"].str.lower() == test_source.lower()]

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty — check test_source matching")

    # -------------------------
    # Features / labels
    # -------------------------
    feature_cols = [c for c in df.columns if not c.startswith("meta_")]

    X_train = train_df[feature_cols]
    y_train = train_df["meta_er_status"]

    X_test = test_df[feature_cols]
    y_test = test_df["meta_er_status"]

    # Remove NaNs
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    # -------------------------
    # Single training run
    # -------------------------
    metrics = run_classifier(X_train, y_train, X_test, y_test)

    metrics.update({
        "adjuster": adjuster,
        "k": n_studies,
        "test_source": test_source
    })

    # -------------------------
    # Save output
    # -------------------------
    out_dir = os.path.join(args.outdir, adjuster)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(
        out_dir,
        f"{os.path.basename(args.csv).replace('.csv','')}-metrics.csv"
    )

    pd.DataFrame([metrics]).to_csv(out_file, index=False)

    print(f"Saved metrics to: {out_file}")

if __name__ == "__main__":
    main()