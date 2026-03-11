#!/usr/bin/env python3

import os
import re
import time
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
import functools

print = functools.partial(print, flush=True)

# -------------------------
# Filename parsing
# -------------------------
def parse_filename(filename):
    basename = os.path.basename(filename).replace(".csv", "")

    pattern = r"^(?P<adjuster>.+)_(?P<n>\d+)studies_test_(?P<test>.+)$"
    match = re.match(pattern, basename)

    if not match:
        raise ValueError(f"Cannot parse filename: {basename}")

    return (
        match.group("adjuster"),
        int(match.group("n")),
        match.group("test")
    )

# -------------------------
# Classifier function
# -------------------------
def run_classifier(X_train, y_train, X_test, y_test, metric, random_state=42, n_jobs=1):
    """Train classifier and compute metrics."""
    start_time = time.time()
    
    # Train model
    model = HistGradientBoostingClassifier(
        max_iter=50, 
        random_state=random_state
    )

    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f" Training completed in {train_time:.2f} seconds")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan
    
    perm_start_time = time.time()
    if metric.lower() == "mcc":
        # Get permutation importance using MCC
        scorer = make_scorer(matthews_corrcoef)
    elif metric.lower() == "roc_auc":
        # Get permutation importance using ROC_AUC
        scorer = make_scorer(roc_auc_score)
    else:
        raise ValueError(f"Unknown metric: {metric}, choose 'roc_auc' or 'mcc'.")
    
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=3,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scorer
    )

    perm_time = time.time() - perm_start_time
    print(f"    Permtuation importance completed in {perm_time:.2f} seconds")

    results =  {
        "ROC_AUC": auc,
        "MCC": mcc,
        "perm_importances_mean": perm_importance.importances_mean.tolist(),
        "perm_importances_std": perm_importance.importances_std.tolist(),
        'feature_names': X_train.columns.tolist(),
        'train_time': train_time,
        'perm_time': perm_time
    }

    return results

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run bootstrapped classifier on adjusted dataset")
    parser.add_argument("--csv", required=True, help="Input adjusted CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--metric", default="roc_auc", help="Which metric to use in permutation importance: mcc or roc_auc")
    parser.add_argument("--n_jobs", type=int, default=1)
    
    args = parser.parse_args()
    
    # -------------------------
    # Load data
    # -------------------------
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    
    df = pd.read_csv(args.csv)
    adjuster, n_studies, test_source = parse_filename(args.csv)
    print(f"Adjuster: {adjuster}, n_studies: {n_studies}, test_source: {test_source}")
    
    # -------------------------
    # Split train/test
    # -------------------------
    train_df = df[df['meta_source'].str.lower() != test_source.lower()]
    test_df = df[df['meta_source'].str.lower() == test_source.lower()]
    
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty — check test_source matching")
    
    feature_cols = [c for c in df.columns if not c.startswith("meta_")]
    X_train, y_train = train_df[feature_cols], train_df['meta_er_status']
    X_test, y_test = test_df[feature_cols], test_df['meta_er_status']

    # Remove NaNs
    mask_train = y_train.notna()
    mask_test = y_test.notna()
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    # -------------------------
    # Run classifier and calculate feature importance
    # -------------------------
    results = run_classifier(X_train, y_train, X_test, y_test, metric = args.metric, n_jobs=args.n_jobs)

    results["adjuster"] = adjuster
    results["n_studies"] = n_studies
    results["test_source"] = test_source

    # -------------------------
    # Convert lists to JSON strings for CSV
    # -------------------------
    for key in ["feature_names", "perm_importances_mean", "perm_importances_std"]:
        results[key] = json.dumps(results[key])
    # -------------------------
    # Save results
    # -------------------------
    out_dir = os.path.join(args.outdir, adjuster)
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, f"{adjuster}_{n_studies}_{test_source}_feature_importance.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\nSaved feature importance: {out_file}")

if __name__ == "__main__":
    main()

# When you need to read it: 
# import pandas as pd
# import json

# df = pd.read_csv("feature_importance_single_row.csv")
# df["feature_names"] = df["feature_names"].apply(json.loads)
# df["perm_importances_mean"] = df["perm_importances_mean"].apply(json.loads)
# df["perm_importances_std"] = df["perm_importances_std"].apply(json.loads)
