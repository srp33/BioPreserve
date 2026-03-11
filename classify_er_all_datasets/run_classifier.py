import os
import sys
import re
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.utils import resample
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
def run_classifier(X_train, y_train, X_test, y_test, random_state=42):
    """Train classifier and compute metrics."""
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
# Bootstrap wrapper
# -------------------------
def run_bootstraps(X_train, y_train, X_test, y_test, adjuster, chunk=0, chunk_size=10):
    """
    Run bootstrapped classifier evaluations for a chunk.
    Returns a list of metric dicts.
    """
    results = []
    chunk_seed_base = 10_000 + chunk * chunk_size
    rng = np.random.default_rng(chunk_seed_base)

    for local_iter in range(chunk_size):
        global_iter = chunk * chunk_size + local_iter
        bootstrap_seed = int(rng.integers(1_000_000_000))
        
        print(f"\n=== Bootstrap global iter {global_iter} (chunk {chunk}, local {local_iter}) ===")
        
        X_boot, y_boot = resample(
            X_train,
            y_train,
            replace=True,
            n_samples=len(X_train),
            stratify=y_train,
            random_state=bootstrap_seed
        )

        metrics = run_classifier(X_boot, y_boot, X_test, y_test, random_state=bootstrap_seed)
        metrics.update({
            "bootstrap_global": global_iter,
            "bootstrap_chunk": chunk,
            "bootstrap_local": local_iter,
            "bootstrap_seed": bootstrap_seed,
            "adjuster": adjuster
        })
        results.append(metrics)
    
    return results

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run bootstrapped classifier on adjusted dataset")
    parser.add_argument("--csv", required=True, help="Input adjusted CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--chunk", type=int, default=0, help="Bootstrap chunk index (0-based)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of bootstraps per chunk")
    
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
    # Run bootstraps
    # -------------------------
    results = run_bootstraps(
        X_train, y_train, X_test, y_test,
        adjuster=adjuster,
        chunk=args.chunk,
        chunk_size=args.chunk_size
    )

    # -------------------------
    # Save results
    # -------------------------
    out_dir = os.path.join(args.outdir, adjuster)
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, f"{os.path.basename(args.csv).replace('.csv','')}_chunk{args.chunk}_metrics.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\nSaved classifier metrics (bootstrap chunk): {out_file}")

if __name__ == "__main__":
    main()