#!/usr/bin/env python3
"""
Quick test of multi-label DBA on top 3 genes.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize


def evaluate_single_label(X_train, X_test, y_train, y_test):
    """Evaluate classification performance for a single label."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # ElasticNet classifier
    clf = SGDClassifier(
        loss='log_loss', penalty='elasticnet',
        alpha=0.0001, l1_ratio=0.15,
        max_iter=1000, random_state=42, tol=1e-3
    )
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def multi_label_objective(shifts, X_train, X_test, all_y_train, all_y_test):
    """
    Multi-label objective: average negative MCC across all labels.
    """
    # Adjust test data (shift-only)
    X_test_adjusted = X_test - shifts
    
    # Evaluate on all labels
    total_loss = 0.0
    valid_labels = 0
    
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        
        mcc = evaluate_single_label(X_train, X_test_adjusted, y_train, y_test)
        
        if not np.isnan(mcc):
            total_loss += (-mcc)  # Minimize negative MCC
            valid_labels += 1
    
    # Average over valid labels
    if valid_labels > 0:
        return total_loss / valid_labels
    else:
        return 1e6


def main():
    print("="*60)
    print("Multi-Label DBA Test (Top 3 Genes)")
    print("="*60)
    
    # Load data
    train_genes_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadjusted_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_dba_df = pl.read_csv("outputs/dba_analysis_shift_only/test_genes_dba_direct.csv")
    
    train_meta_df = pl.read_csv("metadata/metadata_train.csv")
    test_meta_df = pl.read_csv("metadata/metadata_test.csv")
    
    # Top 3 genes
    top_genes = ['ERBB2', 'STARD3', 'PGAP3']
    
    X_train = train_genes_df.select(top_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(top_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(top_genes).to_numpy()
    X_test_dba = test_dba_df.select(top_genes).to_numpy()
    
    # Prepare labels (ER, HER2, Chemo)
    metadata_cols = ['meta_er_status', 'meta_her2_status', 'meta_chemotherapy']
    
    all_y_train = {}
    all_y_test = {}
    
    for meta_col in metadata_cols:
        y_train = train_meta_df[meta_col].to_numpy()
        y_test = test_meta_df[meta_col].to_numpy()
        
        # Encode
        le = LabelEncoder()
        all_labels = np.concatenate([
            y_train[~pl.Series(y_train).is_null()],
            y_test[~pl.Series(y_test).is_null()]
        ])
        if len(all_labels) > 0:
            le.fit(all_labels)
            y_train = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_train])
            y_test = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_test])
            all_y_train[meta_col] = y_train
            all_y_test[meta_col] = y_test
    
    print(f"\nOptimizing for {len(all_y_train)} labels: {list(all_y_train.keys())}")
    
    # Baseline scores
    print("\n" + "="*60)
    print("Baseline: Unadjusted")
    print("="*60)
    baseline_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        score = evaluate_single_label(X_train, X_test_unadjusted, y_train, y_test)
        baseline_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    print(f"  Mean: {np.nanmean(list(baseline_scores.values())):.3f}")
    
    # Bayesian shift+scale
    print("\n" + "="*60)
    print("Bayesian (shift+scale)")
    print("="*60)
    bayesian_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        score = evaluate_single_label(X_train, X_test_bayesian, y_train, y_test)
        bayesian_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    print(f"  Mean: {np.nanmean(list(bayesian_scores.values())):.3f}")
    
    # DBA (HER2-only)
    print("\n" + "="*60)
    print("DBA (HER2-only optimization)")
    print("="*60)
    dba_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        score = evaluate_single_label(X_train, X_test_dba, y_train, y_test)
        dba_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    print(f"  Mean: {np.nanmean(list(dba_scores.values())):.3f}")
    
    # Multi-Label DBA
    print("\n" + "="*60)
    print("Multi-Label DBA (optimizing for all labels)")
    print("="*60)
    
    # Initialize with mean-based shifts
    initial_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_train, axis=0)
    print(f"Initial shifts: {initial_shifts}")
    
    # Optimize
    result = minimize(
        multi_label_objective,
        initial_shifts,
        args=(X_train, X_test_unadjusted, all_y_train, all_y_test),
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': True}
    )
    
    optimal_shifts = result.x
    print(f"\nOptimal shifts: {optimal_shifts}")
    print(f"Final loss: {result.fun:.4f}")
    
    # Apply ML-DBA adjustment
    X_test_ml_dba = X_test_unadjusted - optimal_shifts
    
    # Evaluate
    ml_dba_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        score = evaluate_single_label(X_train, X_test_ml_dba, y_train, y_test)
        ml_dba_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    print(f"  Mean: {np.nanmean(list(ml_dba_scores.values())):.3f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Method':<30} {'Mean MCC':>10}")
    print("-"*60)
    print(f"{'Unadjusted':<30} {np.nanmean(list(baseline_scores.values())):>10.3f}")
    print(f"{'Bayesian (shift+scale)':<30} {np.nanmean(list(bayesian_scores.values())):>10.3f}")
    print(f"{'DBA (HER2-only)':<30} {np.nanmean(list(dba_scores.values())):>10.3f}")
    print(f"{'ML-DBA (all labels)':<30} {np.nanmean(list(ml_dba_scores.values())):>10.3f}")
    
    # Shift comparison
    print("\n" + "="*60)
    print("Shift Comparison")
    print("="*60)
    bayesian_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_bayesian, axis=0)
    dba_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_dba, axis=0)
    
    for i, gene in enumerate(top_genes):
        print(f"{gene:10s}: Bayesian={bayesian_shifts[i]:7.3f}, DBA={dba_shifts[i]:7.3f}, ML-DBA={optimal_shifts[i]:7.3f}")


if __name__ == "__main__":
    main()
