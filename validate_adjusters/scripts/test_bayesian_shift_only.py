#!/usr/bin/env python3
"""
Quick test to compare Bayesian shift-only vs DBA shift-only.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


def evaluate_adjustment(X_train, X_test, y_train, y_test):
    """Evaluate classification performance."""
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
    
    # ElasticNet
    clf = SGDClassifier(
        loss='log_loss', penalty='elasticnet',
        alpha=0.0001, l1_ratio=0.15,
        max_iter=1000, random_state=42, tol=1e-3
    )
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def main():
    print("="*60)
    print("Comparing Bayesian Shift-Only vs DBA Shift-Only")
    print("="*60)
    
    # Load data
    train_genes_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadjusted_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_dba_df = pl.read_csv("outputs/dba_analysis_shift_only/test_genes_dba_direct.csv")
    
    train_meta_df = pl.read_csv("metadata/metadata_train.csv")
    test_meta_df = pl.read_csv("metadata/metadata_test.csv")
    
    # Get labels
    train_y = train_meta_df['meta_her2_status'].to_numpy()
    test_y = test_meta_df['meta_her2_status'].to_numpy()
    
    # Encode
    le = LabelEncoder()
    all_labels = np.concatenate([
        train_y[~pl.Series(train_y).is_null()],
        test_y[~pl.Series(test_y).is_null()]
    ])
    le.fit(all_labels)
    
    train_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
    test_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
    
    # Top 3 genes
    top_genes = ['ERBB2', 'STARD3', 'PGAP3']
    
    X_train = train_genes_df.select(top_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(top_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(top_genes).to_numpy()
    X_test_dba = test_dba_df.select(top_genes).to_numpy()
    
    # Compute Bayesian shift-only (manually)
    # Bayesian parameters from the existing run
    bayesian_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_bayesian, axis=0)
    X_test_bayesian_shift_only = X_test_unadjusted - bayesian_shifts
    
    print("\n1. Unadjusted")
    score = evaluate_adjustment(X_train, X_test_unadjusted, train_y_encoded, test_y_encoded)
    print(f"   Score: {score:.3f}")
    
    print("\n2. Bayesian (shift + scale)")
    score = evaluate_adjustment(X_train, X_test_bayesian, train_y_encoded, test_y_encoded)
    print(f"   Score: {score:.3f}")
    
    print("\n3. Bayesian (shift-only, manual)")
    score = evaluate_adjustment(X_train, X_test_bayesian_shift_only, train_y_encoded, test_y_encoded)
    print(f"   Score: {score:.3f}")
    print(f"   Shifts: {bayesian_shifts}")
    
    print("\n4. DBA (shift-only, optimized)")
    score = evaluate_adjustment(X_train, X_test_dba, train_y_encoded, test_y_encoded)
    print(f"   Score: {score:.3f}")
    
    # Get DBA shifts
    dba_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_dba, axis=0)
    print(f"   Shifts: {dba_shifts}")
    
    print("\n" + "="*60)
    print("Shift Comparison:")
    print("="*60)
    for i, gene in enumerate(top_genes):
        print(f"{gene:10s}: Bayesian={bayesian_shifts[i]:7.3f}, DBA={dba_shifts[i]:7.3f}, Diff={abs(bayesian_shifts[i]-dba_shifts[i]):7.3f}")


if __name__ == "__main__":
    main()
