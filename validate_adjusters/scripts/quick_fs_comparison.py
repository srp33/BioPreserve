#!/usr/bin/env python3
"""
Quick comparison with feature selection using fixed C values.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler


def evaluate_with_lasso(X_train, X_test, y_train, y_test, C=0.1):
    """Evaluate with L1 regularization."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filt = X_train[mask_train]
    y_train_filt = y_train[mask_train]
    X_test_filt = X_test[mask_test]
    y_test_filt = y_test[mask_test]
    
    if len(np.unique(y_train_filt)) < 2 or len(np.unique(y_test_filt)) < 2:
        return np.nan, 0
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_filt)
    X_test_sc = scaler.transform(X_test_filt)
    
    clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                            random_state=42, max_iter=1000)
    clf.fit(X_train_sc, y_train_filt)
    y_pred = clf.predict(X_test_sc)
    
    n_feat = np.sum(np.abs(clf.coef_[0]) > 1e-5)
    return matthews_corrcoef(y_test_filt, y_pred), n_feat


def main():
    print("="*80)
    print("Quick Feature Selection Comparison")
    print("="*80)
    
    # Load data
    train_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadj_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bay_ss_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bay_so_df = pl.read_csv("adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv")
    
    train_meta = pl.read_csv("metadata/metadata_train.csv")
    test_meta = pl.read_csv("metadata/metadata_test.csv")
    
    common_genes = [g for g in train_df.columns 
                   if g in test_unadj_df.columns 
                   and g in test_bay_ss_df.columns
                   and g in test_bay_so_df.columns]
    
    print(f"Genes: {len(common_genes)}")
    
    X_train = train_df.select(common_genes).to_numpy()
    X_test_unadj = test_unadj_df.select(common_genes).to_numpy()
    X_test_bay_ss = test_bay_ss_df.select(common_genes).to_numpy()
    X_test_bay_so = test_bay_so_df.select(common_genes).to_numpy()
    
    # Focus on key binary labels
    labels_to_test = ['meta_er_status', 'meta_her2_status', 'meta_chemotherapy']
    
    all_y_train = {}
    all_y_test = {}
    
    for label in labels_to_test:
        y_tr = train_meta[label].to_numpy()
        y_te = test_meta[label].to_numpy()
        
        le = LabelEncoder()
        all_vals = np.concatenate([
            y_tr[~pl.Series(y_tr).is_null()],
            y_te[~pl.Series(y_te).is_null()]
        ])
        if len(all_vals) > 0 and len(np.unique(all_vals)) >= 2:
            le.fit(all_vals)
            y_tr = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_tr])
            y_te = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_te])
            all_y_train[label] = y_tr
            all_y_test[label] = y_te
    
    # Test different C values
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for C in C_values:
        print(f"\n{'='*80}")
        print(f"C = {C} (stronger regularization = smaller C)")
        print(f"{'='*80}")
        
        for method_name, X_test in [
            ('Unadjusted', X_test_unadj),
            ('Bayesian (shift+scale)', X_test_bay_ss),
            ('Bayesian (shift-only)', X_test_bay_so)
        ]:
            print(f"\n{method_name}:")
            scores = []
            for label in all_y_train.keys():
                y_tr = all_y_train[label]
                y_te = all_y_test[label]
                score, n_feat = evaluate_with_lasso(X_train, X_test, y_tr, y_te, C=C)
                scores.append(score)
                print(f"  {label:30s}: {score:.3f} ({n_feat:2d} features)")
            print(f"  {'Mean':30s}: {np.nanmean(scores):.3f}")


if __name__ == "__main__":
    main()
