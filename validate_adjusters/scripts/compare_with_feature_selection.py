#!/usr/bin/env python3
"""
Compare Bayesian vs ML-DBA with aggressive feature selection.
Uses Lasso (L1) with cross-validated alpha selection for automatic feature selection.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_classification_with_feature_selection(X_train, X_test, y_train, y_test):
    """
    Evaluate classification with automatic feature selection via Lasso.
    Uses simple LogisticRegression with strong L1 penalty.
    """
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan, 0
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Use LogisticRegression with strong L1 penalty for aggressive feature selection
    from sklearn.linear_model import LogisticRegression
    
    clf = LogisticRegression(
        penalty='l1',
        C=0.1,  # Strong regularization
        solver='saga',
        random_state=42,
        max_iter=2000,
        tol=1e-3
    )
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    # Count selected features
    if len(clf.coef_.shape) == 1:
        n_selected = np.sum(np.abs(clf.coef_) > 1e-5)
    else:
        # Multiclass: count features used in any class
        n_selected = np.sum(np.any(np.abs(clf.coef_) > 1e-5, axis=0))
    
    return matthews_corrcoef(y_test_filtered, y_pred), n_selected


def evaluate_regression_with_feature_selection(X_train, X_test, y_train, y_test):
    """Evaluate regression with automatic feature selection via LassoCV."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(y_train_filtered) < 10 or len(y_test_filtered) < 10:
        return np.nan, 0
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_filtered)
    X_test_scaled = scaler_X.transform(X_test_filtered)
    
    # Use LassoCV for automatic alpha selection
    reg = LassoCV(cv=5, random_state=42, max_iter=2000)
    reg.fit(X_train_scaled, y_train_filtered)
    y_pred = reg.predict(X_test_scaled)
    
    # Count selected features
    n_selected = np.sum(np.abs(reg.coef_) > 1e-5)
    
    return r2_score(y_test_filtered, y_pred), n_selected


def multi_label_objective_with_fs(shifts, X_train, X_test, all_y_train, all_y_test, 
                                   continuous_labels, label_weights):
    """
    Multi-label objective with feature selection.
    """
    # Adjust test data (shift-only)
    X_test_adjusted = X_test - shifts
    
    total_loss = 0.0
    total_weight = 0.0
    
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        weight = label_weights.get(label_name, 1.0)
        
        if label_name in continuous_labels:
            score, _ = evaluate_regression_with_feature_selection(
                X_train, X_test_adjusted, y_train, y_test)
        else:
            score, _ = evaluate_classification_with_feature_selection(
                X_train, X_test_adjusted, y_train, y_test)
        
        if not np.isnan(score):
            total_loss += weight * (-score)
            total_weight += weight
    
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return 1e6


def main():
    print("="*80)
    print("Bayesian vs ML-DBA: With Aggressive Feature Selection")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadjusted_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_shift_scale_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_shift_only_df = pl.read_csv("adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv")
    
    train_meta_df = pl.read_csv("metadata/metadata_train.csv")
    test_meta_df = pl.read_csv("metadata/metadata_test.csv")
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_shift_scale_df.columns
                   and g in test_bayesian_shift_only_df.columns]
    
    print(f"Common genes: {len(common_genes)}")
    
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(common_genes).to_numpy()
    X_test_bayesian_shift_scale = test_bayesian_shift_scale_df.select(common_genes).to_numpy()
    X_test_bayesian_shift_only = test_bayesian_shift_only_df.select(common_genes).to_numpy()
    
    # Prepare metadata labels
    metadata_cols = [col for col in train_meta_df.columns 
                    if col.startswith('meta_') and col != 'meta_source']
    
    continuous_labels = ['meta_age_at_diagnosis']
    
    print(f"\nMetadata labels: {len(metadata_cols)}")
    
    all_y_train = {}
    all_y_test = {}
    
    for meta_col in metadata_cols:
        y_train = train_meta_df[meta_col].to_numpy()
        y_test = test_meta_df[meta_col].to_numpy()
        
        if meta_col in continuous_labels:
            all_y_train[meta_col] = y_train
            all_y_test[meta_col] = y_test
        else:
            le = LabelEncoder()
            all_labels = np.concatenate([
                y_train[~pl.Series(y_train).is_null()],
                y_test[~pl.Series(y_test).is_null()]
            ])
            if len(all_labels) > 0 and len(np.unique(all_labels)) >= 2:
                le.fit(all_labels)
                y_train = np.array([le.transform([v])[0] if v is not None else np.nan 
                                   for v in y_train])
                y_test = np.array([le.transform([v])[0] if v is not None else np.nan 
                                  for v in y_test])
                all_y_train[meta_col] = y_train
                all_y_test[meta_col] = y_test
    
    print(f"Valid labels: {len(all_y_train)}")
    
    # Evaluate all methods with feature selection
    results = {}
    feature_counts = {}
    
    # 1. Unadjusted
    print("\n" + "="*80)
    print("1. Evaluating: Unadjusted (with feature selection)")
    print("="*80)
    unadjusted_scores = {}
    unadjusted_features = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score, n_feat = evaluate_regression_with_feature_selection(
                X_train, X_test_unadjusted, y_train, y_test)
        else:
            score, n_feat = evaluate_classification_with_feature_selection(
                X_train, X_test_unadjusted, y_train, y_test)
        unadjusted_scores[label_name] = score
        unadjusted_features[label_name] = n_feat
        print(f"  {label_name}: {score:.3f} ({n_feat} features)")
    results['Unadjusted'] = unadjusted_scores
    feature_counts['Unadjusted'] = unadjusted_features
    
    # 2. Bayesian shift+scale
    print("\n" + "="*80)
    print("2. Evaluating: Bayesian (shift+scale) with feature selection")
    print("="*80)
    bayesian_ss_scores = {}
    bayesian_ss_features = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score, n_feat = evaluate_regression_with_feature_selection(
                X_train, X_test_bayesian_shift_scale, y_train, y_test)
        else:
            score, n_feat = evaluate_classification_with_feature_selection(
                X_train, X_test_bayesian_shift_scale, y_train, y_test)
        bayesian_ss_scores[label_name] = score
        bayesian_ss_features[label_name] = n_feat
        print(f"  {label_name}: {score:.3f} ({n_feat} features)")
    results['Bayesian (shift+scale)'] = bayesian_ss_scores
    feature_counts['Bayesian (shift+scale)'] = bayesian_ss_features
    
    # 3. Bayesian shift-only
    print("\n" + "="*80)
    print("3. Evaluating: Bayesian (shift-only) with feature selection")
    print("="*80)
    bayesian_so_scores = {}
    bayesian_so_features = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score, n_feat = evaluate_regression_with_feature_selection(
                X_train, X_test_bayesian_shift_only, y_train, y_test)
        else:
            score, n_feat = evaluate_classification_with_feature_selection(
                X_train, X_test_bayesian_shift_only, y_train, y_test)
        bayesian_so_scores[label_name] = score
        bayesian_so_features[label_name] = n_feat
        print(f"  {label_name}: {score:.3f} ({n_feat} features)")
    results['Bayesian (shift-only)'] = bayesian_so_scores
    feature_counts['Bayesian (shift-only)'] = bayesian_so_features
    
    # 4. ML-DBA with feature selection
    print("\n" + "="*80)
    print("4. Optimizing: ML-DBA (shift-only) with feature selection")
    print("="*80)
    
    label_weights = {name: 1.0 for name in all_y_train.keys()}
    initial_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_bayesian_shift_only, axis=0)
    
    print("Optimizing (this may take a few minutes)...")
    result = minimize(
        multi_label_objective_with_fs,
        initial_shifts,
        args=(X_train, X_test_unadjusted, all_y_train, all_y_test, 
              continuous_labels, label_weights),
        method='L-BFGS-B',
        options={'maxiter': 30, 'ftol': 1e-3}
    )
    
    optimal_shifts = result.x
    print(f"Optimization complete. Final loss: {result.fun:.4f}")
    
    X_test_ml_dba = X_test_unadjusted - optimal_shifts
    
    print("\nEvaluating ML-DBA...")
    ml_dba_scores = {}
    ml_dba_features = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score, n_feat = evaluate_regression_with_feature_selection(
                X_train, X_test_ml_dba, y_train, y_test)
        else:
            score, n_feat = evaluate_classification_with_feature_selection(
                X_train, X_test_ml_dba, y_train, y_test)
        ml_dba_scores[label_name] = score
        ml_dba_features[label_name] = n_feat
        print(f"  {label_name}: {score:.3f} ({n_feat} features)")
    results['ML-DBA (shift-only)'] = ml_dba_scores
    feature_counts['ML-DBA (shift-only)'] = ml_dba_features
    
    # Save results
    output_dir = Path("outputs/bayesian_vs_ml_dba_with_fs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    comparison_data = []
    for label_name in all_y_train.keys():
        row = {'Label': label_name.replace('meta_', '')}
        for method in results.keys():
            row[method] = results[method][label_name]
            row[f'{method}_features'] = feature_counts[method][label_name]
        comparison_data.append(row)
    
    comparison_df = pl.DataFrame(comparison_data)
    comparison_df.write_csv(output_dir / "comparison_with_feature_selection.csv")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics (with Feature Selection)")
    print("="*80)
    
    methods = list(results.keys())
    for method in methods:
        scores = [results[method][label] for label in all_y_train.keys() 
                 if not np.isnan(results[method][label])]
        features = [feature_counts[method][label] for label in all_y_train.keys() 
                   if not np.isnan(results[method][label])]
        print(f"\n{method}:")
        print(f"  Mean score: {np.mean(scores):.3f}")
        print(f"  Median score: {np.median(scores):.3f}")
        print(f"  Mean features selected: {np.mean(features):.1f}/{len(common_genes)}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
