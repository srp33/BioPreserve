#!/usr/bin/env python3
"""
Multi-Label Decision Boundary Alignment.

Key idea: Instead of optimizing for a single label, optimize shift/scale
parameters that work well across ALL metadata labels simultaneously.

This is more comparable to the Bayesian adjuster, which also adjusts for
all metadata at once.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier, LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from pathlib import Path


def evaluate_classification(X_train, X_test, y_train, y_test, classifier='histgradient'):
    """Evaluate classification performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    # Select classifier
    if classifier == 'histgradient':
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=50, max_depth=5)
    elif classifier == 'elasticnet':
        # Standardize for linear models
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
    elif classifier == 'logistic':
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    elif classifier == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_train_filtered, y_train_filtered)
    y_pred = clf.predict(X_test_filtered)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def evaluate_regression(X_train, X_test, y_train, y_test, regressor='histgradient'):
    """Evaluate regression performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(X_train_filtered) < 10 or len(X_test_filtered) < 10:
        return np.nan
    
    # Select regressor
    if regressor == 'histgradient':
        reg = HistGradientBoostingRegressor(random_state=42, max_iter=50, max_depth=5)
    elif regressor in ['elasticnet', 'logistic', 'randomforest']:
        # Use Ridge for all linear-based methods
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        reg = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")
    
    reg.fit(X_train_filtered, y_train_filtered)
    y_pred = reg.predict(X_test_filtered)
    
    return r2_score(y_test_filtered, y_pred)


def multi_label_objective(params, X_train, X_test, labels_train, labels_test, 
                          label_types, optimize_scale=True, weights=None, 
                          iteration_counter=None, classifier='histgradient'):
    """
    Multi-label objective function.
    
    Optimizes shift/scale parameters to maximize performance across ALL labels.
    
    Args:
        params: [shift_1, ..., shift_n] or [shift_1, ..., shift_n, scale_1, ..., scale_n]
        labels_train: dict of {label_name: encoded_values}
        labels_test: dict of {label_name: encoded_values}
        label_types: dict of {label_name: 'classification' or 'regression'}
        optimize_scale: If False, fix scales to 1.0
        weights: dict of {label_name: weight} for combining scores
        iteration_counter: dict to track iterations
        classifier: Classifier to use for optimization ('histgradient', 'elasticnet', 'logistic', 'randomforest')
    """
    n_genes = X_train.shape[1]
    
    if optimize_scale:
        shifts = params[:n_genes]
        scales = params[n_genes:]
    else:
        shifts = params
        scales = np.ones(n_genes)
    
    # Adjust test data
    X_test_adjusted = (X_test - shifts) / scales
    
    # Evaluate on all labels
    scores = {}
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        if label_type == 'classification':
            score = evaluate_classification(X_train, X_test_adjusted, y_train, y_test, classifier)
        else:
            score = evaluate_regression(X_train, X_test_adjusted, y_train, y_test, classifier)
        
        if not np.isnan(score):
            scores[label_name] = score
    
    # Combine scores (weighted average)
    if not scores:
        return 1e6
    
    if weights is None:
        weights = {k: 1.0 for k in scores.keys()}
    
    # Normalize weights
    total_weight = sum(weights.get(k, 1.0) for k in scores.keys())
    
    combined_score = sum(
        scores[k] * weights.get(k, 1.0) / total_weight 
        for k in scores.keys()
    )
    
    # Track progress
    if iteration_counter is not None:
        iteration_counter['count'] += 1
        if iteration_counter['count'] % 10 == 0:
            print(f"    Iteration {iteration_counter['count']}: score = {combined_score:.4f}")
    
    # Minimize negative score
    return -combined_score


def find_multi_label_alignment(X_train, X_test, labels_train, labels_test,
                               label_types, optimize_scale=True, weights=None,
                               method='direct', classifier='histgradient'):
    """
    Find optimal shift/scale parameters for multi-label alignment.
    
    Args:
        method: 'direct' (optimize with specified classifier) or 
                'closed_form' (linear closed-form solution)
        classifier: Classifier to use for optimization ('histgradient', 'elasticnet', 'logistic', 'randomforest')
    """
    n_genes = X_train.shape[1]
    
    if method == 'closed_form':
        return find_closed_form_alignment(
            X_train, X_test, labels_train, labels_test,
            label_types, optimize_scale, weights
        )
    
    # Direct optimization method (original)
    # Initialize shifts to align means
    initial_shifts = np.nanmean(X_test, axis=0) - np.nanmean(X_train, axis=0)
    initial_scales = np.ones(n_genes)
    
    if optimize_scale:
        initial_params = np.concatenate([initial_shifts, initial_scales])
        bounds = [(None, None)] * n_genes + [(0.01, 100)] * n_genes
    else:
        initial_params = initial_shifts
        bounds = [(None, None)] * n_genes
    
    print(f"  Initial shifts: {initial_shifts[:5]}")
    if optimize_scale:
        print(f"  Initial scales: {initial_scales[:5]}")
    else:
        print(f"  Scales fixed to 1.0 (shift-only optimization)")
    print(f"  Optimization classifier: {classifier}")
    
    # Optimize
    print("  Optimizing parameters across all labels...")
    iteration_counter = {'count': 0}
    
    result = minimize(
        multi_label_objective,
        initial_params,
        args=(X_train, X_test, labels_train, labels_test, label_types, 
              optimize_scale, weights, iteration_counter, classifier),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False, 'ftol': 1e-4}
    )
    
    if optimize_scale:
        optimal_shifts = result.x[:n_genes]
        optimal_scales = result.x[n_genes:]
    else:
        optimal_shifts = result.x
        optimal_scales = np.ones(n_genes)
    
    print(f"  Optimal shifts: {optimal_shifts[:5]}")
    print(f"  Optimal scales: {optimal_scales[:5]}")
    print(f"  Final loss: {result.fun:.4f}")
    
    return optimal_shifts, optimal_scales, result


def find_closed_form_alignment(X_train, X_test, labels_train, labels_test,
                                label_types, optimize_scale=True, weights=None):
    """
    Find optimal shift/scale using closed-form linear solution.
    
    For each label, train linear models on train and test data, then solve
    for shift/scale that aligns the decision boundaries across all labels.
    """
    n_genes = X_train.shape[1]
    
    print("  Training linear models on each label...")
    
    # Collect coefficients from all labels
    W_sum = np.zeros((n_genes, n_genes))  # Σ w_k w_k^T
    delta_sum = np.zeros(n_genes)  # Σ w_k * delta_k
    scale_estimates = []
    
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        # Filter out NaN labels
        mask_train = ~np.isnan(y_train)
        mask_test = ~np.isnan(y_test)
        
        X_train_filtered = X_train[mask_train]
        y_train_filtered = y_train[mask_train]
        X_test_filtered = X_test[mask_test]
        y_test_filtered = y_test[mask_test]
        
        if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2 or len(X_test_filtered) < 10:
            continue
        
        # Standardize
        scaler_train = StandardScaler()
        scaler_test = StandardScaler()
        X_train_scaled = scaler_train.fit_transform(X_train_filtered)
        X_test_scaled = scaler_test.fit_transform(X_test_filtered)
        
        # Train linear models
        if label_type == 'classification':
            clf_train = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
            clf_test = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
            
            clf_train.fit(X_train_scaled, y_train_filtered)
            clf_test.fit(X_test_scaled, y_test_filtered)
            
            w_train = clf_train.coef_[0]
            b_train = clf_train.intercept_[0]
            w_test = clf_test.coef_[0]
            b_test = clf_test.intercept_[0]
        else:
            # Regression
            reg_train = Ridge(alpha=1.0, random_state=42)
            reg_test = Ridge(alpha=1.0, random_state=42)
            
            reg_train.fit(X_train_scaled, y_train_filtered)
            reg_test.fit(X_test_scaled, y_test_filtered)
            
            w_train = reg_train.coef_
            b_train = reg_train.intercept_
            w_test = reg_test.coef_
            b_test = reg_test.intercept_
        
        # Compute scale estimate (element-wise ratio of coefficients)
        if optimize_scale:
            scale_est = np.where(
                np.abs(w_train) > 1e-6,
                w_test / w_train,
                1.0
            )
            scale_est = np.clip(scale_est, 0.1, 10.0)
            scale_estimates.append(scale_est)
        
        # Compute misalignment on original (unscaled) data
        # We need to account for the standardization
        # The shift in original space that aligns means
        mean_train = np.nanmean(X_train, axis=0)
        mean_test = np.nanmean(X_test, axis=0)
        
        # Weight for this label
        label_weight = weights.get(label_name, 1.0) if weights else 1.0
        
        # Add to accumulator (weighted by label importance)
        W_sum += label_weight * np.outer(w_train, w_train)
        
        # The shift that would align the intercepts
        # Simplified: use mean difference as proxy
        delta_sum += label_weight * w_train * (b_train - b_test)
    
    # Solve for optimal shift
    # Add small regularization for numerical stability
    W_sum_reg = W_sum + 1e-6 * np.eye(n_genes)
    
    try:
        optimal_shifts = np.linalg.solve(W_sum_reg, delta_sum)
    except np.linalg.LinAlgError:
        print("  Warning: Singular matrix, using mean shift")
        optimal_shifts = np.nanmean(X_test, axis=0) - np.nanmean(X_train, axis=0)
    
    # Compute optimal scales (median of per-label estimates)
    if optimize_scale and scale_estimates:
        optimal_scales = np.median(scale_estimates, axis=0)
    else:
        optimal_scales = np.ones(n_genes)
    
    print(f"  Closed-form shifts: {optimal_shifts[:5]}")
    print(f"  Closed-form scales: {optimal_scales[:5]}")
    
    # Create a dummy result object for consistency
    class ClosedFormResult:
        def __init__(self):
            self.fun = 0.0
            self.success = True
    
    return optimal_shifts, optimal_scales, ClosedFormResult()


def evaluate_all_labels(X_train, X_test, labels_train, labels_test, label_types, classifier='histgradient'):
    """Evaluate performance on all labels."""
    results = {}
    
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        if label_type == 'classification':
            score = evaluate_classification(X_train, X_test, y_train, y_test, classifier)
            metric = 'MCC'
        else:
            score = evaluate_regression(X_train, X_test, y_train, y_test, classifier)
            metric = 'R²'
        
        results[label_name] = {'score': score, 'metric': metric}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label Decision Boundary Alignment"
    )
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes-unadjusted", required=True)
    parser.add_argument("--test-genes-bayesian", required=True)
    parser.add_argument("--test-genes-effective", required=False)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--continuous-metadata", nargs='+', required=True,
                       help="List of continuous metadata columns")
    parser.add_argument("--shift-only", action="store_true",
                       help="Only optimize shifts, fix scales to 1.0")
    parser.add_argument("--training-classifier", default="histgradient",
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       help="Classifier to use for DBA optimization")
    parser.add_argument("--evaluation-classifiers", nargs='+',
                       default=['histgradient'],
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       help="Classifiers to use for evaluation")
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Label Decision Boundary Alignment")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv(args.train_genes)
    test_unadjusted_df = pl.read_csv(args.test_genes_unadjusted)
    test_bayesian_df = pl.read_csv(args.test_genes_bayesian)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    # Load effective shift data if provided
    test_effective_df = None
    if args.test_genes_effective:
        test_effective_df = pl.read_csv(args.test_genes_effective)
        print(f"  Loaded effective shift data: {len(test_effective_df.columns)} genes")
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_df.columns]
    
    print(f"Common genes: {len(common_genes)}")
    
    # Extract gene data
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(common_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(common_genes).to_numpy()
    
    if test_effective_df is not None:
        X_test_effective = test_effective_df.select(common_genes).to_numpy()
    
    # Prepare labels
    print("\nPreparing labels...")
    continuous_metadata = set(args.continuous_metadata)
    
    labels_train = {}
    labels_test = {}
    label_types = {}
    
    for col in train_meta_df.columns:
        if col.startswith('meta_'):
            label_name = col
            
            train_vals = train_meta_df[col].to_numpy()
            test_vals = test_meta_df[col].to_numpy()
            
            if col in continuous_metadata:
                # Regression
                labels_train[label_name] = train_vals.astype(float)
                labels_test[label_name] = test_vals.astype(float)
                label_types[label_name] = 'regression'
            else:
                # Classification - encode labels
                le = LabelEncoder()
                all_labels = np.concatenate([
                    train_vals[~pl.Series(train_vals).is_null()],
                    test_vals[~pl.Series(test_vals).is_null()]
                ])
                le.fit(all_labels)
                
                labels_train[label_name] = np.array([
                    le.transform([v])[0] if v is not None else np.nan 
                    for v in train_vals
                ])
                labels_test[label_name] = np.array([
                    le.transform([v])[0] if v is not None else np.nan 
                    for v in test_vals
                ])
                label_types[label_name] = 'classification'
    
    print(f"Labels: {list(labels_train.keys())}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    all_results = {}
    
    # 1. Baseline (no adjustment)
    print("\n" + "="*60)
    print("1. Baseline (no adjustment)")
    print("="*60)
    
    # Evaluate with all evaluation classifiers
    for eval_clf in args.evaluation_classifiers:
        print(f"\n  Evaluating with {eval_clf}:")
        results = evaluate_all_labels(X_train, X_test_unadjusted, 
                                      labels_train, labels_test, label_types, eval_clf)
        all_results[f'Unadjusted ({eval_clf})'] = results
        for label, data in results.items():
            print(f"    {label}: {data['score']:.3f} ({data['metric']})")
    
    # 2. Bayesian adjustment (shift+scale)
    print("\n" + "="*60)
    print("2. Bayesian adjustment (shift+scale)")
    print("="*60)
    results = evaluate_all_labels(X_train, X_test_bayesian, 
                                  labels_train, labels_test, label_types)
    all_results['Bayesian (shift+scale)'] = results
    for label, data in results.items():
        print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # 2b. Bayesian effective shift-only (if provided)
    if test_effective_df is not None:
        print("\n" + "="*60)
        print("2b. Bayesian effective shift-only")
        print("="*60)
        results = evaluate_all_labels(X_train, X_test_effective, 
                                      labels_train, labels_test, label_types)
        all_results['Bayesian (effective shift)'] = results
        for label, data in results.items():
            print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # 3. Multi-label DBA (closed-form)
    print("\n" + "="*60)
    print("3. Multi-Label DBA (closed-form linear solution)")
    print("="*60)
    shifts_dba_cf, scales_dba_cf, result_dba_cf = find_multi_label_alignment(
        X_train, X_test_unadjusted, labels_train, labels_test,
        label_types, optimize_scale=not args.shift_only, method='closed_form'
    )
    
    X_test_dba_cf = (X_test_unadjusted - shifts_dba_cf) / scales_dba_cf
    results = evaluate_all_labels(X_train, X_test_dba_cf, 
                                  labels_train, labels_test, label_types)
    all_results['Multi-Label DBA (closed-form)'] = results
    for label, data in results.items():
        print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # Save adjusted data
    adjusted_df = pl.DataFrame({
        gene: X_test_dba_cf[:, i] for i, gene in enumerate(common_genes)
    })
    adjusted_df.write_csv(output_dir / "test_genes_multi_label_dba_closed_form.csv")
    
    # 4. Multi-label DBA (direct optimization)
    print("\n" + "="*60)
    print("4. Multi-Label DBA (direct optimization)")
    print("="*60)
    shifts_dba, scales_dba, result_dba = find_multi_label_alignment(
        X_train, X_test_unadjusted, labels_train, labels_test,
        label_types, optimize_scale=not args.shift_only, method='direct',
        classifier=args.training_classifier
    )
    
    X_test_dba = (X_test_unadjusted - shifts_dba) / scales_dba
    results = evaluate_all_labels(X_train, X_test_dba, 
                                  labels_train, labels_test, label_types)
    all_results['Multi-Label DBA (direct)'] = results
    for label, data in results.items():
        print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # Save adjusted data
    adjusted_df = pl.DataFrame({
        gene: X_test_dba[:, i] for i, gene in enumerate(common_genes)
    })
    adjusted_df.write_csv(output_dir / "test_genes_multi_label_dba_direct.csv")
    
    # Also save with classifier-specific name for Snakemake compatibility
    adjusted_df.write_csv(output_dir / f"test_genes_dba_{args.training_classifier}.csv")
    
    # Save parameters
    params_df = pl.DataFrame({
        'gene': common_genes,
        'multi_label_dba_cf_shift': shifts_dba_cf,
        'multi_label_dba_cf_scale': scales_dba_cf,
        'multi_label_dba_direct_shift': shifts_dba,
        'multi_label_dba_direct_scale': scales_dba,
    })
    params_df.write_csv(output_dir / "multi_label_dba_parameters.csv")
    
    # Also save with classifier-specific name for Snakemake compatibility
    params_df.write_csv(output_dir / f"dba_parameters_{args.training_classifier}.csv")
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON ACROSS ALL LABELS")
    print("="*60)
    
    # Prepare data for CSV
    comparison_data = []
    for method_name, method_results in all_results.items():
        for label_name, label_data in method_results.items():
            comparison_data.append({
                'method': method_name,
                'label': label_name,
                'metric': label_data['metric'],
                'score': label_data['score']
            })
    
    comparison_df = pl.DataFrame(comparison_data)
    comparison_df.write_csv(output_dir / "multi_label_comparison.csv")
    
    # Print summary table
    print(f"\n{'Method':<30} {'Label':<25} {'Metric':<6} {'Score':>8}")
    print("-" * 75)
    for row in comparison_data:
        print(f"{row['method']:<30} {row['label']:<25} {row['metric']:<6} {row['score']:>8.3f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
