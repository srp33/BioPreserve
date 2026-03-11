#!/usr/bin/env python3
"""
Decision Boundary Alignment for Cross-Dataset Classification.

Key idea: Train linear classifiers on both datasets, then find shift/scale
parameters that align the decision boundaries. This directly optimizes for
classification performance rather than metadata prediction.

Approach:
1. Train logistic regression on train dataset → get w_train, b_train
2. Train logistic regression on test dataset → get w_test, b_test
3. Find shift s and scale σ for each gene that:
   - Aligns the coefficient vectors
   - Minimizes classification error on adjusted test data
"""

import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from pathlib import Path


def train_linear_classifier(X, y, classifier_type='logistic'):
    """Train a linear classifier and return coefficients."""
    mask = ~np.isnan(y)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(np.unique(y_filtered)) < 2:
        return None, None, None
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif classifier_type == 'elasticnet':
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    clf.fit(X_scaled, y_filtered)
    
    # Get coefficients
    w = clf.coef_[0]  # shape: (n_features,)
    b = clf.intercept_[0]
    
    return clf, w, b, scaler


def objective_function(params, X_train, X_test, y_train, y_test, w_train, 
                       classifier_type='logistic', alpha_coef=0.1):
    """
    Objective function for optimization.
    
    params: [shift_1, ..., shift_n, scale_1, ..., scale_n]
    
    Minimizes:
    - Classification error on adjusted test data
    - Coefficient alignment penalty (optional)
    """
    n_genes = X_train.shape[1]
    shifts = params[:n_genes]
    scales = params[n_genes:]
    
    # Adjust test data
    X_test_adjusted = (X_test - shifts) / scales
    
    # Train classifier on train data
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test_adjusted[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return 1e6
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Train and evaluate
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    else:
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    # Primary objective: maximize MCC (minimize negative MCC)
    mcc = matthews_corrcoef(y_test_filtered, y_pred)
    loss = -mcc
    
    # Optional: Add coefficient alignment penalty
    if alpha_coef > 0:
        w_new = clf.coef_[0]
        # Penalize difference between new coefficients and original train coefficients
        coef_penalty = np.sum((w_new - w_train)**2)
        loss += alpha_coef * coef_penalty
    
    return loss


def find_optimal_alignment(X_train, X_test, y_train, y_test, 
                          classifier_type='logistic', method='direct', 
                          optimize_scale=True):
    """
    Find optimal shift and scale parameters to align decision boundaries.
    
    Methods:
    - 'direct': Directly optimize for classification performance
    - 'coefficient': Initialize from coefficient ratios, then optimize
    - 'hybrid': Use coefficient alignment as regularization
    
    Args:
        optimize_scale: If False, fix scales to 1.0 and only optimize shifts
    """
    n_genes = X_train.shape[1]
    
    # Train classifiers on both datasets to get initial coefficients
    print("  Training classifiers on both datasets...")
    clf_train, w_train, b_train, scaler_train = train_linear_classifier(
        X_train, y_train, classifier_type
    )
    clf_test, w_test, b_test, scaler_test = train_linear_classifier(
        X_test, y_test, classifier_type
    )
    
    if clf_train is None or clf_test is None:
        return None, None, None
    
    print(f"  Train coefficients: {w_train[:5]}")
    print(f"  Test coefficients: {w_test[:5]}")
    
    # Initialize parameters
    if method == 'coefficient':
        # Initialize scale from coefficient ratios
        # w_train ≈ w_test / scale → scale ≈ w_test / w_train
        initial_scales = np.where(
            np.abs(w_train) > 1e-6,
            w_test / w_train,
            1.0
        )
        # Clip to reasonable range
        initial_scales = np.clip(initial_scales, 0.1, 10.0)
    else:
        initial_scales = np.ones(n_genes)
    
    # Initialize shifts to align means
    initial_shifts = np.nanmean(X_test, axis=0) - np.nanmean(X_train, axis=0)
    
    if optimize_scale:
        initial_params = np.concatenate([initial_shifts, initial_scales])
    else:
        # Only optimize shifts, fix scales to 1.0
        initial_params = initial_shifts
    
    print(f"  Initial shifts: {initial_shifts[:5]}")
    if optimize_scale:
        print(f"  Initial scales: {initial_scales[:5]}")
    else:
        print(f"  Scales fixed to 1.0 (shift-only optimization)")
    
    # Set alpha_coef based on method
    alpha_coef = 0.1 if method == 'hybrid' else 0.0
    
    # Set up bounds
    if optimize_scale:
        bounds = [(None, None)] * n_genes + [(0.01, 100)] * n_genes  # shifts unbounded, scales positive
    else:
        bounds = [(None, None)] * n_genes  # only shifts, unbounded
    
    # Optimize
    print("  Optimizing shift and scale parameters...")
    
    # Wrapper for shift-only optimization
    if not optimize_scale:
        def objective_shift_only(shifts):
            # Fix scales to 1.0
            params = np.concatenate([shifts, np.ones(n_genes)])
            return objective_function(params, X_train, X_test, y_train, y_test, 
                                    w_train, classifier_type, alpha_coef)
        
        result = minimize(
            objective_shift_only,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': True}
        )
        optimal_shifts = result.x
        optimal_scales = np.ones(n_genes)
    else:
        result = minimize(
            objective_function,
            initial_params,
            args=(X_train, X_test, y_train, y_test, w_train, classifier_type, alpha_coef),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': True}
        )
        optimal_shifts = result.x[:n_genes]
        optimal_scales = result.x[n_genes:]
    
    print(f"  Optimal shifts: {optimal_shifts[:5]}")
    print(f"  Optimal scales: {optimal_scales[:5]}")
    print(f"  Final loss: {result.fun:.4f}")
    
    return optimal_shifts, optimal_scales, result


def apply_adjustment(X, shifts, scales):
    """Apply shift and scale adjustment."""
    return (X - shifts) / scales


def evaluate_adjustment(X_train, X_test_adjusted, y_train, y_test, classifier_type='logistic'):
    """Evaluate classification performance after adjustment."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test_adjusted[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Train and evaluate
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif classifier_type == 'elasticnet':
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        X_train_scaled = X_train_filtered
        X_test_scaled = X_test_filtered
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    return matthews_corrcoef(y_test_filtered, y_pred)



def compute_cv_ceiling(X_test, y_test, classifier_type='logistic', n_splits=20):
    """Compute CV ceiling on test set."""
    mask = ~np.isnan(y_test)
    X_filtered = X_test[mask]
    y_filtered = y_test[mask]
    
    if len(np.unique(y_filtered)) < 2:
        return np.nan
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in kfold.split(X_filtered, y_filtered):
        X_tr, X_te = X_filtered[train_idx], X_filtered[test_idx]
        y_tr, y_te = y_filtered[train_idx], y_filtered[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        if classifier_type == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        elif classifier_type == 'elasticnet':
            clf = SGDClassifier(
                loss='log_loss', penalty='elasticnet',
                alpha=0.0001, l1_ratio=0.15,
                max_iter=1000, random_state=42, tol=1e-3
            )
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            X_tr = X_filtered[train_idx]
            X_te = X_filtered[test_idx]
        
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
    
    return matthews_corrcoef(all_y_true, all_y_pred)


def create_comparison_figure(results, gene_names, output_path):
    """Create figure comparing different adjustment methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Performance comparison
    ax = axes[0, 0]
    methods = list(results.keys())
    scores = [results[m]['score'] for m in methods]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60', '#95a5a6']
    
    bars = ax.bar(range(len(methods)), scores, color=colors[:len(methods)], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2., score + 0.02,
                f'{score:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('MCC Score', fontsize=13, fontweight='bold')
    ax.set_title('Decision Boundary Alignment Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylim(0, max(scores) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    if 'CV Ceiling' in results:
        ax.axhline(y=results['CV Ceiling']['score'], color='gray', 
                  linestyle='--', alpha=0.5, linewidth=2)
    
    # Panel 2: Parameter comparison (shifts)
    ax = axes[0, 1]
    n_genes_to_show = min(10, len(gene_names))
    x_pos = np.arange(n_genes_to_show)
    width = 0.25
    
    for i, method in enumerate(['Bayesian', 'DBA (direct)', 'DBA (coefficient)']):
        if method in results and results[method].get('shifts') is not None:
            shifts = results[method]['shifts'][:n_genes_to_show]
            ax.bar(x_pos + i*width, shifts, width, label=method, alpha=0.8)
    
    ax.set_ylabel('Shift Parameter', fontsize=11, fontweight='bold')
    ax.set_title('Shift Parameters by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(gene_names[:n_genes_to_show], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Parameter comparison (scales)
    ax = axes[1, 0]
    for i, method in enumerate(['Bayesian', 'DBA (direct)', 'DBA (coefficient)']):
        if method in results and results[method].get('scales') is not None:
            scales = results[method]['scales'][:n_genes_to_show]
            ax.bar(x_pos + i*width, scales, width, label=method, alpha=0.8)
    
    ax.set_ylabel('Scale Parameter', fontsize=11, fontweight='bold')
    ax.set_title('Scale Parameters by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(gene_names[:n_genes_to_show], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for method in methods:
        if method == 'CV Ceiling':
            table_data.append([method, f"{results[method]['score']:.3f}", "N/A", "Test set only"])
        else:
            score = results[method]['score']
            ceiling = results.get('CV Ceiling', {}).get('score', np.nan)
            pct = f"{score/ceiling*100:.1f}%" if not np.isnan(ceiling) else "N/A"
            desc = results[method].get('description', '')
            table_data.append([method, f"{score:.3f}", pct, desc])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'MCC', '% of Ceiling', 'Description'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Decision Boundary Alignment: Comprehensive Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Decision Boundary Alignment for cross-dataset classification"
    )
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes-unadjusted", required=True)
    parser.add_argument("--test-genes-bayesian", required=True)
    parser.add_argument("--test-genes-effective", required=False,
                       help="Bayesian effective shift-only adjusted data (optional)")
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--target-label", default="meta_her2_status")
    parser.add_argument("--n-top-genes", type=int, default=10,
                       help="Number of top genes to use (0 = all genes)")
    parser.add_argument("--classifier", default="elasticnet",
                       choices=['logistic', 'elasticnet', 'rf'])
    parser.add_argument("--cv-folds", type=int, default=20,
                       help="Number of CV folds for ceiling estimation")
    parser.add_argument("--shift-only", action="store_true",
                       help="Only optimize shifts, fix scales to 1.0")
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Decision Boundary Alignment")
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
    
    # Get labels
    train_y = train_meta_df[args.target_label].to_numpy()
    test_y = test_meta_df[args.target_label].to_numpy()
    
    # Encode
    le = LabelEncoder()
    all_labels = np.concatenate([
        train_y[~pl.Series(train_y).is_null()],
        test_y[~pl.Series(test_y).is_null()]
    ])
    le.fit(all_labels)
    
    train_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
    test_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_df.columns]
    
    print(f"Common genes: {len(common_genes)}")
    
    # Feature selection if requested
    if args.n_top_genes > 0 and args.n_top_genes < len(common_genes):
        print(f"\nSelecting top {args.n_top_genes} genes...")
        X_train_all = train_genes_df.select(common_genes).to_numpy()
        mask = ~np.isnan(train_y_encoded)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_all[mask], train_y_encoded[mask])
        importances = rf.feature_importances_
        
        top_genes = [common_genes[i] for i in np.argsort(importances)[-args.n_top_genes:][::-1]]
        print(f"Top genes: {top_genes}")
    else:
        top_genes = common_genes
    
    # Extract data
    X_train = train_genes_df.select(top_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(top_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(top_genes).to_numpy()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = {}
    
    # 1. Baseline (no adjustment)
    print("\n" + "="*60)
    print("1. Baseline (no adjustment)")
    print("="*60)
    score = evaluate_adjustment(X_train, X_test_unadjusted, 
                               train_y_encoded, test_y_encoded, args.classifier)
    results['Unadjusted'] = {
        'score': score,
        'description': 'No adjustment',
        'shifts': None,
        'scales': None
    }
    print(f"Score: {score:.3f}")
    
    # 2. Bayesian adjustment
    print("\n" + "="*60)
    print("2. Bayesian adjustment (shift+scale)")
    print("="*60)
    score = evaluate_adjustment(X_train, X_test_bayesian, 
                               train_y_encoded, test_y_encoded, args.classifier)
    
    # Get Bayesian parameters (approximate from data)
    bayesian_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_bayesian, axis=0)
    bayesian_scales = np.nanstd(X_test_unadjusted, axis=0) / (np.nanstd(X_test_bayesian, axis=0) + 1e-8)
    
    results['Bayesian (shift+scale)'] = {
        'score': score,
        'description': 'Metadata-based shift+scale',
        'shifts': bayesian_shifts,
        'scales': bayesian_scales
    }
    print(f"Score: {score:.3f}")
    
    # 2b. Bayesian effective shift-only (if provided)
    if test_effective_df is not None:
        print("\n" + "="*60)
        print("2b. Bayesian effective shift-only")
        print("="*60)
        X_test_effective = test_effective_df.select(top_genes).to_numpy()
        score = evaluate_adjustment(X_train, X_test_effective, 
                                   train_y_encoded, test_y_encoded, args.classifier)
        
        # Get effective shifts
        effective_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_effective, axis=0)
        
        results['Bayesian (effective shift)'] = {
            'score': score,
            'description': 'Metadata-based effective shift',
            'shifts': effective_shifts,
            'scales': np.ones(len(top_genes))
        }
        print(f"Score: {score:.3f}")
        print(f"Effective shifts: {effective_shifts[:3]}")
    
    # 3. Decision Boundary Alignment (direct optimization)
    print("\n" + "="*60)
    print("3. Decision Boundary Alignment (direct)")
    print("="*60)
    shifts_direct, scales_direct, result_direct = find_optimal_alignment(
        X_train, X_test_unadjusted, train_y_encoded, test_y_encoded,
        classifier_type=args.classifier, method='direct',
        optimize_scale=not args.shift_only
    )
    
    if shifts_direct is not None:
        X_test_dba_direct = apply_adjustment(X_test_unadjusted, shifts_direct, scales_direct)
        score = evaluate_adjustment(X_train, X_test_dba_direct, 
                                   train_y_encoded, test_y_encoded, args.classifier)
        results['DBA (direct)'] = {
            'score': score,
            'description': 'Direct optimization',
            'shifts': shifts_direct,
            'scales': scales_direct
        }
        print(f"Score: {score:.3f}")
        
        # Save adjusted data
        adjusted_df = pl.DataFrame({
            gene: X_test_dba_direct[:, i] for i, gene in enumerate(top_genes)
        })
        adjusted_df.write_csv(output_dir / "test_genes_dba_direct.csv")
    
    # 4. Decision Boundary Alignment (coefficient-initialized)
    print("\n" + "="*60)
    print("4. Decision Boundary Alignment (coefficient-initialized)")
    print("="*60)
    
    if not args.shift_only:
        shifts_coef, scales_coef, result_coef = find_optimal_alignment(
            X_train, X_test_unadjusted, train_y_encoded, test_y_encoded,
            classifier_type=args.classifier, method='coefficient',
            optimize_scale=True
        )
    else:
        print("  Skipping (shift-only mode)")
        shifts_coef = None
        scales_coef = None
    
    if shifts_coef is not None:
        X_test_dba_coef = apply_adjustment(X_test_unadjusted, shifts_coef, scales_coef)
        score = evaluate_adjustment(X_train, X_test_dba_coef, 
                                   train_y_encoded, test_y_encoded, args.classifier)
        results['DBA (coefficient)'] = {
            'score': score,
            'description': 'Coef-initialized',
            'shifts': shifts_coef,
            'scales': scales_coef
        }
        print(f"Score: {score:.3f}")
        
        # Save adjusted data
        adjusted_df = pl.DataFrame({
            gene: X_test_dba_coef[:, i] for i, gene in enumerate(top_genes)
        })
        adjusted_df.write_csv(output_dir / "test_genes_dba_coefficient.csv")
    
    # 5. Decision Boundary Alignment (hybrid)
    print("\n" + "="*60)
    print("5. Decision Boundary Alignment (hybrid)")
    print("="*60)
    
    if not args.shift_only:
        shifts_hybrid, scales_hybrid, result_hybrid = find_optimal_alignment(
            X_train, X_test_unadjusted, train_y_encoded, test_y_encoded,
            classifier_type=args.classifier, method='hybrid',
            optimize_scale=True
        )
    else:
        print("  Skipping (shift-only mode)")
        shifts_hybrid = None
        scales_hybrid = None
    
    if shifts_hybrid is not None:
        X_test_dba_hybrid = apply_adjustment(X_test_unadjusted, shifts_hybrid, scales_hybrid)
        score = evaluate_adjustment(X_train, X_test_dba_hybrid, 
                                   train_y_encoded, test_y_encoded, args.classifier)
        results['DBA (hybrid)'] = {
            'score': score,
            'description': 'Hybrid (coef penalty)',
            'shifts': shifts_hybrid,
            'scales': scales_hybrid
        }
        print(f"Score: {score:.3f}")
        
        # Save adjusted data
        adjusted_df = pl.DataFrame({
            gene: X_test_dba_hybrid[:, i] for i, gene in enumerate(top_genes)
        })
        adjusted_df.write_csv(output_dir / "test_genes_dba_hybrid.csv")
    
    # 6. CV Ceiling
    print("\n" + "="*60)
    print(f"6. CV Ceiling (test set only, {args.cv_folds} folds)")
    print("="*60)
    cv_score = compute_cv_ceiling(X_test_unadjusted, test_y_encoded, 
                                  args.classifier, n_splits=args.cv_folds)
    results['CV Ceiling'] = {
        'score': cv_score,
        'description': f'{args.cv_folds}-fold CV',
        'shifts': None,
        'scales': None
    }
    print(f"Score: {cv_score:.3f}")
    
    # Create comparison figure
    print("\n" + "="*60)
    print("Creating comparison figure...")
    print("="*60)
    create_comparison_figure(results, top_genes, output_dir / "dba_comparison.png")
    
    # Save results to CSV
    results_data = []
    for method, data in results.items():
        results_data.append({
            'method': method,
            'score': data['score'],
            'description': data['description']
        })
    
    results_df = pl.DataFrame(results_data)
    results_df.write_csv(output_dir / "dba_results.csv")
    
    # Save parameters
    if shifts_direct is not None:
        params_df = pl.DataFrame({
            'gene': top_genes,
            'bayesian_shift': bayesian_shifts,
            'bayesian_scale': bayesian_scales,
            'dba_direct_shift': shifts_direct,
            'dba_direct_scale': scales_direct,
            'dba_coef_shift': shifts_coef if shifts_coef is not None else [np.nan]*len(top_genes),
            'dba_coef_scale': scales_coef if scales_coef is not None else [np.nan]*len(top_genes),
            'dba_hybrid_shift': shifts_hybrid if shifts_hybrid is not None else [np.nan]*len(top_genes),
            'dba_hybrid_scale': scales_hybrid if scales_hybrid is not None else [np.nan]*len(top_genes),
        })
        params_df.write_csv(output_dir / "dba_parameters.csv")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method, data in results.items():
        pct = f"({data['score']/cv_score*100:.1f}% of ceiling)" if method != 'CV Ceiling' else ""
        print(f"{method:25s}: {data['score']:.3f} {pct}")
    print("="*60)


if __name__ == "__main__":
    main()
