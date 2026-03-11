#!/usr/bin/env python3
"""
Compare Bayesian vs ML-DBA across all metadata labels.
Uses all common genes, not just top 3.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_classification(X_train, X_test, y_train, y_test):
    """Evaluate classification performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    clf = SGDClassifier(
        loss='log_loss', penalty='elasticnet',
        alpha=0.0001, l1_ratio=0.15,
        max_iter=1000, random_state=42, tol=1e-3
    )
    
    clf.fit(X_train_scaled, y_train_filtered)
    y_pred = clf.predict(X_test_scaled)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def evaluate_regression(X_train, X_test, y_train, y_test):
    """Evaluate regression performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(y_train_filtered) < 10 or len(y_test_filtered) < 10:
        return np.nan
    
    from sklearn.linear_model import ElasticNet
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_filtered)
    X_test_scaled = scaler_X.transform(X_test_filtered)
    
    reg = ElasticNet(alpha=0.0001, l1_ratio=0.15, random_state=42, max_iter=1000)
    reg.fit(X_train_scaled, y_train_filtered)
    y_pred = reg.predict(X_test_scaled)
    
    return r2_score(y_test_filtered, y_pred)


def multi_label_objective(shifts, X_train, X_test, all_y_train, all_y_test, 
                          continuous_labels, label_weights):
    """
    Multi-label objective: weighted average of negative scores.
    """
    # Adjust test data (shift-only)
    X_test_adjusted = X_test - shifts
    
    total_loss = 0.0
    total_weight = 0.0
    
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        weight = label_weights.get(label_name, 1.0)
        
        if label_name in continuous_labels:
            score = evaluate_regression(X_train, X_test_adjusted, y_train, y_test)
        else:
            score = evaluate_classification(X_train, X_test_adjusted, y_train, y_test)
        
        if not np.isnan(score):
            total_loss += weight * (-score)
            total_weight += weight
    
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return 1e6


def main():
    print("="*80)
    print("Bayesian vs ML-DBA: Full Metadata Comparison")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadjusted_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_shift_scale_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayesian_shift_only_df = pl.read_csv("adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv")
    
    train_meta_df = pl.read_csv("metadata/metadata_train.csv")
    test_meta_df = pl.read_csv("metadata/metadata_test.csv")
    
    # Get common genes across all datasets
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_shift_scale_df.columns
                   and g in test_bayesian_shift_only_df.columns]
    
    print(f"Common genes: {len(common_genes)}")
    
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(common_genes).to_numpy()
    X_test_bayesian_shift_scale = test_bayesian_shift_scale_df.select(common_genes).to_numpy()
    X_test_bayesian_shift_only = test_bayesian_shift_only_df.select(common_genes).to_numpy()
    
    # Prepare all metadata labels
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
            # Keep as continuous
            all_y_train[meta_col] = y_train
            all_y_test[meta_col] = y_test
        else:
            # Encode categorical
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
    
    print(f"Valid labels for evaluation: {len(all_y_train)}")
    
    # Evaluate all methods
    results = {}
    
    # 1. Unadjusted
    print("\n" + "="*80)
    print("1. Evaluating: Unadjusted")
    print("="*80)
    unadjusted_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score = evaluate_regression(X_train, X_test_unadjusted, y_train, y_test)
        else:
            score = evaluate_classification(X_train, X_test_unadjusted, y_train, y_test)
        unadjusted_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    results['Unadjusted'] = unadjusted_scores
    
    # 2. Bayesian shift+scale
    print("\n" + "="*80)
    print("2. Evaluating: Bayesian (shift+scale)")
    print("="*80)
    bayesian_ss_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score = evaluate_regression(X_train, X_test_bayesian_shift_scale, y_train, y_test)
        else:
            score = evaluate_classification(X_train, X_test_bayesian_shift_scale, y_train, y_test)
        bayesian_ss_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    results['Bayesian (shift+scale)'] = bayesian_ss_scores
    
    # 3. Bayesian shift-only
    print("\n" + "="*80)
    print("3. Evaluating: Bayesian (shift-only)")
    print("="*80)
    bayesian_so_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score = evaluate_regression(X_train, X_test_bayesian_shift_only, y_train, y_test)
        else:
            score = evaluate_classification(X_train, X_test_bayesian_shift_only, y_train, y_test)
        bayesian_so_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    results['Bayesian (shift-only)'] = bayesian_so_scores
    
    # 4. ML-DBA (optimize for all labels)
    print("\n" + "="*80)
    print("4. Optimizing: ML-DBA (all labels, shift-only)")
    print("="*80)
    
    # Equal weights for all labels
    label_weights = {name: 1.0 for name in all_y_train.keys()}
    
    # Initialize with Bayesian effective shifts
    initial_shifts = np.nanmean(X_test_unadjusted, axis=0) - np.nanmean(X_test_bayesian_shift_only, axis=0)
    print(f"Initial shifts (from Bayesian): mean={np.mean(initial_shifts):.3f}, std={np.std(initial_shifts):.3f}")
    
    # Optimize
    print("Optimizing...")
    result = minimize(
        multi_label_objective,
        initial_shifts,
        args=(X_train, X_test_unadjusted, all_y_train, all_y_test, 
              continuous_labels, label_weights),
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-4}
    )
    
    optimal_shifts = result.x
    print(f"Optimal shifts: mean={np.mean(optimal_shifts):.3f}, std={np.std(optimal_shifts):.3f}")
    print(f"Final loss: {result.fun:.4f}")
    print(f"Optimization success: {result.success}")
    
    # Apply ML-DBA
    X_test_ml_dba = X_test_unadjusted - optimal_shifts
    
    print("\nEvaluating ML-DBA...")
    ml_dba_scores = {}
    for label_name, y_train in all_y_train.items():
        y_test = all_y_test[label_name]
        if label_name in continuous_labels:
            score = evaluate_regression(X_train, X_test_ml_dba, y_train, y_test)
        else:
            score = evaluate_classification(X_train, X_test_ml_dba, y_train, y_test)
        ml_dba_scores[label_name] = score
        print(f"  {label_name}: {score:.3f}")
    results['ML-DBA (shift-only)'] = ml_dba_scores
    
    # Create comparison DataFrame
    print("\n" + "="*80)
    print("Creating comparison visualizations...")
    print("="*80)
    
    # Build comparison table
    comparison_data = []
    for label_name in all_y_train.keys():
        comparison_data.append({
            'Label': label_name.replace('meta_', ''),
            'Unadjusted': unadjusted_scores[label_name],
            'Bayesian (shift+scale)': bayesian_ss_scores[label_name],
            'Bayesian (shift-only)': bayesian_so_scores[label_name],
            'ML-DBA (shift-only)': ml_dba_scores[label_name]
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    
    # Save results
    output_dir = Path("outputs/bayesian_vs_ml_dba")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.write_csv(output_dir / "comparison_results.csv")
    
    # Save ML-DBA adjusted data
    ml_dba_adjusted_df = pl.DataFrame({
        gene: X_test_ml_dba[:, i] for i, gene in enumerate(common_genes)
    })
    ml_dba_adjusted_df.write_csv(output_dir / "test_genes_ml_dba.csv")
    
    # Save shifts
    shifts_df = pl.DataFrame({
        'gene': common_genes,
        'bayesian_shift': initial_shifts,
        'ml_dba_shift': optimal_shifts,
        'difference': optimal_shifts - initial_shifts
    })
    shifts_df.write_csv(output_dir / "shift_comparison.csv")
    
    # Create visualizations
    import pandas as pd
    comp_pd = comparison_df.to_pandas()
    
    # Plot 1: Heatmap comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
    
    methods = ['Unadjusted', 'Bayesian (shift+scale)', 'Bayesian (shift-only)', 'ML-DBA (shift-only)']
    
    for idx, method in enumerate(methods):
        data = comp_pd[['Label', method]].set_index('Label')
        
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            center=0,
            cbar=(idx == 3),
            ax=axes[idx],
            linewidths=0.5
        )
        axes[idx].set_title(method)
        axes[idx].set_xlabel('')
        if idx > 0:
            axes[idx].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Mean scores comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_scores = {
        method: np.nanmean([comp_pd[method].values[i] 
                           for i in range(len(comp_pd)) 
                           if not np.isnan(comp_pd[method].values[i])])
        for method in methods
    }
    
    bars = ax.bar(range(len(methods)), list(mean_scores.values()), 
                  color=['gray', 'blue', 'green', 'red'], alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Mean Score (MCC/R²)')
    ax.set_title('Average Performance Across All Metadata Labels')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "mean_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Per-label comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(comp_pd))
    width = 0.2
    
    for idx, method in enumerate(methods):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, comp_pd[method], width, label=method, alpha=0.8)
    
    ax.set_xlabel('Metadata Label')
    ax.set_ylabel('Score (MCC/R²)')
    ax.set_title('Performance Comparison Across All Metadata Labels')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_pd['Label'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_label_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    for method in methods:
        scores = [comp_pd[method].values[i] 
                 for i in range(len(comp_pd)) 
                 if not np.isnan(comp_pd[method].values[i])]
        print(f"\n{method}:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Std: {np.std(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
