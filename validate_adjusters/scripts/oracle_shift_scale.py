#!/usr/bin/env python3
"""
Oracle shift-scale adjustment: Find the best possible shift and scale parameters
that maximize cross-dataset HER2 prediction performance.

This establishes the ceiling for what shift-scale methods can achieve.

Approach:
1. For each gene, try different shift/scale parameters
2. Evaluate cross-dataset HER2 prediction performance
3. Select parameters that maximize performance
4. Compare to: (a) no adjustment, (b) Bayesian adjustment, (c) CV ceiling
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_her2_prediction(X_train, y_train, X_test, y_test):
    """
    Train on train, predict on test. Return MCC.
    """
    # Filter out missing labels
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    # Check if we have enough data
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    # Train and evaluate
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train_filtered, y_train_filtered)
    y_pred = clf.predict(X_test_filtered)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def grid_search_shift_scale(train_gene, test_gene, X_train_other, y_train, X_test_other, y_test,
                            shift_range=(-5, 5, 21), scale_range=(0.5, 2.0, 16)):
    """
    Grid search over shift and scale parameters for a single gene.
    
    Returns: (best_shift, best_scale, best_score)
    """
    shifts = np.linspace(shift_range[0], shift_range[1], shift_range[2])
    scales = np.linspace(scale_range[0], scale_range[1], scale_range[2])
    
    best_score = -np.inf
    best_shift = 0
    best_scale = 1
    
    for shift in shifts:
        for scale in scales:
            # Apply shift and scale to test gene
            test_gene_adjusted = (test_gene - shift) / scale
            
            # Combine with other genes
            X_train_full = np.column_stack([train_gene, X_train_other])
            X_test_full = np.column_stack([test_gene_adjusted, X_test_other])
            
            # Evaluate
            score = evaluate_her2_prediction(X_train_full, y_train, X_test_full, y_test)
            
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_shift = shift
                best_scale = scale
    
    return best_shift, best_scale, best_score


def oracle_adjustment_per_gene(train_genes_df, test_genes_df, train_meta_df, test_meta_df,
                               target_genes=None, n_other_genes=5):
    """
    For each target gene, find oracle shift/scale while keeping other top genes fixed.
    
    Parameters:
    -----------
    target_genes : list of str
        Genes to optimize (e.g., ['ERBB2', 'STARD3'])
    n_other_genes : int
        Number of other top genes to include in the model
    
    Returns:
    --------
    dict: {gene: {'shift': float, 'scale': float, 'score': float}}
    """
    # Get HER2 labels
    train_y = train_meta_df['meta_her2_status'].to_numpy()
    test_y = test_meta_df['meta_her2_status'].to_numpy()
    
    # Encode labels
    le = LabelEncoder()
    all_labels = np.concatenate([
        train_y[~pl.Series(train_y).is_null()],
        test_y[~pl.Series(test_y).is_null()]
    ])
    le.fit(all_labels)
    
    train_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
    test_y_encoded = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns if g in test_genes_df.columns]
    
    # If no target genes specified, use top predictive genes
    if target_genes is None:
        from sklearn.ensemble import RandomForestClassifier
        X_train_all = train_genes_df.select(common_genes).to_numpy()
        mask = ~np.isnan(train_y_encoded)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_all[mask], train_y_encoded[mask])
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        target_genes = [common_genes[i] for i in top_indices[:3]]  # Top 3
    
    print(f"Optimizing shift/scale for: {target_genes}")
    
    results = {}
    
    for idx, target_gene in enumerate(target_genes):
        print(f"[{idx+1}/{len(target_genes)}] Optimizing {target_gene}...")
        
        # Get other top genes (excluding target)
        other_genes = [g for g in common_genes if g != target_gene][:n_other_genes]
        
        # Get gene data
        train_target = train_genes_df[target_gene].to_numpy()
        test_target = test_genes_df[target_gene].to_numpy()
        
        train_other = train_genes_df.select(other_genes).to_numpy()
        test_other = test_genes_df.select(other_genes).to_numpy()
        
        # Grid search
        print(f"  {target_gene}: searching...")
        best_shift, best_scale, best_score = grid_search_shift_scale(
            train_target, test_target,
            train_other, train_y_encoded,
            test_other, test_y_encoded
        )
        
        print(f"  {target_gene}: shift={best_shift:.3f}, scale={best_scale:.3f}, MCC={best_score:.3f}")
        
        results[target_gene] = {
            'shift': best_shift,
            'scale': best_scale,
            'score': best_score
        }
    
    return results


def apply_oracle_adjustment(test_genes_df, oracle_params):
    """
    Apply oracle shift/scale parameters to test data.
    """
    adjusted_df = test_genes_df.clone()
    
    for gene, params in oracle_params.items():
        if gene in adjusted_df.columns:
            gene_values = adjusted_df[gene].to_numpy()
            adjusted_values = (gene_values - params['shift']) / params['scale']
            adjusted_df = adjusted_df.with_columns([
                pl.lit(adjusted_values).alias(gene)
            ])
    
    return adjusted_df


def compare_adjustments(train_genes_df, test_genes_df, test_unadjusted_df,
                       test_bayesian_df, train_meta_df, test_meta_df,
                       oracle_params, output_path):
    """
    Compare performance of:
    1. No adjustment
    2. Bayesian adjustment
    3. Oracle adjustment (using only the genes that were optimized)
    4. CV ceiling (train on test, test on test)
    """
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
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_genes_df.columns 
                   and g in test_unadjusted_df.columns
                   and g in test_bayesian_df.columns]
    
    # For oracle, use only top 10 genes (the ones used during optimization)
    # Get top genes from RF
    from sklearn.ensemble import RandomForestClassifier
    X_train_all = train_genes_df.select(common_genes).to_numpy()
    mask = ~np.isnan(train_y_encoded)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_all[mask], train_y_encoded[mask])
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    top_10_genes = [common_genes[i] for i in top_indices]
    
    print(f"Using top 10 genes for all comparisons: {top_10_genes}")
    
    # Prepare data - use only top 10 genes for fair comparison
    X_train = train_genes_df.select(top_10_genes).to_numpy()
    X_test_oracle = test_genes_df.select(top_10_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(top_10_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(top_10_genes).to_numpy()
    
    results = {}
    
    # 1. No adjustment
    print("Evaluating: No adjustment")
    results['No adjustment'] = evaluate_her2_prediction(
        X_train, train_y_encoded, X_test_unadjusted, test_y_encoded
    )
    
    # 2. Bayesian adjustment
    print("Evaluating: Bayesian adjustment")
    results['Bayesian'] = evaluate_her2_prediction(
        X_train, train_y_encoded, X_test_bayesian, test_y_encoded
    )
    
    # 3. Oracle adjustment
    print("Evaluating: Oracle adjustment")
    results['Oracle'] = evaluate_her2_prediction(
        X_train, train_y_encoded, X_test_oracle, test_y_encoded
    )
    
    # 4. CV ceiling (train and test on test set)
    print("Evaluating: CV ceiling")
    from sklearn.model_selection import StratifiedKFold
    test_mask = ~np.isnan(test_y_encoded)
    X_test_filtered = X_test_unadjusted[test_mask]
    y_test_filtered = test_y_encoded[test_mask]
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in kfold.split(X_test_filtered, y_test_filtered):
        X_tr, X_te = X_test_filtered[train_idx], X_test_filtered[test_idx]
        y_tr, y_te = y_test_filtered[train_idx], y_test_filtered[test_idx]
        
        clf = HistGradientBoostingClassifier(random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
    
    results['CV Ceiling'] = matthews_corrcoef(all_y_true, all_y_pred)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    scores = [results[m] for m in methods]
    colors = ['lightcoral', 'steelblue', 'gold', 'lightgreen']
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('MCC Score', fontsize=12)
    ax.set_title('HER2 Status Prediction: Comparison of Adjustment Methods\n(Using top 10 genes)', 
                fontsize=14, pad=20)
    ax.set_ylim(0, max(scores) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation text
    gap_bayesian = results['CV Ceiling'] - results['Bayesian']
    gap_oracle = results['CV Ceiling'] - results['Oracle']
    improvement = results['Oracle'] - results['Bayesian']
    pct_captured = (1 - gap_oracle/gap_bayesian) * 100 if gap_bayesian > 0 else 0
    
    text = f"""
    Gap (Bayesian → Ceiling): {gap_bayesian:.3f}
    Gap (Oracle → Ceiling): {gap_oracle:.3f}
    Improvement (Bayesian → Oracle): {improvement:.3f}
    
    Oracle captures {pct_captured:.1f}% of possible improvement
    
    Genes used: {', '.join(top_10_genes[:3])}...
    """
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def plot_oracle_parameters(oracle_params, output_path):
    """
    Visualize the oracle shift and scale parameters.
    """
    genes = list(oracle_params.keys())
    shifts = [oracle_params[g]['shift'] for g in genes]
    scales = [oracle_params[g]['scale'] for g in genes]
    scores = [oracle_params[g]['score'] for g in genes]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Shifts
    ax1 = axes[0]
    bars = ax1.bar(genes, shifts, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Optimal Shift', fontsize=12)
    ax1.set_title('Oracle Shift Parameters', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, shift in zip(bars, shifts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{shift:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Panel 2: Scales
    ax2 = axes[1]
    bars = ax2.bar(genes, scales, color='coral', alpha=0.8, edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No scaling')
    ax2.set_ylabel('Optimal Scale', fontsize=12)
    ax2.set_title('Oracle Scale Parameters', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, scale in zip(bars, scales):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{scale:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Panel 3: Scores
    ax3 = axes[2]
    bars = ax3.bar(genes, scores, color='gold', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('MCC Score', fontsize=12)
    ax3.set_title('Performance with Oracle Parameters', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Find oracle shift/scale parameters for HER2 prediction"
    )
    parser.add_argument("--train-genes", required=True, help="Train gene expression CSV")
    parser.add_argument("--test-genes-unadjusted", required=True, help="Test gene expression CSV (unadjusted)")
    parser.add_argument("--test-genes-bayesian", required=True, help="Test gene expression CSV (Bayesian adjusted)")
    parser.add_argument("--train-metadata", required=True, help="Train metadata CSV")
    parser.add_argument("--test-metadata", required=True, help="Test metadata CSV")
    parser.add_argument("--target-genes", nargs="+", default=None,
                       help="Genes to optimize (default: top 3 from RF)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Oracle Shift-Scale Analysis for HER2 Status")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv(args.train_genes)
    test_genes_unadjusted_df = pl.read_csv(args.test_genes_unadjusted)
    test_genes_bayesian_df = pl.read_csv(args.test_genes_bayesian)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find oracle parameters
    print("\nFinding oracle shift/scale parameters...")
    oracle_params = oracle_adjustment_per_gene(
        train_genes_df, test_genes_unadjusted_df,
        train_meta_df, test_meta_df,
        target_genes=args.target_genes
    )
    
    # Apply oracle adjustment
    print("\nApplying oracle adjustment...")
    test_genes_oracle_df = apply_oracle_adjustment(test_genes_unadjusted_df, oracle_params)
    
    # Save oracle-adjusted data
    oracle_output = output_dir / "test_genes_oracle_adjusted.csv"
    test_genes_oracle_df.write_csv(oracle_output)
    print(f"Saved oracle-adjusted data to: {oracle_output}")
    
    # Save oracle parameters
    params_output = output_dir / "oracle_parameters.csv"
    params_df = pl.DataFrame([
        {'gene': gene, **params}
        for gene, params in oracle_params.items()
    ])
    params_df.write_csv(params_output)
    print(f"Saved oracle parameters to: {params_output}")
    
    # Plot oracle parameters
    print("\nPlotting oracle parameters...")
    plot_oracle_parameters(oracle_params, output_dir / "oracle_parameters.png")
    
    # Compare all methods
    print("\nComparing adjustment methods...")
    comparison_results = compare_adjustments(
        train_genes_df, test_genes_oracle_df, test_genes_unadjusted_df,
        test_genes_bayesian_df, train_meta_df, test_meta_df,
        oracle_params, output_dir / "adjustment_comparison.png"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for method, score in comparison_results.items():
        print(f"{method:20s}: MCC = {score:.4f}")
    
    gap_bayesian = comparison_results['CV Ceiling'] - comparison_results['Bayesian']
    gap_oracle = comparison_results['CV Ceiling'] - comparison_results['Oracle']
    improvement = comparison_results['Oracle'] - comparison_results['Bayesian']
    pct_captured = (1 - gap_oracle/gap_bayesian) * 100 if gap_bayesian > 0 else 0
    
    print(f"\nGap (Bayesian → Ceiling): {gap_bayesian:.4f}")
    print(f"Gap (Oracle → Ceiling): {gap_oracle:.4f}")
    print(f"Improvement (Bayesian → Oracle): {improvement:.4f}")
    print(f"Oracle captures {pct_captured:.1f}% of possible improvement")
    
    print("\n" + "="*60)
    print(f"Outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
