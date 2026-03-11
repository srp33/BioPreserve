#!/usr/bin/env python3
"""
Debug why adjustment performance is poor.

Questions to answer:
1. Does using all genes (vs top 10) hurt performance?
2. Does Logistic Regression vs Gradient Boosting make a difference?
3. What are the actual Bayesian shift/scale parameters?
4. How do they compare to oracle parameters?
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_with_different_classifiers(X_train, y_train, X_test, y_test, name=""):
    """
    Evaluate with multiple classifiers to see if choice matters.
    """
    # Filter out missing labels
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return {}
    
    results = {}
    
    # Scale data for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # 1. Logistic Regression (L2 only)
    lr = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
    lr.fit(X_train_scaled, y_train_filtered)
    y_pred = lr.predict(X_test_scaled)
    results['Logistic (L2)'] = matthews_corrcoef(y_test_filtered, y_pred)
    
    # 2. ElasticNet (L1 + L2) - using SGDClassifier with elasticnet penalty
    # Try different l1_ratio values
    for l1_ratio in [0.15, 0.5, 0.85]:
        en = SGDClassifier(
            loss='log_loss',  # logistic regression
            penalty='elasticnet',
            alpha=0.0001,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=42,
            tol=1e-3
        )
        en.fit(X_train_scaled, y_train_filtered)
        y_pred = en.predict(X_test_scaled)
        results[f'ElasticNet (l1={l1_ratio:.2f})'] = matthews_corrcoef(y_test_filtered, y_pred)
    
    # 3. Gradient Boosting (no scaling needed)
    gb = HistGradientBoostingClassifier(random_state=42)
    gb.fit(X_train_filtered, y_train_filtered)
    y_pred = gb.predict(X_test_filtered)
    results['Gradient Boosting'] = matthews_corrcoef(y_test_filtered, y_pred)
    
    # 4. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_filtered, y_train_filtered)
    y_pred = rf.predict(X_test_filtered)
    results['Random Forest'] = matthews_corrcoef(y_test_filtered, y_pred)
    
    return results


def compare_gene_subsets(train_genes_df, test_unadjusted_df, test_bayesian_df,
                         train_meta_df, test_meta_df):
    """
    Compare performance with different gene subsets.
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
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_df.columns]
    
    print(f"Total common genes: {len(common_genes)}")
    
    # Get top genes
    X_train_all = train_genes_df.select(common_genes).to_numpy()
    mask = ~np.isnan(train_y_encoded)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_all[mask], train_y_encoded[mask])
    importances = rf.feature_importances_
    
    # Different gene subsets
    gene_subsets = {
        'Top 1 (ERBB2 only)': [common_genes[np.argmax(importances)]],
        'Top 3': [common_genes[i] for i in np.argsort(importances)[-3:][::-1]],
        'Top 10': [common_genes[i] for i in np.argsort(importances)[-10:][::-1]],
        'Top 20': [common_genes[i] for i in np.argsort(importances)[-20:][::-1]],
        'All genes': common_genes
    }
    
    results = []
    
    for subset_name, genes in gene_subsets.items():
        print(f"\n{'='*60}")
        print(f"Testing: {subset_name} ({len(genes)} genes)")
        print(f"{'='*60}")
        
        if subset_name != 'All genes':
            print(f"Genes: {genes[:5]}{'...' if len(genes) > 5 else ''}")
        
        # Prepare data
        X_train = train_genes_df.select(genes).to_numpy()
        X_test_unadj = test_unadjusted_df.select(genes).to_numpy()
        X_test_bayes = test_bayesian_df.select(genes).to_numpy()
        
        # Test unadjusted
        print("\nUnadjusted:")
        unadj_results = evaluate_with_different_classifiers(
            X_train, train_y_encoded, X_test_unadj, test_y_encoded
        )
        for clf, score in unadj_results.items():
            print(f"  {clf:25s}: {score:.4f}")
            results.append({
                'gene_subset': subset_name,
                'n_genes': len(genes),
                'adjustment': 'Unadjusted',
                'classifier': clf,
                'mcc': score
            })
        
        # Test Bayesian adjusted
        print("\nBayesian adjusted:")
        bayes_results = evaluate_with_different_classifiers(
            X_train, train_y_encoded, X_test_bayes, test_y_encoded
        )
        for clf, score in bayes_results.items():
            print(f"  {clf:25s}: {score:.4f}")
            results.append({
                'gene_subset': subset_name,
                'n_genes': len(genes),
                'adjustment': 'Bayesian',
                'classifier': clf,
                'mcc': score
            })
    
    return pl.DataFrame(results)


def extract_bayesian_parameters(bayesian_params_path, oracle_params_path, output_path):
    """
    Compare Bayesian vs Oracle parameters for key genes.
    """
    # Load parameters
    bayesian_df = pl.read_csv(bayesian_params_path)
    oracle_df = pl.read_csv(oracle_params_path)
    
    # Get common genes
    common_genes = [g for g in oracle_df['gene'] if g in bayesian_df['gene']]
    
    # Extract shift and scale from Bayesian
    # Bayesian format: test_mean_slope, test_mean_intercept
    # Oracle format: shift, scale
    # Relationship: adjusted = (original - intercept) / slope
    
    comparison = []
    for gene in common_genes:
        bayes_row = bayesian_df.filter(pl.col('gene') == gene)
        oracle_row = oracle_df.filter(pl.col('gene') == gene)
        
        if len(bayes_row) > 0 and len(oracle_row) > 0:
            bayes_slope = bayes_row['test_mean_slope'][0]
            bayes_intercept = bayes_row['test_mean_intercept'][0]
            
            oracle_shift = oracle_row['shift'][0]
            oracle_scale = oracle_row['scale'][0]
            
            comparison.append({
                'gene': gene,
                'bayesian_shift': bayes_intercept,
                'bayesian_scale': bayes_slope,
                'oracle_shift': oracle_shift,
                'oracle_scale': oracle_scale,
                'shift_diff': abs(bayes_intercept - oracle_shift),
                'scale_diff': abs(bayes_slope - oracle_scale),
                'oracle_score': oracle_row['score'][0]
            })
    
    comp_df = pl.DataFrame(comparison)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    comp_pd = comp_df.to_pandas()
    
    # Panel 1: Shift comparison
    ax1 = axes[0, 0]
    x = np.arange(len(comp_pd))
    width = 0.35
    ax1.bar(x - width/2, comp_pd['bayesian_shift'], width, label='Bayesian', alpha=0.8)
    ax1.bar(x + width/2, comp_pd['oracle_shift'], width, label='Oracle', alpha=0.8)
    ax1.set_xlabel('Gene')
    ax1.set_ylabel('Shift Parameter')
    ax1.set_title('Shift Parameter Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_pd['gene'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Scale comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, comp_pd['bayesian_scale'], width, label='Bayesian', alpha=0.8)
    ax2.bar(x + width/2, comp_pd['oracle_scale'], width, label='Oracle', alpha=0.8)
    ax2.set_xlabel('Gene')
    ax2.set_ylabel('Scale Parameter')
    ax2.set_title('Scale Parameter Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_pd['gene'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No scaling')
    
    # Panel 3: Parameter differences
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, comp_pd['shift_diff'], width, label='Shift diff', alpha=0.8)
    ax3.bar(x + width/2, comp_pd['scale_diff'], width, label='Scale diff', alpha=0.8)
    ax3.set_xlabel('Gene')
    ax3.set_ylabel('Absolute Difference')
    ax3.set_title('Parameter Differences (|Bayesian - Oracle|)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_pd['gene'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Oracle scores
    ax4 = axes[1, 1]
    bars = ax4.bar(x, comp_pd['oracle_score'], alpha=0.8, color='gold', edgecolor='black')
    ax4.set_xlabel('Gene')
    ax4.set_ylabel('Oracle MCC Score')
    ax4.set_title('Oracle Performance (when optimizing each gene)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comp_pd['gene'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, comp_pd['oracle_score']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return comp_df


def main():
    parser = argparse.ArgumentParser(
        description="Debug adjustment performance issues"
    )
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes-unadjusted", required=True)
    parser.add_argument("--test-genes-bayesian", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--bayesian-params", required=True,
                       help="Bayesian parameters CSV")
    parser.add_argument("--oracle-params", required=True,
                       help="Oracle parameters CSV")
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Debugging Adjustment Performance")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv(args.train_genes)
    test_unadjusted_df = pl.read_csv(args.test_genes_unadjusted)
    test_bayesian_df = pl.read_csv(args.test_genes_bayesian)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Question 1: Does gene subset matter?
    print("\n" + "="*60)
    print("QUESTION 1: Does using all genes hurt performance?")
    print("="*60)
    
    results_df = compare_gene_subsets(
        train_genes_df, test_unadjusted_df, test_bayesian_df,
        train_meta_df, test_meta_df
    )
    
    results_df.write_csv(output_dir / "gene_subset_comparison.csv")
    
    # Question 2: Compare Bayesian vs Oracle parameters
    print("\n" + "="*60)
    print("QUESTION 2: How do Bayesian parameters compare to Oracle?")
    print("="*60)
    
    param_comparison = extract_bayesian_parameters(
        args.bayesian_params,
        args.oracle_params,
        output_dir / "parameter_comparison.png"
    )
    
    param_comparison.write_csv(output_dir / "parameter_comparison.csv")
    
    print("\n" + "="*60)
    print("Parameter Comparison:")
    print("="*60)
    print(param_comparison)
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
