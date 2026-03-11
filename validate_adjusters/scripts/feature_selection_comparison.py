#!/usr/bin/env python3
"""
Compare adjustment methods with aggressive feature selection.

For each metadata label:
1. Select top K most predictive genes using Random Forest
2. Evaluate shift+scale vs shift-only on those genes
3. Test multiple values of K (3, 5, 10, 20, 50)
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def select_top_genes(X_train, y_train, n_genes, is_continuous=False):
    """
    Select top N genes using Random Forest feature importance.
    
    Returns: list of gene indices
    """
    mask = ~np.isnan(y_train)
    X_filtered = X_train[mask]
    y_filtered = y_train[mask]
    
    if len(np.unique(y_filtered)) < 2:
        return []
    
    if is_continuous:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    
    rf.fit(X_filtered, y_filtered)
    importances = rf.feature_importances_
    
    # Get top N gene indices
    top_indices = np.argsort(importances)[::-1][:n_genes]
    return top_indices.tolist()


def evaluate_with_genes(X_train, X_test, y_train, y_test, gene_indices, is_continuous=False):
    """
    Evaluate classification/regression performance using selected genes.
    """
    if len(gene_indices) == 0:
        return np.nan
    
    X_train_selected = X_train[:, gene_indices]
    X_test_selected = X_test[:, gene_indices]
    
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train_selected[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test_selected[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2:
        return np.nan
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    if is_continuous:
        # Ridge regression
        clf = Ridge(alpha=1.0, random_state=42)
        clf.fit(X_train_scaled, y_train_filtered)
        y_pred = clf.predict(X_test_scaled)
        return r2_score(y_test_filtered, y_pred)
    else:
        # ElasticNet classifier
        if len(np.unique(y_test_filtered)) < 2:
            return np.nan
        
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
        clf.fit(X_train_scaled, y_train_filtered)
        y_pred = clf.predict(X_test_scaled)
        return matthews_corrcoef(y_test_filtered, y_pred)


def main():
    parser = argparse.ArgumentParser(
        description="Compare adjustment methods with feature selection"
    )
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--train-unadjusted", required=True)
    parser.add_argument("--test-unadjusted", required=True)
    parser.add_argument("--train-shift-scale", required=True)
    parser.add_argument("--test-shift-scale", required=True)
    parser.add_argument("--train-shift-only", required=True)
    parser.add_argument("--test-shift-only", required=True)
    parser.add_argument("--continuous-metadata", nargs="*", default=[])
    parser.add_argument("--n-genes", nargs="+", type=int, default=[3, 5, 10, 20, 50])
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    print("="*80)
    print("Feature Selection Comparison: Shift+Scale vs Shift-Only")
    print("="*80)
    
    # Load metadata
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    # Load gene expression data
    train_unadj_df = pl.read_csv(args.train_unadjusted)
    test_unadj_df = pl.read_csv(args.test_unadjusted)
    train_ss_df = pl.read_csv(args.train_shift_scale)
    test_ss_df = pl.read_csv(args.test_shift_scale)
    train_so_df = pl.read_csv(args.train_shift_only)
    test_so_df = pl.read_csv(args.test_shift_only)
    
    # Get common genes
    common_genes = [g for g in train_unadj_df.columns 
                    if g in test_unadj_df.columns 
                    and g in train_ss_df.columns 
                    and g in test_ss_df.columns
                    and g in train_so_df.columns
                    and g in test_so_df.columns]
    
    print(f"\nUsing {len(common_genes)} common genes")
    print(f"Testing with {len(args.n_genes)} different feature set sizes: {args.n_genes}")
    
    # Convert to numpy
    X_train_unadj = train_unadj_df.select(common_genes).to_numpy()
    X_test_unadj = test_unadj_df.select(common_genes).to_numpy()
    X_train_ss = train_ss_df.select(common_genes).to_numpy()
    X_test_ss = test_ss_df.select(common_genes).to_numpy()
    X_train_so = train_so_df.select(common_genes).to_numpy()
    X_test_so = test_so_df.select(common_genes).to_numpy()
    
    results = []
    
    # Process each metadata label
    for meta_col in train_meta_df.columns:
        print(f"\nProcessing: {meta_col}")
        
        is_continuous = meta_col in args.continuous_metadata
        metric = "R2" if is_continuous else "MCC"
        
        # Get labels
        train_y = train_meta_df[meta_col].to_numpy()
        test_y = test_meta_df[meta_col].to_numpy()
        
        # Encode if categorical
        if not is_continuous:
            train_y_clean = train_y[~pl.Series(train_y).is_null()]
            test_y_clean = test_y[~pl.Series(test_y).is_null()]
            
            if len(train_y_clean) == 0 or len(test_y_clean) == 0:
                print(f"  Skipping {meta_col}: no valid labels")
                continue
            
            le = LabelEncoder()
            all_labels = np.concatenate([train_y_clean, test_y_clean])
            le.fit(all_labels)
            
            train_y = np.array([le.transform([v])[0] if v is not None else np.nan 
                               for v in train_y])
            test_y = np.array([le.transform([v])[0] if v is not None else np.nan 
                              for v in test_y])
        
        # Try different numbers of genes
        for n_genes in args.n_genes:
            print(f"  Testing with top {n_genes} genes...")
            
            # Select genes based on unadjusted data
            gene_indices = select_top_genes(X_train_unadj, train_y, n_genes, is_continuous)
            
            if len(gene_indices) == 0:
                print(f"    Could not select genes")
                continue
            
            selected_gene_names = [common_genes[i] for i in gene_indices]
            
            # Evaluate unadjusted
            score_unadj = evaluate_with_genes(
                X_train_unadj, X_test_unadj, train_y, test_y, 
                gene_indices, is_continuous
            )
            
            # Evaluate shift+scale
            score_ss = evaluate_with_genes(
                X_train_ss, X_test_ss, train_y, test_y,
                gene_indices, is_continuous
            )
            
            # Evaluate shift-only
            score_so = evaluate_with_genes(
                X_train_so, X_test_so, train_y, test_y,
                gene_indices, is_continuous
            )
            
            print(f"    Unadjusted: {score_unadj:.3f}, Shift+Scale: {score_ss:.3f}, Shift-Only: {score_so:.3f}")
            
            results.append({
                'metadata_label': meta_col,
                'metric': metric,
                'n_genes': n_genes,
                'selected_genes': ','.join(selected_gene_names[:5]) + ('...' if len(selected_gene_names) > 5 else ''),
                'score_unadjusted': score_unadj,
                'score_shift_scale': score_ss,
                'score_shift_only': score_so,
                'shift_only_advantage': score_so - score_ss
            })
    
    # Save results
    results_df = pl.DataFrame(results)
    results_df.write_csv(args.output)
    
    print("\n" + "="*80)
    print(f"Results saved to: {args.output}")
    print(f"Total comparisons: {len(results)}")
    print("="*80)


if __name__ == "__main__":
    main()
