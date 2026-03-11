#!/usr/bin/env python3
"""
Run k-fold cross-validation on the test set to establish performance ceiling.

This shows the best possible performance achievable on the test set,
independent of batch correction quality.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder


def run_cv_on_test_set(genes_path, metadata_path, continuous_cols, n_splits=5, random_state=42):
    """Run k-fold CV on test set to get performance ceiling."""
    
    # Load gene expression and metadata
    genes_df = pl.read_csv(genes_path)
    metadata_df = pl.read_csv(metadata_path)
    
    # Merge on index (assuming same order)
    if len(genes_df) != len(metadata_df):
        raise ValueError(f"Gene expression ({len(genes_df)}) and metadata ({len(metadata_df)}) have different lengths")
    
    # Combine horizontally
    df = pl.concat([genes_df, metadata_df], how="horizontal")
    
    # Separate features and metadata
    metadata_cols = [col for col in df.columns if col.startswith('meta_')]
    gene_cols = [col for col in df.columns if not col.startswith('meta_')]
    
    X = df.select(gene_cols).to_numpy()
    
    results = {}
    
    print(f"Running {n_splits}-fold CV on test set...")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(gene_cols)}")
    print(f"Metadata columns: {len(metadata_cols)}")
    print()
    
    # For each metadata column, run CV
    for meta_col in metadata_cols:
        # Skip meta_source (batch label)
        if meta_col == 'meta_source':
            continue
            
        # Get target values
        y_series = df[meta_col]
        
        # Skip if all missing
        if y_series.is_null().all():
            print(f"Skipping {meta_col}: all values missing")
            continue
        
        # Filter out missing values
        mask = ~y_series.is_null()
        mask_np = mask.to_numpy()
        X_filtered = X[mask_np]
        y_filtered = y_series.filter(mask).to_numpy()
        
        if len(y_filtered) < n_splits:
            print(f"Skipping {meta_col}: too few samples ({len(y_filtered)})")
            continue
        
        # Determine if continuous or categorical
        is_continuous = meta_col in continuous_cols
        
        if is_continuous:
            # Regression task - accumulate all predictions
            model = HistGradientBoostingRegressor(random_state=random_state)
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            all_y_true = []
            all_y_pred = []
            
            for train_idx, test_idx in kfold.split(X_filtered):
                X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
                y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
            
            # Compute single R² over all accumulated predictions
            score = r2_score(all_y_true, all_y_pred)
            
            results[meta_col] = {
                'metric': 'R²',
                'score': score,
                'n_samples': len(all_y_true)
            }
            print(f"{meta_col:40s} R² = {score:.4f} (n={len(all_y_true)})")
            
        else:
            # Classification task - accumulate all predictions
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_filtered)
            
            # Skip if only one class
            if len(np.unique(y_encoded)) < 2:
                print(f"Skipping {meta_col}: only one class")
                continue
            
            model = HistGradientBoostingClassifier(random_state=random_state)
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            all_y_true = []
            all_y_pred = []
            
            try:
                for train_idx, test_idx in kfold.split(X_filtered, y_encoded):
                    X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
                    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                
                # Compute single MCC over all accumulated predictions
                score = matthews_corrcoef(all_y_true, all_y_pred)
                
                results[meta_col] = {
                    'metric': 'MCC',
                    'score': score,
                    'n_samples': len(all_y_true)
                }
                print(f"{meta_col:40s} MCC = {score:.4f} (n={len(all_y_true)})")
                
            except ValueError as e:
                print(f"Skipping {meta_col}: {e}")
                continue
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run k-fold CV on test set to establish performance ceiling"
    )
    parser.add_argument(
        "--genes",
        required=True,
        help="Path to test set gene expression CSV file"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to test set metadata CSV file"
    )
    parser.add_argument(
        "--continuous-metadata",
        nargs="+",
        default=["meta_age_at_diagnosis"],
        help="Metadata columns to treat as continuous (default: meta_age_at_diagnosis)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV file for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Run CV
    results = run_cv_on_test_set(
        args.genes,
        args.metadata,
        args.continuous_metadata,
        args.n_splits,
        args.random_state
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Test Set Performance Ceiling (aggregated k-fold CV predictions)")
    print("="*80)
    
    mcc_scores = [r for r in results.values() if r['metric'] == 'MCC']
    r2_scores = [r for r in results.values() if r['metric'] == 'R²']
    
    if mcc_scores:
        avg_mcc = np.mean([r['score'] for r in mcc_scores])
        print(f"\nAverage MCC across all classification tasks: {avg_mcc:.4f}")
        print(f"This represents the performance ceiling on the test set.")
    
    if r2_scores:
        avg_r2 = np.mean([r['score'] for r in r2_scores])
        print(f"\nAverage R² across all regression tasks: {avg_r2:.4f}")
    
    # Save results if requested
    if args.output:
        import pandas as pd
        
        rows = []
        for col, res in results.items():
            rows.append({
                'metadata_column': col,
                'metric': res['metric'],
                'score': res['score'],
                'n_samples': res['n_samples']
            })
        
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
