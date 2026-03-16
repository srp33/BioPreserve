#!/usr/bin/env python3
"""
Run k-fold cross-validation on the test set to establish performance ceiling.

This shows the best possible performance achievable on the test set,
independent of batch correction quality.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_classifier(classifier_name, random_state=42):
    """Get classifier by name."""
    if classifier_name == 'Gradient Boosting':
        return HistGradientBoostingClassifier(random_state=random_state, max_iter=100)
    elif classifier_name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10, n_jobs=-1)
    elif classifier_name == 'Logistic (L2)':
        return LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, n_jobs=-1)
    elif classifier_name == 'ElasticNet (l1=0.15)':
        return SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15,
                           max_iter=1000, random_state=random_state, tol=1e-3, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


def get_regressor(classifier_name, random_state=42):
    """Get regressor by name."""
    if classifier_name == 'Gradient Boosting':
        return HistGradientBoostingRegressor(random_state=random_state, max_iter=100)
    elif classifier_name == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10, n_jobs=-1)
    elif classifier_name in ['Logistic (L2)', 'ElasticNet (l1=0.15)']:
        # Use Ridge for regression
        return Ridge(alpha=1.0, random_state=random_state)
    else:
        raise ValueError(f"Unknown regressor: {classifier_name}")


def run_cv_on_test_set(genes_path, metadata_path, continuous_cols, classifier_names, n_splits=5, random_state=42, label_filter=None):
    """Run k-fold CV on test set to get performance ceiling per classifier.
    
    Args:
        n_splits: Number of folds. Use -1 or 0 for Leave-One-Out CV.
        label_filter: If provided, only evaluate these metadata columns.
    """
    
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
    if label_filter:
        metadata_cols = [col for col in metadata_cols if col in label_filter]
    gene_cols = [col for col in df.columns if not col.startswith('meta_')]
    
    X = df.select(gene_cols).to_numpy()
    
    results = {}
    
    # Determine CV strategy
    use_loocv = n_splits <= 0
    cv_description = "Leave-One-Out CV" if use_loocv else f"{n_splits}-fold CV"
    
    print(f"Running {cv_description} on test set...")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(gene_cols)}")
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Classifiers: {classifier_names}")
    print()
    
    # For each classifier and metadata column, run CV
    for classifier_name in classifier_names:
        print(f"\n{'='*80}")
        print(f"Classifier: {classifier_name}")
        print(f"{'='*80}")
        
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
                model = get_regressor(classifier_name, random_state)
                
                # Choose CV strategy
                if use_loocv:
                    kfold = LeaveOneOut()
                else:
                    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                all_y_true = []
                all_y_pred = []
                
                # Need scaling for linear models
                needs_scaling = classifier_name in ['Logistic (L2)', 'ElasticNet (l1=0.15)']
                
                for train_idx, test_idx in kfold.split(X_filtered):
                    X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
                    y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]
                    
                    if needs_scaling:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                
                # Compute single R² over all accumulated predictions
                score = r2_score(all_y_true, all_y_pred)
                
                key = (classifier_name, meta_col)
                results[key] = {
                    'classifier': classifier_name,
                    'label': meta_col,
                    'metric': 'R²',
                    'score': score,
                    'n_samples': len(all_y_true)
                }
                print(f"  {meta_col:40s} R² = {score:.4f} (n={len(all_y_true)})")
                
            else:
                # Classification task - accumulate all predictions
                # Encode labels
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_filtered)
                
                # Skip if only one class
                if len(np.unique(y_encoded)) < 2:
                    print(f"  Skipping {meta_col}: only one class")
                    continue
                
                model = get_classifier(classifier_name, random_state)
                
                # Choose CV strategy
                # Note: For LOOCV with classification, we can't use StratifiedKFold
                # LeaveOneOut doesn't guarantee stratification but ensures maximum data usage
                if use_loocv:
                    kfold = LeaveOneOut()
                else:
                    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                all_y_true = []
                all_y_pred = []
                
                # Need scaling for linear models
                needs_scaling = classifier_name in ['Logistic (L2)', 'ElasticNet (l1=0.15)']
                
                try:
                    # For LOOCV, we don't pass y to split()
                    if use_loocv:
                        split_iter = kfold.split(X_filtered)
                    else:
                        split_iter = kfold.split(X_filtered, y_encoded)
                    
                    for train_idx, test_idx in split_iter:
                        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
                        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                        
                        if needs_scaling:
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        all_y_true.extend(y_test)
                        all_y_pred.extend(y_pred)
                    
                    # Compute single MCC over all accumulated predictions
                    score = matthews_corrcoef(all_y_true, all_y_pred)
                    
                    key = (classifier_name, meta_col)
                    results[key] = {
                        'classifier': classifier_name,
                        'label': meta_col,
                        'metric': 'MCC',
                        'score': score,
                        'n_samples': len(all_y_true)
                    }
                    print(f"  {meta_col:40s} MCC = {score:.4f} (n={len(all_y_true)})")
                    
                except ValueError as e:
                    print(f"  Skipping {meta_col}: {e}")
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
        "--classifiers",
        nargs="+",
        default=["Gradient Boosting", "Random Forest", "Logistic (L2)", "ElasticNet (l1=0.15)"],
        help="Classifiers to evaluate (default: all 4)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Specific metadata labels to evaluate (default: all meta_ columns)"
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
        args.classifiers,
        args.n_splits,
        args.random_state,
        args.labels
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Test Set Performance Ceiling (aggregated k-fold CV predictions)")
    print("="*80)
    
    # Group by classifier
    for classifier_name in args.classifiers:
        classifier_results = {k: v for k, v in results.items() if v['classifier'] == classifier_name}
        
        mcc_scores = [r for r in classifier_results.values() if r['metric'] == 'MCC']
        r2_scores = [r for r in classifier_results.values() if r['metric'] == 'R²']
        
        print(f"\n{classifier_name}:")
        if mcc_scores:
            avg_mcc = np.mean([r['score'] for r in mcc_scores])
            print(f"  Average MCC: {avg_mcc:.4f}")
        
        if r2_scores:
            avg_r2 = np.mean([r['score'] for r in r2_scores])
            print(f"  Average R²: {avg_r2:.4f}")
    
    # Save results if requested
    if args.output:
        import pandas as pd
        
        rows = []
        for key, res in results.items():
            rows.append({
                'classifier': res['classifier'],
                'metadata_column': res['label'],
                'metric': res['metric'],
                'score': res['score'],
                'n_samples': res['n_samples']
            })
        
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
