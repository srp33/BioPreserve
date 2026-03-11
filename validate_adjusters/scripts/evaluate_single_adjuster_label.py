#!/usr/bin/env python3
"""
Evaluate a single adjuster on a single label.
Used for parallelization - each job evaluates one (adjuster, label) pair.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder
import json


def evaluate_adjuster_label(X_train, X_test, y_train, y_test, classifier_name, is_continuous):
    """Evaluate a single adjuster on a single label."""
    # Filter out NaN labels
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    # Check if we have enough data
    if len(X_train_filtered) < 10 or len(X_test_filtered) < 10:
        return np.nan
    
    if not is_continuous and len(np.unique(y_train_filtered)) < 2:
        return np.nan
    
    # Select model
    if is_continuous:
        if classifier_name == "Gradient Boosting":
            model = HistGradientBoostingRegressor(random_state=42)
        else:
            model = HistGradientBoostingRegressor(random_state=42)
    else:
        if classifier_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=1)
        elif classifier_name == "Gradient Boosting":
            model = HistGradientBoostingClassifier(random_state=42)
        elif classifier_name == "Logistic (L2)":
            model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42, n_jobs=1)
        elif classifier_name == "ElasticNet (l1=0.15)":
            model = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=0.0001,
                l1_ratio=0.15,
                max_iter=1000,
                random_state=42,
                tol=1e-3,
                n_jobs=1
            )
        else:
            model = HistGradientBoostingClassifier(random_state=42)
    
    # Train and predict
    model.fit(X_train_filtered, y_train_filtered)
    y_pred = model.predict(X_test_filtered)
    
    # Calculate metric
    if is_continuous:
        score = r2_score(y_test_filtered, y_pred)
        metric = "R²"
    else:
        score = matthews_corrcoef(y_test_filtered, y_pred)
        metric = "MCC"
    
    return score, metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--adjuster-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--continuous-metadata", nargs='+', default=[])
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    # Load data
    train_genes_df = pl.read_csv(args.train_genes)
    test_genes_df = pl.read_csv(args.test_genes)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    # Get common genes (exclude metadata columns)
    common_genes = [
        g for g in train_genes_df.columns 
        if g in test_genes_df.columns and not g.startswith('meta_')
    ]
    
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test = test_genes_df.select(common_genes).to_numpy()
    
    # Get labels
    train_y = train_meta_df[args.label].to_numpy()
    test_y = test_meta_df[args.label].to_numpy()
    
    # Check if continuous
    is_continuous = args.label in args.continuous_metadata
    
    # Encode if categorical
    if not is_continuous:
        le = LabelEncoder()
        all_labels = np.concatenate([
            train_y[~pl.Series(train_y).is_null()],
            test_y[~pl.Series(test_y).is_null()]
        ])
        le.fit(all_labels)
        
        train_y = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
        test_y = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
    else:
        train_y = train_y.astype(float)
        test_y = test_y.astype(float)
    
    # Evaluate
    result = evaluate_adjuster_label(X_train, X_test, train_y, test_y, 
                                     args.classifier, is_continuous)
    
    if isinstance(result, tuple):
        score, metric = result
    else:
        score = result
        metric = "R²" if is_continuous else "MCC"
    
    # Save result
    output_data = {
        'adjuster': args.adjuster_name,
        'label': args.label,
        'classifier': args.classifier,
        'score': float(score) if not np.isnan(score) else None,
        'metric': metric
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    main()
