#!/usr/bin/env python3
"""
DBA Classifier Comparison Analysis.

Train multi-label DBA with different classifiers, then evaluate each
adjustment across ALL classifiers to see if adjustments are classifier-specific
or general.

This answers: "Does the choice of optimization classifier matter?"
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path

# Import from multi_label_dba
import sys
sys.path.insert(0, str(Path(__file__).parent))
from multi_label_dba import (
    find_multi_label_alignment,
    evaluate_all_labels,
    LabelEncoder
)


def main():
    parser = argparse.ArgumentParser(
        description="DBA Classifier Comparison Analysis"
    )
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes-unadjusted", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--continuous-metadata", nargs='+', required=True)
    parser.add_argument("--shift-only", action="store_true")
    parser.add_argument("--training-classifiers", nargs='+',
                       default=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'])
    parser.add_argument("--evaluation-classifiers", nargs='+',
                       default=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'])
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*70)
    print("DBA Classifier Comparison Analysis")
    print("="*70)
    print(f"Training classifiers: {args.training_classifiers}")
    print(f"Evaluation classifiers: {args.evaluation_classifiers}")
    
    # Load data
    print("\nLoading data...")
    train_genes_df = pl.read_csv(args.train_genes)
    test_unadjusted_df = pl.read_csv(args.test_genes_unadjusted)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns]
    
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(common_genes).to_numpy()
    
    # Prepare labels
    print("Preparing labels...")
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
                labels_train[label_name] = train_vals.astype(float)
                labels_test[label_name] = test_vals.astype(float)
                label_types[label_name] = 'regression'
            else:
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
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train DBA with each training classifier (or load if already exists)
    dba_adjustments = {}
    
    for train_clf in args.training_classifiers:
        adjusted_file = output_dir / f"test_genes_dba_{train_clf}.csv"
        params_file = output_dir / f"dba_parameters_{train_clf}.csv"
        
        if adjusted_file.exists() and params_file.exists():
            print(f"\nLoading pre-trained DBA for {train_clf}...")
            adjusted_df = pl.read_csv(adjusted_file)
            X_test_adjusted = adjusted_df.select(common_genes).to_numpy()
            
            params_df = pl.read_csv(params_file)
            shifts = params_df['shift'].to_numpy()
            scales = params_df['scale'].to_numpy()
            
            dba_adjustments[train_clf] = {
                'shifts': shifts,
                'scales': scales,
                'X_test_adjusted': X_test_adjusted
            }
        else:
            print("\n" + "="*70)
            print(f"Training DBA with {train_clf}")
            print("="*70)
            
            shifts, scales, result = find_multi_label_alignment(
                X_train, X_test_unadjusted, labels_train, labels_test,
                label_types, optimize_scale=not args.shift_only,
                classifier=train_clf
            )
            
            X_test_adjusted = (X_test_unadjusted - shifts) / scales
            dba_adjustments[train_clf] = {
                'shifts': shifts,
                'scales': scales,
                'X_test_adjusted': X_test_adjusted
            }
            
            # Save adjusted data
            adjusted_df = pl.DataFrame({
                gene: X_test_adjusted[:, i] for i, gene in enumerate(common_genes)
            })
            adjusted_df.write_csv(adjusted_file)
            
            # Save parameters
            params_df = pl.DataFrame({
                'gene': common_genes,
                'shift': shifts,
                'scale': scales,
            })
            params_df.write_csv(params_file)
    
    # Evaluate each DBA adjustment with each evaluation classifier
    print("\n" + "="*70)
    print("Cross-Evaluation Matrix")
    print("="*70)
    
    comparison_data = []
    
    for train_clf in args.training_classifiers:
        X_test_adjusted = dba_adjustments[train_clf]['X_test_adjusted']
        
        for eval_clf in args.evaluation_classifiers:
            print(f"\nDBA trained with {train_clf}, evaluated with {eval_clf}:")
            
            results = evaluate_all_labels(
                X_train, X_test_adjusted,
                labels_train, labels_test, label_types,
                classifier=eval_clf
            )
            
            for label_name, label_data in results.items():
                comparison_data.append({
                    'training_classifier': train_clf,
                    'evaluation_classifier': eval_clf,
                    'label': label_name,
                    'metric': label_data['metric'],
                    'score': label_data['score']
                })
                print(f"  {label_name}: {label_data['score']:.3f} ({label_data['metric']})")
    
    # Save comparison results
    comparison_df = pl.DataFrame(comparison_data)
    comparison_df.write_csv(output_dir / "classifier_comparison.csv")
    
    # Create summary: average performance across all labels for each combination
    print("\n" + "="*70)
    print("Summary: Average Performance")
    print("="*70)
    print(f"{'Training Clf':<15} {'Eval Clf':<15} {'Avg Score':>10}")
    print("-" * 42)
    
    for train_clf in args.training_classifiers:
        for eval_clf in args.evaluation_classifiers:
            subset = [d for d in comparison_data 
                     if d['training_classifier'] == train_clf 
                     and d['evaluation_classifier'] == eval_clf
                     and not np.isnan(d['score'])]
            if subset:
                avg_score = np.mean([d['score'] for d in subset])
                print(f"{train_clf:<15} {eval_clf:<15} {avg_score:>10.3f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
