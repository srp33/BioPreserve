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
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing evaluation results, don't re-evaluate")
    
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
    
    # If aggregate-only mode, just load existing evaluation results
    if args.aggregate_only:
        print("\n" + "="*70)
        print("Aggregate-Only Mode: Loading existing evaluation results")
        print("="*70)
        
        comparison_data = []
        
        # Look for evaluation JSON files in the parent directory structure
        # Pattern: outputs/classifier_results/{eval_clf}/DBA_Shift_Only_Trained_{train_clf}__{label}.json
        results_base = output_dir.parent.parent / "classifier_results"
        
        if not results_base.exists():
            print(f"ERROR: Results directory not found: {results_base}")
            print("Cannot use --aggregate-only mode without existing evaluation results")
            return
        
        for eval_clf_dir in results_base.iterdir():
            if not eval_clf_dir.is_dir():
                continue
            
            eval_clf_name = eval_clf_dir.name
            
            # Map classifier directory names to internal names
            clf_map = {
                'Random_Forest': 'randomforest',
                'Gradient_Boosting': 'histgradient',
                'Logistic_L2': 'logistic',
                'ElasticNet_l1_0.15': 'elasticnet'
            }
            eval_clf = clf_map.get(eval_clf_name, eval_clf_name.lower())
            
            # Find DBA evaluation files
            pattern = "DBA_Shift_Only_Trained_" if "shift_only" in str(output_dir) else "DBA_Shift_Scale_Trained_"
            
            for json_file in eval_clf_dir.glob(f"{pattern}*.json"):
                try:
                    import json
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract training classifier from filename
                    # Format: DBA_Shift_Only_Trained_{train_clf}__{label}.json
                    filename = json_file.stem
                    parts = filename.split('__')
                    if len(parts) == 2:
                        train_clf_part = parts[0].replace(pattern, '')
                        label = parts[1]
                        
                        comparison_data.append({
                            'training_classifier': train_clf_part,
                            'evaluation_classifier': eval_clf,
                            'label': data['label'],
                            'metric': data['metric'],
                            'score': data['score'] if data['score'] is not None else float('nan')
                        })
                except Exception as e:
                    print(f"  Warning: Could not load {json_file}: {e}")
        
        if not comparison_data:
            print("ERROR: No evaluation results found!")
            print(f"Looked in: {results_base}")
            return
        
        print(f"\nLoaded {len(comparison_data)} evaluation results")
        
        # Save comparison results
        comparison_df = pl.DataFrame(comparison_data)
        comparison_df.write_csv(output_dir / "classifier_comparison.csv")
        
        # Create summary
        print("\n" + "="*70)
        print("Summary: Average Performance")
        print("="*70)
        print(f"{'Training Clf':<15} {'Eval Clf':<15} {'Avg Score':>10}")
        print("-" * 42)
        
        for train_clf in sorted(set(d['training_classifier'] for d in comparison_data)):
            for eval_clf in sorted(set(d['evaluation_classifier'] for d in comparison_data)):
                subset = [d for d in comparison_data 
                         if d['training_classifier'] == train_clf 
                         and d['evaluation_classifier'] == eval_clf
                         and not np.isnan(d['score'])]
                if subset:
                    avg_score = np.mean([d['score'] for d in subset])
                    print(f"{train_clf:<15} {eval_clf:<15} {avg_score:>10.3f}")
        
        print("\n" + "="*70)
        print(f"Results saved to: {output_dir / 'classifier_comparison.csv'}")
        print("="*70)
        return
    
    # Original training/evaluation mode below
    # Train DBA with each training classifier (or load if already exists)
    dba_adjustments = {}
    
    # Track genes used by each DBA (they should all be the same, but check)
    dba_genes = None
    
    for train_clf in args.training_classifiers:
        adjusted_file = output_dir / f"test_genes_dba_{train_clf}.csv"
        params_file = output_dir / f"dba_parameters_{train_clf}.csv"
        
        if adjusted_file.exists() and params_file.exists():
            print(f"\nLoading pre-trained DBA for {train_clf}...")
            adjusted_df = pl.read_csv(adjusted_file)
            
            # Use genes from the DBA file (not common_genes from original inputs)
            dba_genes_this = [col for col in adjusted_df.columns if not col.startswith('meta_')]
            
            if dba_genes is None:
                dba_genes = dba_genes_this
                print(f"  Using {len(dba_genes)} genes from DBA outputs")
            elif set(dba_genes) != set(dba_genes_this):
                print(f"  WARNING: Gene mismatch! Expected {len(dba_genes)}, got {len(dba_genes_this)}")
                # Use intersection
                dba_genes = [g for g in dba_genes if g in dba_genes_this]
                print(f"  Using intersection: {len(dba_genes)} genes")
            
            X_test_adjusted = adjusted_df.select(dba_genes).to_numpy()
            
            params_df = pl.read_csv(params_file)
            # Use the direct optimization parameters (not closed-form)
            shifts = params_df['multi_label_dba_direct_shift'].to_numpy()
            scales = params_df['multi_label_dba_direct_scale'].to_numpy()
            
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
    
    # Subset X_train to match the genes used in DBA
    if dba_genes is not None:
        print(f"\nSubsetting X_train to {len(dba_genes)} genes used in DBA")
        gene_indices = [i for i, g in enumerate(common_genes) if g in dba_genes]
        X_train_subset = X_train[:, gene_indices]
    else:
        X_train_subset = X_train
    
    comparison_data = []
    
    for train_clf in args.training_classifiers:
        X_test_adjusted = dba_adjustments[train_clf]['X_test_adjusted']
        
        for eval_clf in args.evaluation_classifiers:
            print(f"\nDBA trained with {train_clf}, evaluated with {eval_clf}:")
            
            results = evaluate_all_labels(
                X_train_subset, X_test_adjusted,
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
