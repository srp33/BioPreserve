import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
import sys
import os
# Add the parent directory (scripts) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.utils import HashCache
except ImportError:
    from utils import HashCache
from itertools import permutations

def print_now(*args, **kwargs):
    """Print a message to the console with flushing."""
    print(*args, flush=True, **kwargs)

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate a standard set of classification metrics."""
    # Return NaN for metrics that fail if only one class is present.
    if len(pd.unique(y_true)) < 2:
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'ROC AUC': np.nan,
            'Sensitivity (TPR)': np.nan,
            'Specificity (TNR)': np.nan,
            'MCC (Matthews Corr Coef)': np.nan,
        }

    # Use confusion matrix for robust calculation of TPR/TNR.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_proba),
        'Sensitivity (TPR)': (tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'Specificity (TNR)': (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'MCC (Matthews Corr Coef)': matthews_corrcoef(y_true, y_pred),
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }
    return metrics

def _process_fold(train_index, test_index, X, y, model, eval_sources, dataset_name):
    """
    Process a single cross-validation fold.
    
    Train a model, then evaluate it on the whole test set and, if specified,
    on subsets of the test set based on the `eval_sources` series.
    """
    clf_model = clone(model)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf_model.fit(X_train, y_train)

    # Define subsets for evaluation. Always evaluate on the full test set.
    subsets_to_evaluate = {dataset_name: (X_test, y_test)}

    # If sources are provided, also evaluate on subsets from each source.
    if eval_sources is not None:
        test_sources = eval_sources.iloc[test_index]
        for source_name in test_sources.unique():
            mask = (test_sources == source_name)
            subset_label = f"{dataset_name} on {source_name}"
            subsets_to_evaluate[subset_label] = (X_test[mask], y_test[mask])

    #{
        #"Combined": [full_x_test (1/3 rows, all gene columns), full y_test true er_status column]
        #Dataset 1: [subset of x test ()]
        # Dataset 2: 
    #}
    # Calculate metrics for each defined subset and return structured dicts.
    fold_results = []
    for name, (X_sub, y_sub) in subsets_to_evaluate.items():
        if len(y_sub) == 0:
            continue
        y_pred_sub = clf_model.predict(X_sub)
        y_proba_sub = clf_model.predict_proba(X_sub)[:, 1]
        metrics = calculate_metrics(y_sub, y_pred_sub, y_proba_sub)

        for metric_name, score in metrics.items():
            fold_results.append({
                "metric": metric_name,
                "dataset": name,
                "score": score
            })
    return fold_results

def run_analysis(df, dataset_name, train_source, test_source, output_file, adjustment_name, n_repeats, n_splits, repeat_offset, evaluate_by_source):
    """
    Run a complete cross-validation analysis for a given dataset.
    
    This function handles both standard CV and CV with evaluation on subsets
    (controlled by `evaluate_by_source`). Results are averaged per repeat
    and saved to a CSV file.
    """
    if n_repeats <= 0:
        return
    print_now(f"\nRunning classification for {dataset_name} with {n_repeats} repeats (offset: {repeat_offset})...")

    # Prepare data based on whether subsets will be evaluated.
    y = df['meta_er_status']
    eval_sources = df['meta_source'] if evaluate_by_source else None
    
    cols_to_drop = ['meta_er_status', 'meta_source']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).select_dtypes(include=[np.number])

    # Setup model and cross-validation.
    random_seed = 42 + repeat_offset
    model = HistGradientBoostingClassifier(max_iter=100, random_state=random_seed)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
    splits = list(cv.split(X, y))

    # Process all folds in parallel.
    fold_results_list = Parallel(n_jobs=-1)(
        delayed(_process_fold)(train_idx, test_idx, X, y, model, eval_sources, dataset_name) 
        for train_idx, test_idx in splits
    )

    # Aggregate results.
    all_fold_results = [item for sublist in fold_results_list for item in sublist]
    if not all_fold_results:
        print_now(f"No results generated for {dataset_name}. Skipping save.")
        return
    df_all_folds = pd.DataFrame(all_fold_results)

    # Add a 'repeat' column to enable grouping by repeat.
    fold_indices = np.repeat(np.arange(len(splits)), [len(f) for f in fold_results_list])
    df_all_folds['repeat'] = (fold_indices // n_splits) + repeat_offset

    # Group by repeat, metric, and dataset, then average scores over the splits.
    df_repeats = df_all_folds.groupby(['repeat', 'metric', 'dataset'])['score'].mean().reset_index()

    # Add metadata columns and format for saving.
    df_repeats.rename(columns={'score': 'value'}, inplace=True)
    df_repeats['train'] = f"{train_source}"
    df_repeats['test'] = f"{test_source}"
    df_repeats['classifier'] = 'HistGradientBoosting'
    df_repeats['adjuster'] = adjustment_name
    df_repeats['column'] = 'meta_er_status'
    
    # Reorder columns and append to the output file.
    output_cols = ['train', 'test', 'metric', 'classifier', 'adjuster', 'dataset', 'column', 'value']
    df_repeats[output_cols].to_csv(output_file, mode='a', header=False, index=False, float_format='%.4f')
    print_now(f"Classification results for {dataset_name} saved.")

def run_inter_source_analysis(df, train_source, test_source, output_file, adjustment_name, n_repeats, repeat_offset):
    """
    Train a model on one data source and test it on another.

    This function performs a direct train-test evaluation, repeated `n_repeats`
    times with different random seeds for the model to account for stochasticity.
    """
    if n_repeats <= 0:
        return

    dataset_name = f"Train on {train_source}, Test on {test_source}"
    print_now(f"\nRunning analysis for {dataset_name} with {n_repeats} repeats (offset: {repeat_offset})...")

    # Split data into train and test sets based on source.
    df_train = df[df['meta_source'] == train_source]
    df_test = df[df['meta_source'] == test_source]

    if df_train.empty or df_test.empty:
        print_now(f"Warning: Data for source '{train_source}' or '{test_source}' is empty. Skipping.")
        return

    # Prepare data for modeling.
    cols_to_drop = ['meta_er_status', 'meta_source']
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns]).select_dtypes(include=[np.number])
    y_train = df_train['meta_er_status']
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns]).select_dtypes(include=[np.number])
    y_test = df_test['meta_er_status']

    # Run the analysis for each repeat.
    results_list = []
    for i in range(n_repeats):
        current_repeat_num = repeat_offset + i
        model = HistGradientBoostingClassifier(max_iter=100, random_state=42 + current_repeat_num)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        for metric_name, score in metrics.items():
            results_list.append({
                "metric": metric_name,
                "dataset": dataset_name,
                "value": score
            })

    if not results_list:
        print_now(f"No results generated for {dataset_name}. Skipping save.")
        return

    # Format results into a DataFrame for saving.
    df_results = pd.DataFrame(results_list)
    df_results['train'] = f"{train_source}"
    df_results['test'] = f"{test_source}"
    df_results['classifier'] = 'HistGradientBoosting'
    df_results['adjuster'] = adjustment_name
    df_results['column'] = 'meta_er_status'
    
    # Reorder columns and append to the output file.
    output_cols = ['train', 'test', 'metric', 'classifier', 'adjuster', 'dataset', 'column', 'value']
    df_results[output_cols].to_csv(output_file, mode='a', header=False, index=False, float_format='%.4f')
    print_now(f"Analysis results for {dataset_name} saved.")

def generate_summary(detailed_file, summary_file):
    """Read detailed results and generate a summary with mean, std, and SEM."""
    print_now("\n" + "="*60 + "\nGENERATING SUMMARY METRICS")
    try:
        df = pd.read_csv(detailed_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print_now(f"Warning: Detailed results file '{detailed_file}' not found or empty. Cannot generate summary.")
        return

    # Group by experiment identifiers and calculate aggregate stats.
    summary = df.groupby(['adjuster', 'dataset', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    summary['sem'] = df.groupby(['adjuster', 'dataset', 'metric'])['value'].apply(lambda x: x.std() / np.sqrt(x.count()) if x.count() > 1 else 0.0).values
    
    summary.rename(columns={'dataset': 'evaluation'}, inplace=True)
    
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    summary.to_csv(summary_file, index=False, float_format='%.4f')
    print_now(f"Summary metrics saved to: {summary_file}")
    print_now("="*60)

def print_confusion_matrix(detailed_file, matrix_file):
    # Filter confusion matrix values from metrics CSV file
    metrics_full = pd.read_csv(detailed_file, index_col=False)
    matrix_values = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    metrics_filtered = metrics_full[metrics_full['metric'].isin(matrix_values)]
    
    # Create the .txt file
    os.makedirs(os.path.dirname(matrix_file), exist_ok=True)

    # Average the confusion matrix values between folds
    metrics_average = metrics_filtered.groupby(['train', 'test', 'metric'], as_index=False)['value'].mean()

    # Save to .txt file
    metrics_average.to_csv(matrix_file, sep='\t', index=False)

    print_now(f"Confusion matrix values saved to: {matrix_file}")

def main():
    """Parse arguments and run the classification pipeline."""
    parser = argparse.ArgumentParser(description="Run HistGradientBoosting classification on gene expression data for ER status.")
    parser.add_argument('--input-data', required=True, help='Path to the combined gene expression data file.')
    parser.add_argument('--output', required=True, help='Path for the detailed output CSV file.')
    parser.add_argument('--confusion-matrix', required=True, help='Path for the confusion matrices .txt file.')
    parser.add_argument('--summary', required=True, help='Path for the summary metrics CSV file.')
    parser.add_argument('--adjustment', default='Unadjusted', help='Name of the adjustment method used.')
    parser.add_argument('--cache-dir', default='./.cache', help='Directory to store cache files.')
    parser.add_argument('--force-rerun', action='store_true', help='Force re-computation even if cache is valid.')
    parser.add_argument('-n', '--n-repeats', type=int, default=10, help='Number of repeats for cross-validation.')
    parser.add_argument('--n-splits', type=int, default=3, help='Number of splits for cross-validation.')
    args = parser.parse_args()
        
    hash_cache = HashCache(
        hash_dir=args.cache_dir, 
        cache_filename=f"hgb_er_status_cache.json",
        write_over=args.force_rerun
    )

    # Ensure output file exists and has a header.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
        pd.DataFrame(columns=['metric', 'classifier', 'adjuster', 'dataset', 'column', 'value']).to_csv(args.output, index=False)
    
    print_now(f"=== HistGradientBoostingClassifier ER Status Classification ({args.adjustment}) ===")
    
    try:
        print_now(f"Loading combined data from {args.input_data}")
        df_combined = pd.read_csv(args.input_data)
        if 'meta_er_status' not in df_combined.columns or 'meta_source' not in df_combined.columns:
            raise ValueError("'meta_er_status' and 'meta_source' are required columns.")
    except Exception as e:
        print_now(f"Error loading or validating data: {e}")
        return
    
    source_names = df_combined['meta_source'].unique()
    
    # Define all classification runs.
    runs = []

    # 1. Standard cross-validation runs.
    runs.append({
        "type": "cv",
        "name": "Combined",
        "data": df_combined,
        "evaluate_by_source": True,
        "dataset_names": ['Combined'] + [f'Combined on {s}' for s in source_names]
    })
    if len(source_names) > 1:
        for name in source_names:
            runs.append({
                "type": "cv",
                "name": name,
                "data": df_combined[df_combined['meta_source'] == name],
                "evaluate_by_source": False,
                "dataset_names": [name]
            })

    # 2. Inter-source runs (train on one, test on another).
    if len(source_names) > 1:
        for train_src, test_src in permutations(source_names, 2):
            run_name = f"Train {train_src} - Test {test_src}"
            dataset_name = f"Train on {train_src}, Test on {test_src}"
            runs.append({
                "type": "inter_source",
                "name": run_name,
                "train": train_src ,
                "test": test_src,
                "train_source": train_src,
                "test_source": test_src,
                "dataset_names": [dataset_name]
            })

    # Execute all defined runs.
    # for run in runs:
    #     print_now("\n" + "="*60 + f"\nPREPARING RUN: {run['name']} ({args.adjustment})")
        
    #     key = f"{run['name']}|{os.path.basename(args.input_data)}|{args.adjustment}"
    #     run_identifier = {'adjuster': args.adjustment, 'dataset': run['dataset_names']}
    #     count_config = {
    #         'primary_grouping_col': 'dataset',
    #         'primary_grouping_val': run['dataset_names'][0],
    #         'count_col': 'metric'
    #     }

    #     with hash_cache.check_and_manage_repeats(
    #         key=key,
    #         input_paths=[args.input_data],
    #         n_repeats_requested=args.n_repeats,
    #         output_file=args.output,
    #         run_identifier=run_identifier,
    #         count_config=count_config
    #     ) as (action, n_to_run, n_existing):
    #         if action in ["RUN_FULL", "RUN_PARTIAL"]:
    #             if run['type'] == 'cv':
    #                 run_analysis(
    #                     df=run['data'],
    #                     dataset_name=run['name'],
    #                     train_source=run['train'],
    #                     test_source=run['test'],
    #                     output_file=args.output,
    #                     adjustment_name=args.adjustment,
    #                     n_repeats=n_to_run,
    #                     n_splits=args.n_splits,
    #                     repeat_offset=n_existing,
    #                     evaluate_by_source=run['evaluate_by_source']
    #                 )
    #             elif run['type'] == 'inter_source':
    #                 run_inter_source_analysis(
    #                     df=df_combined,
    #                     train_source=run['train_source'],
    #                     test_source=run['test_source'],
    #                     output_file=args.output,
    #                     adjustment_name=args.adjustment,
    #                     n_repeats=n_to_run,
    #                     repeat_offset=n_existing
    #                 )

    # hash_cache._save_hashes()
    generate_summary(args.output, args.summary)
    print_confusion_matrix(args.output, args.confusion_matrix)
    
    print_now("\nPipeline finished.")
    print_now(f"Detailed results are in: {args.output}")
    print_now(f"Summary metrics are in: {args.summary}")
    print_now("="*60)

if __name__ == "__main__":
    main()