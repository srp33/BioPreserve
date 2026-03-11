import argparse
import os
import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
from filelock import FileLock

RANDOM_SEED = 42

# Add the parent directory (scripts) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_write_to_csv(df, output_file):
    lock_path = output_file + '.lock'
    with FileLock(lock_path):
        df.to_csv(output_file, mode='a', header=False, index=False, float_format='%.4f')

def print_now(*args, **kwargs):
    """Print a message to the console with flushing."""
    print(*args, flush=True, **kwargs)

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate a standard set of classification metrics."""
    # PENDING: Handle both pandas Series and numpy arrays. Might fix AttributeError.
    if isinstance(y_pred, np.ndarray):
        # For numpy arrays, use pandas isnull to handle both numeric and non-numeric data
        y_pred_cleaned = y_pred[~pd.isnull(y_pred)]
    else:
        # For pandas Series, use the dropna() method
        y_pred_cleaned = y_pred.dropna()

    # Use confusion matrix for robust calculation of TPR/TNR
    print_now(f"DEBUG calculate_metrics: y_true dtype={type(y_true)} shape={getattr(y_true, 'shape', 'no shape')} unique={np.unique(y_true)}")
    print_now(f"DEBUG calculate_metrics: y_pred dtype={type(y_pred)} shape={getattr(y_pred, 'shape', 'no shape')} unique={np.unique(y_pred)}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    # Return NaN for metrics that fail if only one class is present.
    if len(pd.unique(y_true)) < 2 or len(pd.unique(y_pred_cleaned)) < 2:
        return {
            'ROC AUC': 0.5,
            'True Negative': tn,
            'False Positive': fp,
            'False Negative': fn,
            'True Positive': tp
        }

    metrics = {
        'ROC AUC' : roc_auc_score(y_true, y_proba),
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }

    return metrics

def run_single_dataset(filepath, source, output_file, pred_col, adjustment, classifier, clf_model, n_splits, current_repeat):
    """Generate metrics for a single dataset."""
    # Load pandas dataframe
    df = pd.DataFrame()
    df = load_dataframe(filepath, pred_col) # Give df an index - double check

    # Separate data into training and testing
    y = df[pred_col]
    print_now(f"DEBUG single_dataset: y dtype={y.dtype} unique={y.unique()}")
    cols_to_drop = [pred_col]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).select_dtypes(include=[np.number])

    # Reset index to ensure continuous indexing for CV splits
    df_reset = df.reset_index(drop=True)
    y = df_reset[pred_col]
    X = df_reset.drop(columns=[col for col in cols_to_drop if col in df_reset.columns]).select_dtypes(include=[np.number])

    # Create dataframe for y predictions
    predictions = pd.DataFrame(index=df_reset.index, columns=['y_predicted', 'y_probability'])

    # Clone the model
    model = clone(clf_model)
    cv_random_seed = RANDOM_SEED + current_repeat
    cv = RepeatedStratifiedKFold(n_splits=n_splits, random_state=cv_random_seed)
    try:
        splits = list(cv.split(X,y))
    except ValueError as e:
        print_now(f"Error splitting data: {e} for {filepath}")
        print_now(f"y shape: {y.shape} for {filepath}")
        print_now(f"y unique: {y.unique()} for {filepath}")
        print_now(f"X shape: {X.shape} for {filepath}")
        raise e

    # Fit the model, test, and calculate metrics
    for train_index, test_index in splits:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
    
        # Predict off of x test to get y test predictions
        y_pred = model.predict(X_test)
        print_now(f"DEBUG single_dataset: y_pred dtype={type(y_pred)} shape={y_pred.shape} unique={np.unique(y_pred)}")
        predictions.loc[test_index, 'y_predicted'] = y_pred

        # Take the second column of probabilistic predictions from x test
        y_proba = model.predict_proba(X_test)[:,1]
        predictions.loc[test_index, 'y_probability'] = y_proba

    print_now(f"Shapes: Y : {y.shape} Y pred: {y_pred.shape} Y proba: {y_proba.shape} Uniques: Y : {np.unique(y)} Y pred: {np.unique(y_pred)} for dataset: {filepath}")

    if y.isnull().any():
        print_now(f"Dtype: {y.dtype}")
        # PENDING: Use pandas .isnull() to avoid TypeError on non-numeric data.
        raise ValueError(f"Found {y.isnull().sum()} NaN(s) in y for dataset: {filepath}")
    
    y_predicted_series = predictions['y_predicted']
    if y_predicted_series.isnull().any():
        print_now(f"Dtype: {y_predicted_series.dtype}")
        # PENDING: Use pandas .isnull() for the Series, not the temporary y_pred variable.
        raise ValueError(f"Found {y_predicted_series.isnull().sum()} NaN(s) in y_pred for dataset: {filepath}")

    # PENDING: Convert to a numeric type that supports integers and NaNs. Might fix ValueError.
    y_predicted_series = y_predicted_series.astype(float).astype('Int64')

    # Generate the metrics from y, y predictions, and y probabilistic predictions
    metrics = calculate_metrics(y, y_predicted_series, predictions['y_probability'])
    metrics_df = pd.DataFrame([metrics])

    # Add other columns
    
    metrics_df['Classifier'] = classifier
    metrics_df['Adjustment'] = adjustment
    metrics_df['Prediction'] = pred_col
    metrics_df['Train'] = source
    metrics_df['Test'] = source
    # Use parent directory + filename for uniqueness
    path_parts = Path(filepath).parts[-2:]  # Get last 2 parts (dir/file.csv)
    file_id = "_".join(path_parts).replace("/", "_").replace(".", "_")
    metrics_df['Run_ID'] = f"single_{file_id}_{current_repeat}"

    # Reorder columns and append to output file
    output_cols = ['Train', 'Test', 'ROC AUC', 'True Negative', 'False Negative', 'False Positive', 'True Positive', 
    'Classifier', 'Adjustment', 'Prediction', 'Run_ID']
    safe_write_to_csv(metrics_df[output_cols], output_file)
    print_now(f"Classification results for {filepath} saved.")



def run_paired_datasetsset(filepath, output_file, pred_col, source_col, adjustment, classifier, clf_model, n_splits, current_repeat):
    """Generate metrics for combined datasets
    with each training and testing combination."""
    # Load pandas dataframe
    df = load_dataframe(filepath, pred_col)

    # Ensure source column has two unique values
    sources = df[source_col].unique()
    if len(sources) != 2:
        raise ValueError(f"Expected exactly 2 unique sources in '{source_col}', found: {sources}")
    source1, source2 = sources

    # Define train/test combinations
    combinations = [
        (f'{source1};{source2}', f'{source1};{source2}'),
        (source1, source2),
        (source2, source1),
        (source1, source1),
        (source2, source2)
    ]

    # Drop unwanted columns to get features
    meta_columns = [col for col in df.columns if "meta_" in col]
    cols_to_drop = [pred_col, source_col] + meta_columns
    feature_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).select_dtypes(include=[np.number])

    results = []

    for train_key, test_key in combinations:
        # Select training data
        if ';' in train_key:
            train_df = df
        else:
            train_df = df[df[source_col] == train_key]

        # Select testing data
        if ';' in test_key:
            test_df = df
        else:
            test_df = df[df[source_col] == test_key]

        should_split = test_key in train_key

        # Get features and targets
        X_train = feature_df.loc[train_df.index]
        y_train = train_df[pred_col]

        X_test = feature_df.loc[test_df.index]
        y_test = test_df[pred_col]
    
        # Create empty dataframe for y predictions
        predictions = pd.DataFrame(index=df.index, columns=['y_predicted', 'y_probability'])

        # Clone model
        model = clone(clf_model)

        if should_split:
            # Use cross-validation splitting on the training data
            cv_random_seed = RANDOM_SEED + current_repeat
            cv = RepeatedStratifiedKFold(n_splits=n_splits, random_state=cv_random_seed)
            splits = list(cv.split(X_train, y_train))

            # Fit the model, test, and calculate metrics using cross-validation
            for train_index, val_index in splits:
                X_train_fold = X_train.iloc[train_index]
                X_val_fold = X_train.iloc[val_index]
                y_train_fold = y_train.iloc[train_index]
                y_val_fold = y_train.iloc[val_index]
                model.fit(X_train_fold, y_train_fold)
            
                # Predict on validation fold
                y_pred = model.predict(X_val_fold)
                predictions.loc[y_val_fold.index, 'y_predicted'] = y_pred

                # Take the second column of probabilistic predictions
                y_proba = model.predict_proba(X_val_fold)[:,1]
                predictions.loc[y_val_fold.index, 'y_probability'] = y_proba

            # Use the training data indices for evaluation (since we did CV on training data)
            y_true = y_train
            y_pred_all = predictions.loc[y_train.index, 'y_predicted'].values
            y_proba_all = predictions.loc[y_train.index, 'y_probability'].values
        else:
            # Fit the model and create predictions without cross-validation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            predictions.loc[y_test.index, 'y_predicted'] = y_pred
            predictions.loc[y_test.index, 'y_probability'] = y_proba

            y_true = y_test
            y_pred_all = y_pred
            y_proba_all = y_proba

        print_now(f"Shapes: Y true: {y_true.shape} Y pred: {y_pred_all.shape} Y proba: {y_proba_all.shape} Uniques: Y true: {np.unique(y_true)} Y pred: {np.unique(y_pred_all)} for dataset: {filepath}")

        # Define subsets for evaluation - always evaluate on full dataset
        subsets_to_evaluate = {test_key: (y_true, y_pred_all, y_proba_all)}
        
        # If we're evaluating combined datasets, also evaluate on individual source subsets
        if ';' in test_key and should_split:
            # Get the source information for the evaluation data
            eval_sources = df.loc[y_true.index, source_col] if should_split else test_df[source_col]
            for source_name in eval_sources.unique():
                mask = (eval_sources == source_name)
                subset_label = source_name
                subsets_to_evaluate[subset_label] = (
                    y_true[mask], 
                    y_pred_all[mask] if isinstance(y_pred_all, pd.Series) else y_pred_all[mask.values],
                    y_proba_all[mask] if isinstance(y_proba_all, pd.Series) else y_proba_all[mask.values]
                )

        # Calculate metrics for each subset
        for subset_name, (y_true_sub, y_pred_sub, y_proba_sub) in subsets_to_evaluate.items():
            if len(y_true_sub) == 0:
                continue
                
            # Ensure consistent data types for metrics calculation
            try:
                y_true_clean = y_true_sub.astype(int)
                y_pred_clean = y_pred_sub.astype(int)
            except (ValueError, TypeError) as e:
                print_now(f"Could not convert to int, using original data: {e}")
                y_true_clean = y_true_sub
                y_pred_clean = y_pred_sub
            
            # Calculate metrics
            metrics = calculate_metrics(y_true_clean, y_pred_clean, y_proba_sub)
            metrics_df = pd.DataFrame([metrics])
        
            # Add other columns
            metrics_df['Classifier'] = classifier
            metrics_df['Adjustment'] = adjustment
            metrics_df['Prediction'] = pred_col
            metrics_df['Train'] = train_key
            metrics_df['Test'] = subset_name  # Use subset name instead of test_key
            # Use parent directory + filename for uniqueness
            path_parts = Path(filepath).parts[-2:]  # Get last 2 parts (dir/file.csv)
            file_id = "_".join(path_parts).replace("/", "_").replace(".", "_")
            metrics_df['Run_ID'] = f"combined_{file_id}_{current_repeat}"

            # Reorder columns and append to results
            output_cols = ['Train', 'Test', 'ROC AUC', 'True Negative', 'False Negative', 'False Positive', 'True Positive', 
            'Classifier', 'Adjustment', 'Prediction', 'Run_ID']
            results.append(metrics_df[output_cols].copy())
    
    # Make sure the whole run completes before writing to the file.
    for metrics_df_selection in results:
        safe_write_to_csv(metrics_df_selection, output_file)

    print_now(f"All classification results for {filepath} saved.")

def load_dataframe(filename, pred_col):
    """Read the file into a pandas dataframe and check it has the required columns."""
    output_parts = [f"Loading {filename}"]
    
    try:
        # Check file size first
        file_size = os.path.getsize(filename)
        output_parts.append(f"File size: {file_size} bytes")
        
        if file_size == 0:
            output_parts.append("ERROR: File is empty (0 bytes)")
            print_now(" | ".join(output_parts))
            raise ValueError(f"File {filename} is empty")
        
        df = pd.read_csv(filename, low_memory=False)

        if pred_col not in df.columns :
            meta_cols = [col for col in df.columns if col.startswith('meta_')]
            raise ValueError(f"{pred_col} not found in dataframe. Meta columns: {meta_cols}")

        # DEBUG: Check original data
        output_parts.append(f"Original {pred_col} dtype={df[pred_col].dtype}, unique values={df[pred_col].unique()}")

        # Remove all non [0,1] values (This should be done in the preprocessing script)
        original_size = len(df)
        df = df[(df[pred_col] == 0) | (df[pred_col] == 1)]
        if len(df) < original_size:
            output_parts.append(f"WARNING: Removed {original_size - len(df)} rows with non-binary values in {pred_col}. After filtering, shape={df.shape}, {pred_col} dtype={df[pred_col].dtype}, unique values={df[pred_col].unique()}")
        
        print_now(" | ".join(output_parts))
        return df
        
    except Exception as e:
        output_parts.append(f"ERROR: {str(e)}")
        # Force flush the error message before re-raising
        error_msg = " | ".join(output_parts)
        print_now(error_msg)
        sys.stdout.flush()
        sys.stderr.flush()
        raise

def generate_runs(single_list, single_names, combined_list, n_repeats):
    """Return a list of dictionaries of run parameters for each combination of datasets."""
    runs = []
    for current_repeat in range(n_repeats):
        for source, filepath in zip(single_names, single_list):
            # Use parent directory + filename for uniqueness
            path_parts = Path(filepath).parts[-2:]  # Get last 2 parts (dir/file.csv)
            file_id = "_".join(path_parts).replace("/", "_").replace(".", "_")
            runs.append({
                "type": "single",
                "filename": filepath,
                "source": source,
                "current_repeat": current_repeat,
                "run_id": f"single_{file_id}_{current_repeat}"
            })

        for filepath in combined_list:
            # Use parent directory + filename for uniqueness
            path_parts = Path(filepath).parts[-2:]  # Get last 2 parts (dir/file.csv)
            file_id = "_".join(path_parts).replace("/", "_").replace(".", "_")
            runs.append({
                "type": "combined",
                "filename": filepath,
                "current_repeat": current_repeat,
                "run_id": f"combined_{file_id}_{current_repeat}"
            })

    return runs

def filter_runs(runs, output_file):
    """Filter out runs that have already been completed."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print_now("No existing output file found. Running all runs.")
        return runs
    
    try:
        existing_df = pd.read_csv(output_file, header=0, names=['Train', 'Test', 'ROC AUC', 'True Negative', 'False Negative', 'False Positive', 'True Positive', 'Classifier', 'Adjustment', 'Prediction', 'Run_ID'])
        completed_ids = set(existing_df.get('Run_ID', []))
        filtered_runs = [run for run in runs if run['run_id'] not in completed_ids]
        print_now(f"Filtered {len(runs)} runs down to {len(filtered_runs)} remaining runs.")
        return filtered_runs
    except Exception as e:
        print_now(f"Error reading output file for filtering: {e}. Running all runs.")
        return runs

def execute_run(args, run, model):
    """Perform a single run based on its type."""
    if run['type'] == 'single':
        run_single_dataset(
            filepath=run['filename'],
            source=run['source'],
            output_file=args.output,
            pred_col=args.prediction_column,
            adjustment=args.adjustment,
            classifier=args.classifier,
            clf_model=model,
            n_splits=args.n_splits,
            current_repeat=run['current_repeat']
        )

    elif run['type'] == 'combined':
        run_paired_datasetsset(
            filepath=run['filename'],
            output_file=args.output,
            pred_col=args.prediction_column,
            source_col=args.source_column,
            adjustment=args.adjustment,
            classifier=args.classifier,
            clf_model=model,
            n_splits=args.n_splits,
            current_repeat=run['current_repeat']
        )

def initialize_model(classifier):
    random_seed = RANDOM_SEED
    if classifier == 'HistGradientBoosting':
        model = HistGradientBoostingClassifier(max_iter=100, random_state=random_seed)
    elif classifier == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=random_seed)
    else:
        print_now(f"{classifier} is not currently recognized. Please try 'HistGradientBoosting' or 'RandomForest'.")

    return model

def main():
    """Parse arguments and run the classification pipeline."""
    parser = argparse.ArgumentParser(description="Run HistGradientBoosting classification on gene expression for ER status.")
    parser.add_argument('--combined-list', nargs='*', required=True, help='List of combined gene expression data filenames.')
    parser.add_argument('--single-list', nargs='*', required=True, help='List of individual gene expression data filenames.')
    parser.add_argument('--single-source-names', nargs='*', required=True, help='List of dataset names for single source files.')
    parser.add_argument('--output', required=True, help='Path for the detailed output CSV file.')
    parser.add_argument('--adjustment', required=True, help='Type of adjustment done to CSV file.')
    parser.add_argument('--classifier', default= 'HistGradientBoosting', help='Name of classifier algorithm used.')
    parser.add_argument('--prediction-column', default='meta_er_status', help='Name of the prediction column, y.')
    parser.add_argument('--source-column', default='meta_source', help='Name of the source column.')
    parser.add_argument('--n', '--n-repeats', type=int, default=10, help='Number of repeats for cross-validation.')
    parser.add_argument('--n-splits', type=int, default=3, help='Number of splits for cross-validation.')
    parser.add_argument('--force-rerun', action='store_true', help='Force re-computation even if cache is valid.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for parallel execution.')
    args = parser.parse_args()

    # Create output file from output_file path and create header
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
        pd.DataFrame(columns=['Train', 'Test', 'ROC AUC', 'True Negative', 'False Negative', 'False Positive', 'True Positive', 
    'Classifier', 'Adjustment', 'Prediction', 'Run_ID']).to_csv(args.output, index=False)
    
    print_now(f"=== HistGradientBoostingClassifier ER Status Classification ({args.adjustment}) ===")

    # Set up model
    model = initialize_model(args.classifier)
    
    # Generate a list of dictionaries with the parameters for each run
    runs = generate_runs(args.single_list, args.single_source_names, args.combined_list, args.n)

    if not args.force_rerun:
        runs = filter_runs(runs, args.output)

    # Execute runs in parallel
    n_jobs = min(len(runs), os.cpu_count() or 1, args.num_workers)
    print_now(f"Running {len(runs)} runs in parallel with {n_jobs} jobs. args.num_workers: {args.num_workers}. os.cpu_count: {os.cpu_count()}")
    if runs:
        Parallel(n_jobs=n_jobs)(
            delayed(execute_run)(args, run, model) for run in runs
        )
    print_now("\nPipeline finished.")
    print_now(f"Detailed results are in: {args.output}")
    print_now("="*60)

if __name__ == "__main__":
    main()

