import pandas as pd
import numpy as np
import os
import argparse

def print_now(*args, **kwargs):
    """Print a message to the console with flushing."""
    print(*args, flush=True, **kwargs)
    
    
def generate_summary(detailed_file, summary_file):
    """Read detailed results and generate a summary with mean, std, and SEM."""
    print_now("\n" + "="*60 + "\nGENERATING SUMMARY METRICS")
    try:
        df = pd.read_csv(detailed_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print_now(f"Warning: Detailed results file '{detailed_file}' not found or empty. Cannot generate summary.")
        return

    # Columns to aggregate (metrics)
    metric_cols = ['ROC AUC', 'True Negative', 'False Positive', 'False Negative', 'True Positive']

    # Grouping columns 
    group_cols = ['Train', 'Test', 'Classifier', 'Adjustment', 'Prediction']

    # Aggregate
    summary_mean = df.groupby(group_cols)[metric_cols].mean().reset_index()
    summary_std = df.groupby(group_cols)[metric_cols].std().reset_index()
    summary_sem = df.groupby(group_cols)[metric_cols].sem().reset_index()

    # Rename columns
    summary_mean = summary_mean.rename(columns={col: f"{col} Mean" for col in metric_cols})
    summary_std = summary_std.rename(columns={col: f"{col} Std" for col in metric_cols})
    summary_sem = summary_sem.rename(columns={col: f"{col} SEM" for col in metric_cols})

    summary = summary_mean.merge(summary_std, on=group_cols).merge(summary_sem, on=group_cols)

    # Save summary
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    summary.to_csv(summary_file, index=False, float_format='%.4f')

    print_now(f"Summary metrics saved to: {summary_file}")
    print_now("="*60)

def main():
    """Parse arguments and generate the summary file."""
    parser = argparse.ArgumentParser(description="Confusion matrix for gene expression data for ER status.")
    parser.add_argument('--output', required=True, help='Path for the detailed output CSV file.')
    parser.add_argument('--summary', required=True, help='Path for the summary CSV file.')

    args = parser.parse_args()
    
    generate_summary(args.output, args.summary)

    print_now(f"Summary metrics are in: {args.summary}")
    print_now("="*60)

if __name__ == "__main__":
    main()