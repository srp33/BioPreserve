import pandas as pd
import os
import argparse

def print_now(*args, **kwargs):
    """Print a message to the console with flushing."""
    print(*args, flush=True, **kwargs)

def print_confusion_matrix(detailed_file, matrix_file):
    """Create a tab separated file with the averaged confusion matrix values per fold"""
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
    """Parse arguments and print the confusion matrix."""
    parser = argparse.ArgumentParser(description="Confusion matrix for gene expression data for ER status.")
    parser.add_argument('--output', required=True, help='Path for the detailed output CSV file.')
    parser.add_argument('--confusion-matrix', required=True, help='Path for the confusion matrices .txt file.')

    args = parser.parse_args()

    print_confusion_matrix(args.output, args.confusion_matrix)

    print_now("Printing finished.\n")
    print_now(f"Confusion matrix metrics are in: {args.confusion_matrix}")
    print_now("="*60)

if __name__ == "__main__":
    main()