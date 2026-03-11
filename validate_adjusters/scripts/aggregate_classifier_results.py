#!/usr/bin/env python3
"""
Aggregate individual evaluation results and create heatmap.
"""

import argparse
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_heatmap(results_df, classifier_name, output_path, cv_results_df=None):
    """Create heatmap from aggregated results."""
    # Pivot to create matrix
    pivot_df = results_df.pivot(
        index='adjuster',
        columns='label',
        values='score'
    )
    
    # Convert to numpy for plotting
    matrix = pivot_df.to_numpy()
    adjusters = pivot_df['adjuster'].to_list()
    labels = [col for col in pivot_df.columns if col != 'adjuster']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), max(8, len(adjusters) * 0.4)))
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-0.5,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=adjusters,
        cbar_kws={'label': 'Score (MCC or R²)'},
        ax=ax
    )
    
    ax.set_title(f'{classifier_name} Performance Across Adjusters and Labels', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metadata Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjuster', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-files", nargs='+', required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--output-heatmap", required=True)
    parser.add_argument("--output-csv", required=True)
    
    args = parser.parse_args()
    
    # Load all results
    all_results = []
    for result_file in args.result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    if not all_results:
        print("Error: No results loaded")
        return
    
    # Create DataFrame
    results_df = pl.DataFrame(all_results)
    
    # Save CSV
    results_df.write_csv(args.output_csv)
    print(f"Results saved to: {args.output_csv}")
    
    # Create heatmap
    create_heatmap(results_df, args.classifier, args.output_heatmap)


if __name__ == "__main__":
    main()
