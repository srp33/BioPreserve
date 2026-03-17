#!/usr/bin/env python3
"""
Aggregate individual JSON evaluation results and create heatmap.
Reads all JSON files from classifier results directory and creates visualization.
"""

import argparse
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_json_results(results_dir):
    """Load all JSON result files from directory."""
    results_dir = Path(results_dir)
    results = []
    
    for json_file in results_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def strip_prefix(adjuster_name):
    """Strip METHOD_ or ORACLE_ prefix from adjuster name for display."""
    # Handle the special oracle names that are already formatted
    if adjuster_name.startswith("ORACLE:"):
        return adjuster_name
    
    # Strip prefixes from file-based names
    if adjuster_name.startswith("METHOD_"):
        return adjuster_name[7:]  # Remove "METHOD_"
    elif adjuster_name.startswith("ORACLE_"):
        return adjuster_name[7:]  # Remove "ORACLE_"
    
    return adjuster_name


def format_adjuster_name(adjuster_name):
    """Format adjuster name for better display."""
    # Map internal names to display names
    name_map = {
        'unadjusted': 'Unadjusted',
        'bayesian_shift_scale': 'Bayesian (Shift+Scale)',
        'bayesian_effective_shift_only': 'Bayesian (Effective Shift-Only)',
        'multi_label_dba_shift_only': 'ORACLE: Multi-Label DBA (Shift-Only)',
        'multi_label_dba_shift_scale': 'ORACLE: Multi-Label DBA (Shift+Scale)',
        'aligned': 'Aligned',
        'mnn': 'MNN',
        'min_mean': 'Min-Mean',
        'log_transformed': 'Log-Transformed',
        'log_combat': 'Log-ComBat',
        'gmm': 'GMM',
    }
    
    # Strip prefix first
    clean_name = strip_prefix(adjuster_name)
    
    # If it's already a formatted DBA name (from the evaluation script), return as-is
    if clean_name.startswith("DBA:") or clean_name.startswith("ORACLE:"):
        return clean_name
    
    # Return mapped name or original
    return name_map.get(clean_name, clean_name)


def format_label_name(label):
    """Format metadata label for display."""
    label_map = {
        "meta_er_status": "ER status",
        "meta_menopause_status": "Menopause",
        "meta_sex": "Sex",
        "meta_age_at_diagnosis": "Age (R²)",
        "meta_chemotherapy": "Chemo",
        "meta_histological_type": "Hist. type",
        "meta_her2_status": "HER2",
        "meta_age_at_diagnosis_combined_lt50": "Age <50",
        "meta_age_at_diagnosis_combined_50_69": "Age 50-69",
        "meta_age_at_diagnosis_combined_ge70": "Age ≥70",
    }
    
    return label_map.get(label, label)


def create_heatmap(results_df, classifier_name, cv_results_df, output_path, output_txt_path=None):
    """Create heatmap from results DataFrame."""
    # Format adjuster names
    results_df = results_df.with_columns(
        pl.col('adjuster').map_elements(format_adjuster_name, return_dtype=pl.Utf8).alias('adjuster_display')
    )
    
    # Add CV results as a row
    if cv_results_df is not None:
        # Filter CV results for this classifier only
        cv_filtered = cv_results_df.filter(pl.col('classifier') == classifier_name)
        
        cv_rows = []
        for row in cv_filtered.iter_rows(named=True):
            cv_rows.append({
                'adjuster_display': 'CV Ceiling (Test Set)',
                'label': row['metadata_column'],
                'score': row['score']
            })
        
        cv_df = pl.DataFrame(cv_rows)
        
        # Combine with results
        results_with_cv = pl.concat([
            results_df.select(['adjuster_display', 'label', 'score']),
            cv_df
        ])
    else:
        results_with_cv = results_df.select(['adjuster_display', 'label', 'score'])
    
    # Pivot to create heatmap matrix
    pivot = results_with_cv.pivot(
        values="score",
        index="adjuster_display",
        on="label"
    )
    
    # Convert to pandas for plotting
    heatmap_df = pivot.to_pandas().set_index("adjuster_display")
    
    # Format column names
    heatmap_df.columns = [format_label_name(c) for c in heatmap_df.columns]
    
    # Sort rows: CV ceiling first, then all others by average performance
    cv_rows = heatmap_df[heatmap_df.index.str.contains("CV Ceiling", na=False)]
    non_cv_rows = heatmap_df[~heatmap_df.index.str.contains("CV Ceiling", na=False)]
    
    # Calculate average performance across all labels for sorting
    if len(non_cv_rows) > 0:
        # Compute mean across columns (labels), ignoring NaN values
        non_cv_rows = non_cv_rows.copy()
        non_cv_rows['_mean_score'] = non_cv_rows.mean(axis=1, skipna=True)
        non_cv_rows = non_cv_rows.sort_values('_mean_score', ascending=False)
        non_cv_rows = non_cv_rows.drop(columns=['_mean_score'])
    
    import pandas as pd
    heatmap_df = pd.concat([cv_rows, non_cv_rows])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(heatmap_df.columns) * 1.1), 
                                    max(3, len(heatmap_df) * 0.6 + 1.5)))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'MCC / R²'}
    )
    
    ax.set_title(
        f"Cross-Dataset Classification Performance: {classifier_name}\n"
        f"(MCC for categorical, R² for continuous metadata)",
        pad=15,
        fontsize=12
    )
    ax.set_xlabel("Metadata Label", fontsize=10)
    ax.set_ylabel("Adjustment Method", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved heatmap to: {output_path}")
    
    # Save text table if path provided
    if output_txt_path:
        with open(output_txt_path, 'w') as f:
            f.write(f"Cross-Dataset Classification Performance: {classifier_name}\n")
            f.write(f"(MCC for categorical, R² for continuous metadata)\n")
            f.write("=" * 120 + "\n\n")
            
            # Get column widths
            col_width = 10
            method_width = max(35, max(len(str(idx)) for idx in heatmap_df.index) + 2)
            
            # Header
            header = f"{'Method':<{method_width}}"
            for col in heatmap_df.columns:
                header += f"{col:>{col_width}}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Data rows
            for idx, row in heatmap_df.iterrows():
                line = f"{idx:<{method_width}}"
                for val in row:
                    if np.isnan(val):
                        line += f"{'NaN':>{col_width}}"
                    else:
                        line += f"{val:>{col_width}.2f}"
                f.write(line + "\n")
            
            f.write("\n")
        
        print(f"  Saved text table to: {output_txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate JSON results and create heatmap"
    )
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing JSON result files")
    parser.add_argument("--classifier", required=True,
                       help="Classifier name for plot title")
    parser.add_argument("--cv-results", required=False,
                       help="Path to CV results CSV file")
    parser.add_argument("--output-heatmap", required=True,
                       help="Output path for heatmap PNG")
    parser.add_argument("--output-csv", required=True,
                       help="Output path for aggregated CSV")
    parser.add_argument("--output-txt", required=False,
                       help="Output path for text table (optional)")
    
    args = parser.parse_args()
    
    print(f"\nAggregating results for: {args.classifier}")
    print(f"  Reading from: {args.results_dir}")
    
    # Load all JSON results
    results = load_json_results(args.results_dir)
    
    if not results:
        print(f"  WARNING: No JSON files found in {args.results_dir}")
        # Create empty outputs
        empty_df = pl.DataFrame({
            'adjuster': [],
            'label': [],
            'classifier': [],
            'score': [],
            'metric': []
        })
        empty_df.write_csv(args.output_csv)
        
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', fontsize=14)
        ax.set_title(f"No results for {args.classifier}")
        plt.savefig(args.output_heatmap, dpi=150, bbox_inches="tight")
        plt.close()
        return
    
    print(f"  Found {len(results)} result files")
    
    # Convert to DataFrame
    results_df = pl.DataFrame(results)
    
    # Save aggregated CSV
    results_df.write_csv(args.output_csv)
    print(f"  Saved CSV to: {args.output_csv}")
    
    # Load CV results if provided
    cv_results_df = None
    if args.cv_results:
        cv_results_df = pl.read_csv(args.cv_results)
        print(f"  Loaded CV results from: {args.cv_results}")
    
    # Create heatmap
    output_txt = getattr(args, 'output_txt', None)
    create_heatmap(results_df, args.classifier, cv_results_df, args.output_heatmap, output_txt)
    
    print(f"  Complete!")


if __name__ == "__main__":
    main()
