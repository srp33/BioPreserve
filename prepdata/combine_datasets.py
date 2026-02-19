# combine_datasets.py - Simplified version with gene ID conversion moved upstream

import argparse
import os
import pandas as pd

def combine_datasets_and_report(input1_path, input2_path, output_path):
    """Combine two datasets and return a single formatted status message."""
    try:
        # Load datasets
        df1 = pd.read_csv(input1_path, low_memory=False)
        df2 = pd.read_csv(input2_path, low_memory=False)
        
        # Validate required columns
        missing_er_status = [i for i, df in enumerate([df1, df2], 1) if 'meta_er_status' not in df.columns]
        if missing_er_status:
            return f"ERROR: 'meta_er_status' column missing in dataset(s) {', '.join(map(str, missing_er_status))}", False
        
        # Get common genes
        genes1 = [col for col in df1.columns if not col.startswith('meta_') and col != "Sample_ID"]
        genes2 = [col for col in df2.columns if not col.startswith('meta_') and col != "Sample_ID"]
        common_genes = list(set(genes1) & set(genes2))
        
        if not common_genes:
            return f"ERROR: No common genes found between datasets. Dataset 1 genes sample: {genes1[:10]}, Dataset 2 genes sample: {genes2[:10]}", False
        
        # Add source identifiers and combine
        df1['meta_source'] = os.path.basename(os.path.dirname(input1_path))
        df2['meta_source'] = os.path.basename(os.path.dirname(input2_path))
        
        keep_cols = common_genes + ['meta_er_status', 'meta_source']
        df_combined = pd.concat([df1[keep_cols], df2[keep_cols]], ignore_index=True)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_combined.to_csv(output_path, index=False)
        
        # Build status message
        msg = f"SUCCESS: {df_combined.shape[0]} samples, {len(common_genes)} genes"
        
        # Add warnings
        warnings = []
        if len(common_genes) < 10:
            warnings.append(f"very few genes ({len(common_genes)})")
            msg += f" | Common genes: {common_genes}"
        elif len(common_genes) < 100:
            warnings.append(f"low gene count ({len(common_genes)})")
        if df_combined.shape[0] < 20:
            warnings.append(f"small sample size ({df_combined.shape[0]})")
        
        if warnings:
            msg += f" ⚠️ {', '.join(warnings)}"
        
        return msg, True
        
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", False

def main():
    """Main function to parse arguments and orchestrate the dataset combination."""
    parser = argparse.ArgumentParser(description="Combine two gene expression datasets based on common genes.")
    parser.add_argument('--input1', required=True, help='Path to the first data file.')
    parser.add_argument('--input2', required=True, help='Path to the second data file.')
    parser.add_argument('--output', required=True, help='Path for the combined output CSV file.')

    args = parser.parse_args()

    message, success = combine_datasets_and_report(args.input1, args.input2, args.output)
    print(message, flush=True)
    
    return 0 if success else 1

if __name__ == "__main__":
    main()