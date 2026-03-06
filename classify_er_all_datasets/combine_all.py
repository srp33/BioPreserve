import pandas as pd
from pathlib import Path
import argparse

def print_now(*args, **kwargs):
    """Prints immediately to stdout (e.g., for real-time CLI feedback)."""
    print(*args, flush=True, **kwargs)

def combine_gold_unadjusted_files(gold_dir: Path, output_file: Path):
    """
    Combines all 'unadjusted.csv' files from GSE subfolders in 'gold_dir'.
    Each file gets a 'meta_source' column from its parent folder (e.g., GSE1234).
    Saves combined file to 'output_file'.
    """

    # Collect GSE folders
    gse_files = list(gold_dir.glob("gse*/unadjusted.csv"))
    metabric_file = list(gold_dir.glob("metabric/unadjusted.csv"))
    drop_files = ['gse115577', 'gse123845', 'gse163882']
    gse_files = [x for x in gse_files if x.parent.name not in drop_files]

    unadjusted_files = gse_files + metabric_file
    
    if not unadjusted_files:
        raise FileNotFoundError(f"No unadjusted.csv files found in {gold_dir}/GSE*/")

    dfs = []

    for f in unadjusted_files:
        df = pd.read_csv(f, low_memory=False)
        gse_id = f.parent.name
        df['meta_source'] = gse_id
        print_now(f"Loaded {f} with shape {df.shape}")
        dfs.append(df)

    # Shared expression columns
    expr_cols_sets = [set(c for c in df.columns if not c.startswith('meta_')) for df in dfs]
    shared_expr_cols = set.intersection(*expr_cols_sets)

    # All meta columns
    all_meta_cols = set()
    for df in dfs:
        meta_cols = {c for c in df.columns if c.startswith('meta_')}
        all_meta_cols.update(meta_cols)
        
    # Final columns = shared expression + all meta
    final_cols = list(shared_expr_cols.union(all_meta_cols))


    print_now(f"Using {len(final_cols)} columns (meta + shared expression).")

    # Fill in missing columns for each df
    for i, df in enumerate(dfs):
        missing = set(final_cols) - set(df.columns)
        for col in missing:
            df[col] = pd.NA
        dfs[i] = df[final_cols]

    combined_df = pd.concat(dfs, ignore_index=True)

    # DEBUG: Confirm the meta_source column exists and has expected GSE IDs
    print_now("Unique meta_source values:", combined_df['meta_source'].unique())

    # DEBUG: Checking for meta columns and their values
    meta_cols = [col for col in combined_df.columns if col.startswith("meta_")]
    for col in meta_cols:
        print_now(f"{col}: unique values ->", combined_df[col].unique()[:10])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print_now(f"Saved combined data to {output_file} with shape {combined_df.shape}")

def main():
    parser = argparse.ArgumentParser(description="Combine all unadjusted.csv files from gold directory.")
    parser.add_argument('--input-dir', type=Path, required=True, help='Path to gold directory containing GSE subfolders')
    parser.add_argument('--output-file', type=Path, required=True, help='Path to save the combined CSV output')
    args = parser.parse_args()

    combine_gold_unadjusted_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()