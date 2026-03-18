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
        df = pd.read_csv(f, index_col=0, low_memory=False)
        df.index.name = "meta_Sample_ID"
        df = df.reset_index()
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
    final_cols = sorted(all_meta_cols) + sorted(shared_expr_cols)

    print_now(f"Using {len(final_cols)} columns ( {len(all_meta_cols)} meta + {len(shared_expr_cols)} shared expression).")

    # Fill in missing columns for each df
    dfs = [df.reindex(columns=final_cols) for df in dfs]

    combined_df = pd.concat(dfs, ignore_index=True)

    # DEBUG: Confirm presence of meta_Sample_ID as a column
    print_now("meta_Sample_ID head:", combined_df["meta_Sample_ID"].head())
    print_now("Sample ID NA values: ", combined_df["meta_Sample_ID"].isna().sum())

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