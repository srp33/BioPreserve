import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, linregress, f_oneway


def run_tests(file_path, meta_cols, outdir):
    df = pd.read_csv(file_path)

    # Validate metadata columns
    for col in meta_cols:
        if col not in df.columns:
            raise ValueError(f"Metadata column '{col}' not found in {file_path}")

    metadata = df[meta_cols]

    # Identify gene columns
    gene_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("meta")]
    genes = df[gene_cols].apply(pd.to_numeric, errors='coerce')

    # Output DataFrames
    tstat_df = pd.DataFrame(index=gene_cols)
    slope_df = pd.DataFrame(index=gene_cols)
    fstat_df = pd.DataFrame(index=gene_cols)
    pval_df = pd.DataFrame(index=gene_cols)

    base_name = Path(file_path).stem
    os.makedirs(outdir, exist_ok=True)

    for meta_col in meta_cols:
        labels = metadata[meta_col].dropna()

        # Align genes with non-null metadata
        valid_idx = labels.index
        gene_data = genes.loc[valid_idx]
        labels = labels.loc[valid_idx]

        unique_vals = labels.unique()

        # -------------------------
        # Binary → Welch t-test
        # -------------------------
        if len(unique_vals) == 2:
            group1 = gene_data[labels == unique_vals[0]]
            group2 = gene_data[labels == unique_vals[1]]

            if group1.shape[0] < 2 or group2.shape[0] < 2:
                print(f"[Skipping t-test] {meta_col} (not enough samples)")
                continue

            t_stats, p_vals = ttest_ind(
                group1,
                group2,
                axis=0,
                equal_var=False,
                nan_policy="omit"
            )

            tstat_df[meta_col] = t_stats
            pval_df[meta_col] = p_vals

        # ------------------------
        # Multi-class -> ANOVA
        # ------------------------
        elif all(isinstance(x, int) for x in unique_vals):
            f_stats = []
            pvals_list = []

            for gene in gene_cols:
                groups = [gene_data.loc[labels == val, gene].dropna().values for val in unique_vals]
                groups = [g for g in groups if len(g) > 0]

                if len(groups) < 2:
                    f_stats.append(np.nan)
                    pvals_list.append(np.nan)
                    continue

                try:
                    f_stat, p_val = f_oneway(*groups)
                    f_stats.append(f_stat)
                    pvals_list.append(p_val)
                except Exception:
                    f_stats.append(np.nan)
                    pvals_list.append(np.nan)

            tstat_df[meta_col] = f_stats
            pval_df[meta_col] = pvals_list

        # -------------------------
        # Continuous → Linear regression
        # -------------------------
        else:
            slopes = []
            pvals_list = []

            x = labels.values.astype(float)

            for gene in gene_cols:
                y = gene_data[gene].values
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 3:
                    slopes.append(np.nan)
                    pvals_list.append(np.nan)
                    continue

                try:
                    res = linregress(x[mask], y[mask])
                    slopes.append(res.slope)
                    pvals_list.append(res.pvalue)
                except Exception:
                    slopes.append(np.nan)
                    pvals_list.append(np.nan)

            slope_df[meta_col] = slopes
            pval_df[meta_col] = pvals_list

    # -------------------------
    # Save outputs
    # -------------------------
    save_df_with_append_and_targets(tstat_df, outdir, base_name, "tstats")
    save_df_with_append_and_targets(slope_df, outdir, base_name, "slopes")
    save_df_with_append_and_targets(fstat_df, outdir, base_name, "fstats")
    save_df_with_append_and_targets(pval_df, outdir, base_name, "pvalues")


def save_df_with_append_and_targets(df, outdir, base_name, suffix):
    """
    Save a DataFrame to a CSV, appending new columns if the file exists,
    and also save one CSV per column (target).
    """
    os.makedirs(outdir, exist_ok=True)
    combined_path = os.path.join(outdir, f"{base_name}-{suffix}.csv")

    # Append columns if file exists
    if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
        existing = pd.read_csv(combined_path, index_col=0)
        # Overwrite columns if they exist
        for col in df.columns:
            existing[col] = df[col]
        combined = existing
        combined.index.name = "Gene"
        combined.to_csv(combined_path)
        print(f"[Updated combined] {combined_path}")
    else:
        df.index.name = "Gene"
        df.to_csv(combined_path)
        print(f"[Saved combined] {combined_path}")
        combined = df

    # Save individual target/column CSVs
    for target in df.columns:
        target_path = os.path.join(outdir, f"{base_name}-{target}-{suffix}.csv")
        combined[[target]].to_csv(target_path)
        print(f"[Saved target] {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Gene-wise tests for metadata columns.")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--meta-cols", nargs="+", required=True, help="Metadata columns to test")
    parser.add_argument("--outdir", required=True, help="Output directory")

    args = parser.parse_args()

    print(f"\nProcessing {args.csv}...")
    run_tests(args.csv, args.meta_cols, args.outdir)


if __name__ == "__main__":
    main()