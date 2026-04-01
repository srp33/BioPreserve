#!/usr/bin/env python3

import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(
        description="Subset expression + metadata dataset into train/test using selected genes"
    )

    parser.add_argument("--expression", required=True,
                        help="Input adjusted CSV containing metadata + gene expression")

    parser.add_argument("--genes", required=True,
                        help="CSV file with selected genes (must contain a 'gene' column)")

    parser.add_argument("--train_out", required=True,
                        help="Output path for train CSV")

    parser.add_argument("--test_out", required=True,
                        help="Output path for test CSV")

    parser.add_argument("--train", required=True,
                        help="Value of meta_source corresponding to training set")

    parser.add_argument("--test", required=True,
                        help="Value of meta_source corresponding to test set")

    parser.add_argument("--split", action="store_true",
                        help="Whether to split into train/test")

    args = parser.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(args.expression, low_memory=False)

    if "meta_source" not in df.columns:
        raise ValueError("meta_source column not found in input dataset.")

    # -------------------------
    # Load selected genes
    # -------------------------
    genes_df = pd.read_csv(args.genes, index_col=False)

    if "gene" not in genes_df.columns:
        print("Columns: ", genes_df.columns)
        raise ValueError("Selected genes file must contain a 'gene' column.")

    selected_genes = genes_df['gene'].astype(str).tolist()

    # -------------------------
    # Identify columns
    # -------------------------
    meta_cols = [c for c in df.columns if c.startswith("meta_")]

    # Keep only genes that exist in dataset
    available_genes = [g for g in selected_genes if g in df.columns]

    missing = set(selected_genes) - set(available_genes)
    if missing:
        print(f"Warning: {len(missing)} selected genes not found in dataset (showing up to 5):")
        print(list(missing)[:5])

    cols_to_keep = meta_cols + available_genes
    df_subset = df[cols_to_keep]

    # -------------------------
    # Split train/test
    # -------------------------
    train_df = df_subset[df_subset["meta_source"] == args.train]
    test_df = df_subset[df_subset["meta_source"] == args.test]

    print(f"Train samples: {train_df.shape[0]}")
    print(f"Test samples: {test_df.shape[0]}")
    print(f"Genes retained: {len(available_genes)}")

    # -------------------------
    # Write outputs
    # -------------------------
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print(f"Saved train dataset: {args.train_out}")
    print(f"Saved test dataset: {args.test_out}")

if __name__ == "__main__":
    main()