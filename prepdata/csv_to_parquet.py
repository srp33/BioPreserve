#!/usr/bin/env python3
"""
Convert a CSV file to Parquet format and emit a list of metadata columns.

Column names are sanitized (characters like = / are replaced with _) so they
are safe for use as Snakemake wildcards and filesystem paths.

Outputs:
  - <output>.parquet  — the data in Parquet format (with sanitized column names)
  - <output_dir>/meta_columns.txt — one metadata column name per line
    (excluding meta_source and meta_Sample_ID)
"""

import argparse
import re
from pathlib import Path
import polars as pl


def sanitize_col(name: str) -> str:
    """Replace characters that are unsafe for Snakemake wildcards / filesystem paths."""
    return re.sub(r"[=/\\{}]", "_", name)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the .parquet file")
    parser.add_argument("--columns-file", type=Path, required=True,
                        help="Path to write metadata column list")
    args = parser.parse_args()

    print(f"Reading {args.input} ...", flush=True)
    df = pl.read_csv(args.input, infer_schema_length=10000)
    print(f"  Shape: {df.shape}", flush=True)

    # Sanitize column names so they are safe for Snakemake wildcards & paths
    rename_map = {}
    for c in df.columns:
        safe = sanitize_col(c)
        if safe != c:
            rename_map[c] = safe
    if rename_map:
        print(f"  Sanitized {len(rename_map)} column names:", flush=True)
        for old, new in rename_map.items():
            print(f"    {old} -> {new}", flush=True)
        df = df.rename(rename_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"  Wrote {args.output}", flush=True)

    # Emit metadata column list (skip identifiers/grouping cols)
    skip = {"meta_source", "meta_Sample_ID"}
    meta_cols = [c for c in df.columns if c.startswith("meta_") and c not in skip]

    args.columns_file.parent.mkdir(parents=True, exist_ok=True)
    args.columns_file.write_text("\n".join(meta_cols) + "\n")
    print(f"  Wrote {len(meta_cols)} column names to {args.columns_file}", flush=True)


if __name__ == "__main__":
    main()
