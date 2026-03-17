#!/usr/bin/env python3
"""
Aggregate heatmap results across all classifiers, producing an average-performance
heatmap and text table.
"""

import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


LABEL_MAP = {
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

ADJUSTER_MAP = {
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


def format_adjuster(name):
    """Strip prefix and map to display name."""
    for prefix in ("METHOD_", "ORACLE_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.startswith("DBA:") or name.startswith("ORACLE:"):
        return name
    return ADJUSTER_MAP.get(name, name)


def main():
    parser = argparse.ArgumentParser(
        description="Average heatmap results across classifiers"
    )
    parser.add_argument("--input-csvs", nargs="+", required=True,
                        help="Per-classifier results CSVs")
    parser.add_argument("--cv-results", required=False,
                        help="CV ceiling results CSV")
    parser.add_argument("--output-heatmap", required=True)
    parser.add_argument("--output-txt", required=True)
    args = parser.parse_args()

    # Load and concatenate all per-classifier results
    frames = []
    for csv_path in args.input_csvs:
        df = pl.read_csv(csv_path)
        frames.append(df)
    all_df = pl.concat(frames)

    # Format adjuster names
    all_df = all_df.with_columns(
        pl.col("adjuster").map_elements(format_adjuster, return_dtype=pl.Utf8).alias("method")
    )

    # Average score across classifiers for each (method, label)
    avg_df = (
        all_df
        .group_by(["method", "label"])
        .agg(pl.col("score").mean().alias("score"))
    )

    # Add CV ceiling averaged across classifiers
    if args.cv_results:
        cv = pl.read_csv(args.cv_results)
        cv_avg = (
            cv
            .group_by("metadata_column")
            .agg(pl.col("score").mean().alias("score"))
            .rename({"metadata_column": "label"})
            .with_columns(pl.lit("CV Ceiling (Test Set)").alias("method"))
            .select(["method", "label", "score"])
        )
        avg_df = pl.concat([avg_df, cv_avg])

    # Pivot
    pivot = avg_df.pivot(values="score", index="method", on="label")
    pdf = pivot.to_pandas().set_index("method")
    pdf.columns = [LABEL_MAP.get(c, c) for c in pdf.columns]

    # Drop Sex column if all zeros / NaN
    if "Sex" in pdf.columns and (pdf["Sex"].fillna(0).abs() < 0.01).all():
        pdf = pdf.drop(columns=["Sex"])

    # Sort: CV ceiling first, then by row mean descending
    cv_rows = pdf[pdf.index.str.contains("CV Ceiling", na=False)]
    rest = pdf[~pdf.index.str.contains("CV Ceiling", na=False)].copy()
    rest["_mean"] = rest.mean(axis=1, skipna=True)
    rest = rest.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    import pandas as pd
    pdf = pd.concat([cv_rows, rest])

    # ---- Heatmap ----
    fig, ax = plt.subplots(figsize=(max(10, len(pdf.columns) * 1.1),
                                    max(3, len(pdf) * 0.6 + 1.5)))
    sns.heatmap(pdf, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "MCC / R²"})
    ax.set_title("Cross-Dataset Performance Averaged Across All Classifiers\n"
                 "(MCC for categorical, R² for continuous metadata)", pad=15, fontsize=12)
    ax.set_xlabel("Metadata Label", fontsize=10)
    ax.set_ylabel("Adjustment Method", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(args.output_heatmap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {args.output_heatmap}")

    # ---- Text table ----
    col_w = 10
    meth_w = max(40, max(len(str(i)) for i in pdf.index) + 2)
    lines = [
        "Cross-Dataset Performance Averaged Across All Classifiers",
        "(MCC for categorical, R² for continuous metadata)",
        "=" * 120, "",
    ]
    header = f"{'Method':<{meth_w}}" + "".join(f"{c:>{col_w}}" for c in pdf.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for idx, row in pdf.iterrows():
        line = f"{idx:<{meth_w}}"
        for v in row:
            line += f"{v:>{col_w}.2f}" if not np.isnan(v) else f"{'NaN':>{col_w}}"
        lines.append(line)
    lines.append("")

    with open(args.output_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved text:    {args.output_txt}")


if __name__ == "__main__":
    main()
