import argparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def plot_heatmap(results_path, output_path, continuous_metadata):
    results_df = pl.read_csv(results_path)

    pivot = results_df.pivot(values="score_mean", index="adjuster", on="metadata_label")

    label_map = {
        "meta_er_status": "ER status",
        "meta_menopause_status": "Menopause",
        "meta_sex": "Sex",
        "meta_age_at_diagnosis": "Age (R\u00b2)",
        "meta_chemotherapy": "Chemo",
        "meta_histological_type": "Hist. type",
        "meta_her2_status": "HER2",
        "meta_age_at_diagnosis_combined_lt50": "Age <50",
        "meta_age_at_diagnosis_combined_50_69": "Age 50-69",
        "meta_age_at_diagnosis_combined_ge70": "Age \u226570",
    }
    adjuster_map = {
        "unadjusted": "unadjusted",
        "bayesian_shift_scale": "bayesian_shift_scale",
    }

    heatmap_df = pivot.to_pandas().set_index("adjuster")
    heatmap_df.columns = [label_map.get(c, c) for c in heatmap_df.columns]
    heatmap_df.index = [adjuster_map.get(i, i) for i in heatmap_df.index]

    er_col = "ER status"
    if er_col in heatmap_df.columns:
        heatmap_df = heatmap_df.sort_values(er_col, ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(heatmap_df.columns) * 1.1), max(3, len(heatmap_df) * 1.2 + 1.5)))

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
    )

    ax.set_title(
        "Cross-Dataset Classification Performance\n(MCC, except Age which uses R\u00b2)",
        pad=15,
    )
    ax.set_xlabel("Metadata Label")
    ax.set_ylabel("Adjuster")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return f"SUCCESS: heatmap saved to {output_path}"


def main():
    parser = argparse.ArgumentParser(description="Plot cross-dataset classification performance heatmap.")
    parser.add_argument("--results", required=True, help="Path to classification results CSV.")
    parser.add_argument("--output", required=True, help="Output path for heatmap PNG.")
    parser.add_argument("--continuous-metadata", nargs="*", default=[],
                        help="Metadata columns treated as continuous (for reference only).")
    args = parser.parse_args()

    message = plot_heatmap(args.results, args.output, args.continuous_metadata)
    print(message, flush=True)


if __name__ == "__main__":
    main()
