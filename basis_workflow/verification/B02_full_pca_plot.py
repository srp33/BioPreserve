#!/usr/bin/env python3
"""
Full-genome PCA before/after alignment, colored by dataset and subtype.

Outputs:
  full_pca_alignment.png — 2x2 grid: before/after × dataset/subtype
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_dataset_with_meta(csv_path):
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_df = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    gene_df = df.drop(columns=meta_cols, errors="ignore").select_dtypes(include=[np.number])
    return gene_df, meta_df


def log_transform(df):
    return np.log(df - df.min(axis=0) + 1.0)


def subtype(er, her2):
    if pd.isna(er) or pd.isna(her2): return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Full-genome PCA before/after alignment.")
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ga, ma = load_dataset_with_meta(args.dataset_a)
    gb, mb = load_dataset_with_meta(args.dataset_b)
    ga = log_transform(ga)
    gb = log_transform(gb)

    aligned_df = pd.read_csv(args.aligned, index_col=0, low_memory=False)
    gc = aligned_df.select_dtypes(include=[np.number])
    mc = [c for c in aligned_df.columns if c.startswith("meta_")]
    if mc:
        gc = gc.drop(columns=mc, errors="ignore")

    common = sorted(set(ga.columns) & set(gb.columns) & set(gc.columns))

    er_col_a = "meta_er_status" if "meta_er_status" in ma.columns else "meta_ER_status"
    er_col_b = "meta_er_status" if "meta_er_status" in mb.columns else "meta_ER_status"
    her2_col_a = "meta_her2_status" if "meta_her2_status" in ma.columns else "meta_HER2_status"
    her2_col_b = "meta_her2_status" if "meta_her2_status" in mb.columns else "meta_HER2_status"

    er_a = ma[er_col_a].values if er_col_a in ma.columns else np.full(len(ma), np.nan)
    er_b = mb[er_col_b].values if er_col_b in mb.columns else np.full(len(mb), np.nan)
    her2_a = ma[her2_col_a].values if her2_col_a in ma.columns else np.full(len(ma), np.nan)
    her2_b = mb[her2_col_b].values if her2_col_b in mb.columns else np.full(len(mb), np.nan)

    color_map = {
        "ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange",
        "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for col, (Y, title) in enumerate([
        (gb[common].values, "Before alignment"),
        (gc[common].values, "After alignment"),
    ]):
        comb = np.vstack([ga[common].values, Y])
        ds = ["A"] * len(ga) + ["B"] * len(Y)
        st = [subtype(e, h) for e, h in zip(
            np.concatenate([er_a, er_b]), np.concatenate([her2_a, her2_b]))]
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(comb)
        n = len(ga)

        # Top: by dataset
        ax = axes[0, col]
        for d, c, m in [("A", "steelblue", "o"), ("B", "coral", "^")]:
            mask = np.array([x == d for x in ds])
            kw = dict(c=c, marker=m, alpha=0.5, s=25, label=f"Dataset {d}")
            if m == "o":
                kw["edgecolors"] = "white"
                kw["linewidths"] = 0.3
            ax.scatter(coords[mask, 0], coords[mask, 1], **kw)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"{title} — by dataset")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Bottom: by subtype
        ax = axes[1, col]
        for s in ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]:
            for d, m in [("A", "o"), ("B", "^")]:
                mask = np.array([(x == s and y == d) for x, y in zip(st, ds)])
                if mask.sum() == 0:
                    continue
                kw = dict(c=color_map[s], marker=m, alpha=0.5, s=25, label=f"{d} {s}")
                if m == "o":
                    kw["edgecolors"] = "white"
                    kw["linewidths"] = 0.3
                ax.scatter(coords[mask, 0], coords[mask, 1], **kw)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"{title} — by subtype")
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(True, alpha=0.2)

    plt.suptitle("PCA on full gene expression — before vs after alignment", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
