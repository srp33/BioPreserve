"""PCA visualization of alignment quality."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _subtype(er, her2):
    if pd.isna(er) or pd.isna(her2): return "Unknown"
    if er == 1 and her2 == 0: return "ER+/HER2-"
    if er == 1 and her2 == 1: return "ER+/HER2+"
    if er == 0 and her2 == 0: return "ER-/HER2-"
    if er == 0 and her2 == 1: return "ER-/HER2+"
    return "Unknown"


COLOR_MAP = {
    "ER+/HER2-": "steelblue", "ER+/HER2+": "darkorange",
    "ER-/HER2-": "mediumseagreen", "ER-/HER2+": "crimson", "Unknown": "gray",
}


def full_pca(ref_log, tgt_log_raw, tgt_log_aligned, meta_ref, meta_tgt,
             output_path, gene_subset=None, title_prefix=""):
    """2×2 PCA: before/after × dataset/subtype.

    Parameters
    ----------
    ref_log, tgt_log_raw, tgt_log_aligned : pd.DataFrame
        Log-transformed expression matrices.
    meta_ref, meta_tgt : pd.DataFrame
        Metadata with meta_er_status, meta_her2_status columns.
    output_path : str
        Where to save the PNG.
    gene_subset : set or None
        If provided, restrict PCA to these genes.
    title_prefix : str
        Prefix for the plot title.
    """
    common = sorted(set(ref_log.columns) & set(tgt_log_raw.columns) & set(tgt_log_aligned.columns))
    if gene_subset:
        common = [g for g in common if g in gene_subset]

    er_col_r = "meta_er_status" if "meta_er_status" in meta_ref.columns else "meta_ER_status"
    er_col_t = "meta_er_status" if "meta_er_status" in meta_tgt.columns else "meta_ER_status"
    her2_col_r = "meta_her2_status" if "meta_her2_status" in meta_ref.columns else "meta_HER2_status"
    her2_col_t = "meta_her2_status" if "meta_her2_status" in meta_tgt.columns else "meta_HER2_status"

    er_a = meta_ref[er_col_r].values if er_col_r in meta_ref.columns else np.full(len(meta_ref), np.nan)
    er_b = meta_tgt[er_col_t].values if er_col_t in meta_tgt.columns else np.full(len(meta_tgt), np.nan)
    her2_a = meta_ref[her2_col_r].values if her2_col_r in meta_ref.columns else np.full(len(meta_ref), np.nan)
    her2_b = meta_tgt[her2_col_t].values if her2_col_t in meta_tgt.columns else np.full(len(meta_tgt), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for col, (Y, title) in enumerate([
        (tgt_log_raw[common].values, "Before alignment"),
        (tgt_log_aligned[common].values, "After alignment"),
    ]):
        comb = np.vstack([ref_log[common].values, Y])
        ds = ["A"] * len(ref_log) + ["B"] * len(Y)
        st = [_subtype(e, h) for e, h in zip(np.concatenate([er_a, er_b]), np.concatenate([her2_a, her2_b]))]
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(comb)
        n = len(ref_log)

        ax = axes[0, col]
        for d, c, m in [("A", "steelblue", "o"), ("B", "coral", "^")]:
            mask = np.array([x == d for x in ds])
            kw = dict(c=c, marker=m, alpha=0.5, s=25, label=f"Dataset {d}")
            if m == "o": kw["edgecolors"] = "white"; kw["linewidths"] = 0.3
            ax.scatter(coords[mask, 0], coords[mask, 1], **kw)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"{title} — by dataset"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

        ax = axes[1, col]
        for s in ["ER+/HER2-", "ER+/HER2+", "ER-/HER2-", "ER-/HER2+"]:
            for d, m in [("A", "o"), ("B", "^")]:
                mask = np.array([(x == s and y == d) for x, y in zip(st, ds)])
                if mask.sum() == 0: continue
                kw = dict(c=COLOR_MAP[s], marker=m, alpha=0.5, s=25, label=f"{d} {s}")
                if m == "o": kw["edgecolors"] = "white"; kw["linewidths"] = 0.3
                ax.scatter(coords[mask, 0], coords[mask, 1], **kw)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"{title} — by subtype")
        ax.legend(fontsize=6, ncol=2, loc="best"); ax.grid(True, alpha=0.2)

    n_genes = len(common)
    plt.suptitle(f"{title_prefix}PCA on {n_genes} genes — before vs after alignment", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
