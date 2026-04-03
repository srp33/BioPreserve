#!/usr/bin/env python3
"""
Rank genes by Mahalanobis distance relative to Bayesian Shift-Scale.

For each adjuster:
  1. Load unadjusted (log-transformed) and adjusted test data (samples × genes CSVs).
  2. Reconstruct per-gene affine parameters: scale α = std_adj / std_raw,
     shift β = mean_adj − α · mean_raw  (via reconstruct_params from param_utils).
  3. Reconstruct the Bayesian Shift-Scale affine parameters for the same genes.
  4. Load the Bayesian posterior precision matrix for each gene.
  5. For each gene/adjuster pair, compute one Mahalanobis distance between
     [α_adjuster, β_adjuster] and [α_bayesian, β_bayesian], using that gene's
     Bayesian posterior precision as the weighting metric.
  6. Rank genes descending — largest distance = worst agreement with Bayesian Shift-Scale.

Outputs (in --output-dir):
  <adjuster>_gene_ranking.csv      per-adjuster gene rankings
  all_adjusters_gene_ranking.csv   long-form: adjuster | gene | mahalanobis_distance | scale | shift
  mahalanobis_distances_wide.csv   wide-form: gene | <adjuster1> | <adjuster2> | ...
  scales_wide.csv                  wide-form: gene | <adjuster1> | ...
  shifts_wide.csv                  wide-form: gene | <adjuster1> | ...

Usage example:
  python rank_genes_mahalanobis.py \\
    --unadjusted data/log_transformed-2_studies-test_metabric.csv \\
    --bayesian-params data/precision_matrix.csv \\
    --adjusters \\
        bayesian:data/bayesian-shift-scale-2_studies-test_metabric.csv \\
        combat:data/combat-2_studies-test_metabric.csv \\
        gmm:data/gmm-2_studies-test_metabric.csv \\
    --output-dir results/
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from mahalanobis import mahalanobis_distances, parse_bayesian_params
from param_utils import reconstruct_params


def load_gene_matrix(path: str) -> tuple[list[str], np.ndarray]:
    """Load a samples×genes CSV.

    Columns prefixed with 'meta_' (sample metadata) are skipped.
    The header is read first so that schema overrides can be applied on the
    full read, preventing Polars from mis-inferring integer columns that
    contain float values (e.g. meta_age_at_diagnosis).

    Returns
    -------
    gene_names : list[str]
    arr : np.ndarray, shape (Genes, Samples)   ← transposed so genes are rows
    """
    with open(path, newline="") as handle:
        header = next(csv.reader(handle))
    schema_overrides = {
        c: pl.String if c.startswith("meta_") else pl.Float64
        for c in header
    }
    df = pl.read_csv(path, schema_overrides=schema_overrides)
    gene_cols = [c for c in df.columns if not c.startswith("meta_")]
    df = df.select(gene_cols)
    gene_names = df.columns
    arr = df.to_numpy().T  # samples×genes → genes×samples
    return gene_names, arr


def compute_adjuster_params(
    unadjusted_path: str,
    adjusted_path: str,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Reconstruct per-gene scale (α) and shift (β) for one adjuster.

    Uses reconstruct_params from param_utils:
      α = std_adj / std_raw
      β = mean_adj − α · mean_raw

    Parameters
    ----------
    unadjusted_path : str
        CSV of raw (unadjusted) data, samples × genes.
    adjusted_path : str
        CSV of adjuster output, samples × genes.

    Returns
    -------
    gene_names : list[str]   genes present in both files
    scales : dict[str, float]   per-gene α
    shifts : dict[str, float]   per-gene β
    """
    raw_genes, y_raw = load_gene_matrix(unadjusted_path)
    adj_genes, y_adjusted = load_gene_matrix(adjusted_path)

    adj_gene_set = set(adj_genes)
    common_genes = [g for g in raw_genes if g in adj_gene_set]
    if not common_genes:
        raise ValueError("No genes in common between unadjusted and adjusted files.")

    raw_idx = {g: i for i, g in enumerate(raw_genes)}
    adj_idx = {g: i for i, g in enumerate(adj_genes)}

    y_raw_common = y_raw[[raw_idx[g] for g in common_genes]]       # (n_genes, n_samples)
    y_adj_common = y_adjusted[[adj_idx[g] for g in common_genes]]  # (n_genes, n_samples)

    params = reconstruct_params(y_raw_common, y_adj_common)
    scales = {g: float(params["alpha"][i]) for i, g in enumerate(common_genes)}
    shifts = {g: float(params["beta"][i]) for i, g in enumerate(common_genes)}

    return common_genes, scales, shifts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank genes by Mahalanobis distance for each adjuster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--unadjusted",
        required=True,
        metavar="PATH",
        help="Path to unadjusted (log-transformed) test gene expression CSV (samples × genes).",
    )
    parser.add_argument(
        "--adjusters",
        nargs="+",
        metavar="NAME:PATH",
        required=True,
        help=(
            "One or more adjuster entries in NAME:PATH format. "
            "Each PATH is a samples×genes CSV of the adjuster's test output."
        ),
    )
    parser.add_argument(
        "--bayesian-params",
        required=True,
        metavar="PATH",
        help=(
            "Path to Bayesian posterior mean/precision CSV, e.g. precision_matrix.csv."
        ),
    )
    parser.add_argument(
        "--bayesian-split",
        choices=("train", "test"),
        default="test",
        help="Which split's Bayesian posterior parameters to use.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where output CSVs are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse NAME:PATH pairs
    adjuster_map: dict[str, str] = {}
    for item in args.adjusters:
        if ":" not in item:
            parser.error(f"Each --adjusters entry must be NAME:PATH, got: {item!r}")
        name, path = item.split(":", 1)
        adjuster_map[name] = path

    bayesian_params = parse_bayesian_params(
        args.bayesian_params,
        split=args.bayesian_split,
    )
    print(
        f"Loaded Bayesian posterior params for {len(bayesian_params)} genes "
        f"from {args.bayesian_params} ({args.bayesian_split} split)."
    )
    if "bayesian" not in adjuster_map:
        parser.error("A 'bayesian' adjuster entry is required to define the reference.")

    try:
        bayesian_gene_names, bayesian_scales, bayesian_shifts = compute_adjuster_params(
            args.unadjusted,
            adjuster_map["bayesian"],
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"ERROR loading Bayesian reference: {exc} — aborting.\n", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded Bayesian reference params for {len(bayesian_gene_names)} genes.")

    all_results: list[pl.DataFrame] = []

    for adjuster_name, adjusted_path in adjuster_map.items():
        print(f"── {adjuster_name}")
        print(f"   adjusted data : {adjusted_path}")

        try:
            gene_names, scales, shifts = compute_adjuster_params(
                args.unadjusted, adjusted_path
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"   ERROR: {exc} — aborting.\n", file=sys.stderr)
            sys.exit(1)

        print(f"   genes reconstructed : {len(gene_names)}")
        scored_genes = sorted(
            set(gene_names)
            & set(bayesian_scales)
            & set(bayesian_shifts)
            & set(bayesian_params)
        )
        if not scored_genes:
            raise ValueError(
                f"No genes in common between reconstructed parameters and "
                f"Bayesian reference/precision params for adjuster '{adjuster_name}'."
            )
        print(f"   genes scored        : {len(scored_genes)}")

        per_gene_params = {
            gene: {
                "mean": np.array([bayesian_scales[gene], bayesian_shifts[gene]]),
                "precision": bayesian_params[gene]["precision"],
            }
            for gene in scored_genes
        }
        dist_df = mahalanobis_distances(scales, shifts, per_gene_params)
        # Attach adjuster label and the reconstructed parameters
        scale_vals = [scales.get(g, float("nan")) for g in dist_df["gene"].to_list()]
        shift_vals = [shifts.get(g, float("nan")) for g in dist_df["gene"].to_list()]
        dist_df = dist_df.with_columns([
            pl.lit(adjuster_name).alias("adjuster"),
            pl.Series("scale", scale_vals),
            pl.Series("shift", shift_vals),
        ]).select(["adjuster", "gene", "mahalanobis_distance", "scale", "shift"])

        out_path = output_dir / f"{adjuster_name}_gene_ranking.csv"
        dist_df.write_csv(out_path)
        print(f"   saved → {out_path}")

        print("   top 5 (worst alignment):")
        for row in dist_df.head(5).iter_rows(named=True):
            print(
                f"     {row['gene']:<20s}  dist={row['mahalanobis_distance']:.4f}"
                f"  scale={row['scale']:.4f}  shift={row['shift']:.4f}"
            )
        print()

        all_results.append(dist_df)

    if not all_results:
        print("No results produced — check paths and gene overlap.")
        sys.exit(1)

    # Long-form combined output
    combined = pl.concat(all_results)
    combined_path = output_dir / "all_adjusters_gene_ranking.csv"
    combined.write_csv(combined_path)
    print(f"Long-form results → {combined_path}")

    # Wide-form: gene × adjuster, values = mahalanobis_distance
    pivot = (
        combined.pivot(
            values="mahalanobis_distance",
            index="gene",
            on="adjuster",
            aggregate_function="first",
        )
        .sort("gene")
    )
    pivot_path = output_dir / "mahalanobis_distances_wide.csv"
    pivot.write_csv(pivot_path)
    print(f"Wide-form distances → {pivot_path}")

    # Wide-form: gene × adjuster, values = scale
    scale_pivot = (
        combined.pivot(
            values="scale",
            index="gene",
            on="adjuster",
            aggregate_function="first",
        )
        .sort("gene")
    )
    scale_path = output_dir / "scales_wide.csv"
    scale_pivot.write_csv(scale_path)
    print(f"Wide-form scales    → {scale_path}")

    # Wide-form: gene × adjuster, values = shift
    shift_pivot = (
        combined.pivot(
            values="shift",
            index="gene",
            on="adjuster",
            aggregate_function="first",
        )
        .sort("gene")
    )
    shift_path = output_dir / "shifts_wide.csv"
    shift_pivot.write_csv(shift_path)
    print(f"Wide-form shifts    → {shift_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
