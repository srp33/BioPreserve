import numpy as np
import polars as pl


def parse_bayesian_params(params_path: str, split: str = "test") -> dict[str, dict]:

    """Parse the Bayesian adjustment CSV and return per-gene means and precision matrices.

    Parameters
    ----------
    params_path : str
        Path to the precision_matrix.csv produced by fit_and_shift.py.
    split : {"train", "test"}
        Which split's parameters to load.

    Returns
    -------
    dict mapping gene name -> {"mean": np.ndarray shape (2,), "precision": np.ndarray shape (2, 2)}
        mean[0] is the posterior mean slope (scale), mean[1] is the posterior mean intercept (shift).
    """

    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    df = pl.read_csv(params_path)
    params = {}
    for row in df.iter_rows(named=True):
        gene = row["gene"]
        mean = np.array([row[f"{split}_mean_slope"], row[f"{split}_mean_intercept"]])
        precision = np.array([
            [row[f"{split}_prec_00"], row[f"{split}_prec_01"]],
            [row[f"{split}_prec_10"], row[f"{split}_prec_11"]],
        ])
        params[gene] = {"mean": mean, "precision": precision}

    return params


def mahalanobis_distances(
    adjuster_scales: dict[str, float],
    adjuster_shifts: dict[str, float],
    reference_params: dict[str, dict],
) -> pl.DataFrame:

    """Compute Mahalanobis distance between adjuster parameters and a reference point.

    For each gene present in all three inputs, computes:

        d = sqrt( (x - mu)^T @ Precision @ (x - mu) )

    where x = [adjuster_scale, adjuster_shift] and mu is the reference mean vector.
    Genes missing from any input are silently skipped.

    Parameters
    ----------
    adjuster_scales : dict[str, float]
        Per-gene scale (slope) applied by the adjuster.
    adjuster_shifts : dict[str, float]
        Per-gene shift (intercept) applied by the adjuster.
    reference_params : dict[str, dict]
        Per-gene reference parameters. Each value has keys "mean" (shape (2,))
        and "precision" (shape (2, 2)).

    Returns
    -------
    pl.DataFrame
        Columns ["gene", "mahalanobis_distance"], sorted descending so the worst
        alignments appear first.
    """

    genes = sorted(
        set(adjuster_scales) & set(adjuster_shifts) & set(reference_params)
    )

    rows = []
    for gene in genes:
        x = np.array([adjuster_scales[gene], adjuster_shifts[gene]])
        mu = reference_params[gene]["mean"]
        precision = reference_params[gene]["precision"]
        diff = x - mu
        dist = float(np.sqrt(diff @ precision @ diff))
        rows.append({"gene": gene, "mahalanobis_distance": dist})

    return pl.DataFrame(rows).sort("mahalanobis_distance", descending=True)
