import argparse
import gc
import logging
import os
import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# Data class for all loaded data
@dataclass
class _LoadedData:
    X_train: np.ndarray
    y_train: np.ndarray
    gene_names: list
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_df: pl.DataFrame
    X_train_df: pl.DataFrame
    X_test_df: pl.DataFrame
    y_train_df: pl.DataFrame


def _fit_bayesian_regression(x, y, slope_median, slope_log_sd, intercept_median, intercept_sd, resid_sd, draws, cores=1):

    """Given the appropriate statistics, fit the Bayesian regression required to compute the shift and scale."""

    with pymc.Model():
        slope = pymc.Lognormal("slope", mu=np.log(slope_median), sigma=slope_log_sd)
        intercept = pymc.Normal("intercept", mu=intercept_median, sigma=intercept_sd)
        epsilon = pymc.HalfNormal("epsilon", sigma=resid_sd)
        mu_y = intercept + slope * x
        pymc.Normal("y_obs", mu=mu_y, sigma=epsilon, observed=y)
        parameter_samples = pymc.sample(draws=draws, cores=cores, progressbar=False)

    return parameter_samples


def _has_converged(trace, rhat_threshold):

    """Detect if a gene has converged. Returns (converged, worst_variable, worst_rhat)."""

    rhat = az.rhat(trace)
    worst_var, worst_val = None, 0.0
    for var in rhat.data_vars: # type: ignore
        val = float(rhat[var].max()) # type: ignore
        if val > worst_val:
            worst_val = val
            worst_var = var
    return worst_val <= rhat_threshold, worst_var, worst_val


def _extract_posterior_precisions(trace):

    """Get precisions after the sampling process is completed."""

    slopes = trace.posterior["slope"].values.flatten()
    intercepts = trace.posterior["intercept"].values.flatten()
    samples = np.vstack([slopes, intercepts])
    mean = np.mean(samples, axis=1)
    cov_matrix = np.cov(samples)
    jitter = np.trace(cov_matrix) * 1e-6 * np.eye(2)
    precision = np.linalg.inv(cov_matrix + jitter)
    return mean, precision


def _process_gene_worker(args):

    """Worker for per-gene Bayesian fitting.

    Returns (gene, params_dict, None) on success or (gene, None, reason_str) on failure.
    """

    (gene,
     train_x, train_y, test_x, test_y,
     slope_median, slope_log_sd, intercept_median, intercept_sd,
     train_resid_sd, test_resid_sd,
     draws, rhat_threshold, max_divergences) = args

    # Suppress noisy PyMC/pytensor output from worker processes
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    train_samples = _fit_bayesian_regression(
        train_x, train_y,
        slope_median=slope_median, slope_log_sd=slope_log_sd,
        intercept_median=intercept_median, intercept_sd=intercept_sd,
        resid_sd=train_resid_sd, draws=draws, cores=1,
    )
    train_divergences = int(train_samples.sample_stats["diverging"].sum().values)

    test_samples = _fit_bayesian_regression(
        test_x, test_y,
        slope_median=slope_median, slope_log_sd=slope_log_sd,
        intercept_median=intercept_median, intercept_sd=intercept_sd,
        resid_sd=test_resid_sd, draws=draws, cores=1,
    )
    test_divergences = int(test_samples.sample_stats["diverging"].sum().values)

    if train_divergences > max_divergences or test_divergences > max_divergences:
        return gene, None, f"divergences exceeded {max_divergences} (train={train_divergences}, test={test_divergences})"

    train_converged, train_worst_var, train_worst_rhat = _has_converged(train_samples, rhat_threshold)
    test_converged, test_worst_var, test_worst_rhat = _has_converged(test_samples, rhat_threshold)

    if not train_converged or not test_converged:
        details = []
        if not train_converged:
            details.append(f"train {train_worst_var}={train_worst_rhat:.3f}")
        if not test_converged:
            details.append(f"test {test_worst_var}={test_worst_rhat:.3f}")
        return gene, None, f"R-hat exceeded {rhat_threshold}: {', '.join(details)}"

    train_mean, train_precision = _extract_posterior_precisions(train_samples)
    test_mean, test_precision = _extract_posterior_precisions(test_samples)

    del train_samples, test_samples
    gc.collect()

    params = {
        "train": {"mean": train_mean.tolist(), "precision": train_precision.tolist()},
        "test": {"mean": test_mean.tolist(), "precision": test_precision.tolist()},
    }
    return gene, params, None


def _load_data(train_path, test_path):

    """Load and split train/test CSVs into a _LoadedData dataclass."""

    gse_clean = pl.read_csv(train_path)
    metabric_clean = pl.read_csv(test_path)

    X_train_df = gse_clean.select(cs.starts_with("meta"))
    y_train_df = gse_clean.drop(cs.starts_with("meta"))
    X_test_df = metabric_clean.select(cs.starts_with("meta"))
    y_test_df = metabric_clean.drop(cs.starts_with("meta"))

    return _LoadedData(
        X_train=X_train_df.to_numpy(),
        y_train=y_train_df.to_numpy(),
        gene_names=y_train_df.columns,
        X_test=X_test_df.to_numpy(),
        y_test=y_test_df.to_numpy(),
        y_test_df=y_test_df,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        y_train_df=y_train_df,
    )


def _fit_linear_regression(X_train, y_train, X_test, y_test, gene_names):

    """Fit a linear regression model and return predictions and per-gene metrics."""

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred, multioutput="raw_values")
    test_r2 = r2_score(y_test, y_test_pred, multioutput="raw_values")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput="raw_values"))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="raw_values"))

    all_metrics = {
        gene: {
            "train_r2": float(train_r2[i]),
            "test_r2": float(test_r2[i]),
            "train_rmse": float(train_rmse[i]),
            "test_rmse": float(test_rmse[i]),
        }
        for i, gene in enumerate(gene_names)
    }

    return y_train_pred, y_test_pred, all_metrics


def _fit_bayesian_regressions(y_train, y_train_pred, y_test, y_test_pred, gene_names,
                               slope_log_sd, intercept_sd, draws, rhat_threshold, max_divergences, n_jobs=1):

    """Fit per-gene Bayesian regressions and return converged posterior parameters.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 uses all available CPUs.
    """

    train_std = y_train.std(axis=0)
    sd_ratio = np.where(train_std == 0, 1.0, y_test.std(axis=0) / train_std)
    sd_ratio = np.maximum(sd_ratio, 1e-6)
    intercept_difference = y_test.mean(axis=0) - y_train.mean(axis=0)
    train_residuals_sd = np.maximum((y_train_pred - y_train).std(axis=0), 1e-6)
    # Test residual SD estimated from data rather than a fixed constant
    test_residuals_sd = np.maximum((y_test_pred - y_test).std(axis=0), 1e-6)

    work_items = [
        (
            gene,
            y_train_pred[:, i], y_train[:, i],
            y_test_pred[:, i], y_test[:, i],
            float(sd_ratio[i]), slope_log_sd,
            float(intercept_difference[i]), intercept_sd,
            float(train_residuals_sd[i]), float(test_residuals_sd[i]),
            draws, rhat_threshold, max_divergences,
        )
        for i, gene in enumerate(gene_names)
    ]

    all_params = {}
    failed_genes = []
    n_workers = n_jobs if n_jobs > 0 else os.cpu_count()

    # Parallelize across genes; n_workers=1 falls back to sequential
    if n_workers == 1:
        for idx, item in enumerate(work_items):
            gene = item[0]
            # #9: use logger instead of print
            logger.info("Fitting gene %d/%d: %s", idx + 1, len(gene_names), gene)
            _, params, reason = _process_gene_worker(item)
            if params is None:
                logger.warning("  Dropping %s: %s", gene, reason)
                failed_genes.append(gene)
            else:
                all_params[gene] = params
    else:
        logger.info("Fitting %d genes across %d workers", len(gene_names), n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_gene_worker, item) for item in work_items]
            for future, item in zip(futures, work_items):
                gene = item[0]
                try:
                    _, params, reason = future.result()
                except Exception as exc:
                    logger.warning("  Dropping %s: worker raised %s", gene, exc)
                    failed_genes.append(gene)
                    continue
                if params is None:
                    logger.warning("  Dropping %s: %s", gene, reason)
                    failed_genes.append(gene)
                else:
                    all_params[gene] = params

    if failed_genes:
        logger.warning(
            "Dropped %d/%d genes due to non-convergence: %s",
            len(failed_genes), len(gene_names), failed_genes,
        )

    return all_params


def _save_params_and_metrics(all_params, all_metrics, params_path, metrics_converged_path):

    """Write Bayesian parameters and converged metrics to CSV files."""

    params_rows = []
    for gene, p in all_params.items():
        tm, tp = p["train"]["mean"], p["train"]["precision"]
        em, ep = p["test"]["mean"], p["test"]["precision"]
        params_rows.append({
            "gene": gene,
            "train_mean_slope": tm[0], "train_mean_intercept": tm[1],
            "train_prec_00": tp[0][0], "train_prec_01": tp[0][1],
            "train_prec_10": tp[1][0], "train_prec_11": tp[1][1],
            "test_mean_slope": em[0], "test_mean_intercept": em[1],
            "test_prec_00": ep[0][0], "test_prec_01": ep[0][1],
            "test_prec_10": ep[1][0], "test_prec_11": ep[1][1],
        })
    pl.DataFrame(params_rows).write_csv(params_path)

    metrics_converged = [{"gene": gene, **all_metrics[gene]} for gene in all_params]
    pl.DataFrame(metrics_converged).write_csv(metrics_converged_path)


def _shift_and_scale(y_test_df, all_params):

    """Apply shift and scale to test gene expression using posterior means."""

    shifted = {}
    for gene in all_params:
        y = y_test_df[gene].to_numpy()
        scale = all_params[gene]["test"]["mean"][0]
        shift = all_params[gene]["test"]["mean"][1]
        shifted[gene] = (y - shift) / scale
    return pl.DataFrame(shifted)


def fit_and_shift(train_path, test_path, params_path, metrics_converged_path,
                  slope_log_sd, intercept_sd, draws, rhat_threshold, max_divergences,
                  n_jobs=1,
                  X_train_path=None, X_test_path=None, y_train_path=None, y_test_path=None):

    """
    Run the full adjuster: load - linear regression - Bayesian regression - shift-scale.

    Parameters
    ----------
    train_path : str
        Path to cleaned train CSV (gse_clean.csv).
    test_path : str
        Path to cleaned test CSV (metabric_clean.csv).
    params_path : str
        Output path for Bayesian parameters CSV.
    metrics_converged_path : str
        Output path for converged metrics CSV.
    slope_log_sd, intercept_sd : float
        Bayesian prior hyperparameters. Test residual SD is estimated from
        data automatically (per-gene test prediction residuals).
    draws : int
        Number of MCMC draws.
    rhat_threshold : float
        R-hat convergence threshold.
    max_divergences : int
        Maximum allowed divergences per split before a gene is dropped.
        Use 0 to drop genes with any divergence; small values like 5 are
        less strict and recommended for noisy genes.
    n_jobs : int
        Number of parallel workers for per-gene MCMC. -1 uses all CPUs. Default 1.
    X_train_path, X_test_path : str, optional
        Output paths for metadata splits.
    y_train_path, y_test_path : str, optional
        Output paths for unadjusted gene expression splits.

    Returns
    -------
    pl.DataFrame
        Shifted and scaled test gene expression data (converged genes only).
    """

    data = _load_data(train_path, test_path)

    if X_train_path:
        data.X_train_df.write_csv(X_train_path)
    if X_test_path:
        data.X_test_df.write_csv(X_test_path)
    if y_train_path:
        data.y_train_df.write_csv(y_train_path)
    if y_test_path:
        data.y_test_df.write_csv(y_test_path)

    y_train_pred, y_test_pred, all_metrics = _fit_linear_regression(
        data.X_train, data.y_train, data.X_test, data.y_test, data.gene_names
    )
    all_params = _fit_bayesian_regressions(
        data.y_train, y_train_pred, data.y_test, y_test_pred, data.gene_names,
        slope_log_sd, intercept_sd, draws, rhat_threshold, max_divergences, n_jobs=n_jobs,
    )
    _save_params_and_metrics(all_params, all_metrics, params_path, metrics_converged_path)
    return _shift_and_scale(data.y_test_df, all_params)


def main():

    # Configure logging for the CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Split, fit linear and Bayesian regression, and shift-scale test gene expression data."
    )
    parser.add_argument("--train", required=True, help="Path to cleaned train CSV.")
    parser.add_argument("--test", required=True, help="Path to cleaned test CSV.")
    parser.add_argument("--params", required=True, help="Output path for Bayesian parameters CSV.")
    parser.add_argument("--metrics-converged", required=True, help="Output path for converged metrics CSV.")
    parser.add_argument("--output", required=True, help="Output path for shifted gene expression CSV.")
    parser.add_argument("--slope-log-sd", type=float, required=True)
    parser.add_argument("--intercept-sd", type=float, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--rhat-threshold", type=float, required=True)
    parser.add_argument(
        "--max-divergences", type=int, required=True,
        # #12: document footgun of 0
        help="Max divergences per split before a gene is dropped. "
             "Use 0 to drop on any divergence; small values like 5 are less strict.",
    )
    # Expose parallelism via CLI
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers for per-gene MCMC. -1 uses all CPUs. Default: 1.",
    )
    parser.add_argument("--X-train-output", required=True, help="Output path for training metadata CSV.")
    parser.add_argument("--X-test-output", required=True, help="Output path for test metadata CSV.")
    parser.add_argument("--y-train-output", required=True, help="Output path for unadjusted training gene expression CSV.")
    parser.add_argument("--y-test-output", required=True, help="Output path for unadjusted test gene expression CSV.")
    args = parser.parse_args()

    y_test_shifted = fit_and_shift(
        args.train, args.test, args.params, args.metrics_converged,
        args.slope_log_sd, args.intercept_sd,
        args.draws, args.rhat_threshold, args.max_divergences,
        n_jobs=args.n_jobs,
        X_train_path=args.X_train_output,
        X_test_path=args.X_test_output,
        y_train_path=args.y_train_output,
        y_test_path=args.y_test_output,
    )
    y_test_shifted.write_csv(args.output)
    logger.info("SUCCESS: shifted %d genes", len(y_test_shifted.columns))


if __name__ == "__main__":

    main()
