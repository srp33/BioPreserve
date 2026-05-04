"""
ComBat batch correction — faithful port of R sva::ComBat (Johnson et al. 2007).

Supports reference batch mode where the reference dataset is unchanged.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-8


def _aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    if s2 <= 0 or m <= 0:
        return 1.1
    return max((2 * s2 + m ** 2) / s2, 1.1)


def _bprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    if s2 <= 0 or m <= 0:
        return EPS
    return max((m * s2 + m ** 3) / s2, EPS)


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data_batch, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    """Iterative EB solver matching R's it.sol."""
    n = s_data_batch.shape[1]
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(500):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.sum((s_data_batch - g_new[:, np.newaxis]) ** 2, axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), EPS)),
            np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), EPS)),
        )
        g_old = g_new
        d_old = d_new
        if change < conv:
            break
    return g_new, d_new


def combat_correct(dat, batch_labels, ref_batch=None):
    """ComBat batch correction — faithful port of R sva::ComBat.

    Parameters
    ----------
    dat : np.ndarray
        Gene expression matrix, shape (n_genes, n_samples).
    batch_labels : np.ndarray
        Batch label per sample.
    ref_batch : int or None
        If set, use this batch as reference (its data won't change).

    Returns
    -------
    corrected : np.ndarray
        Corrected expression matrix, same shape as dat.
    """
    dat = dat.copy()
    n_genes, n_samples = dat.shape
    batches_unique = np.unique(batch_labels)
    n_batch = len(batches_unique)

    batches = [np.where(batch_labels == b)[0] for b in batches_unique]
    n_batches = np.array([len(b) for b in batches])

    # Zero-variance genes within any batch
    zero_rows = set()
    for b_idx in batches:
        if len(b_idx) > 1:
            v = np.var(dat[:, b_idx], axis=1)
            zero_rows |= set(np.where(v == 0)[0])
    keep_rows = sorted(set(range(n_genes)) - zero_rows)

    if zero_rows:
        logger.info(f"  [COMBAT] {len(zero_rows)} zero-variance genes excluded")

    dat_orig = dat.copy()
    dat = dat[keep_rows, :]
    n_genes_kept = dat.shape[0]

    # Design matrix
    design = np.zeros((n_samples, n_batch))
    for i, b_idx in enumerate(batches):
        design[b_idx, i] = 1.0

    ref = None
    if ref_batch is not None:
        ref = list(batches_unique).index(ref_batch)
        design[:, ref] = 1.0

    # Standardize
    B_hat = np.linalg.lstsq(design, dat.T, rcond=None)[0]

    if ref is not None:
        grand_mean = B_hat[ref, :]
    else:
        grand_mean = (n_batches / n_samples) @ B_hat

    stand_mean = np.outer(grand_mean, np.ones(n_samples))

    if ref is not None:
        ref_idx = batches[ref]
        resid = dat[:, ref_idx] - (design[ref_idx, :] @ B_hat).T
        var_pooled = np.mean(resid ** 2, axis=1)
    else:
        resid = dat - (design @ B_hat).T
        var_pooled = np.mean(resid ** 2, axis=1)

    var_pooled = np.maximum(var_pooled, EPS)
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)[:, np.newaxis]

    # Batch effect estimates
    gamma_hat = np.zeros((n_batch, n_genes_kept))
    delta_hat = np.zeros((n_batch, n_genes_kept))
    for i, b_idx in enumerate(batches):
        gamma_hat[i, :] = np.mean(s_data[:, b_idx], axis=1)
        if len(b_idx) > 1:
            delta_hat[i, :] = np.var(s_data[:, b_idx], axis=1, ddof=1)
        else:
            delta_hat[i, :] = 1.0

    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    a_prior = np.array([_aprior(delta_hat[i, :]) for i in range(n_batch)])
    b_prior = np.array([_bprior(delta_hat[i, :]) for i in range(n_batch)])

    # EB estimation
    gamma_star = np.zeros((n_batch, n_genes_kept))
    delta_star = np.zeros((n_batch, n_genes_kept))
    for i, b_idx in enumerate(batches):
        g_star, d_star = _it_sol(
            s_data[:, b_idx], gamma_hat[i, :], delta_hat[i, :],
            gamma_bar[i], t2[i], a_prior[i], b_prior[i],
        )
        gamma_star[i, :] = g_star
        delta_star[i, :] = d_star

    if ref is not None:
        gamma_star[ref, :] = 0.0
        delta_star[ref, :] = 1.0

    # Apply correction
    bayesdata = s_data.copy()
    for i, b_idx in enumerate(batches):
        bayesdata[:, b_idx] = (
            (s_data[:, b_idx] - gamma_star[i, :][:, np.newaxis])
            / np.sqrt(delta_star[i, :])[:, np.newaxis]
        )

    corrected = bayesdata * np.sqrt(var_pooled)[:, np.newaxis] + stand_mean

    if ref is not None:
        corrected[:, batches[ref]] = dat[:, batches[ref]]

    result = dat_orig.copy()
    result[keep_rows, :] = corrected
    return result
