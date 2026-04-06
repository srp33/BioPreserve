"""
ComBat batch correction — faithful port of R sva::ComBat (Johnson et al. 2007)
with support for sample-level Optimal Transport weights.

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


def _it_sol(s_data_batch, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, weights=None):
    """Iterative EB solver matching R's it.sol with optional weights."""
    if weights is None:
        weights = np.ones(s_data_batch.shape[1])
        
    V1 = np.sum(weights)
    V2 = np.sum(weights**2)
    n_eff = (V1**2) / V2 if V2 > EPS else 0
    n = max(n_eff, 2.0)  # Use effective sample size
    
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(500):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        # Weighted sum of squares
        sum2 = np.sum(weights * (s_data_batch - g_new[:, np.newaxis]) ** 2, axis=1)
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


def combat_correct(dat, batch_labels, ref_batch=None, weights=None):
    """ComBat batch correction with optional sample weights.

    Parameters
    ----------
    dat : np.ndarray
        Gene expression matrix, shape (n_genes, n_samples).
    batch_labels : np.ndarray
        Batch label per sample.
    ref_batch : int or None
        If set, use this batch as reference (its data won't change).
    weights : np.ndarray or None
        Sample weights (e.g. from Optimal Transport). shape (n_samples,).

    Returns
    -------
    corrected : np.ndarray
        Corrected expression matrix, same shape as dat.
    """
    dat = dat.copy()
    n_genes, n_samples = dat.shape
    batches_unique = np.unique(batch_labels)
    n_batch = len(batches_unique)

    if weights is None:
        weights = np.ones(n_samples)

    batches = [np.where(batch_labels == b)[0] for b in batches_unique]

    # Zero-variance genes within any batch
    zero_rows = set()
    for b_idx in batches:
        if len(b_idx) > 1:
            # Weighted variance check
            w_b = weights[b_idx]
            v1 = np.sum(w_b)
            if v1 > EPS:
                mu = np.sum(dat[:, b_idx] * w_b, axis=1) / v1
                v = np.sum(w_b * (dat[:, b_idx] - mu[:, np.newaxis])**2, axis=1) / v1
                zero_rows |= set(np.where(v < EPS)[0])
            
    keep_rows = sorted(set(range(n_genes)) - zero_rows)

    if zero_rows:
        logger.info(f"  [COMBAT] {len(zero_rows)} near-zero-variance genes excluded")

    dat_orig = dat.copy()
    dat = dat[keep_rows, :]
    n_genes_kept = dat.shape[0]

    ref = None
    if ref_batch is not None:
        if ref_batch in batches_unique:
            ref = list(batches_unique).index(ref_batch)

    # 1. Weighted Grand Mean & Standardization
    # If ref_batch is provided, grand mean is just the weighted mean of the reference batch
    if ref is not None:
        ref_idx = batches[ref]
        w_ref = weights[ref_idx]
        v1_ref = np.sum(w_ref)
        grand_mean = np.sum(dat[:, ref_idx] * w_ref, axis=1) / v1_ref if v1_ref > EPS else np.mean(dat[:, ref_idx], axis=1)
    else:
        v1_all = np.sum(weights)
        grand_mean = np.sum(dat * weights, axis=1) / v1_all if v1_all > EPS else np.mean(dat, axis=1)

    stand_mean = np.outer(grand_mean, np.ones(n_samples))

    # 2. Weighted Pooled Variance
    if ref is not None:
        ref_idx = batches[ref]
        w_ref = weights[ref_idx]
        v1_ref = np.sum(w_ref)
        v2_ref = np.sum(w_ref**2)
        if v1_ref > EPS:
            var_pooled = np.sum(w_ref * (dat[:, ref_idx] - stand_mean[:, ref_idx])**2, axis=1) / v1_ref
            # Apply unbiased correction if possible
            n_eff = (v1_ref**2) / v2_ref if v2_ref > EPS else 0
            if n_eff > 1:
                var_pooled *= n_eff / (n_eff - 1)
        else:
            var_pooled = np.var(dat[:, ref_idx], axis=1)
    else:
        v1_all = np.sum(weights)
        v2_all = np.sum(weights**2)
        if v1_all > EPS:
            var_pooled = np.sum(weights * (dat - stand_mean)**2, axis=1) / v1_all
            n_eff = (v1_all**2) / v2_all if v2_all > EPS else 0
            if n_eff > 1:
                var_pooled *= n_eff / (n_eff - 1)
        else:
            var_pooled = np.var(dat, axis=1)

    var_pooled = np.maximum(var_pooled, EPS)
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)[:, np.newaxis]

    # 3. Weighted Batch Effect Estimates
    gamma_hat = np.zeros((n_batch, n_genes_kept))
    delta_hat = np.zeros((n_batch, n_genes_kept))
    
    for i, b_idx in enumerate(batches):
        w_b = weights[b_idx]
        v1_b = np.sum(w_b)
        v2_b = np.sum(w_b**2)
        
        if v1_b > EPS:
            gamma_hat[i, :] = np.sum(s_data[:, b_idx] * w_b, axis=1) / v1_b
        else:
            gamma_hat[i, :] = np.mean(s_data[:, b_idx], axis=1)
            
        if len(b_idx) > 1:
            if v1_b > EPS:
                # Weighted variance
                d_biased = np.sum(w_b * (s_data[:, b_idx] - gamma_hat[i, :][:, np.newaxis])**2, axis=1) / v1_b
                n_eff_b = (v1_b**2) / v2_b if v2_b > EPS else 0
                if n_eff_b > 1:
                    d_biased *= n_eff_b / (n_eff_b - 1)
                delta_hat[i, :] = np.maximum(d_biased, EPS)
            else:
                delta_hat[i, :] = np.maximum(np.var(s_data[:, b_idx], axis=1, ddof=1), EPS)
        else:
            delta_hat[i, :] = 1.0

    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    a_prior = np.array([_aprior(delta_hat[i, :]) for i in range(n_batch)])
    b_prior = np.array([_bprior(delta_hat[i, :]) for i in range(n_batch)])

    # 4. EB estimation
    gamma_star = np.zeros((n_batch, n_genes_kept))
    delta_star = np.zeros((n_batch, n_genes_kept))
    for i, b_idx in enumerate(batches):
        g_star, d_star = _it_sol(
            s_data[:, b_idx], gamma_hat[i, :], delta_hat[i, :],
            gamma_bar[i], t2[i], a_prior[i], b_prior[i],
            weights=weights[b_idx]
        )
        gamma_star[i, :] = g_star
        delta_star[i, :] = d_star

    if ref is not None:
        gamma_star[ref, :] = 0.0
        delta_star[ref, :] = 1.0

    # 5. Apply correction
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
