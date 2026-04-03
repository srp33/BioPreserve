"""
gmm_adjust_vectorized.py - Vectorized GMM batch normalization

Vectorized 2-component Gaussian Mixture Model with mini-batch processing
and optional parallelization. Matches the refactored R implementation.
"""

import numpy as np
from scipy.stats import norm, rankdata
from multiprocessing import Pool, cpu_count


def simple_fallback(gene_exp):
    """Simple fallback: log-transform and quantile normalization"""
    gene_exp = np.asarray(gene_exp)
    if gene_exp.size == 0:
        return gene_exp
    
    min_val = np.nanmin(gene_exp)
    x_transformed = np.log(gene_exp - min_val + 1.0)
    n_valid = np.sum(~np.isnan(x_transformed))
    
    if n_valid > 1:
        ranks = rankdata(x_transformed, method='average', nan_policy='omit')
        quantiles = ranks / (n_valid + 1.0)
        return norm.ppf(quantiles)
    else:
        return x_transformed


def compute_gaussian_pdf(data, means, variances, weights):
    """
    Compute Gaussian PDF for multiple genes.
    
    Computes the probability density function of a Gaussian distribution:
    PDF(x | μ, σ²) = (1 / (σ√(2π))) * exp(-(x-μ)² / (2σ²))
    
    Parameters:
    -----------
    data : ndarray [n_genes × n_samples]
    means : ndarray [n_genes, 1] of means (μ)
    variances : ndarray [n_genes, 1] of variances (σ²)
    weights : ndarray [n_genes, 1] of mixture weights
    
    Returns:
    --------
    ndarray [n_genes × n_samples] of weighted PDFs
    """
    centered = data - means  # (x - μ)
    scaled = centered**2 / variances  # (x - μ)² / σ²
    sds = np.sqrt(variances)  # σ
    return np.exp(-0.5 * scaled) * (weights / (sds * np.sqrt(2 * np.pi)))


def update_variance_coupled(data, means, responsibilities, Nk, variance_other, alpha_v, hyperprior_strength=0.0):
    """
    Update variance with coupled Inverse-Gamma prior and dynamic hyperprior.
    
    Computes posterior mean of Inverse-Gamma distribution where the prior
    mean is coupled to the other component's variance. This encourages
    similar variances across components for stability.
    
    Parameters:
    -----------
    data : ndarray [n_genes × n_samples]
    means : ndarray [n_genes, 1] of component means
    responsibilities : ndarray [n_genes × n_samples]
    Nk : ndarray [n_genes] of effective sample counts
    variance_other : ndarray [n_genes] of other component's variances
    alpha_v : float, Inverse-Gamma shape parameter
    hyperprior_strength : float or ndarray, strength of hyperprior regularization
    
    Returns:
    --------
    ndarray [n_genes] of updated variances
    """
    # Compute sum of squared residuals weighted by responsibilities
    centered = data - means
    S_k = (responsibilities * centered**2).sum(axis=1)
    
    # 1. Base Prior (Standard Inverse-Gamma coupling)
    beta_base = (alpha_v - 1) * variance_other * 1.0
    
    # 2. Hyperprior Injection
    if np.any(hyperprior_strength > 0):
        # The strength acts as 'alpha' count for the hyperprior
        alpha_hyperprior = hyperprior_strength
        # The 'beta' is derived assuming variance ratio is exactly 1.0
        beta_hyperprior = hyperprior_strength * variance_other * 1.0
        
        alpha_prior = alpha_v + alpha_hyperprior
        beta0 = beta_base + beta_hyperprior
    else:
        alpha_prior = alpha_v
        beta0 = beta_base
    
    # 3. Posterior Update
    alpha_post = alpha_prior + 0.5 * Nk
    beta_post = beta0 + 0.5 * S_k
    
    # Posterior mean: beta_post / (alpha_post - 1)
    return np.maximum(beta_post / (alpha_post - 1), 1e-6)


def fit_gmm_batch(data_chunk, weight_alpha=None, variance_alpha=None, max_iter=100, tol=1e-4, 
                  hyperprior_strength=None, hyperprior_decay_rate=None):
    """
    Fit GMM to multiple genes simultaneously (Fully Vectorized - No Loops).
    """
    n_genes, n_samples = data_chunk.shape
    K = 2
    eps = 1e-12
    
    # Initialize parameters [n_genes × K]
    weights = np.full((n_genes, K), 0.5)
    
    # --- VECTORIZED INITIALIZATION (Replaces Loop) ---
    # Use Mean +/- 0.5 * SD instead of sorting for quantiles
    # This is O(N) instead of O(N log N) and fully vectorized
    means = np.zeros((n_genes, K))
    variances = np.zeros((n_genes, K))
    
    row_means = np.nanmean(data_chunk, axis=1)
    row_vars = np.nanvar(data_chunk, axis=1)
    row_sds = np.sqrt(row_vars)
    
    # Initialize components split around the mean
    means[:, 0] = row_means - 0.5 * row_sds
    means[:, 1] = row_means + 0.5 * row_sds
    
    # Clamp variances to avoid singularities
    variances[:, 0] = np.maximum(row_vars, 1e-6)
    variances[:, 1] = variances[:, 0].copy()
    # ------------------------------------------------
    
    # Set priors
    if weight_alpha is None:
        weight_alpha = 3.0 + n_samples / 100.0
    if variance_alpha is None:
        variance_alpha = 6.0 + n_samples / 50.0
    alpha_v = max(variance_alpha, 1.01)
    
    # Set hyperprior defaults
    if hyperprior_strength is None:
        # Default to equal weight with the data
        hyperprior_strength = n_samples
    
    if hyperprior_decay_rate is None:
        hyperprior_decay_rate = 2.0
    
    # Track convergence
    converged = np.zeros(n_genes, dtype=bool)
    log_likelihood_old = np.full(n_genes, -np.inf)
    active_genes = np.arange(n_genes)
    
    for iteration in range(max_iter):
        if len(active_genes) == 0:
            break
        
        # E-step
        data_active = data_chunk[active_genes, :]
        
        pdf_k1 = compute_gaussian_pdf(data_active, means[active_genes, 0:1], 
                                      variances[active_genes, 0:1], weights[active_genes, 0:1])
        pdf_k2 = compute_gaussian_pdf(data_active, means[active_genes, 1:2], 
                                      variances[active_genes, 1:2], weights[active_genes, 1:2])
        
        pdf_sums = pdf_k1 + pdf_k2 + eps
        resp_k1 = pdf_k1 / pdf_sums
        resp_k2 = pdf_k2 / pdf_sums
        
        # M-step
        Nk1 = resp_k1.sum(axis=1)
        Nk2 = resp_k2.sum(axis=1)
        
        variances_old = variances[active_genes, :].copy()
        
        # Update weights
        weights[active_genes, 0] = (Nk1 + weight_alpha) / (n_samples + K * weight_alpha)
        weights[active_genes, 1] = (Nk2 + weight_alpha) / (n_samples + K * weight_alpha)
        
        # Update means
        means[active_genes, 0] = (data_active * resp_k1).sum(axis=1) / (Nk1 + eps)
        means[active_genes, 1] = (data_active * resp_k2).sum(axis=1) / (Nk2 + eps)
        
        # Calculate dynamic hyperprior strength based on component separation
        # 1. Calculate Normalized Separation
        # separation = difference in means / scale of variance
        mean_sep = means[active_genes, 1] - means[active_genes, 0]
        var_scale = np.sqrt(variances_old[:, 0] + variances_old[:, 1])
        normalized_sep = mean_sep / (var_scale + 1e-12)
        
        # 2. Calculate Dynamic Strength
        # Decay formula: Strength * e^(-rate * separation)
        if hyperprior_strength > 0:
            current_strength = hyperprior_strength * np.exp(-hyperprior_decay_rate * normalized_sep)
        else:
            current_strength = 0.0
        
        # Update variances with dynamic hyperprior
        variances[active_genes, 0] = update_variance_coupled(
            data_active, means[active_genes, 0:1], resp_k1, Nk1, variances_old[:, 1], alpha_v, current_strength
        )
        variances[active_genes, 1] = update_variance_coupled(
            data_active, means[active_genes, 1:2], resp_k2, Nk2, variances_old[:, 0], alpha_v, current_strength
        )
        
        # --- VECTORIZED CONVERGENCE CHECK (Replaces Loop) ---
        current_ll = np.log(pdf_sums).sum(axis=1)
        
        # Calculate diff for ALL active genes at once
        delta = np.abs(current_ll - log_likelihood_old[active_genes])
        
        # Boolean mask of who converged this step
        newly_converged = delta < tol
        
        # Update global state
        # Map active indices back to global indices
        converged_indices = active_genes[newly_converged]
        converged[converged_indices] = True
        
        # Update history
        log_likelihood_old[active_genes] = current_ll
        
        # Update active set
        active_genes = active_genes[~newly_converged]
        # ----------------------------------------------------
    
    return {'means': means, 'variances': variances, 'weights': weights}


def get_gmm_responsibilities(data, genes_are_rows=False, log_transform=True, **kwargs):
    """
    Fits GMM and returns responsibilities.
    Matches the signature expected by BASIS dictionary building.
    """
    if not genes_are_rows:
        data = data.T
    
    n_genes, n_samples = data.shape
    
    if log_transform:
        min_vals = np.nanmin(data, axis=1, keepdims=True)
        data = np.log(data - min_vals + 1.0)
    
    # Use existing vectorized fitting logic
    params = fit_gmm_batch(data, **kwargs)
    
    # Final E-step to get responsibilities
    means = params['means']
    variances = params['variances']
    weights = params['weights']
    eps = 1e-12
    
    pdf_k1 = compute_gaussian_pdf(data, means[:, 0:1], variances[:, 0:1], weights[:, 0:1])
    pdf_k2 = compute_gaussian_pdf(data, means[:, 1:2], variances[:, 1:2], weights[:, 1:2])
    
    pdf_sums = pdf_k1 + pdf_k2 + eps
    resp_k1 = pdf_k1 / pdf_sums
    resp_k2 = pdf_k2 / pdf_sums
    
    # BASIS expects [lower, upper] based on means
    swap_needed = means[:, 0] > means[:, 1]
    if np.any(swap_needed):
        r1_final = resp_k1.copy()
        r2_final = resp_k2.copy()
        r1_final[swap_needed, :] = resp_k2[swap_needed, :]
        r2_final[swap_needed, :] = resp_k1[swap_needed, :]
        return {"responsibilities": (r1_final, r2_final), "params": params}
    
    return {"responsibilities": (resp_k1, resp_k2), "params": params}


def apply_gmm_transforms_batch(data_chunk, gmm_params, mean_mean_zero, mean1_zero,
                               unit_var, means_at_1, output_counts):
    """
    Apply GMM transformations to a batch of genes (vectorized).
    """
    n_genes, n_samples = data_chunk.shape
    means = gmm_params['means'].copy()
    variances = gmm_params['variances'].copy()
    
    # Ensure lower component first
    swap_needed = means[:, 0] > means[:, 1]
    if np.any(swap_needed):
        means[swap_needed, :] = means[swap_needed, :][:, [1, 0]]
        variances[swap_needed, :] = variances[swap_needed, :][:, [1, 0]]
    
    transformed = data_chunk.copy()
    
    # Apply transformations (row-wise operations)
    if mean1_zero:
        transformed = transformed - means[:, 0:1]
    
    if mean_mean_zero:
        mean_centers = 0.5 * (means[:, 0] + means[:, 1])
        transformed = transformed - mean_centers[:, np.newaxis]
    
    if unit_var:
        variances_total = 0.5 * variances.sum(axis=1) + 0.25 * (means[:, 1] - means[:, 0])**2
        scale_factors = np.sqrt(np.maximum(variances_total, 1e-9))
        transformed = transformed / scale_factors[:, np.newaxis]
    
    if means_at_1:
        scale_factors = (means[:, 1] - means[:, 0]) / 2
        transformed = transformed / scale_factors[:, np.newaxis]
    
    if output_counts:
        transformed = np.round(np.exp(transformed) * 250)
    
    return transformed


def bimodal_normalize(data, weight_alpha=None, variance_alpha=None, mean_mean_zero=True,
                     unit_var=True, mean1_zero=False, diff_exp=False, means_at_1=False,
                     output_counts=False, log_transform=True, debug=False,
                     num_workers=None, chunk_size=200, hyperprior_strength=None, 
                     hyperprior_decay_rate=None):
    """
    Bimodal normalization using GMM with mini-batch vectorization.
    """
    if diff_exp and unit_var:
        raise ValueError("Unit variance not allowed for diff exp")
    if diff_exp and means_at_1:
        raise ValueError("Means at 1 not allowed for diff exp")
    if means_at_1 and unit_var:
        raise ValueError("Cannot have both means_at_1 and unit_var")
    if means_at_1 and not mean_mean_zero:
        raise ValueError("Cannot have means_at_1 without mean_mean_zero")
    
    n_genes, n_samples = data.shape
    bimodal_data = np.full((n_genes, n_samples), np.nan)
    
    # --- VECTORIZED LOG TRANSFORM (Replaces Loop) ---
    if log_transform:
        # Calculate min per row, keeping dimensions (n_genes, 1) for broadcasting
        min_vals = np.nanmin(data, axis=1, keepdims=True)
        # Broadcasting: (n_genes, n_samples) - (n_genes, 1)
        data = np.log(data - min_vals + 1.0)
    # ------------------------------------------------
    
    # Identify degenerate genes
    gene_vars = np.var(data, axis=1)
    degenerate = np.isnan(gene_vars) | (gene_vars < 1e-8)
    
    good_genes = np.where(~degenerate)[0]
    bad_genes = np.where(degenerate)[0]
    
    if debug:
        print(f"Processing {len(good_genes)} non-degenerate genes with GMM")
        if len(bad_genes) > 0:
            print(f"Using fallback for {len(bad_genes)} degenerate genes")
    
    # Apply fallback to degenerate genes
    for g in bad_genes:
        bimodal_data[g, :] = simple_fallback(data[g, :])
    
    if len(good_genes) == 0:
        return bimodal_data
    
    # Create chunks from good genes only
    n_chunks = int(np.ceil(len(good_genes) / chunk_size))
    chunks = np.array_split(good_genes, n_chunks)
    
    if debug:
        print(f"Processing in {len(chunks)} chunks of size ~{chunk_size}")
    
    # Process chunks
    use_parallel = num_workers is not None and num_workers != 1 and len(chunks) > 1
    
    if use_parallel:
        num_cores = cpu_count() if num_workers == -1 else min(num_workers, cpu_count())
        if debug:
            print(f"Using {num_cores} cores for parallel processing")
        
        def process_chunk(chunk_idx):
            chunk_data = data[chunk_idx, :]
            gmm_params = fit_gmm_batch(chunk_data, weight_alpha, variance_alpha, 
                                     hyperprior_strength=hyperprior_strength, 
                                     hyperprior_decay_rate=hyperprior_decay_rate)
            return apply_gmm_transforms_batch(chunk_data, gmm_params, mean_mean_zero,
                                             mean1_zero, unit_var, means_at_1, output_counts)
        
        with Pool(num_cores) as pool:
            chunk_results = pool.map(process_chunk, chunks)
        
        for i, chunk_idx in enumerate(chunks):
            bimodal_data[chunk_idx, :] = chunk_results[i]
    else:
        if debug:
            print("Sequential chunk processing")
        
        for chunk_idx in chunks:
            chunk_data = data[chunk_idx, :]
            gmm_params = fit_gmm_batch(chunk_data, weight_alpha, variance_alpha,
                                     hyperprior_strength=hyperprior_strength,
                                     hyperprior_decay_rate=hyperprior_decay_rate)
            bimodal_data[chunk_idx, :] = apply_gmm_transforms_batch(
                chunk_data, gmm_params, mean_mean_zero, mean1_zero,
                unit_var, means_at_1, output_counts)
    
    return bimodal_data


def gmm_adjust(data, batch, genes_are_rows=False, weight_alpha=None, variance_alpha=None,
              hyperprior_strength=None, hyperprior_decay_rate=2.0, # Added params here
              mean_mean_zero=True, mean1_zero=False, unit_var=True, diff_exp=False,
              means_at_1=False, output_counts=False, log_transform=True, debug=False,
              num_workers=None, chunk_size=200):
    """
    GMM adjustment for multiple batches.
    
    Parameters:
    -----------
    data : ndarray, gene expression data
    batch : array-like, batch labels for each sample
    genes_are_rows : bool, if True data is [genes × samples], else [samples × genes]
    
    Returns:
    --------
    adjusted_data : ndarray, same shape as input
    """
    batch = np.asarray(batch)
    
    # Work in [genes × samples] orientation
    needs_transpose_back = False
    if not genes_are_rows:
        if debug:
            print("Transposing to [genes × samples] orientation")
        data = data.T
        needs_transpose_back = True
    
    n_genes, n_samples = data.shape
    adjusted_data = np.full((n_genes, n_samples), np.nan)
    
    if debug:
        print(f"Batch: {batch}")
    
    batch_levels = np.unique(batch)
    for b in batch_levels:
        batch_indices = np.where(batch == b)[0]
        if debug:
            print(f"Batch indices: {batch_indices}")
        
        batch_data = data[:, batch_indices]
        batch_adjusted = bimodal_normalize(
            batch_data,
            weight_alpha=weight_alpha,
            variance_alpha=variance_alpha,
            hyperprior_strength=hyperprior_strength,
            hyperprior_decay_rate=hyperprior_decay_rate,
            mean_mean_zero=mean_mean_zero,
            unit_var=unit_var,
            mean1_zero=mean1_zero,
            diff_exp=diff_exp,
            means_at_1=means_at_1,
            output_counts=output_counts,
            log_transform=log_transform,
            debug=debug,
            num_workers=num_workers,
            chunk_size=chunk_size
        )
        adjusted_data[:, batch_indices] = batch_adjusted
    
    if needs_transpose_back:
        if debug:
            print("Transposing back to [samples × genes]")
        adjusted_data = adjusted_data.T
    
    return adjusted_data