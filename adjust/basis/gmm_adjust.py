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


def update_variance_coupled(data, means, responsibilities, Nk, variance_other, alpha_v, 
                           hyperprior_strength=0.0):
    """
    Update variance with coupled Inverse-Gamma prior and optional hyperprior.
    
    Computes posterior mean of Inverse-Gamma distribution where the prior
    mean is coupled to the other component's variance with ratio=1.0. This encourages
    similar variances across components for stability.
    
    With hyperprior_strength > 0, adds a hierarchical prior on beta that
    enforces variance ratio = 1.0, with strength proportional to hyperprior_strength.
    
    Parameters:
    -----------
    data : ndarray [n_genes × n_samples]
    means : ndarray [n_genes, 1] of component means
    responsibilities : ndarray [n_genes × n_samples]
    Nk : ndarray [n_genes] of effective sample counts
    variance_other : ndarray [n_genes] of other component's variances
    alpha_v : float, Inverse-Gamma shape parameter
    hyperprior_strength : float, strength of hyperprior enforcing ratio=1.0 (default 0.0)
                         Acts as pseudo-observations favoring ratio=1.0
    
    Returns:
    --------
    ndarray [n_genes] of updated variances
    """
    # Compute sum of squared residuals weighted by responsibilities
    centered = data - means
    S_k = (responsibilities * centered**2).sum(axis=1)
    
    # Prior: beta0 = (alpha - 1) * variance_other * 1.0 (fixed ratio)
    # This couples the prior to the other component's variance with ratio=1.0
    
    # Hyperprior: Adds pseudo-observations enforcing ratio=1.0
    # hyperprior_strength acts as additional prior observations
    if np.any(hyperprior_strength > 0):
        # Add hyperprior pseudo-observations to both alpha and beta
        # This enforces ratio=1.0 with strength proportional to hyperprior_strength
        alpha_hyperprior = hyperprior_strength
        beta_hyperprior = hyperprior_strength * variance_other * 1.0  # Fixed ratio = 1.0
        
        # Base prior (with fixed ratio=1.0)
        beta_base = (alpha_v - 1) * variance_other * 1.0
        
        # Combined prior (base + hyperprior)
        alpha_prior = alpha_v + alpha_hyperprior
        beta0 = beta_base + beta_hyperprior
    else:
        # No hyperprior: use fixed ratio=1.0 with base alpha
        alpha_prior = alpha_v
        beta0 = (alpha_v - 1) * variance_other * 1.0
    
    # Posterior parameters
    alpha_post = alpha_prior + 0.5 * Nk
    beta_post = beta0 + 0.5 * S_k
    
    # Posterior mean: beta_post / (alpha_post - 1)
    return np.maximum(beta_post / (alpha_post - 1), 1e-6)


def fit_gmm_batch(data_chunk, weight_alpha=None, variance_alpha=None, 
                  hyperprior_strength=None, hyperprior_decay_rate=2.0, max_iter=200, tol=1e-5):
    """
    Fit GMM to multiple genes simultaneously (Fully Vectorized - No Loops).
    
    Parameters:
    -----------
    hyperprior_strength : float, maximum strength of hyperprior enforcing variance ratio = 1.0
                         (default 0.0 = no hyperprior). Acts as pseudo-observations.
                         Strength decays exponentially with mean separation.
    hyperprior_decay_rate : float, rate of exponential decay for hyperprior strength
                           (default 2.0). Higher values = faster decay with separation.
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
        weight_alpha = 4.0 + n_samples / 30.0
    if variance_alpha is None:
        variance_alpha = 6.0 + n_samples / 50.0
    alpha_v = max(variance_alpha, 1.01)

    if hyperprior_strength is None:
        hyperprior_strength = n_samples
    
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
        
        # Calculate mean separation (normalized by variance scale)
        mean_sep = means[active_genes, 1] - means[active_genes, 0]
        var_scale = np.sqrt(variances_old[:, 0] + variances_old[:, 1])
        normalized_sep = mean_sep / (var_scale + eps)
        
        # Compute hyperprior strength (decays with mean separation)
        # When separation is small, strongly enforce ratio = 1.0
        # When separation is large, allow more flexibility
        if hyperprior_strength > 0:
            # Exponential decay: strength * exp(-decay_rate * separation)
            current_hyperprior_strength = hyperprior_strength * np.exp(-hyperprior_decay_rate * normalized_sep)
        else:
            current_hyperprior_strength = 0.0
        
        # Update variances with fixed ratio = 1.0
        # Lower component: coupled to upper with ratio = 1.0
        variances[active_genes, 0] = update_variance_coupled(
            data_active, means[active_genes, 0:1], resp_k1, Nk1, variances_old[:, 1], 
            alpha_v, current_hyperprior_strength
        )
        # Upper component: coupled to lower with ratio = 1.0
        variances[active_genes, 1] = update_variance_coupled(
            data_active, means[active_genes, 1:2], resp_k2, Nk2, variances_old[:, 0], 
            alpha_v, current_hyperprior_strength
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
    
    # Calculate final responsibilities for all genes
    final_resp_k1 = np.zeros((n_genes, n_samples))
    final_resp_k2 = np.zeros((n_genes, n_samples))
    
    pdf_k1 = compute_gaussian_pdf(data_chunk, means[:, 0:1], 
                                  variances[:, 0:1], weights[:, 0:1])
    pdf_k2 = compute_gaussian_pdf(data_chunk, means[:, 1:2], 
                                  variances[:, 1:2], weights[:, 1:2])
    
    pdf_sums = pdf_k1 + pdf_k2 + eps
    final_resp_k1 = pdf_k1 / pdf_sums
    final_resp_k2 = pdf_k2 / pdf_sums
    
    return {'means': means, 'variances': variances, 'weights': weights, 
            'responsibilities': (final_resp_k1, final_resp_k2)}


def apply_gmm_transforms_batch(data_chunk, gmm_params, mean_mean_zero, mean1_zero,
                               unit_var, means_at_1, output_counts):
    """
    Apply GMM transformations to a batch of genes (vectorized).
    
    Parameters:
    -----------
    data_chunk : ndarray [n_genes × n_samples]
    gmm_params : dict with 'means', 'variances', 'weights'
    
    Returns:
    --------
    transformed : ndarray [n_genes × n_samples]
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


def bimodal_normalize(data, weight_alpha=None, variance_alpha=None,
                     hyperprior_strength=100.0, hyperprior_decay_rate=2.0, mean_mean_zero=True,
                     unit_var=True, mean1_zero=False, diff_exp=False, means_at_1=False,
                     output_counts=False, log_transform=True, debug=False,
                     num_workers=None, chunk_size=200):
    """
    Bimodal normalization using GMM with mini-batch vectorization.
    
    Parameters:
    -----------
    data : ndarray [n_genes × n_samples]
    chunk_size : int, number of genes per chunk
    num_workers : int, number of parallel workers (-1 for all cores)
    
    Returns:
    --------
    bimodal_data : ndarray [n_genes × n_samples]
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
                                      hyperprior_strength, hyperprior_decay_rate)
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
                                      hyperprior_strength, hyperprior_decay_rate)
            bimodal_data[chunk_idx, :] = apply_gmm_transforms_batch(
                chunk_data, gmm_params, mean_mean_zero, mean1_zero,
                unit_var, means_at_1, output_counts)
    
    return bimodal_data


def get_gmm_responsibilities(data, genes_are_rows=False, weight_alpha=None, variance_alpha=None,
                            hyperprior_strength=None, hyperprior_decay_rate=2.0,
                            log_transform=True, debug=False, chunk_size=200):
    """
    Get GMM responsibilities for genes without applying transformations.
    
    Returns:
    --------
    dict with keys:
        'responsibilities': tuple of (resp_lower, resp_upper) arrays [n_genes × n_samples]
        'means': array [n_genes × 2] of component means
        'variances': array [n_genes × 2] of component variances
        'weights': array [n_genes × 2] of component weights
    """
    # Work in [genes × samples] orientation
    if not genes_are_rows:
        data = data.T
    
    n_genes, n_samples = data.shape
    
    # Log transform if needed
    if log_transform:
        min_vals = np.nanmin(data, axis=1, keepdims=True)
        data = np.log(data - min_vals + 1.0)
    
    # Fit GMM
    gmm_params = fit_gmm_batch(data, weight_alpha, variance_alpha, 
                              hyperprior_strength, hyperprior_decay_rate)
    
    # Get responsibilities
    resp_lower, resp_upper = gmm_params['responsibilities']
    means = gmm_params['means'].copy()
    variances = gmm_params['variances'].copy()
    weights = gmm_params['weights'].copy()
    
    # Ensure lower component is actually lower
    swap_needed = means[:, 0] > means[:, 1]
    if np.any(swap_needed):
        # Swap responsibilities
        temp = resp_lower[swap_needed, :].copy()
        resp_lower[swap_needed, :] = resp_upper[swap_needed, :]
        resp_upper[swap_needed, :] = temp
        
        # Swap means, variances, and weights
        means[swap_needed, :] = means[swap_needed, :][:, [1, 0]]
        variances[swap_needed, :] = variances[swap_needed, :][:, [1, 0]]
        weights[swap_needed, :] = weights[swap_needed, :][:, [1, 0]]
    
    if not genes_are_rows:
        resp_lower = resp_lower.T
        resp_upper = resp_upper.T
    
    return {
        'responsibilities': (resp_lower, resp_upper),
        'means': means,
        'variances': variances,
        'weights': weights
    }


def adjust(data, batch, genes_are_rows=False, weight_alpha=None, variance_alpha=None,
              hyperprior_strength=None, hyperprior_decay_rate=2.0,
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
            hyperprior_strength=None,
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
