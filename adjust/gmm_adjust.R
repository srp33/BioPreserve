# ============================================================================
# gmm_adjust.R - 1D Gaussian Mixture Model with posterior-mean priors
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(foreach)
  library(doParallel)
})

# -----------------------------------------------------------------------------
# SIMPLE FALLBACK FOR DEGENERATE GENES
# -----------------------------------------------------------------------------

simple_fallback <- function(gene_exp) {
  min_val <- min(gene_exp, na.rm = TRUE)
  x_transformed <- log(gene_exp - min_val + 1)
  n_valid <- sum(!is.na(x_transformed))
  if (n_valid > 1) {
    quantiles <- rank(x_transformed, na.last = "keep", ties.method = "average") / (n_valid + 1)
    qnorm(quantiles)
  } else x_transformed
}


# -----------------------------------------------------------------------------
# HELPER: COMPUTE GAUSSIAN PDF
# -----------------------------------------------------------------------------

#' Compute Gaussian PDF for multiple genes
#' 
#' Computes the probability density function of a Gaussian distribution:
#' PDF(x | μ, σ²) = (1 / (σ√(2π))) * exp(-(x-μ)² / (2σ²))
#' 
#' @param data Matrix [n_genes × n_samples] of data points
#' @param means Vector [n_genes] of means (μ)
#' @param variances Vector [n_genes] of variances (σ²)
#' @param weights Vector [n_genes] of mixture weights
#' @return Matrix [n_genes × n_samples] of weighted PDFs
compute_gaussian_pdf <- function(data, means, variances, weights) {
  centered <- data - means  # (x - μ)
  scaled <- centered^2 / variances  # (x - μ)² / σ²
  sds <- sqrt(variances)  # σ
  exp(-0.5 * scaled) * (weights / (sds * sqrt(2 * pi)))
}

#' Update variance with coupled Inverse-Gamma prior and dynamic hyperprior
#' 
#' Computes posterior mean of Inverse-Gamma distribution where the prior
#' mean is coupled to the other component's variance. This encourages
#' similar variances across components for stability. Includes dynamic
#' hyperprior that enforces variance similarity when components are close.
#' 
#' @param data Matrix [n_genes × n_samples] of data points
#' @param means Vector [n_genes] of component means
#' @param responsibilities Matrix [n_genes × n_samples] of responsibilities
#' @param Nk Vector [n_genes] of effective sample counts
#' @param variance_other Vector [n_genes] of other component's variances
#' @param alpha_v Inverse-Gamma shape parameter
#' @param hyperprior_strength Vector [n_genes] of hyperprior strengths (default 0.0)
#' @return Vector [n_genes] of updated variances
update_variance_coupled <- function(data, means, responsibilities, Nk, variance_other, alpha_v, hyperprior_strength = 0.0) {
  eps <- 1e-12
  
  # Compute sum of squared residuals weighted by responsibilities
  centered <- data - means
  S_k <- rowSums(responsibilities * centered^2)
  
  # 1. Base Prior (Standard Inverse-Gamma coupling)
  beta_base <- (alpha_v - 1) * variance_other * 1.0
  
  # 2. Hyperprior Injection
  if (any(hyperprior_strength > 0)) {
    # The strength acts as 'alpha' count for the hyperprior
    alpha_hyperprior <- hyperprior_strength
    # The 'beta' is derived assuming variance ratio is exactly 1.0
    beta_hyperprior <- hyperprior_strength * variance_other * 1.0
    
    alpha_prior <- alpha_v + alpha_hyperprior
    beta0 <- beta_base + beta_hyperprior
  } else {
    alpha_prior <- alpha_v
    beta0 <- beta_base
  }
  
  # 3. Posterior Update
  alpha_post <- alpha_prior + 0.5 * Nk
  beta_post <- beta0 + 0.5 * S_k
  
  # Posterior mean: beta_post / (alpha_post - 1)
  pmax(beta_post / (alpha_post - 1), 1e-6)
}


# -----------------------------------------------------------------------------
# VECTORIZED MINI-BATCH GMM FITTING
# -----------------------------------------------------------------------------

#' Fit GMM to multiple genes simultaneously (fully vectorized)
#' @param data_chunk Matrix [n_genes × n_samples] for this chunk
#' @param weight_alpha Prior for mixture weights
#' @param variance_alpha Prior for variances
#' @param hyperprior_strength Maximum weight of variance regularization (default: n_samples)
#' @param hyperprior_decay_rate Controls how fast regularization vanishes as components separate (default: 2.0)
#' @param max_iter Maximum EM iterations
#' @param tol Convergence tolerance
#' @return List with means, variances, weights matrices [n_genes × 2]
fit_gmm_batch <- function(data_chunk, weight_alpha = NULL, variance_alpha = NULL, 
                         hyperprior_strength = NULL, hyperprior_decay_rate = NULL,
                         max_iter = 100, tol = 1e-6) {
  # data_chunk is [n_genes × n_samples]
  n_genes <- nrow(data_chunk)
  n_samples <- ncol(data_chunk)
  K <- 2
  eps <- 1e-12
  
  # Initialize parameters for all genes [n_genes × K]
  means <- matrix(0, nrow = n_genes, ncol = K)
  variances <- matrix(0, nrow = n_genes, ncol = K)
  weights <- matrix(0.5, nrow = n_genes, ncol = K)
  
  # Vectorized initialization using mean ± 0.5*sd (faster than quantiles)
  row_sums <- rowSums(data_chunk)
  row_sq_sums <- rowSums(data_chunk^2)
  global_means <- row_sums / n_samples
  global_vars <- (row_sq_sums - (row_sums^2 / n_samples)) / (n_samples - 1)
  global_vars <- pmax(global_vars, 1e-6)
  global_sds <- sqrt(global_vars)
  
  means[, 1] <- global_means - 0.5 * global_sds
  means[, 2] <- global_means + 0.5 * global_sds
  variances[, 1] <- global_vars
  variances[, 2] <- global_vars
  
  # Set priors
  if (is.null(weight_alpha)) weight_alpha <- 3.0 + n_samples / 100.0
  if (is.null(variance_alpha)) variance_alpha <- 6.0 + n_samples / 50.0
  alpha_v <- max(variance_alpha, 1.01)
  
  # Set hyperprior defaults
  if (is.null(hyperprior_strength)) {
    # Default to equal weight with the data
    hyperprior_strength <- n_samples
  }
  if (is.null(hyperprior_decay_rate)) {
    hyperprior_decay_rate <- 2.0
  }
  
  # Track convergence per gene
  converged <- rep(FALSE, n_genes)
  log_likelihood_old <- rep(-Inf, n_genes)
  active_genes <- 1:n_genes
  
  for (iter in 1:max_iter) {
    if (length(active_genes) == 0) break
    
    # ========================================================================
    # VECTORIZED E-STEP: Compute responsibilities for all active genes at once
    # ========================================================================
    # Extract active gene data [n_active_genes × n_samples]
    data_active <- data_chunk[active_genes, , drop = FALSE]
    n_active <- length(active_genes)
    
    # Compute PDFs for both components using helper function
    pdf_k1 <- compute_gaussian_pdf(data_active, means[active_genes, 1], 
                                   variances[active_genes, 1], weights[active_genes, 1])
    pdf_k2 <- compute_gaussian_pdf(data_active, means[active_genes, 2], 
                                   variances[active_genes, 2], weights[active_genes, 2])
    
    # Normalize to get responsibilities [n_active × n_samples]
    pdf_sums <- pdf_k1 + pdf_k2 + eps
    resp_k1 <- pdf_k1 / pdf_sums
    resp_k2 <- pdf_k2 / pdf_sums
    
    # Transpose for compatibility with M-step [n_samples × n_active]
    resp_k1_t <- t(resp_k1)
    resp_k2_t <- t(resp_k2)
    
    # ========================================================================
    # VECTORIZED M-STEP: Update parameters for all active genes
    # ========================================================================
    
    # Compute Nk for all genes [n_active × K]
    Nk1 <- colSums(resp_k1_t)
    Nk2 <- colSums(resp_k2_t)
    
    variances_old <- variances[active_genes, , drop = FALSE]
    
    # 1) Update weights [n_active × K]
    weights[active_genes, 1] <- (Nk1 + weight_alpha) / (n_samples + K * weight_alpha)
    weights[active_genes, 2] <- (Nk2 + weight_alpha) / (n_samples + K * weight_alpha)
    
    # 2) Update means [n_active × K]
    weighted_sums1 <- rowSums(data_active * resp_k1)
    means[active_genes, 1] <- weighted_sums1 / (Nk1 + eps)
    
    weighted_sums2 <- rowSums(data_active * resp_k2)
    means[active_genes, 2] <- weighted_sums2 / (Nk2 + eps)
    
    # 2.5) Calculate dynamic hyperprior strength based on component separation
    # Calculate normalized separation: difference in means / scale of variance
    mean_sep <- means[active_genes, 2] - means[active_genes, 1]
    var_scale <- sqrt(variances_old[, 1] + variances_old[, 2])
    normalized_sep <- mean_sep / (var_scale + 1e-12)
    
    # Calculate dynamic strength: Strength * e^(-rate * separation)
    if (hyperprior_strength > 0) {
      current_strength <- hyperprior_strength * exp(-hyperprior_decay_rate * normalized_sep)
    } else {
      current_strength <- rep(0.0, n_active)
    }
    
    # 3) Update variances with coupling and dynamic hyperprior [n_active × K]
    variances[active_genes, 1] <- update_variance_coupled(
      data_active, means[active_genes, 1], resp_k1, Nk1, variances_old[, 2], alpha_v, current_strength
    )
    variances[active_genes, 2] <- update_variance_coupled(
      data_active, means[active_genes, 2], resp_k2, Nk2, variances_old[, 1], alpha_v, current_strength
    )
    
    # ========================================================================
    # VECTORIZED CONVERGENCE CHECK
    # ========================================================================
    log_likelihoods <- rowSums(log(pdf_sums))
    
    # Compute delta for all active genes at once
    delta <- abs(log_likelihoods - log_likelihood_old[active_genes])
    newly_converged <- delta < tol
    
    # Update convergence status
    converged[active_genes[newly_converged]] <- TRUE
    log_likelihood_old[active_genes] <- log_likelihoods
    
    # Update active genes (only non-converged)
    active_genes <- which(!converged)
  }
  
  return(list(means = means, variances = variances, weights = weights))
}

#' Apply GMM transformations to a batch of genes (fully vectorized)
#' @param data_chunk Matrix [n_genes × n_samples]
apply_gmm_transforms_batch <- function(data_chunk, gmm_params, mean_mean_zero, mean1_zero,
                                      unit_var, means_at_1, output_counts) {
  # data_chunk is [n_genes × n_samples]
  n_genes <- nrow(data_chunk)
  n_samples <- ncol(data_chunk)
  
  means <- gmm_params$means
  variances <- gmm_params$variances
  
  # Vectorized: Ensure lower component first for all genes
  swap_needed <- means[, 1] > means[, 2]
  if (any(swap_needed)) {
    means[swap_needed, ] <- means[swap_needed, c(2, 1)]
    variances[swap_needed, ] <- variances[swap_needed, c(2, 1)]
  }
  
  transformed <- data_chunk
  
  # All operations work on rows (genes)
  if (mean1_zero) {
    # Subtract first mean from each gene (row-wise)
    transformed <- transformed - means[, 1]
  }
  
  if (mean_mean_zero) {
    # Center means around zero
    mean_centers <- 0.5 * (means[, 1] + means[, 2])
    transformed <- transformed - mean_centers
  }
  
  if (unit_var) {
    # Scale to unit variance
    variances_total <- 0.5 * rowSums(variances) + 0.25 * (means[, 2] - means[, 1])^2
    scale_factors <- sqrt(pmax(variances_total, 1e-9))
    transformed <- transformed / scale_factors
  }
  
  if (means_at_1) {
    # Scale so means are at ±1 
    scale_factors <- (means[, 2] - means[, 1]) / 2
    transformed <- transformed / scale_factors
  }
  
  if (output_counts) {
    # Convert to count-like data
    transformed <- round(exp(transformed) * 250)
  }
  
  return(transformed)
}

#' Setup parallel cluster with fallback
#' @param num_workers Number of workers (-1 for all cores)
#' @param debug Print debug messages
#' @return Cluster object
setup_parallel_cluster <- function(num_workers, debug = FALSE) {
  num_cores <- if (num_workers == -1) detectCores() else min(num_workers, detectCores())
  if (debug) message("Using ", num_cores, " cores for parallel processing")
  
  if (.Platform$OS.type == "unix") {
    tryCatch({
      parallel::makeForkCluster(num_cores)
    }, error = function(e) {
      if (debug) cat("Fork cluster failed, using PSOCK:", e$message, "\n")
      parallel::makePSOCKcluster(num_cores)
    })
  } else {
    parallel::makePSOCKcluster(num_cores)
  }
}

#' Bimodal normalization using GMM with mini-batch vectorization and optional parallelization
#' @param data Matrix in [genes × samples] orientation
bimodal_normalize <- function(data, weight_alpha=NULL, variance_alpha=NULL, hyperprior_strength=NULL, hyperprior_decay_rate=NULL,
                             mean_mean_zero = TRUE, unit_var = TRUE, 
                             mean1_zero = FALSE, diff_exp = FALSE, means_at_1 = FALSE, 
                             output_counts = FALSE, log_transform = TRUE, debug = FALSE, 
                             num_workers = NULL, chunk_size = 200) {
  # Expect data in [genes × samples] orientation
  gene_names <- rownames(data)
  if (is.null(gene_names)) gene_names <- paste0("Gene", 1:nrow(data))

  if (diff_exp && unit_var) stop("Unit variance not allowed for diff exp")
  if (diff_exp && means_at_1) stop("Means at 1 not allowed for diff exp")
  if (means_at_1 && unit_var) stop("Cannot have both means_at_1 and unit_var")
  if (means_at_1 && !mean_mean_zero) stop("Cannot have means_at_1 without mean_mean_zero")
  
  n_genes <- nrow(data)
  n_samples <- ncol(data)
  bimodal_data <- matrix(NA, nrow = n_genes, ncol = n_samples)
  
  # Vectorized log transform
  if (log_transform) {
    row_mins <- apply(data, 1, min, na.rm = TRUE)
    data <- log(sweep(data, 1, row_mins, "-") + 1)
  }
  
  # Vectorized variance calculation (faster than apply)
  row_sums <- rowSums(data, na.rm = TRUE)
  row_sq_sums <- rowSums(data^2, na.rm = TRUE)
  n_valid <- rowSums(!is.na(data))
  gene_vars <- (row_sq_sums - (row_sums^2 / n_valid)) / (n_valid - 1)
  degenerate <- is.na(gene_vars) | gene_vars < 1e-8
  
  good_genes <- which(!degenerate)
  bad_genes <- which(degenerate)
  
  if (debug) {
    message("Processing ", length(good_genes), " non-degenerate genes with GMM")
    if (length(bad_genes) > 0) {
      message("Using fallback for ", length(bad_genes), " degenerate genes")
    }
  }
  
  # Apply fallback to degenerate genes
  if (length(bad_genes) > 0) {
    for (g in bad_genes) {
      bimodal_data[g, ] <- simple_fallback(data[g, ])
    }
  }
  
  # Fail fast
  if (length(good_genes) == 0) {
    rownames(bimodal_data) <- gene_names
    colnames(bimodal_data) <- colnames(data)
    return(bimodal_data)
  }
  
  # Create chunks from good genes only
  # Use larger chunks for sequential processing to reduce overhead
  effective_chunk_size <- if (is.null(num_workers) || num_workers == 1) {
    min(chunk_size * 5, length(good_genes))  # 5x larger chunks for sequential
  } else {
    chunk_size
  }
  
  chunks <- split(good_genes, ceiling(seq_along(good_genes) / effective_chunk_size))
  
  if (debug) message("Processing in ", length(chunks), " chunks of size ~", effective_chunk_size)
  
  # Decide on parallelization
  use_parallel <- !is.null(num_workers) && (num_workers != 1) && length(chunks) > 1
  
  if (use_parallel) {
    cl <- setup_parallel_cluster(num_workers, debug)
    registerDoParallel(cl)
    on.exit(stopCluster(cl), add = TRUE)
    
    # Process chunks in parallel
    # Explicitly export helper functions to avoid namespace collisions
    chunk_results <- foreach(i = seq_along(chunks), .packages = c("stats"),
                             .export = c("fit_gmm_batch", "apply_gmm_transforms_batch", 
                                       "compute_gaussian_pdf", "update_variance_coupled"),
                             .errorhandling = 'pass') %dopar% {
      chunk_idx <- chunks[[i]]
      chunk_data <- data[chunk_idx, , drop = FALSE]
      gmm_params <- fit_gmm_batch(chunk_data, weight_alpha, variance_alpha, hyperprior_strength, hyperprior_decay_rate)
      apply_gmm_transforms_batch(chunk_data, gmm_params, mean_mean_zero, 
                                mean1_zero, unit_var, means_at_1, output_counts)
    }
    
    # Combine results
    for (i in seq_along(chunks)) {
      chunk_idx <- chunks[[i]]
      result_chunk <- chunk_results[[i]]
      if (!is.null(result_chunk) && !inherits(result_chunk, "error")) {
        bimodal_data[chunk_idx, ] <- result_chunk
      } else {
        # Fallback for failed chunks
        for (g in chunk_idx) {
          bimodal_data[g, ] <- simple_fallback(data[g, ])
        }
      }
    }
    
  } else {
    # Sequential processing of chunks
    if (debug) message("Sequential chunk processing")
    
    for (chunk_idx in chunks) {
      chunk_data <- data[chunk_idx, , drop = FALSE]
      gmm_params <- fit_gmm_batch(chunk_data, weight_alpha, variance_alpha, hyperprior_strength, hyperprior_decay_rate)
      bimodal_data[chunk_idx, ] <- apply_gmm_transforms_batch(chunk_data, gmm_params, mean_mean_zero, 
                                                              mean1_zero, unit_var, means_at_1, output_counts)
    }
  }
  
  rownames(bimodal_data) <- gene_names
  colnames(bimodal_data) <- colnames(data)
  return(bimodal_data)
}

#' GMM adjustment for multiple batches
#' 
#' @param data Matrix of gene expression data (samples x genes)
#' @param batch Vector of batch labels for each sample
#' @param weight_alpha Dirichlet prior parameter for GMM weights
#' @param variance_alpha Prior for variances
#' @param hyperprior_strength Maximum weight of variance regularization
#' @param hyperprior_decay_rate Decay rate for variance regularization
#' @param mean_mean_zero If TRUE, center means around zero (default TRUE)
#' @param unit_var If TRUE, scale to unit variance (default TRUE)
#' @param diff_exp If TRUE, adjust first mean to zero for differential expression preservation (default FALSE)
#' @param means_at_1 If TRUE, place means at ±1 (default FALSE)
#' @param output_counts If TRUE, attempt to preserve count structure (default FALSE)
#' @param log_transform If TRUE, apply log-transformation to data (default TRUE). Set to FALSE for pre-transformed data.
#' @param debug If TRUE, print progress messages
#' 
#' @details The function applies GMM-based bimodal normalization to each batch separately.
#' Various transformation options allow control over the final distribution properties.
gmm_adjust <- function(data, batch, genes_are_columns=TRUE, weight_alpha=NULL, variance_alpha=NULL,
                      hyperprior_strength=NULL, hyperprior_decay_rate=2.0,
                      mean_mean_zero = TRUE, mean1_zero = FALSE, unit_var = TRUE, diff_exp = FALSE, 
                      means_at_1 = FALSE, output_counts = FALSE, log_transform = TRUE, 
                      debug = FALSE, num_workers = NULL, chunk_size = 200) {
  
  batch_factor <- as.factor(batch)
  batch_levels <- levels(batch_factor)
  
  # Work in [genes × samples] orientation throughout
  needs_transpose_back <- FALSE
  if (genes_are_columns) {
    if (debug) message("Transposing to [genes × samples] orientation")
    data <- t(data)
    needs_transpose_back <- TRUE
  }
  # Now data is [genes × samples]
  
  adjusted_data <- matrix(NA, nrow = nrow(data), ncol = ncol(data))

  if (debug) message("Batch: ", paste(head(batch), collapse=", "))
  
  for (b in batch_levels) {
    batch_indices <- which(batch == b)
    # Extract batch samples (columns in [genes × samples])
    batch_data <- data[, batch_indices, drop = FALSE]
    
    batch_adjusted <- bimodal_normalize(
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
    
    adjusted_data[, batch_indices] <- batch_adjusted
  }
  
  rownames(adjusted_data) <- rownames(data)
  colnames(adjusted_data) <- colnames(data)

  if (needs_transpose_back) {
    if (debug) message("Transposing back to [samples × genes]")
    adjusted_data <- t(adjusted_data)
  }
  return(adjusted_data)
}