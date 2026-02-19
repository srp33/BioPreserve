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
# GMM MODEL CONSTRUCTOR
# -----------------------------------------------------------------------------

#' 1D 2-component Gaussian Mixture Model
#'
#' @param max_iter Maximum number of EM iterations
#' @param tol Convergence tolerance for log-likelihood
#' @param weight_alpha Dirichlet prior pseudo-count for mixture weights (>= 1.0)
#' @param variance_alpha Inverse-Gamma prior shape parameter for variances (>1 for posterior mean to exist)
GaussianMixture1D <- function(
  max_iter = 100,
  tol = 1e-4,
  weight_alpha = NULL,
  variance_alpha = NULL
) {
  structure(list(
    max_iter = max_iter,
    tol = tol,
    weight_alpha = weight_alpha,
    variance_alpha = variance_alpha,
    means_ = NULL,
    variances_ = NULL,
    weights_ = NULL,
    resp_ = NULL
  ), class = "GaussianMixture1D")
}

# -----------------------------------------------------------------------------
# NORMAL PDF
# -----------------------------------------------------------------------------

normal_pdf <- function(x, mean, sd) {
  exp(-0.5 * ((x - mean) / sd)^2) / (sd * sqrt(2 * pi))
}

# -----------------------------------------------------------------------------
# FIT FUNCTION
# -----------------------------------------------------------------------------

fit.GaussianMixture1D <- function(model, X) {
  X <- as.vector(X)
  n <- length(X)
  K <- 2
  eps <- 1e-12

  # Initialize means at 25th and 75th percentiles
  means <- quantile(X, c(0.25, 0.75), na.rm = TRUE)
  variances <- rep(var(X, na.rm = TRUE) + 1e-6, K)
  weights <- rep(0.5, K)

  if (is.null(model$weight_alpha)) {
    model$weight_alpha <- 3.0 + n / 100.0
  }
  if (is.null(model$variance_alpha)) {
    model$variance_alpha <- 6.0 + n / 50.0
  }

  log_likelihood_old <- -Inf

  alpha_w <- model$weight_alpha
  alpha_v <- max(model$variance_alpha, 1.01)  # ensure posterior mean exists

  for (iter in 1:model$max_iter) {
    # -------------------------------
    # E-step: responsibilities
    # -------------------------------
    pdfs <- matrix(0, nrow = n, ncol = K)
    for (k in 1:K) {
      pdfs[, k] <- weights[k] * normal_pdf(X, means[k], sqrt(variances[k]))
    }
    resp <- pdfs / (rowSums(pdfs) + eps)

    # -------------------------------
    # M-step: posterior mean updates
    # -------------------------------
    Nk <- colSums(resp)
    variances_old <- variances

    # 1) Update weights: posterior mean of Dirichlet
    weights <- (Nk + alpha_w) / (n + K * alpha_w)

    # 2) Update means (weighted ML, no prior)
    for (k in 1:K) {
      means[k] <- sum(resp[, k] * X) / (Nk[k] + eps)
    }

    # 3) Update variances: inverse-gamma prior, coupled across components
    for (k in 1:K) {
      other_k <- 3 - k  # if k=1, other=2; if k=2, other=1
      if (Nk[k] < 1e-8) {
        variances[k] <- variances_old[k]
        next
      }

      S_k <- sum(resp[, k] * (X - means[k])^2)
      v_other <- variances_old[other_k]
      beta0 <- (alpha_v - 1) * v_other

      alpha_post <- alpha_v + 0.5 * Nk[k]
      beta_post <- beta0 + 0.5 * S_k

      if (alpha_post <= 1 + 1e-12) {
        variances[k] <- max(variances_old[k], 1e-6)
      } else {
        variances[k] <- beta_post / (alpha_post - 1)
      }
    }

    variances <- pmax(variances, 1e-6)

    # -------------------------------
    # Convergence check
    # -------------------------------
    log_likelihood <- sum(log(rowSums(pdfs) + eps))
    if (abs(log_likelihood - log_likelihood_old) < model$tol) break
    log_likelihood_old <- log_likelihood
  }

  model$means_ <- means
  model$variances_ <- variances
  model$weights_ <- weights
  model$resp_ <- resp

  return(model)
}

# -----------------------------------------------------------------------------
# INVERSE CDF / QUANTILE FUNCTION
# -----------------------------------------------------------------------------

inverse_cdf_gmm <- function(p, means, variances, weights) {
  p <- as.numeric(p)
  if (any(is.na(means)) || any(is.na(variances)) || any(is.na(weights))) {
    return(rep(NA, length(p)))
  }

  if (length(unique(means)) == 1 || any(variances < 1e-12)) {
    overall_mean <- sum(means * weights)
    overall_var <- sum(variances * weights) + sum((means - overall_mean)^2 * weights)
    return(qnorm(p, mean = overall_mean, sd = sqrt(max(overall_var, 1e-10))))
  }

  stds <- sqrt(pmax(variances, 1e-12))

  gmm_cdf <- function(x) sum(weights * pnorm(x, mean = means, sd = stds))

  min_bound <- min(means - 15 * stds)
  max_bound <- max(means + 15 * stds)
  if ((max_bound - min_bound) < 20) {
    center <- (min_bound + max_bound) / 2
    min_bound <- center - 10
    max_bound <- center + 10
  }

  solve_for_single_p <- function(p_val) {
    if (is.na(p_val)) return(NA)
    if (p_val <= 1e-12) return(min_bound - 5)
    if (p_val >= 1 - 1e-12) return(max_bound + 5)
    root_fun <- function(x) gmm_cdf(x) - p_val
    left <- min_bound; right <- max_bound
    f_left <- root_fun(left); f_right <- root_fun(right)

    expansions <- 0
    while (sign(f_left) == sign(f_right) && expansions < 4) {
      factor <- 2^(expansions + 1)
      left <- min_bound - 10 * factor
      right <- max_bound + 10 * factor
      f_left <- root_fun(left); f_right <- root_fun(right)
      expansions <- expansions + 1
    }

    tryCatch({
      if (sign(f_left) != sign(f_right)) uniroot(root_fun, interval = c(left, right), tol = 1e-5)$root
      else {
        overall_mean <- sum(means * weights)
        overall_var <- sum(variances * weights) + sum((means - overall_mean)^2 * weights)
        qnorm(p_val, mean = overall_mean, sd = sqrt(max(overall_var, 1e-10)))
      }
    }, error = function(e) {
      overall_mean <- sum(means * weights)
      overall_var <- sum(variances * weights) + sum((means - overall_mean)^2 * weights)
      qnorm(p_val, mean = overall_mean, sd = sqrt(max(overall_var, 1e-10)))
    })
  }

  result <- sapply(p, solve_for_single_p)
  bad <- which(is.na(result) | is.infinite(result))
  if (length(bad) > 0) {
    overall_mean <- sum(means * weights)
    overall_var <- sum(variances * weights) + sum((means - overall_mean)^2 * weights)
    result[bad] <- qnorm(p[bad], mean = overall_mean, sd = sqrt(max(overall_var, 1e-10)))
  }
  return(result)
}

# -----------------------------------------------------------------------------
# SIMPLE FALLBACK
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
# GENE TRANSFORM / BIMODAL NORMALIZE / BATCH ADJUST
# -----------------------------------------------------------------------------

get_gene_gmm_transform <- function(
    gene_exp,
    alpha0 = 10,
    nonlinear = TRUE,
    mean_mean_zero = TRUE,
    mean1_zero = FALSE,
    unit_var = TRUE,
    means_at_1 = FALSE,
    diff_exp = FALSE,
    output_counts = FALSE
) {
  if (all(is.na(gene_exp))) return(gene_exp)
  
  if (output_counts && any(gene_exp %% 1 != 0)) {
    warning("Not preserving counts, expression is not integral")
  }

  if (diff_exp && unit_var) stop("Unit variance not allowed for diff exp")
  if (diff_exp && means_at_1) stop("Means at 1 not allowed for diff exp")

  # --- Log-transform ---
  min_val <- min(gene_exp, na.rm = TRUE)
  x_transformed <- log(gene_exp - min_val + 1)
  mean_shift_fallback <- scale(x_transformed, scale = FALSE)[, 1]

  # Check for small variance
  if (var(x_transformed, na.rm = TRUE) < 1e-8) return(mean_shift_fallback)

  # --- Quantiles ---
  ranks <- rank(x_transformed, na.last = "keep", ties.method = "average")
  n_valid <- sum(!is.na(x_transformed))
  quantiles <- ranks / (n_valid + 1)

  # --- Fit 2-component GMM ---
  gmm <- GaussianMixture1D(alpha0 = alpha0)
  gmm <- fit.GaussianMixture1D(gmm, x_transformed)

  qnorm_fallback <- qnorm(quantiles)

  # Validate GMM parameters
  if (any(is.na(gmm$means_)) || any(is.na(gmm$variances_)) || any(is.na(gmm$weights_))) {
    return(qnorm_fallback)
  }

  # Ensure lower component first
  if (gmm$means_[1] > gmm$means_[2]) {
    means <- c(gmm$means_[2], gmm$means_[1])
    variances <- c(gmm$variances_[2], gmm$variances_[1])
    weights <- c(gmm$weights_[2], gmm$weights_[1])
  } else {
    means <- gmm$means_
    variances <- gmm$variances_
    weights <- gmm$weights_
  }

  # Check for small variances, likely not truly bimodal
  if (any(variances < 1e-9)) return(qnorm_fallback)

  # --- Nonlinear mapping to original GMM distribution ---
  if (nonlinear) {
    mapped <- tryCatch({
      inverse_cdf_gmm(quantiles, means = means, variances = variances, weights = weights)
    }, error = function(e) {
      NA
    })

    if (any(is.na(mapped) | is.infinite(mapped))) {
      x_transformed <- mean_shift_fallback
    } else {
      x_transformed <- mapped
    }
  }

  # --- Affine corrections ---
  if (mean1_zero) {
    # Adjusts the first mean to be 0
    if (mean_mean_zero) stop("Cannot have both mean1_zero and mean_mean_zero")
    if (means_at_1) stop("Cannot have both mean1_zero and means_at_1")
    x_transformed <- x_transformed - means[1]
  }

  if (mean_mean_zero) {
    # Centers the means on zero, one on either side
    mean_center <- 0.5 * (means[1] + means[2])
    x_transformed <- x_transformed - mean_center
  }

  if (unit_var) {
    # Scales the variance to be 1, assuming equal weights
    variance <- 0.5 * (variances[1] + variances[2]) + 0.25 * (means[2] - means[1])^2
    scale_factor <- sqrt(max(variance, 1e-9))
    x_transformed <- x_transformed / scale_factor
  }

  if (means_at_1) {
    # Puts the means at +- 1
    if (unit_var) stop("Cannot have both means_at_1 and unit_var")
    if (!mean_mean_zero) stop("Cannot have means_at_1 without mean_mean_zero")

    # Compute safe scaling so component means map to ±1
    scale_factor <- (means[2] - means[1]) / 2
    if (abs(scale_factor) < 1e-6) {
      return(mean_shift_fallback)
    }
    x_transformed <- x_transformed / scale_factor
  }

  if (output_counts) {
    x_transformed = round(exp(x_transformed) * 1000)
  }

  return(x_transformed)
}

#' Bimodal normalization using GMM with parallel processing
bimodal_normalize <- function(data, alpha0 = 10, nonlinear = TRUE, mean_mean_zero = TRUE, 
                             mean1_zero = FALSE, unit_var = TRUE, diff_exp = FALSE, means_at_1 = FALSE, 
                             output_counts = FALSE, debug = FALSE, num_workers = NULL) {
  gene_names <- colnames(data)
  if (is.null(gene_names)) gene_names <- paste0("Gene", 1:ncol(data))
  
  message("%%%%%%% Num workers: ", num_workers)
  # Use parallel processing if num_workers > 1
  if (!is.null(num_workers) && num_workers > 1) {
    num_cores <- if (num_workers == -1) detectCores() else min(num_workers, detectCores())
    message("%%%%%%% Num cores: ", num_cores)
    if (.Platform$OS.type == "unix") {
      cl <- tryCatch({
        parallel::makeForkCluster(num_cores)
      }, error = function(e) {
        if (debug) cat("Fork cluster failed, using PSOCK:", e$message, "\n")
        parallel::makePSOCKcluster(num_cores)
      })
    } else {
      if (debug) cat("Fork cluster successful")
      cl <- parallel::makePSOCKcluster(num_cores)
    }
    registerDoParallel(cl)
    on.exit(stopCluster(cl), add = TRUE)
    
    results <- foreach(i = seq_along(gene_names), .combine = cbind, 
                      .packages = c("stats"), .errorhandling = 'pass') %dopar% {
      gene_exp <- data[, i]
      if (all(is.na(gene_exp)) || all(gene_exp == gene_exp[1], na.rm = TRUE)) {
        print(gene_exp)
        stop("Stopping here")
        return(gene_exp)
      }
      
      tryCatch({
        get_gene_gmm_transform(gene_exp, alpha0, nonlinear, mean_mean_zero, mean1_zero, 
                              unit_var, diff_exp, means_at_1, output_counts)
      }, error = function(e) {
        message("Doing simple fallback")
        simple_fallback(gene_exp)
      })
    }
    bimodal_data <- as.matrix(results)
  } else {
    # Sequential processing
    message("Doing sequential processing!")
    bimodal_data <- matrix(NA, nrow = nrow(data), ncol = ncol(data))
    message("Gene names: ", gene_names)
    for (i in seq_along(gene_names)) {
      gene_exp <- data[, i]
      if (all(is.na(gene_exp)) || all(gene_exp == gene_exp[1], na.rm = TRUE)) {
        message("All NA")
        bimodal_data[, i] <- gene_exp
        next
      }
      
      tryCatch({
        bimodal_data[, i] <- get_gene_gmm_transform(gene_exp, alpha0, nonlinear, mean_mean_zero, 
                                                   mean1_zero, unit_var, diff_exp, means_at_1, output_counts)
      }, error = function(e) {
        message("Got Error, simple fallback")
        message(e)
        bimodal_data[, i] <- simple_fallback(gene_exp)
      })
    }
  }
  
  colnames(bimodal_data) <- gene_names
  rownames(bimodal_data) <- rownames(data)
  return(bimodal_data)
}

#' GMM adjustment for multiple batches
#' 
#' @param data Matrix of gene expression data (samples x genes)
#' @param batch Vector of batch labels for each sample
#' @param alpha0 Dirichlet prior parameter for GMM weights
#' @param nonlinear If TRUE, apply inverse CDF transformation (default TRUE)
#' @param mean_mean_zero If TRUE, center means around zero (default TRUE)
#' @param unit_var If TRUE, scale to unit variance (default TRUE)
#' @param diff_exp If TRUE, adjust first mean to zero for differential expression preservation (default FALSE)
#' @param means_at_1 If TRUE, place means at ±1 (default FALSE)
#' @param output_counts If TRUE, attempt to preserve count structure (default FALSE)
#' @param debug If TRUE, print progress messages
#' 
#' @details The function applies GMM-based bimodal normalization to each batch separately.
#' Various transformation options allow control over the final distribution properties.
gmm_adjust <- function(data, batch, alpha0 = 10, nonlinear = TRUE, mean_mean_zero = TRUE,
                      mean1_zero = FALSE, unit_var = TRUE, diff_exp = FALSE, means_at_1 = FALSE, 
                      output_counts = FALSE, debug = FALSE, num_workers = NULL) {
  batch_factor <- as.factor(batch)
  batch_levels <- levels(batch_factor)
  adjusted_data <- matrix(NA, nrow = nrow(data), ncol = ncol(data))

  if (debug) message("Debug: ",  batch)
  
  for (b in batch_levels) {
    batch_indices <- which(batch == b)
    if (debug) message("Batch indices: ", batch_indices)
    batch_data <- data[batch_indices, , drop = FALSE]
    
    batch_adjusted <- bimodal_normalize(
      batch_data, alpha0, nonlinear, mean_mean_zero, mean1_zero, 
      unit_var, diff_exp, means_at_1, output_counts, debug, num_workers
    )
    
    adjusted_data[batch_indices, ] <- batch_adjusted
  }
  
  colnames(adjusted_data) <- colnames(data)
  rownames(adjusted_data) <- rownames(data)
  return(adjusted_data)
}