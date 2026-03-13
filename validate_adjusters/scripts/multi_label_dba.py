#!/usr/bin/env python3
"""
Multi-Label Decision Boundary Alignment.

Key idea: Instead of optimizing for a single label, optimize shift/scale
parameters that work well across ALL metadata labels simultaneously.

This is more comparable to the Bayesian adjuster, which also adjusts for
all metadata at once.
"""

import argparse
import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier, LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from pathlib import Path


def train_classifiers_for_labels(X_train, labels_train, label_types, classifier='histgradient'):
    """
    Pre-train classifiers for all labels to avoid retraining during optimization.
    
    Returns:
        dict: {label_name: {'clf': trained_classifier, 'scaler': scaler or None, 
                           'mask_train': mask, 'X_train_filtered': filtered_X_train}}
    """
    trained_models = {}
    
    print(f"  Pre-training {len(labels_train)} classifiers (one per label)...", flush=True)
    
    trained_count = 0
    for label_name, y_train in labels_train.items():
        label_type = label_types[label_name]
        mask_train = ~np.isnan(y_train)
        
        X_train_filtered = X_train[mask_train]
        y_train_filtered = y_train[mask_train]
        
        # Skip if not enough classes
        if len(np.unique(y_train_filtered)) < 2:
            trained_models[label_name] = None
            continue
        
        scaler = None
        
        if label_type == 'classification':
            # Select and train classifier
            if classifier == 'histgradient':
                clf = HistGradientBoostingClassifier(random_state=42, max_iter=50, max_depth=5)
            elif classifier == 'elasticnet':
                scaler = StandardScaler()
                X_train_filtered = scaler.fit_transform(X_train_filtered)
                clf = SGDClassifier(
                    loss='log_loss', penalty='elasticnet',
                    alpha=0.0001, l1_ratio=0.15,
                    max_iter=1000, random_state=42, tol=1e-3
                )
            elif classifier == 'logistic':
                scaler = StandardScaler()
                X_train_filtered = scaler.fit_transform(X_train_filtered)
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            elif classifier == 'randomforest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
            else:
                raise ValueError(f"Unknown classifier: {classifier}")
        else:  # regression
            if classifier == 'histgradient':
                clf = HistGradientBoostingRegressor(random_state=42, max_iter=50, max_depth=5)
            else:
                # Use Ridge for all other methods
                scaler = StandardScaler()
                X_train_filtered = scaler.fit_transform(X_train_filtered)
                clf = Ridge(alpha=1.0, random_state=42)
        
        clf.fit(X_train_filtered, y_train_filtered)
        trained_count += 1
        print(f"    ✓ [{trained_count}/{len(labels_train)}] Trained {classifier} for {label_name}", flush=True)
        
        trained_models[label_name] = {
            'clf': clf,
            'scaler': scaler,
            'mask_train': mask_train,
            'label_type': label_type
        }
    
    return trained_models


def soft_mcc(y_true, y_proba, temperature=10.0):
    """
    Compute an improved differentiable approximation of Matthews Correlation Coefficient.
    
    Uses a sigmoid-sharpened version of predicted probabilities to better approximate
    the hard decision boundary at 0.5, while maintaining differentiability.
    
    The temperature parameter controls how sharp the sigmoid is:
    - Higher temperature (e.g., 10-20): Sharper, closer to hard threshold
    - Lower temperature (e.g., 1-5): Smoother, more like original probabilities
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 1 (continuous [0, 1])
        temperature: Sharpness of the sigmoid around 0.5 (higher = sharper)
    
    Returns:
        float: Soft MCC in range [-1, 1]
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Apply sigmoid sharpening around 0.5 decision boundary
    # This makes the soft MCC more sensitive to changes near the threshold
    # sigmoid(temperature * (p - 0.5)) maps:
    #   p=0.5 -> 0.5 (unchanged)
    #   p>0.5 -> closer to 1.0 (sharpened)
    #   p<0.5 -> closer to 0.0 (sharpened)
    p1_sharp = 1.0 / (1.0 + np.exp(-temperature * (y_proba - 0.5)))
    p0_sharp = 1.0 - p1_sharp
    
    # Soft confusion matrix elements using sharpened probabilities
    tp = np.sum(y_true * p1_sharp)
    tn = np.sum((1 - y_true) * p0_sharp)
    fp = np.sum((1 - y_true) * p1_sharp)
    fn = np.sum(y_true * p0_sharp)
    
    # Compute soft MCC
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Avoid division by zero
    if denominator < 1e-8:
        return 0.0
    
    return numerator / denominator


def _is_tree_based(clf):
    """Check if a classifier is tree-based (piecewise constant predictions)."""
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    return isinstance(clf, (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                            RandomForestClassifier, RandomForestRegressor))


def _smoothed_log_loss(clf, X, y_true, scaler=None, n_samples=32, noise_scale=0.03):
    """
    Compute log-loss averaged over noisy perturbations of the input (randomized smoothing).
    
    For tree-based models, predictions are piecewise constant, so the loss surface
    is a staircase with zero gradient almost everywhere. By averaging the loss over
    Gaussian-perturbed inputs, we convolve the staircase with a smooth kernel,
    producing a smooth loss surface that gradient-based optimizers can follow.
    
    Args:
        clf: Trained classifier with predict_proba
        X: Input features (n_samples, n_features)
        y_true: True labels
        scaler: Optional StandardScaler to apply before prediction
        n_samples: Number of noisy copies to average over
        noise_scale: Std dev of Gaussian noise as fraction of per-feature std dev
    
    Returns:
        float: Negative mean log-loss (higher = better)
    """
    eps = 1e-15
    y_int = y_true.astype(int)
    
    # Compute noise scale from feature standard deviations
    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std < eps, 1.0, feature_std)
    sigma = noise_scale * feature_std  # (n_features,)
    
    rng = np.random.RandomState(42)
    total_log_loss = 0.0
    
    for _ in range(n_samples):
        X_noisy = X + rng.randn(*X.shape) * sigma
        X_pred = scaler.transform(X_noisy) if scaler is not None else X_noisy
        y_proba = clf.predict_proba(X_pred)
        y_proba = np.clip(y_proba, eps, 1 - eps)
        p_true = y_proba[np.arange(len(y_int)), y_int]
        total_log_loss += -np.mean(np.log(p_true))
    
    avg_log_loss = total_log_loss / n_samples
    return -avg_log_loss  # negative because higher = better


def _smoothed_mse(clf, X, y_true, scaler=None, n_samples=32, noise_scale=0.03):
    """
    Compute MSE averaged over noisy perturbations (randomized smoothing for regression).
    """
    feature_std = np.std(X, axis=0)
    feature_std = np.where(feature_std < 1e-15, 1.0, feature_std)
    sigma = noise_scale * feature_std
    
    rng = np.random.RandomState(42)
    total_mse = 0.0
    
    for _ in range(n_samples):
        X_noisy = X + rng.randn(*X.shape) * sigma
        X_pred = scaler.transform(X_noisy) if scaler is not None else X_noisy
        y_pred = clf.predict(X_pred)
        total_mse += np.mean((y_true - y_pred) ** 2)
    
    return -(total_mse / n_samples)  # negative because higher = better


def evaluate_with_pretrained(X_test, y_test, trained_model, use_proba=True, temperature=10.0):
    """
    Evaluate using a pre-trained classifier.
    
    During optimization (use_proba=True):
      - Linear models: log-loss (classification) or negative MSE (regression)
      - Tree-based models: same losses but averaged over Gaussian-perturbed inputs
        (randomized smoothing) to produce a smooth loss surface for L-BFGS-B
    
    For final evaluation (use_proba=False): hard MCC / R².
    
    Args:
        X_test: Test features (already adjusted)
        y_test: Test labels
        trained_model: Dict with 'clf', 'scaler', 'label_type'
        use_proba: If True, use smooth loss for optimization
        temperature: Unused, kept for API compatibility
    """
    if trained_model is None:
        return np.nan
    
    clf = trained_model['clf']
    scaler = trained_model['scaler']
    label_type = trained_model['label_type']
    
    mask_test = ~np.isnan(y_test)
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if label_type == 'classification':
        if len(np.unique(y_test_filtered)) < 2:
            return np.nan
    
    # For optimization: use smooth loss functions
    if use_proba:
        tree_based = _is_tree_based(clf)
        
        if label_type == 'classification':
            if tree_based:
                # Randomized smoothing: average log-loss over noisy perturbations
                # Apply scaler before noise so noise is in original feature space
                return _smoothed_log_loss(clf, X_test_filtered, y_test_filtered, scaler)
            else:
                # Linear models: log-loss is already smooth
                X_pred = scaler.transform(X_test_filtered) if scaler is not None else X_test_filtered
                try:
                    y_proba = clf.predict_proba(X_pred)
                    eps = 1e-15
                    y_proba = np.clip(y_proba, eps, 1 - eps)
                    p_true = y_proba[np.arange(len(y_test_filtered)), y_test_filtered.astype(int)]
                    return -(-np.mean(np.log(p_true)))  # negative log-loss
                except Exception as e:
                    print(f"    Warning: log-loss failed, falling back to hard predictions: {e}")
        
        if label_type == 'regression':
            if tree_based:
                return _smoothed_mse(clf, X_test_filtered, y_test_filtered, scaler)
            else:
                X_pred = scaler.transform(X_test_filtered) if scaler is not None else X_test_filtered
                y_pred = clf.predict(X_pred)
                return -np.mean((y_test_filtered - y_pred) ** 2)
    
    # For final evaluation: use hard metrics
    X_pred = scaler.transform(X_test_filtered) if scaler is not None else X_test_filtered
    y_pred = clf.predict(X_pred)
    
    if label_type == 'classification':
        return matthews_corrcoef(y_test_filtered, y_pred)
    else:
        return r2_score(y_test_filtered, y_pred)


def evaluate_classification(X_train, X_test, y_train, y_test, classifier='histgradient'):
    """Evaluate classification performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
        return np.nan
    
    # Select classifier
    if classifier == 'histgradient':
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=50, max_depth=5)
    elif classifier == 'elasticnet':
        # Standardize for linear models
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=0.15,
            max_iter=1000, random_state=42, tol=1e-3
        )
    elif classifier == 'logistic':
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif classifier == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_train_filtered, y_train_filtered)
    y_pred = clf.predict(X_test_filtered)
    
    return matthews_corrcoef(y_test_filtered, y_pred)


def evaluate_regression(X_train, X_test, y_train, y_test, regressor='histgradient'):
    """Evaluate regression performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filtered = X_train[mask_train]
    y_train_filtered = y_train[mask_train]
    X_test_filtered = X_test[mask_test]
    y_test_filtered = y_test[mask_test]
    
    if len(X_train_filtered) < 10 or len(X_test_filtered) < 10:
        return np.nan
    
    # Select regressor
    if regressor == 'histgradient':
        reg = HistGradientBoostingRegressor(random_state=42, max_iter=50, max_depth=5)
    elif regressor in ['elasticnet', 'logistic', 'randomforest']:
        # Use Ridge for all linear-based methods
        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)
        reg = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")
    
    reg.fit(X_train_filtered, y_train_filtered)
    y_pred = reg.predict(X_test_filtered)
    
    return r2_score(y_test_filtered, y_pred)


def multi_label_objective(params, X_train, X_test, labels_train, labels_test, 
                          label_types, optimize_scale=True, weights=None, 
                          iteration_counter=None, classifier='histgradient', trained_models=None,
                          temperature=10.0):
    """
    Multi-label objective function.
    
    Optimizes shift/scale parameters to maximize performance across ALL labels.
    
    Args:
        params: [shift_1, ..., shift_n] or [shift_1, ..., shift_n, scale_1, ..., scale_n]
        labels_train: dict of {label_name: encoded_values}
        labels_test: dict of {label_name: encoded_values}
        label_types: dict of {label_name: 'classification' or 'regression'}
        optimize_scale: If False, fix scales to 1.0
        weights: dict of {label_name: weight} for combining scores
        iteration_counter: dict to track iterations
        classifier: Classifier to use for optimization ('histgradient', 'elasticnet', 'logistic', 'randomforest')
        trained_models: Pre-trained models dict (if None, will train on each iteration - slow!)
        temperature: Temperature for sigmoid sharpening in soft MCC (higher = sharper)
    """
    n_genes = X_train.shape[1]
    
    if optimize_scale:
        shifts = params[:n_genes]
        scales = params[n_genes:]
    else:
        shifts = params
        scales = np.ones(n_genes)
    
    # Adjust test data
    X_test_adjusted = (X_test - shifts) / scales
    
    # Evaluate on all labels
    scores = {}
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        # Use pre-trained model if available (much faster!)
        if trained_models is not None and label_name in trained_models:
            # Use probability-based scoring for smooth gradients during optimization
            score = evaluate_with_pretrained(X_test_adjusted, y_test, trained_models[label_name], 
                                            use_proba=True, temperature=temperature)
        else:
            # Fallback to training on each iteration (slow)
            if label_type == 'classification':
                score = evaluate_classification(X_train, X_test_adjusted, y_train, y_test, classifier)
            else:
                score = evaluate_regression(X_train, X_test_adjusted, y_train, y_test, classifier)
        
        if not np.isnan(score):
            scores[label_name] = score
    
    # Combine scores (weighted average)
    if not scores:
        return 1e6
    
    if weights is None:
        weights = {k: 1.0 for k in scores.keys()}
    
    # Normalize weights
    total_weight = sum(weights.get(k, 1.0) for k in scores.keys())
    
    combined_score = sum(
        scores[k] * weights.get(k, 1.0) / total_weight 
        for k in scores.keys()
    )
    
    # Track progress
    if iteration_counter is not None:
        iteration_counter['count'] += 1
        if iteration_counter['count'] % 10 == 0:
            # Show top 3 weighted labels
            top_labels = sorted(scores.items(), key=lambda x: weights.get(x[0], 1.0) * x[1], reverse=True)[:3]
            label_str = ", ".join([f"{k}={v:.3f}(w={weights.get(k, 1.0):.2f})" for k, v in top_labels])
            print(f"    Iter {iteration_counter['count']}: combined={combined_score:.4f} | {label_str}", flush=True)
    
    # Minimize negative score
    return -combined_score


from pathlib import Path


def load_cv_ceiling_weights(cv_results_path, classifier_name, label_types):
    """
    Load CV ceiling results and compute weights based on achievable performance.
    
    Labels with higher CV ceiling MCC/R² get higher weights in optimization.
    
    Args:
        cv_results_path: Path to test_set_cv_results.csv
        classifier_name: Name of classifier (e.g., 'Gradient Boosting', 'Random Forest')
        label_types: dict of {label_name: 'classification' or 'regression'}
    
    Returns:
        dict: {label_name: weight} where weight is proportional to CV ceiling score
    """
    try:
        import polars as pl
        cv_df = pl.read_csv(cv_results_path)
        
        # Filter for this classifier
        cv_df = cv_df.filter(pl.col('classifier') == classifier_name)
        
        weights = {}
        for label_name in label_types.keys():
            # Find CV ceiling for this label
            label_row = cv_df.filter(pl.col('metadata_column') == label_name)
            
            if len(label_row) == 0:
                # No CV ceiling available, use default weight
                weights[label_name] = 1.0
                continue
            
            score = label_row['score'][0]
            
            # Use absolute value of score as weight (higher ceiling = higher weight)
            # Add small epsilon to avoid zero weights
            weight = max(abs(score), 0.01)
            
            # Square the weight to emphasize high-performing labels even more
            weight = weight ** 2
            
            weights[label_name] = weight
        
        print(f"  Loaded CV ceiling weights for {classifier_name}:", flush=True)
        for label_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {label_name}: {weight:.4f}", flush=True)
        
        return weights
        
    except Exception as e:
        print(f"  Warning: Could not load CV ceiling weights: {e}", flush=True)
        print(f"  Using uniform weights", flush=True)
        return {k: 1.0 for k in label_types.keys()}


def find_multi_label_alignment(X_train, X_test, labels_train, labels_test,
                               label_types, optimize_scale=True, weights=None,
                               method='direct', classifier='histgradient', cv_results_path=None,
                               temperature=10.0):
    """
    Find optimal shift/scale parameters for multi-label alignment.
    
    Args:
        method: 'direct' (optimize with specified classifier) or 
                'closed_form' (linear closed-form solution)
        classifier: Classifier to use for optimization ('histgradient', 'elasticnet', 'logistic', 'randomforest')
        cv_results_path: Path to CV ceiling results CSV for weight calculation
        temperature: Temperature for sigmoid sharpening in soft MCC (higher = sharper)
    """
    n_genes = X_train.shape[1]
    
    if method == 'closed_form':
        return find_closed_form_alignment(
            X_train, X_test, labels_train, labels_test,
            label_types, optimize_scale, weights
        )
    
    # Load CV ceiling-based weights if available and no weights provided
    if weights is None and cv_results_path is not None:
        # Map classifier names
        classifier_name_map = {
            'histgradient': 'Gradient Boosting',
            'elasticnet': 'ElasticNet (l1=0.15)',
            'logistic': 'Logistic (L2)',
            'randomforest': 'Random Forest'
        }
        classifier_display_name = classifier_name_map.get(classifier, classifier)
        weights = load_cv_ceiling_weights(cv_results_path, classifier_display_name, label_types)
    
    # Direct optimization method (original)
    # Initialize shifts to align means (simple baseline)
    mean_alignment_shifts = np.nanmean(X_test, axis=0) - np.nanmean(X_train, axis=0)
    
    # Try to load Bayesian effective shifts as better initialization
    try:
        import polars as pl
        effective_shifts_df = pl.read_csv('adjusters/effective_shifts.csv')
        bayesian_shifts = effective_shifts_df['effective_shift'].to_numpy()
        if len(bayesian_shifts) == n_genes:
            initial_shifts = bayesian_shifts
            print(f"  Using Bayesian effective shifts for initialization")
        else:
            initial_shifts = mean_alignment_shifts
            print(f"  Using mean alignment for initialization (Bayesian shifts wrong size)")
    except:
        initial_shifts = mean_alignment_shifts
        print(f"  Using mean alignment for initialization (Bayesian shifts not available)")
    
    initial_scales = np.ones(n_genes)
    
    if optimize_scale:
        initial_params = np.concatenate([initial_shifts, initial_scales])
        bounds = [(None, None)] * n_genes + [(0.01, 100)] * n_genes
    else:
        initial_params = initial_shifts
        bounds = [(None, None)] * n_genes
    
    print(f"  Initial shifts: {initial_shifts[:5]}", flush=True)
    if optimize_scale:
        print(f"  Initial scales: {initial_scales[:5]}", flush=True)
    else:
        print(f"  Scales fixed to 1.0 (shift-only optimization)", flush=True)
    print(f"  Optimization classifier: {classifier}", flush=True)
    print(f"  Soft MCC temperature: {temperature} (higher = sharper decision boundary)", flush=True)
    
    # Pre-train classifiers for all labels (HUGE speedup!)
    print("  Pre-training classifiers for all labels...", flush=True)
    trained_models = train_classifiers_for_labels(X_train, labels_train, label_types, classifier)
    print(f"  Trained {len([m for m in trained_models.values() if m is not None])} classifiers", flush=True)
    
    # Optimize
    print("  Optimizing parameters across all labels...", flush=True)
    iteration_counter = {'count': 0}
    
    result = minimize(
        multi_label_objective,
        initial_params,
        args=(X_train, X_test, labels_train, labels_test, label_types, 
              optimize_scale, weights, iteration_counter, classifier, trained_models, temperature),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5}
    )
    
    print(f"  Optimization result: {result.message}", flush=True)
    print(f"  Success: {result.success}", flush=True)
    print(f"  Number of iterations: {result.nit}", flush=True)
    print(f"  Number of function evaluations: {result.nfev}", flush=True)
    
    if optimize_scale:
        optimal_shifts = result.x[:n_genes]
        optimal_scales = result.x[n_genes:]
    else:
        optimal_shifts = result.x
        optimal_scales = np.ones(n_genes)
    
    print(f"  Optimal shifts: {optimal_shifts[:5]}", flush=True)
    print(f"  Optimal scales: {optimal_scales[:5]}", flush=True)
    print(f"  Final loss: {result.fun:.4f}", flush=True)
    
    return optimal_shifts, optimal_scales, result


def find_closed_form_alignment(X_train, X_test, labels_train, labels_test,
                                label_types, optimize_scale=True, weights=None):
    """
    Find optimal shift/scale using closed-form linear solution.
    
    For each label, train linear models on train and test data, then solve
    for shift/scale that aligns the decision boundaries across all labels.
    """
    n_genes = X_train.shape[1]
    
    print("  Training linear models on each label...")
    
    # Collect coefficients from all labels
    W_sum = np.zeros((n_genes, n_genes))  # Σ w_k w_k^T
    delta_sum = np.zeros(n_genes)  # Σ w_k * delta_k
    scale_estimates = []
    
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        # Filter out NaN labels
        mask_train = ~np.isnan(y_train)
        mask_test = ~np.isnan(y_test)
        
        X_train_filtered = X_train[mask_train]
        y_train_filtered = y_train[mask_train]
        X_test_filtered = X_test[mask_test]
        y_test_filtered = y_test[mask_test]
        
        if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2 or len(X_test_filtered) < 10:
            continue
        
        # Standardize
        scaler_train = StandardScaler()
        scaler_test = StandardScaler()
        X_train_scaled = scaler_train.fit_transform(X_train_filtered)
        X_test_scaled = scaler_test.fit_transform(X_test_filtered)
        
        # Train linear models
        if label_type == 'classification':
            clf_train = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf_test = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            
            clf_train.fit(X_train_scaled, y_train_filtered)
            clf_test.fit(X_test_scaled, y_test_filtered)
            
            w_train = clf_train.coef_[0]
            b_train = clf_train.intercept_[0]
            w_test = clf_test.coef_[0]
            b_test = clf_test.intercept_[0]
        else:
            # Regression
            reg_train = Ridge(alpha=1.0, random_state=42)
            reg_test = Ridge(alpha=1.0, random_state=42)
            
            reg_train.fit(X_train_scaled, y_train_filtered)
            reg_test.fit(X_test_scaled, y_test_filtered)
            
            w_train = reg_train.coef_
            b_train = reg_train.intercept_
            w_test = reg_test.coef_
            b_test = reg_test.intercept_
        
        # Compute scale estimate (element-wise ratio of coefficients)
        if optimize_scale:
            scale_est = np.where(
                np.abs(w_train) > 1e-6,
                w_test / w_train,
                1.0
            )
            scale_est = np.clip(scale_est, 0.1, 10.0)
            scale_estimates.append(scale_est)
        
        # Compute misalignment on original (unscaled) data
        # We need to account for the standardization
        # The shift in original space that aligns means
        mean_train = np.nanmean(X_train, axis=0)
        mean_test = np.nanmean(X_test, axis=0)
        
        # Weight for this label
        label_weight = weights.get(label_name, 1.0) if weights else 1.0
        
        # Add to accumulator (weighted by label importance)
        W_sum += label_weight * np.outer(w_train, w_train)
        
        # The shift that would align the intercepts
        # Simplified: use mean difference as proxy
        delta_sum += label_weight * w_train * (b_train - b_test)
    
    # Solve for optimal shift
    # Add small regularization for numerical stability
    W_sum_reg = W_sum + 1e-6 * np.eye(n_genes)
    
    try:
        optimal_shifts = np.linalg.solve(W_sum_reg, delta_sum)
    except np.linalg.LinAlgError:
        print("  Warning: Singular matrix, using mean shift")
        optimal_shifts = np.nanmean(X_test, axis=0) - np.nanmean(X_train, axis=0)
    
    # Compute optimal scales (median of per-label estimates)
    if optimize_scale and scale_estimates:
        optimal_scales = np.median(scale_estimates, axis=0)
    else:
        optimal_scales = np.ones(n_genes)
    
    print(f"  Closed-form shifts: {optimal_shifts[:5]}")
    print(f"  Closed-form scales: {optimal_scales[:5]}")
    
    # Create a dummy result object for consistency
    class ClosedFormResult:
        def __init__(self):
            self.fun = 0.0
            self.success = True
    
    return optimal_shifts, optimal_scales, ClosedFormResult()


def evaluate_all_labels(X_train, X_test, labels_train, labels_test, label_types, classifier='histgradient'):
    """Evaluate performance on all labels."""
    results = {}
    
    for label_name in labels_train.keys():
        y_train = labels_train[label_name]
        y_test = labels_test[label_name]
        label_type = label_types[label_name]
        
        if label_type == 'classification':
            score = evaluate_classification(X_train, X_test, y_train, y_test, classifier)
            metric = 'MCC'
        else:
            score = evaluate_regression(X_train, X_test, y_train, y_test, classifier)
            metric = 'R²'
        
        results[label_name] = {'score': score, 'metric': metric}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label Decision Boundary Alignment"
    )
    parser.add_argument("--train-genes", required=True)
    parser.add_argument("--test-genes-unadjusted", required=True)
    parser.add_argument("--test-genes-bayesian", required=True)
    parser.add_argument("--test-genes-effective", required=False)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--continuous-metadata", nargs='+', required=True,
                       help="List of continuous metadata columns")
    parser.add_argument("--shift-only", action="store_true",
                       help="Only optimize shifts, fix scales to 1.0")
    parser.add_argument("--training-classifier", default="histgradient",
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       help="Classifier to use for DBA optimization")
    parser.add_argument("--evaluation-classifiers", nargs='+',
                       default=['histgradient'],
                       choices=['histgradient', 'elasticnet', 'logistic', 'randomforest'],
                       help="Classifiers to use for evaluation")
    parser.add_argument("--temperature", type=float, default=10.0,
                       help="Temperature for sigmoid sharpening in soft MCC (higher = sharper, closer to hard threshold). Default: 10.0")
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*60, flush=True)
    print("Multi-Label Decision Boundary Alignment", flush=True)
    print("="*60, flush=True)
    
    # Load data
    print("\nLoading data...", flush=True)
    train_genes_df = pl.read_csv(args.train_genes)
    print(f"  ✓ Loaded training genes: {train_genes_df.shape}", flush=True)
    test_unadjusted_df = pl.read_csv(args.test_genes_unadjusted)
    print(f"  ✓ Loaded test genes (unadjusted): {test_unadjusted_df.shape}", flush=True)
    test_bayesian_df = pl.read_csv(args.test_genes_bayesian)
    print(f"  ✓ Loaded test genes (Bayesian): {test_bayesian_df.shape}", flush=True)
    train_meta_df = pl.read_csv(args.train_metadata)
    print(f"  ✓ Loaded training metadata: {train_meta_df.shape}", flush=True)
    test_meta_df = pl.read_csv(args.test_metadata)
    print(f"  ✓ Loaded test metadata: {test_meta_df.shape}", flush=True)
    
    # Load effective shift data if provided
    test_effective_df = None
    if args.test_genes_effective:
        test_effective_df = pl.read_csv(args.test_genes_effective)
        print(f"  Loaded effective shift data: {len(test_effective_df.columns)} genes")
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns 
                   if g in test_unadjusted_df.columns 
                   and g in test_bayesian_df.columns]
    
    print(f"Common genes: {len(common_genes)}", flush=True)
    
    # Extract gene data
    X_train = train_genes_df.select(common_genes).to_numpy()
    X_test_unadjusted = test_unadjusted_df.select(common_genes).to_numpy()
    X_test_bayesian = test_bayesian_df.select(common_genes).to_numpy()
    
    if test_effective_df is not None:
        X_test_effective = test_effective_df.select(common_genes).to_numpy()
    
    # Prepare labels
    print("\nPreparing labels...", flush=True)
    continuous_metadata = set(args.continuous_metadata)
    
    labels_train = {}
    labels_test = {}
    label_types = {}
    
    label_count = 0
    for col in train_meta_df.columns:
        if col.startswith('meta_'):
            label_name = col
            label_count += 1
            
            train_vals = train_meta_df[col].to_numpy()
            test_vals = test_meta_df[col].to_numpy()
            
            if col in continuous_metadata:
                # Regression
                labels_train[label_name] = train_vals.astype(float)
                labels_test[label_name] = test_vals.astype(float)
                label_types[label_name] = 'regression'
            else:
                # Classification - encode labels
                le = LabelEncoder()
                all_labels = np.concatenate([
                    train_vals[~pl.Series(train_vals).is_null()],
                    test_vals[~pl.Series(test_vals).is_null()]
                ])
                le.fit(all_labels)
                
                labels_train[label_name] = np.array([
                    le.transform([v])[0] if v is not None else np.nan 
                    for v in train_vals
                ])
                labels_test[label_name] = np.array([
                    le.transform([v])[0] if v is not None else np.nan 
                    for v in test_vals
                ])
                label_types[label_name] = 'classification'
    
    print(f"  ✓ Prepared {label_count} labels", flush=True)
    print(f"Labels: {list(labels_train.keys())}", flush=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {output_dir}", flush=True)
    
    # Store results
    all_results = {}
    
    # 1. Baseline (no adjustment)
    print("\n" + "="*60)
    print("1. Baseline (no adjustment)")
    print("="*60)
    
    # Evaluate with all evaluation classifiers
    for eval_clf in args.evaluation_classifiers:
        print(f"\n  Evaluating with {eval_clf}:")
        results = evaluate_all_labels(X_train, X_test_unadjusted, 
                                      labels_train, labels_test, label_types, eval_clf)
        all_results[f'Unadjusted ({eval_clf})'] = results
        for label, data in results.items():
            print(f"    {label}: {data['score']:.3f} ({data['metric']})")
    
    # 2. Bayesian adjustment (shift+scale)
    print("\n" + "="*60)
    print("2. Bayesian adjustment (shift+scale)")
    print("="*60)
    results = evaluate_all_labels(X_train, X_test_bayesian, 
                                  labels_train, labels_test, label_types)
    all_results['Bayesian (shift+scale)'] = results
    for label, data in results.items():
        print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # 2b. Bayesian effective shift-only (if provided)
    if test_effective_df is not None:
        print("\n" + "="*60)
        print("2b. Bayesian effective shift-only")
        print("="*60)
        results = evaluate_all_labels(X_train, X_test_effective, 
                                      labels_train, labels_test, label_types)
        all_results['Bayesian (effective shift)'] = results
        for label, data in results.items():
            print(f"  {label}: {data['score']:.3f} ({data['metric']})")
    
    # 3. Multi-label DBA (closed-form) - only run if no specific classifier requested
    if args.training_classifier == "histgradient":
        print("\n" + "="*60)
        print("3. Multi-Label DBA (closed-form linear solution)")
        print("="*60)
        shifts_dba_cf, scales_dba_cf, result_dba_cf = find_multi_label_alignment(
            X_train, X_test_unadjusted, labels_train, labels_test,
            label_types, optimize_scale=not args.shift_only, method='closed_form'
        )
        
        X_test_dba_cf = (X_test_unadjusted - shifts_dba_cf) / scales_dba_cf
        results = evaluate_all_labels(X_train, X_test_dba_cf, 
                                      labels_train, labels_test, label_types)
        all_results['Multi-Label DBA (closed-form)'] = results
        for label, data in results.items():
            print(f"  {label}: {data['score']:.3f} ({data['metric']})")
        
        # Save adjusted data
        adjusted_df = pl.DataFrame({
            gene: X_test_dba_cf[:, i] for i, gene in enumerate(common_genes)
        })
        adjusted_df.write_csv(output_dir / "test_genes_multi_label_dba_closed_form.csv")
    else:
        print("\n" + "="*60)
        print(f"3. Skipping closed-form (using {args.training_classifier} for direct optimization)")
        print("="*60)
    
    # 4. Multi-label DBA (direct optimization)
    print("\n" + "="*60, flush=True)
    print("4. Multi-Label DBA (direct optimization)", flush=True)
    print("="*60, flush=True)
    
    # Try to load CV ceiling results for weighting
    cv_results_path = output_dir.parent / "test_set_cv_results.csv"
    if not cv_results_path.exists():
        cv_results_path = None
        print("  Note: CV ceiling results not found, using uniform weights", flush=True)
    
    shifts_dba, scales_dba, result_dba = find_multi_label_alignment(
        X_train, X_test_unadjusted, labels_train, labels_test,
        label_types, optimize_scale=not args.shift_only, method='direct',
        classifier=args.training_classifier, cv_results_path=cv_results_path,
        temperature=args.temperature
    )
    
    X_test_dba = (X_test_unadjusted - shifts_dba) / scales_dba
    print("  Evaluating final performance...", flush=True)
    results = evaluate_all_labels(X_train, X_test_dba, 
                                  labels_train, labels_test, label_types)
    all_results['Multi-Label DBA (direct)'] = results
    for label, data in results.items():
        print(f"  {label}: {data['score']:.3f} ({data['metric']})", flush=True)
    
    # Save adjusted data
    print("  Saving adjusted data...", flush=True)
    adjusted_df = pl.DataFrame({
        gene: X_test_dba[:, i] for i, gene in enumerate(common_genes)
    })
    adjusted_df.write_csv(output_dir / "test_genes_multi_label_dba_direct.csv")
    print(f"    ✓ Saved: test_genes_multi_label_dba_direct.csv", flush=True)
    
    # Also save with classifier-specific name for Snakemake compatibility
    adjusted_df.write_csv(output_dir / f"test_genes_dba_{args.training_classifier}.csv")
    print(f"    ✓ Saved: test_genes_dba_{args.training_classifier}.csv", flush=True)
    
    # Save parameters
    print("  Saving parameters...", flush=True)
    params_data = {
        'gene': common_genes,
        'multi_label_dba_direct_shift': shifts_dba,
        'multi_label_dba_direct_scale': scales_dba,
    }
    
    # Add closed-form parameters if they were computed
    if args.training_classifier == "histgradient":
        params_data['multi_label_dba_cf_shift'] = shifts_dba_cf
        params_data['multi_label_dba_cf_scale'] = scales_dba_cf
    
    params_df = pl.DataFrame(params_data)
    params_df.write_csv(output_dir / "multi_label_dba_parameters.csv")
    print(f"    ✓ Saved: multi_label_dba_parameters.csv", flush=True)
    
    # Also save with classifier-specific name for Snakemake compatibility
    params_df.write_csv(output_dir / f"dba_parameters_{args.training_classifier}.csv")
    print(f"    ✓ Saved: dba_parameters_{args.training_classifier}.csv", flush=True)
    
    # Create comparison table
    print("\n" + "="*60, flush=True)
    print("COMPARISON ACROSS ALL LABELS", flush=True)
    print("="*60, flush=True)
    
    # Prepare data for CSV
    print("  Creating comparison table...", flush=True)
    comparison_data = []
    for method_name, method_results in all_results.items():
        for label_name, label_data in method_results.items():
            comparison_data.append({
                'method': method_name,
                'label': label_name,
                'metric': label_data['metric'],
                'score': label_data['score']
            })
    
    comparison_df = pl.DataFrame(comparison_data)
    comparison_df.write_csv(output_dir / "multi_label_comparison.csv")
    print(f"  ✓ Saved: multi_label_comparison.csv", flush=True)
    
    # Print summary table
    print(f"\n{'Method':<30} {'Label':<25} {'Metric':<6} {'Score':>8}", flush=True)
    print("-" * 75, flush=True)
    for row in comparison_data:
        print(f"{row['method']:<30} {row['label']:<25} {row['metric']:<6} {row['score']:>8.3f}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
