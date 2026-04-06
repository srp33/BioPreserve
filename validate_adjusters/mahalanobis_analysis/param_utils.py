import numpy as np

EPS = np.finfo(float).eps

def reconstruct_params(y_raw, y_adjusted, debug=False):

    """
    Reconstruct affine parameters (alpha, beta) from raw and adjusted data.
    y_raw: (Genes x Samples)
    y_adjusted: (Genes x Samples)
    """

    # Calculate scale (alpha) using standard deviation ratio
    # Add EPS to denominator to prevent division by zero
    std_raw = np.std(y_raw, axis=1) + EPS
    std_adj = np.std(y_adjusted, axis=1)
    alpha_recon = std_adj / std_raw

    # Calculate shift (beta) using mean difference
    mean_raw = np.mean(y_raw, axis=1)
    mean_adj = np.mean(y_adjusted, axis=1)
    beta_recon = mean_adj - alpha_recon * mean_raw

    if debug:
        print(f"DEBUG: Reconstruction complete for {y_raw.shape[0]} genes.")

    return {'alpha': alpha_recon, 'beta': beta_recon}