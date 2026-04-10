"""Sinkhorn Unbalanced Optimal Transport."""

import numpy as np
import scipy.spatial.distance as dist
import scipy.special as sp

EPS = 1e-8


def sinkhorn_uot(X_embed, Y_embed, ot_epsilon=0.01, ot_tau=0.1):
    """Run Sinkhorn UOT, return (w_ref, w_tgt, intersection_mass).

    Parameters
    ----------
    X_embed, Y_embed : np.ndarray
        Sample embeddings, shape (n_samples, n_dims).
    ot_epsilon : float
        Entropy regularization (lower = sharper).
    ot_tau : float
        Mass relaxation (lower = more willing to destroy unmatched mass).
    """
    N_ref, N_tgt = X_embed.shape[0], Y_embed.shape[0]
    C = dist.cdist(X_embed, Y_embed, metric="sqeuclidean")
    C = C / (np.max(C) + EPS)

    log_a = np.log(np.ones(N_ref) / N_ref)
    log_b = np.log(np.ones(N_tgt) / N_tgt)
    f, g = np.zeros(N_ref), np.zeros(N_tgt)
    fi = ot_tau / (ot_tau + ot_epsilon)

    for iteration in range(1000):
        f_prev = f.copy()
        g = fi * (log_b - sp.logsumexp((-C / ot_epsilon) + f[:, None], axis=0))
        f = fi * (log_a - sp.logsumexp((-C / ot_epsilon) + g[None, :], axis=1))
        if np.max(np.abs(f - f_prev)) < 1e-5:
            break

    P = np.exp((-C / ot_epsilon) + f[:, None] + g[None, :])
    w_ref = np.sum(P, axis=1) * N_ref
    w_tgt = np.sum(P, axis=0) * N_tgt
    return w_ref, w_tgt, float(np.sum(P))
