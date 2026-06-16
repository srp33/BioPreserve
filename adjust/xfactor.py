"""
Population-aware cross-platform alignment — single self-contained file.

Aligns dataset A into dataset B's frame, leak-free and population-aware, with no external project files
(only numpy / pandas / scipy / scikit-learn). This is the validated "cross-factor" algorithm: it

  1. discovers binary FACTORS  — reproducibly-bimodal, correlated gene modules (each = a population axis),
  2. fits the noise->slope LAW — within-mode noise ratio predicts the per-gene affine stretch,
  3. for each factor, IDENTIFIES which population matches across datasets using the CROSS-FACTOR ANCHOR
     (the other co-occurring factors give a reference->query value map; it never touches a missing
     population's data — the leak-free contract),
  4. ALIGNS shared populations (per-gene affine, b pinned to the matched reference centroid) and PLACES a
     population absent from one dataset by the cross-domain coupling-fill.

Public API
----------
    align_into_reference(XA, XB, genes) -> XA aligned into XB's frame    # THE method (the ComBat-equivalent
                                                                         # entry point: data in, aligned out)
    discover_factors(XA, XB, genes)     -> [[gene,...], ...]            # the factor-discovery step

This file is ONLY the algorithm. Data loading, the partial-overlap construction, the baseline methods
(ComBat/Harmony/...), the metrics, and the per-factor convenience used to speed up that particular
benchmark all live with the EVALUATION, not here — anything the baseline pipeline also touches is
evaluation infrastructure and is kept out, so this file is exactly the method under test. Shared low-level
helpers (e.g. factor_axis, discover_factors) are intentionally DUPLICATED between this file and the
evaluation so the two stay decoupled.

Inputs are numpy arrays (samples x genes); `genes` is the list of gene names. Determinism: all randomness
is seeded (GaussianMixture random_state=0) — same inputs give identical output.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import theilslopes
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FastICA
from sklearn.linear_model import Lasso

EPS = 1e-9
CLIP = 2.0          # clip on log noise-ratio when applying the law


# ── core statistics ────────────────────────────────────────────────────────────

def factor_axis(R):
    """Oriented leading axis + 2-mode GMM split of a gene module R (n x m). Returns (axis, hi_mask, score)
    with higher score = high mode."""
    Z = (R - R.mean(0)) / (R.std(0) + EPS)
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    axis = Vt[0]; score = Zc @ axis
    if np.corrcoef(score, Z.mean(1))[0, 1] < 0:
        axis, score = -axis, -score
    g = GaussianMixture(2, n_init=3, random_state=0).fit(score.reshape(-1, 1))
    hi = g.predict(score.reshape(-1, 1)) == int(np.argmax(g.means_.ravel()))
    return axis, hi, score


def corr_shrink(M, ridge=0.05):
    """Within-mode correlation via Ledoit-Wolf shrinkage (+ tiny ridge); identity when too little data."""
    n, m = M.shape
    if n < 4 or m < 2:
        return np.eye(m)
    try:
        # store_precision=False: we read only .covariance_; the precision's pinvh+eigh dominated the cost
        cov = LedoitWolf(assume_centered=False, store_precision=False).fit(M).covariance_
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        C = cov / np.outer(d, d)
    except Exception:
        C = np.nan_to_num(np.atleast_2d(np.corrcoef(M, rowvar=False)), nan=0.0)
    np.fill_diagonal(C, 1.0)
    return (1.0 - ridge) * C + ridge * np.eye(m)


def qmap(xtr, ytr, xte):
    """Monotone quantile map y~q(x): match sorted xtr->ytr, interpolate at xte."""
    o = np.argsort(xtr)
    return np.interp(xte, np.asarray(xtr)[o], np.sort(ytr))


def _fit_law(slope, nr):
    """Robust log-log fit  log(slope) ~ b0 + b1*log(noise_ratio)  (Theil-Sen). Returns (b0, b1)."""
    x = np.log(np.clip(nr, .05, 50)); y = np.log(np.clip(slope, .05, 50))
    b1, b0, _, _ = theilslopes(y, x)
    return float(b0), float(b1)


# ── factor discovery ───────────────────────────────────────────────────────────

def bimodality_coef(X):
    """Sarle's bimodality coefficient per column (>0.555 => bimodal/flat)."""
    n = X.shape[0]; m = X.mean(0); s = X.std(0) + 1e-9; z = (X - m) / s
    skew = (z ** 3).mean(0); kurt = (z ** 4).mean(0) - 3.0
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew ** 2 + 1) / np.clip(kurt + corr, 1e-6, None)


def bimodality_coef_vec(x):
    n = len(x)
    if n < 10: return 0.0
    z = (x - x.mean()) / (x.std() + 1e-9)
    skew = (z ** 3).mean(); kurt = (z ** 4).mean() - 3.0
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew ** 2 + 1) / max(kurt + corr, 1e-6))


def _zscore_blocks(train_sets):
    """Per-dataset z-scored copies. z-scoring inverts each gene's per-dataset affine batch (a*x+b, a>0), so
    the residual blocks are batch-invariant — discovery runs on them with NO batch-correction step, which is
    what severs the discover<->correct circularity. Relative within-gene loadings survive z-scoring (the
    per-gene sd cancels in any ratio), so the cross-dataset consistency structure is preserved here."""
    return [((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(float) for X in train_sets]


def _pooled_bc_per_gene(blocks):
    """Unweighted mean Sarle bimodality per gene, over datasets where the gene varies. Bimodality is a SHAPE
    signal, not an estimate to precision-pool: n-weighting would let one large, noisy dataset (overlapping
    modes => low BC) suppress a latent that is cleanly bimodal elsewhere. Each dataset's evidence counts
    equally; datasets where the (residual) gene is flat are skipped, so a factor absent from some sets is not
    penalized."""
    G = blocks[0].shape[1]; acc = np.zeros(G); cnt = np.zeros(G)
    for B in blocks:
        ok = B.std(0) > 1e-3; bc = np.nan_to_num(bimodality_coef(B))
        acc[ok] += bc[ok]; cnt[ok] += 1.0
    return acc / np.clip(cnt, 1e-9, None)


def _pooled_corr(blocks, cols, min_n=5):
    """Precision-weighted Fisher-z pooled correlation over `cols`. Correlation is invariant to each dataset's
    per-gene affine, so this pools cross-dataset evidence with no batch step."""
    zsum, wsum = None, 0.0
    for B in blocks:
        if B.shape[0] < min_n:
            continue
        sub = B[:, cols]
        if np.all(sub.std(0) < 1e-9):
            continue
        R = np.clip(np.nan_to_num(np.corrcoef(sub, rowvar=False)), -0.999, 0.999)
        w = max(B.shape[0] - 3, 1); z = np.arctanh(R)
        zsum = w * z if zsum is None else zsum + w * z; wsum += w
    if zsum is None:
        return np.eye(len(cols))
    C = np.tanh(zsum / wsum); np.fill_diagonal(C, 1.0)
    return np.nan_to_num(C)


def _pooled_score_bc(blocks, score_of):
    """Unweighted mean bimodality of the factor SCORE across datasets (where it varies) — confirms the axis
    is a BINARY latent (a two-population score is bimodal; a continuous or noise axis is not). Unweighted so
    one large noisy dataset can't veto a factor that is cleanly split elsewhere."""
    vals = [bimodality_coef_vec(s) for B, s in zip(blocks, score_of) if B.shape[0] >= 10 and s.std() > 1e-9]
    return float(np.mean(vals)) if vals else 0.0


def _consistent_loading(blocks, score_of):
    """Per-gene cross-dataset-CONSISTENT association with the factor score. Returns (beta_bar, agree):
    beta_bar = precision-weighted mean per-dataset corr(score, gene); agree = weighted fraction of datasets
    whose sign matches the pooled sign. Generative basis: within-gene relative loadings are dataset-
    invariant, so a TRUE member shows a reproducible coefficient everywhere, while a one-dataset artifact
    averages toward zero or flips sign across datasets."""
    G = blocks[0].shape[1]; betas, ws = [], []
    for B, s in zip(blocks, score_of):
        sv = s.std()
        if B.shape[0] < 5 or sv < 1e-9:
            betas.append(None); ws.append(0.0); continue
        sc = (s - s.mean()) / sv; sd_raw = B.std(0)
        beta = ((B - B.mean(0)) * sc[:, None]).mean(0) / (sd_raw + 1e-9)   # = corr(score, gene)
        beta[sd_raw < 1e-6] = 0.0
        betas.append(beta); ws.append(max(B.shape[0] - 3, 1))
    wsum = sum(ws); bsum = np.zeros(G)
    for b, w in zip(betas, ws):
        if b is not None: bsum += w * b
    beta_bar = bsum / max(wsum, 1e-9)
    sgn = np.zeros(G); wsgn = 0.0
    for b, w in zip(betas, ws):
        if b is None: continue
        sgn += w * (np.sign(b) == np.sign(beta_bar)); wsgn += w
    return beta_bar, sgn / max(wsgn, 1e-9)


def _varimax(L, q=200, tol=1e-7):
    """Varimax rotation of a loading matrix L (p x k): rotate to maximize the variance of squared loadings,
    i.e. drive each loading toward 0 or large => SPARSE, localized factors. Real expression data's leading
    principal axes MIX a dominant continuous gradient (library size / tissue) with the binary population
    latents; the latents are ROTATIONS within the top eigen-subspace, not the raw eigenvectors. Rotation
    recovers them (and lets a gene load on several => overlap), while the diffuse global axis stays spread and
    fails the later bimodality gate."""
    p, k = L.shape
    if k < 2:
        return L
    R = np.eye(k); d = 0.0
    for _ in range(q):
        Lam = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lam ** 3 - Lam @ np.diag((Lam ** 2).sum(0)) / p))
        R = u @ vt; d_new = float(s.sum())
        if d != 0 and d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def discover_factors_multi(train_sets, genes, bc_thr=0.555, min_size=3, max_cand=1200,
                           max_factors=40, beta_floor=0.45, agree_frac=0.6, dup_jaccard=0.7,
                           return_support=False):
    """Joint factor discovery from ALL datasets under the generative model
        x_dg = a_dg * ( mu_g + sum_k z_dk * w_kg + eps ) + b_dg .
    Steps (on batch-invariant z-scored residual blocks):
      1. CANDIDATES: the genes whose unweighted-mean bimodality across datasets clears `bc_thr` (a generous,
         tractability prefilter; the real decision is below);
      2. SIGNAL SUBSPACE: the eigenvectors of the pooled, precision-weighted correlation whose eigenvalue
         exceeds the Marchenko-Pastur noise edge — the structure provably above sampling noise;
      3. ROTATE: varimax the signal loadings to localize them into sparse factors (the binary latents are
         rotations within this subspace, not the raw principal axes which mix a global continuous gradient);
      4. CONFIRM + ASSIGN each rotated axis: drop it if diffuse (participation ratio = a global gradient, not
         a module) or if its per-dataset SCORE (signed-loading projection) is not bimodal; otherwise assign a
         gene by the generative CONSISTENCY rule — joined iff its association with the score is reproducible
         across datasets (consistent sign + precision-weighted |corr| above `beta_floor`), since within-gene
         relative loadings are dataset-invariant. Membership runs over ALL genes (the cheap gates come first),
         so a multi-latent gene too smeared to pass the bimodality prefilter is still recovered here.
    A gene may load on several rotated factors (overlap); near-duplicate factors (Jaccard > dup_jaccard) are
    merged out. Factors are returned strongest-first. With return_support=True also returns per factor
    {n_genes, strength, score_bc, n_overlap}."""
    blocks = _zscore_blocks(train_sets)
    n_eff = sum(B.shape[0] for B in blocks)
    floor = max(beta_floor, 4.0 / np.sqrt(max(n_eff, 1)))
    bc_pool = _pooled_bc_per_gene(blocks)
    cand = np.where(bc_pool > bc_thr)[0]
    factors, support = [], []
    if len(cand) >= min_size:
        cand = np.sort(cand[np.argsort(-bc_pool[cand])][:max_cand])
        vals, vecs = np.linalg.eigh(_pooled_corr(blocks, cand))
        mp = (1.0 + np.sqrt(len(cand) / max(n_eff, 1))) ** 2           # Marchenko-Pastur noise edge
        sig = np.where(vals > mp)[0]
        sig = sig[np.argsort(-vals[sig])][:max_factors]               # signal subspace, strongest first
        if len(sig):
            Lr = _varimax(vecs[:, sig] * np.sqrt(vals[sig]))          # rotate to localize the loadings
            order = np.argsort(-(Lr ** 2).sum(0))                     # strongest rotated factor first
            cblocks = [B[:, cand] for B in blocks]
            seen = []
            for j in order:
                strength = float((Lr[:, j] ** 2).sum()); nv = np.sqrt(strength)
                if nv < 1e-9:
                    continue
                l = Lr[:, j] / nv
                pr = (l ** 2).sum() ** 2 / ((l ** 4).sum() + 1e-12)   # participation ratio = effective #genes
                if pr > 50 and pr > 0.5 * len(cand):                  # diffuse global gradient, not a module
                    continue
                score_of = [cb @ l for cb in cblocks]
                sbc = _pooled_score_bc(blocks, score_of)
                if sbc < bc_thr:                                      # rotated axis is not a binary latent
                    continue
                beta_bar, agree = _consistent_loading(blocks, score_of)   # over ALL genes (after the cheap
                member = np.where((np.abs(beta_bar) > floor) & (agree > agree_frac))[0]   # PR + bimodal gates)
                if len(member) < min_size:                            # consistency rule; recovers multi-latent
                    continue                                          # overlap genes excluded from candidates
                ms = set(int(i) for i in member)
                if any(len(ms & s) / len(ms | s) > dup_jaccard for s in seen):
                    continue                                         # near-duplicate of an accepted factor
                ov = sum(1 for i in member if any(int(i) in s for s in seen))
                seen.append(ms)
                factors.append([genes[i] for i in member])
                support.append({"n_genes": int(len(member)), "strength": strength,
                                "score_bc": float(sbc), "n_overlap": int(ov)})
    if not return_support:
        return factors
    return factors, support


def discover_factors(XA, XB, genes, **kwargs):
    return discover_factors_multi([XA, XB], genes, **kwargs)


def discover_factors_sfa(train_sets, genes, bc_thr=0.555, min_size=3, max_cand=1200,
                         max_factors=30, n_iter=8, mf_sweeps=3, dup_corr=0.8, return_support=False):
    """Mean-field VARIATIONAL EM for a BINARY sparse factor model — the step toward the full generative model:
        x_dg = sum_k z_dik w_kg + eps ,   z_dik ~ Bernoulli(pi_dk) ,   w sparse ,   eps ~ N(0, sigma^2)
    (fit on batch-invariant z-scored candidate genes). It is CORRELATED (oblique) and OVERLAPPING by design.
      E-step: mean-field Bernoulli responsibilities rho_dik = sigmoid( [w_k·(residual it explains) - ||w_k||^2/2]
              / sigma^2 + logit pi_dk ) — each latent turns on for the samples whose residual it explains.
      M-step: shared sparse loadings W by Lasso(gene ~ rho) with the penalty DERIVED from the estimated noise
              (universal threshold sigma*sqrt(2 log K / N) — NO hand-set alpha); per-dataset prevalences
              pi_dk = mean rho (partial overlap = pi->0/1); noise sigma^2 from the residual.
    A gene's membership = nonzero loading; factors are de-duplicated by responsibility correlation. Same
    return contract as discover_factors_multi (support: {n_genes, score_bc, n_overlap})."""
    blocks = _zscore_blocks(train_sets)
    N = sum(B.shape[0] for B in blocks)
    bc_pool = _pooled_bc_per_gene(blocks)
    cand = np.where(bc_pool > bc_thr)[0]
    factors, support = [], []
    if len(cand) >= min_size:
        cand = np.sort(cand[np.argsort(-bc_pool[cand])][:max_cand])
        Xc = [B[:, cand] for B in blocks]
        vals, vecs = np.linalg.eigh(_pooled_corr(blocks, cand))
        mp = (1.0 + np.sqrt(len(cand) / max(N, 1))) ** 2
        sig = np.where(vals > mp)[0]; sig = sig[np.argsort(-vals[sig])][:max_factors]
        K = len(sig)
        if K >= 1:
            rho = []                                                       # init responsibilities from signal subspace
            for X in Xc:                                                    # SHARP binary split (var ~ z's 0.25, so
                S = X @ vecs[:, sig]                                        # loadings are correctly scaled — a soft
                rho.append(np.where(S > np.median(S, 0), 0.9, 0.1))         # sigmoid init collapses the E-step)
            W = None; sigma2 = 1.0
            for _ in range(n_iter):
                R = np.vstack(rho); Xs = np.vstack(Xc)
                alpha = np.sqrt(sigma2) * np.sqrt(2.0 * np.log(max(K, 2)) / R.shape[0])   # noise-derived penalty
                W = Lasso(alpha=alpha, max_iter=2000).fit(R, Xs).coef_     # (n_cand x K), shared sparse loadings
                keepc = np.abs(W).sum(0) > 1e-9
                if keepc.sum() < 1:
                    W = None; break
                if keepc.sum() < K:
                    W = W[:, keepc]; rho = [r[:, keepc] for r in rho]; K = W.shape[1]
                sigma2 = float(((Xs - np.vstack(rho) @ W.T) ** 2).mean())
                wn2 = (W ** 2).sum(0)
                for d, X in enumerate(Xc):                                # E-step: mean-field per dataset
                    pi = np.clip(rho[d].mean(0), 1e-3, 1 - 1e-3); lp = np.log(pi / (1 - pi))
                    r = rho[d]; m = r @ W.T
                    for _s in range(mf_sweeps):
                        for k in range(K):
                            excl = X - m + np.outer(r[:, k], W[:, k])
                            nk = 1.0 / (1.0 + np.exp(-((excl @ W[:, k] - 0.5 * wn2[k]) / sigma2 + lp[k])))
                            m += np.outer(nk - r[:, k], W[:, k]); r[:, k] = nk
                    rho[d] = r
            if W is not None:
                K = W.shape[1]; Rall = np.vstack(rho)
                sbcs = np.array([_pooled_score_bc(blocks, [r[:, k] for r in rho]) for k in range(K)])
                kept = []
                for k in np.argsort(-sbcs):
                    if sbcs[k] < bc_thr:
                        continue
                    if any(abs(np.corrcoef(Rall[:, k], Rall[:, j])[0, 1]) > dup_corr for j in kept):
                        continue                                          # de-dup by responsibility correlation
                    member = cand[np.abs(W[:, k]) > 1e-9]
                    if len(member) < min_size:
                        continue
                    ov = sum(1 for i in member if any(int(i) in set(int(x) for x in cand[np.abs(W[:, j]) > 1e-9]) for j in kept))
                    kept.append(k)
                    factors.append([genes[i] for i in member])
                    support.append({"n_genes": int(len(member)), "score_bc": float(sbcs[k]), "n_overlap": int(ov)})
    if not return_support:
        return factors
    return factors, support


def discover_factors_ica(train_sets, genes, bc_thr=0.555, min_size=3, max_cand=1200,
                         max_factors=30, beta_floor=0.45, agree_frac=0.6, dup_jaccard=0.7,
                         return_support=False):
    """Discover factors by INDEPENDENT COMPONENT ANALYSIS on the pooled signal subspace — the principled
    alternative to PCA + varimax + localization-thresholds. Binary latents are maximally NON-Gaussian
    (bimodal), and independent; ICA targets exactly that, so it should find them without the variance-based
    machinery (which surfaces the dense global gradient first) or its hand-tuned gates.

    Steps: z-score (batch-invariant) -> candidates (bimodal prefilter) -> project onto the pooled-correlation
    signal subspace (eigenvalues above the Marchenko-Pastur edge) -> FastICA on the STACKED projections, so
    non-Gaussianity is pooled across datasets -> each independent component whose pooled SCORE is bimodal is a
    binary latent (this gate replaces varimax + the participation-ratio gate) -> genes assigned by the same
    cross-dataset CONSISTENCY rule. Components are ordered most-bimodal first. Same return contract as
    discover_factors_multi."""
    blocks = _zscore_blocks(train_sets)
    n_eff = sum(B.shape[0] for B in blocks)
    floor = max(beta_floor, 4.0 / np.sqrt(max(n_eff, 1)))
    bc_pool = _pooled_bc_per_gene(blocks)
    cand = np.where(bc_pool > bc_thr)[0]
    factors, support = [], []
    if len(cand) >= min_size:
        cand = np.sort(cand[np.argsort(-bc_pool[cand])][:max_cand])
        vals, vecs = np.linalg.eigh(_pooled_corr(blocks, cand))
        mp = (1.0 + np.sqrt(len(cand) / max(n_eff, 1))) ** 2
        sig = np.where(vals > mp)[0]
        sig = sig[np.argsort(-vals[sig])][:max_factors]               # signal subspace, strongest first
        K = len(sig)
        if K >= 1:
            E = vecs[:, sig]
            proj = [B[:, cand] @ E for B in blocks]                   # coordinates in the signal subspace
            try:
                ica = FastICA(n_components=K, whiten="unit-variance", random_state=0, max_iter=1000)
                ica.fit(np.vstack(proj))                              # ICA on the pooled cloud
            except Exception:
                ica = None
            if ica is not None:
                Sd = [ica.transform(p) for p in proj]                # per-dataset independent-component scores
                sbcs = np.array([_pooled_score_bc(blocks, [S[:, k] for S in Sd]) for k in range(K)])
                seen = []
                for k in np.argsort(-sbcs):                           # most-bimodal component first
                    if sbcs[k] < bc_thr:                              # not a binary latent
                        continue
                    score_of = [S[:, k] for S in Sd]
                    beta_bar, agree = _consistent_loading(blocks, score_of)
                    member = np.where((np.abs(beta_bar) > floor) & (agree > agree_frac))[0]
                    if len(member) < min_size:
                        continue
                    ms = set(int(i) for i in member)
                    if any(len(ms & s) / len(ms | s) > dup_jaccard for s in seen):
                        continue
                    ov = sum(1 for i in member if any(int(i) in s for s in seen))
                    seen.append(ms)
                    factors.append([genes[i] for i in member])
                    support.append({"n_genes": int(len(member)), "score_bc": float(sbcs[k]), "n_overlap": int(ov)})
    if not return_support:
        return factors
    return factors, support


# ── noise->slope law (leave-one-factor-out capable) ─────────────────────────────

def law_records(X_tgt, X_ref, factors, idx, min_mode=5):
    """Per-factor (slope, noise-ratio) observations from the dual-mode structure of each factor."""
    recs = {}
    for lid, F in enumerate(factors):
        cols = [idx[f] for f in F]; rows = []
        _, hiT, _ = factor_axis(X_tgt[:, cols]); _, hiR, _ = factor_axis(X_ref[:, cols])
        if min(hiT.sum(), (~hiT).sum(), hiR.sum(), (~hiR).sum()) >= min_mode:
            for c in cols:
                sepT = X_tgt[hiT, c].mean() - X_tgt[~hiT, c].mean()
                sepR = X_ref[hiR, c].mean() - X_ref[~hiR, c].mean()
                sdT = 0.5 * (X_tgt[hiT, c].std() + X_tgt[~hiT, c].std()) + 1e-3
                sdR = 0.5 * (X_ref[hiR, c].std() + X_ref[~hiR, c].std()) + 1e-3
                if abs(sepT) > 0.1 and abs(sepR) > 0.2 and np.sign(sepT) == np.sign(sepR):
                    rows.append((sepR / sepT, sdR / sdT))
        recs[lid] = rows
    return recs


def fit_law(recs, exclude=None):
    """Fit (b0,b1) from pooled records, optionally excluding one factor (leak-free for that factor)."""
    sl = [s for lid, rows in recs.items() if lid != exclude for s, _ in rows]
    nr = [n for lid, rows in recs.items() if lid != exclude for _, n in rows]
    if len(sl) < 3:
        return (0.0, 1.0)
    return _fit_law(np.array(sl), np.array(nr))


# ── affine alignment primitive (b-pinning + coupling-fill) ──────────────────────

def _slope(sd_ref, sd_P, law):
    return np.exp(law[0] + law[1] * np.clip(np.log(sd_ref / sd_P), -CLIP, CLIP))


def align_partition_joint(P, z_target, tgt, ref, law):
    """Align a single-mode partition P (mode z_target) of one factor into the reference frame.
    `tgt`/`ref` carry per-mode mu/sd and a `present` flag. Cases: A reference HAS the matched mode
    (b-pin directly); B reference MISSING it but shares the other (coupling-fill from the dual-mode
    target); C doubly-missing (degenerate fallback). Returns (P_aligned, case)."""
    mu_P = P.mean(0); sd_P = P.std(0) + 1e-3
    z_other = "lo" if z_target == "hi" else "hi"
    if ref["present"][z_target]:
        a = _slope(ref["sd"][z_target], sd_P, law); b = ref["mu"][z_target] - a * mu_P; case = "A"
    elif ref["present"][z_other] and tgt["present"][z_target] and tgt["present"][z_other]:
        a = _slope(ref["sd"][z_other], tgt["sd"][z_other], law)
        b = ref["mu"][z_other] - a * tgt["mu"][z_other]; case = "B"
    else:
        zr = z_target if ref["present"][z_target] else z_other
        sd_ref = ref["sd"][zr] if ref["present"][zr] else sd_P
        a = _slope(sd_ref, sd_P, law)
        b = (ref["mu"][zr] if ref["present"][zr] else mu_P) - a * mu_P; case = "C"
    return P * a + b, case


# ── cross-factor mode identification ────────────────────────────────────────────

def _is_bimodal(X):
    """Does this block contain TWO populations? 2-component GMM beats 1 by BIC AND the components are
    separated (>3 pooled sd) — a single population's leading PC is easily over-split, so BIC alone over-calls."""
    if len(X) < 20 or X.shape[1] < 2:
        return False
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    s = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[0][:, 0].reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(s); g2 = GaussianMixture(2, n_init=2, random_state=0).fit(s)
    if g2.bic(s) >= g1.bic(s):
        return False
    mu = g2.means_.ravel(); sd = np.sqrt(g2.covariances_.ravel())
    return abs(mu[0] - mu[1]) / (0.5 * (sd[0] + sd[1]) + 1e-9) > 3.0


def _corr(X):
    return corr_shrink(X) if (X.shape[0] >= 4 and X.shape[1] >= 2) else np.eye(X.shape[1])


def _cdist(a, b):
    m = min(a.shape[0], b.shape[0]); return float(np.sum((a[:m, :m] - b[:m, :m]) ** 2))


def _prof(x):
    """Standardized gene-mean profile — a scale/location-free expression signature (batch-robust)."""
    return (x - x.mean()) / (x.std() + 1e-9)


def _match(Xz, R, cR):
    """Distance between query mode Xz and reference R as the SAME population: signature + correlation."""
    return np.sum((_prof(Xz.mean(0)) - _prof(R.mean(0))) ** 2) + _cdist(_corr(Xz), cR)


def xfactor_match(Q, R, factor_cols, held, hiQ_L, Lcols, axQ=None, axR=None):
    """Cross-factor anchor: predict where R's (single) population of factor `held` sits in the QUERY frame
    using the OTHER factors' reference->query value map, then return shared_is_hi (which query mode of L
    matches the reference). None if too little cross-factor signal. axQ/axR cache factor_axis per factor so
    the anchor is O(factors) not O(factors^2) — identical result (factor_axis is a pure function)."""
    xR, yQ = [], []
    for lid, cols in factor_cols.items():
        if lid == held or len(cols) < 2:
            continue
        try:
            hQ = (axQ[lid] if axQ is not None and lid in axQ else factor_axis(Q[:, cols]))[1]
            hR = (axR[lid] if axR is not None and lid in axR else factor_axis(R[:, cols]))[1]
        except Exception:
            continue
        if min(hQ.sum(), (~hQ).sum(), hR.sum(), (~hR).sum()) < 5:
            continue
        for c in cols:
            xR += [R[hR, c].mean(), R[~hR, c].mean()]; yQ += [Q[hQ, c].mean(), Q[~hQ, c].mean()]
    if len(xR) < 10:
        return None
    pred = qmap(np.array(xR), np.array(yQ), R[:, Lcols].mean(0))
    dh = np.sum((Q[hiQ_L][:, Lcols].mean(0) - pred) ** 2)
    dl = np.sum((Q[~hiQ_L][:, Lcols].mean(0) - pred) ** 2)
    return dh <= dl


# ── the aligner ─────────────────────────────────────────────────────────────────

def _align_one(Q_cat, R_cat, factor_cols, lid, law, force_xfactor=False, axQ=None, axR=None):
    """Align factor `lid`'s genes of the query into the reference. Detects whether the reference holds one
    population or two; identifies the match (cross-factor anchor when one is missing; correlation/signature
    otherwise). Returns the aligned query block (n_query x n_genes_of_factor) and the query hi mask.
    axQ/axR are optional per-factor factor_axis caches (avoid recomputation; identical result)."""
    eps = 1e-3; cols = factor_cols[lid]
    hiQ = (axQ[lid] if axQ is not None and lid in axQ else factor_axis(Q_cat[:, cols]))[1]; Rb = R_cat[:, cols]
    tgt = {"mu": {"hi": Q_cat[hiQ][:, cols].mean(0), "lo": Q_cat[~hiQ][:, cols].mean(0)},
           "sd": {"hi": Q_cat[hiQ][:, cols].std(0) + eps, "lo": Q_cat[~hiQ][:, cols].std(0) + eps},
           "present": {"hi": True, "lo": True}}
    if (not force_xfactor) and _is_bimodal(Rb):                 # reference has BOTH populations
        hiR = (axR[lid] if axR is not None and lid in axR else factor_axis(Rb))[1]
        flip = _match(Q_cat[hiQ][:, cols], Rb[hiR], _corr(Rb[hiR])) > _match(Q_cat[hiQ][:, cols], Rb[~hiR], _corr(Rb[~hiR]))
        rh, rl = ((~hiR, hiR) if flip else (hiR, ~hiR))
        ref = {"mu": {"hi": Rb[rh].mean(0), "lo": Rb[rl].mean(0)},
               "sd": {"hi": Rb[rh].std(0) + eps, "lo": Rb[rl].std(0) + eps}, "present": {"hi": True, "lo": True}}
    else:                                                       # reference has ONE population
        sh = xfactor_match(Q_cat, R_cat, factor_cols, lid, hiQ, cols, axQ=axQ, axR=axR)
        if sh is None:
            sh = _match(Q_cat[hiQ][:, cols], Rb, _corr(Rb)) <= _match(Q_cat[~hiQ][:, cols], Rb, _corr(Rb))
        m, s = Rb.mean(0), Rb.std(0) + eps
        ref = {"mu": {"hi": m if sh else None, "lo": None if sh else m},
               "sd": {"hi": s if sh else None, "lo": None if sh else s}, "present": {"hi": sh, "lo": not sh}}
    Qc = np.zeros((len(Q_cat), len(cols)))
    Qc[hiQ], _ = align_partition_joint(Q_cat[hiQ][:, cols], "hi", tgt, ref, law)
    Qc[~hiQ], _ = align_partition_joint(Q_cat[~hiQ][:, cols], "lo", tgt, ref, law)
    return Qc, hiQ


def _ls_gene(t, r):
    """Location-scale (1-component limit): match a gene's mean and scale. Robust default for genes with no
    usable binary latent. y = a*t + b with a = sd_r/sd_t, b = mean_r - a*mean_t."""
    a = (r.std() + 1e-9) / (t.std() + 1e-9)
    return a * t + (r.mean() - a * t.mean())


def _binary_gene(t, r, law):
    """Single-gene mode-aware affine (2-component): split each by value, align the modes. Beats location-
    scale only when the gene is genuinely bimodal in BOTH datasets (the per-mode stretch matters)."""
    eps = 1e-3
    hT = factor_axis(t.reshape(-1, 1))[1]; hR = factor_axis(r.reshape(-1, 1))[1]
    ms = lambda x: (np.array([x.mean()]), np.array([x.std() + eps]))
    mTh, sTh = ms(t[hT]); mTl, sTl = ms(t[~hT]); mRh, sRh = ms(r[hR]); mRl, sRl = ms(r[~hR])
    tgt = {"mu": {"hi": mTh, "lo": mTl}, "sd": {"hi": sTh, "lo": sTl}, "present": {"hi": True, "lo": True}}
    ref = {"mu": {"hi": mRh, "lo": mRl}, "sd": {"hi": sRh, "lo": sRl}, "present": {"hi": True, "lo": True}}
    o = np.zeros_like(t)
    o[hT] = align_partition_joint(t[hT].reshape(-1, 1), "hi", tgt, ref, law)[0].ravel()
    o[~hT] = align_partition_joint(t[~hT].reshape(-1, 1), "lo", tgt, ref, law)[0].ravel()
    return o


def batch_shift_magnitude(XA, XB, eps=1e-6):
    """Robust global BATCH magnitude (in pooled-sd units): the MEDIAN per-gene affine shift between the two
    datasets — location |mean_B-mean_A|/pooled_sd combined with scale |log(sd_B/sd_A)|. The median makes it a
    BATCH detector, not a biology detector: a differing population mix shifts only the minority of genes in a
    factor, which the median ignores, while a platform/normalization difference shifts ~every gene. Small M =>
    same platform => any difference is biological => correcting would destroy it."""
    mA, mB = XA.mean(0), XB.mean(0); sA, sB = XA.std(0) + eps, XB.std(0) + eps
    ok = (XA.std(0) > eps) & (XB.std(0) > eps)
    loc = np.abs(mB - mA) / (0.5 * (sA + sB)); scl = np.abs(np.log(sB / sA))
    return float(np.median(np.sqrt(loc[ok] ** 2 + scl[ok] ** 2))) if ok.any() else 0.0


def batch_confidence(M, M0=0.5):
    """Correction weight lambda in [0,1]: ~0 when the batch shift M is tiny (trust RAW — small corrections on
    same-platform data usually hurt, since the differences are biological), -> 1 when M is large (real batch,
    correct fully). M0 is the shift (in pooled-sd units) at which we half-trust the correction. (Fallback used
    only when no factors are available; the factor-based batch_confidence_derived has NO such constant.)"""
    return M * M / (M * M + M0 * M0)


def _factor_modes(X, factors, idx, min_mode=5):
    """Per factor: (gene cols, hi-mask) from ONE factor_axis call; None if a mode is too small. Computed once
    per dataset so the null splits below can subset the mask instead of re-running the GMM."""
    out = []
    for F in factors:
        cols = [idx[g] for g in F if g in idx]
        if len(cols) < 2:
            out.append(None); continue
        try:
            hi = factor_axis(X[:, cols])[1]
        except Exception:
            out.append(None); continue
        out.append((cols, hi) if min(int(hi.sum()), int((~hi).sum())) >= min_mode else None)
    return out


def _midpoint_disp(X1, m1, X2, m2):
    """Median MIX-INVARIANT batch displacement (the JBM location parameter) between two sample sets sharing
    factor structure (m1,m2 = per-factor (cols, hi-mask)). Per factor gene the two modes give the affine
    (slope a=sep2/sep1; midpoint m=1/2(mu_hi+mu_lo) is independent of the mixing weights), and |m2 - a*m1| /
    within-mode-sd is the offset NOT explained by a population-mix difference — i.e. genuine batch."""
    vals = []
    for f1, f2 in zip(m1, m2):
        if f1 is None or f2 is None:
            continue
        cols, h1 = f1; _, h2 = f2
        if min(int(h1.sum()), int((~h1).sum()), int(h2.sum()), int((~h2).sum())) < 3:
            continue
        for c in cols:
            a1h, a1l = X1[h1, c].mean(), X1[~h1, c].mean(); a2h, a2l = X2[h2, c].mean(), X2[~h2, c].mean()
            s1, s2 = a1h - a1l, a2h - a2l
            sd2 = 0.5 * (X2[h2, c].std() + X2[~h2, c].std()) + 1e-6
            if abs(s1) < 0.1 or np.sign(s1) != np.sign(s2):
                continue
            a = s2 / s1
            vals.append(abs(0.5 * (a2h + a2l) - a * 0.5 * (a1h + a1l)) / sd2)
    return float(np.median(vals)) if vals else float("nan")


def batch_confidence_derived(XA, XB, factors, idx, n_splits=2, seed=0):
    """Correction weight with NO hand-set scale: derive the no-batch baseline empirically. The observed batch
    (midpoint displacement on the bimodal genes) is compared to the NULL displacement seen between two random
    halves of a SINGLE dataset (same platform, no batch — only sampling + cohort mix, on the same factor
    genes, reusing the full-data mode mask). lambda = clamp(1 - (M_null/M_obs)^2): ~0 when the cross-dataset
    shift is no bigger than the within-dataset sampling floor (same platform => differences are biological =>
    keep raw), -> 1 when it far exceeds it (real batch). Falls back to the all-gene magnitude with no factors."""
    if not factors:
        return batch_confidence(batch_shift_magnitude(XA, XB))
    mA = _factor_modes(XA, factors, idx); mB = _factor_modes(XB, factors, idx)
    M_obs = _midpoint_disp(XA, mA, XB, mB)
    if not np.isfinite(M_obs):
        return batch_confidence(batch_shift_magnitude(XA, XB))
    rng = np.random.RandomState(seed); nulls = []
    n = len(XA); h = n // 2                                            # null on the QUERY (the data being shrunk):
    for _ in range(n_splits):                                          # REFIT factor_axis per half to match M_obs
        p = rng.permutation(n); X1, X2 = XA[p[:h]], XA[p[h:]]          # (independent splits — mode-assignment
        d = _midpoint_disp(X1, _factor_modes(X1, factors, idx), X2, _factor_modes(X2, factors, idx))  # variability counts)
        if np.isfinite(d):
            nulls.append(d)
    if not nulls:
        return batch_confidence(batch_shift_magnitude(XA, XB))
    M_null = float(np.median(nulls))
    return float(np.clip(1.0 - (M_null / (M_obs + 1e-9)) ** 2, 0.0, 1.0))


def align_all(XA, XB, genes, factors=None, law=None, return_factors=False, shrink_raw=True):
    """Adjust EVERY gene of XA into XB's frame (the batch-confounding eliminator for the full transcriptome),
    model-selected per gene: cross-factor mode-aware for genes in a discovered factor; single-gene mode-aware
    for genes individually bimodal in both datasets; location-scale otherwise (unimodal, or bimodal in one).
    `factors` and `law`, if given, are used as-is (frozen) rather than re-derived — used by align_test.
    With shrink_raw, the whole correction is blended toward RAW by the batch confidence (no batch => keep raw,
    preserving biology); for a real cross-platform batch the weight is ~1 so this is a no-op."""
    idx = {g: i for i, g in enumerate(genes)}
    if factors is None:
        factors = discover_factors(XA, XB, genes)
    out = align_into_reference(XA, XB, genes, factors=factors, law=law)  # factor genes (cross-factor mode-ID)
    fac = {idx[g] for f in factors for g in f}
    law = law if law is not None else fit_law(law_records(XA, XB, factors, idx))
    # per-gene sufficient statistics, computed ONCE (vectorized) instead of recomputed per gene in the loop
    mA, sA = XA.mean(0), XA.std(0); mB, sB = XB.mean(0), XB.std(0)
    bcA, bcB = bimodality_coef(XA), bimodality_coef(XB)
    facmask = np.zeros(len(genes), bool); facmask[list(fac)] = True
    deg = (~facmask) & ((sA < 1e-9) | (sB < 1e-9))                     # degenerate (no spread in one set)
    binm = (~facmask) & (~deg) & (bcA > 0.555) & (bcB > 0.555)         # individually bimodal in both
    lsm = (~facmask) & (~deg) & (~binm)                               # location-scale (the robust default)
    out[:, deg] = XA[:, deg] - mA[deg] + mB[deg]
    a = (sB[lsm] + 1e-9) / (sA[lsm] + 1e-9)                            # one vectorized affine for ALL ls genes
    out[:, lsm] = a * (XA[:, lsm] - mA[lsm]) + mB[lsm]
    for i in np.where(binm)[0]:                                       # only the bimodal genes still need a GMM
        try:
            out[:, i] = _binary_gene(XA[:, i], XB[:, i], law)
        except Exception:
            out[:, i] = (sB[i] + 1e-9) / (sA[i] + 1e-9) * (XA[:, i] - mA[i]) + mB[i]
    if shrink_raw:
        lam = batch_confidence_derived(XA, XB, factors, idx)           # back off when there is no batch effect
        out = lam * out + (1.0 - lam) * XA                             # (scale derived from a within-set null)
    return (out, factors) if return_factors else out


def align_into_reference(XA, XB, genes, factors=None, law=None, return_factors=False):
    """Align the discovered-factor genes of XA into XB's frame (cross-factor mode-ID). With `law=None` the
    noise->slope law is fit leave-one-factor-out; pass a frozen (b0,b1) to use it as-is (align_test)."""
    idx = {g: i for i, g in enumerate(genes)}
    if factors is None:
        factors = discover_factors(XA, XB, genes)
    if not factors:
        return (XA.copy(), factors) if return_factors else XA.copy()
    cat_genes = sorted({g for f in factors for g in f}, key=lambda g: idx[g])
    catalog_cols = [idx[g] for g in cat_genes]; local = {g: i for i, g in enumerate(cat_genes)}
    factor_cols = {lid: [local[g] for g in f] for lid, f in enumerate(factors)}
    recs = None if law is not None else law_records(XA, XB, factors, idx)
    Q_cat = XA[:, catalog_cols]; R_cat = XB[:, catalog_cols]
    axQ = {lid: factor_axis(Q_cat[:, cols]) for lid, cols in factor_cols.items() if len(cols) >= 2}
    axR = {lid: factor_axis(R_cat[:, cols]) for lid, cols in factor_cols.items() if len(cols) >= 2}
    out = XA.copy(); written = set()
    for lid, cols in factor_cols.items():                    # factors are dominant-first (strongest rotation)
        if len(cols) < 2:
            continue
        lw = law if law is not None else fit_law(recs, exclude=lid)
        Qc, _ = _align_one(Q_cat, R_cat, factor_cols, lid, lw, axQ=axQ, axR=axR)
        for j, c in enumerate(cols):                         # write each gene ONCE, by its dominant factor
            gcol = catalog_cols[c]
            if gcol in written:
                continue
            out[:, gcol] = Qc[:, j]; written.add(gcol)
    return (out, factors) if return_factors else out


# ── multiple training sets: merge (fit) + project a test set (transform) ─────────

def _set_snr(X, factors, idx, min_mode=5):
    """A frame's 'cleanliness': median within-factor signal-to-noise (|mode separation| / pooled within-mode
    sd) over all factor genes. Higher = tighter, better-separated populations — nicer to align everyone into
    (the stretch a = sd_ref/sd_target then COMPRESSES noisier sets toward the clean reference rather than
    amplifying everyone toward a noisy one)."""
    vals = []
    for F in factors:
        cols = [idx[g] for g in F]
        try:
            _, hi, _ = factor_axis(X[:, cols])
        except Exception:
            continue
        if min(int(hi.sum()), int((~hi).sum())) < min_mode:
            continue
        for c in cols:
            sep = abs(X[hi, c].mean() - X[~hi, c].mean())
            sd = 0.5 * (X[hi, c].std() + X[~hi, c].std()) + 1e-9
            vals.append(sep / sd)
    return float(np.median(vals)) if vals else 0.0


def pick_reference(train_sets, factors, genes):
    """Choose the common frame to merge into = the CLEANEST set (highest within-factor SNR), so the merged
    space inherits low-noise, well-separated, 'nice' properties — rather than privileging an arbitrary
    anchor. Falls back to the largest set when there is no factor signal to judge cleanliness by."""
    idx = {g: i for i, g in enumerate(genes)}
    if factors:
        snr = [_set_snr(X, factors, idx) for X in train_sets]
        if max(snr) > 0:
            return int(np.argmax(snr))
    return int(np.argmax([len(X) for X in train_sets]))


def merge_training(train_sets, genes, anchor=None, discover=None):
    """Combine N training sets into ONE merged dataset (to train a classifier on), using evidence from
    EVERY set — not just a pair. Three joint steps:

      1. DISCOVER factors jointly across all sets (default discover_factors_sfa): mean-field variational EM
         for a binary, correlated, OVERLAPPING sparse factor model (best mode-ID; sparsity penalty derived
         from noise — no hand-set knob). Pass discover=discover_factors_ica (independent, tight, faster) or
         discover_factors_multi (PCA+varimax) to swap.
      2. CHOOSE the most consistent space (pick_reference): align into the cleanest / lowest-noise set so
         the merged frame has nice properties (override with `anchor` to force a specific set).
      3. FIT one frozen noise->slope law pooled over EVERY set->reference transformation, then align each
         set into the reference with that shared factor set and law (the reference passes through unchanged).

    Row order of `merged` follows the input order of `train_sets` (so caller labels concatenate directly).
    Returns (merged, model); `model` freezes {factors, law, anchor, support} so a held-out test aligns onto
    this exact frame later via align_test (training never re-touched)."""
    idx = {g: i for i, g in enumerate(genes)}
    factors, support = (discover or discover_factors_sfa)(train_sets, genes, return_support=True)
    if anchor is None:
        anchor = pick_reference(train_sets, factors, genes)
    A = train_sets[anchor]
    others = [i for i in range(len(train_sets)) if i != anchor]
    allrows = []                                                       # frozen law pooled over every set->reference
    for i in others:
        for rows in law_records(train_sets[i], A, factors, idx).values():
            allrows.extend(rows)
    law = fit_law({0: allrows}) if allrows else (0.0, 1.0)
    aligned = [A if i == anchor else align_all(train_sets[i], A, genes, factors=factors, law=law)
               for i in range(len(train_sets))]                       # preserve input order
    return np.vstack(aligned), {"factors": factors, "law": law, "anchor": anchor, "support": support}


def align_test(X_test, merged, genes, model):
    """Align a held-out test set onto the merged training frame, holding the training data FIXED (the
    classifier was trained on `merged`; this only adjusts the test). Uses the frozen factors + law."""
    return align_all(X_test, merged, genes, factors=model["factors"], law=model["law"])


if __name__ == "__main__":
    # Self-contained smoke test on synthetic data (no file IO — loading lives with the evaluation).
    # Two datasets share a binary population on genes 0..9; dataset B applies a per-gene affine STRETCH.
    rng = np.random.RandomState(0)
    G, nA, nB = 60, 200, 300
    genes = [f"g{i}" for i in range(G)]
    zA = rng.rand(nA) < 0.5; zB = rng.rand(nB) < 0.4               # the population, present in both
    sep = rng.rand(10) * 2 + 2
    XA = rng.randn(nA, G) * 0.5; XB = rng.randn(nB, G) * 0.5
    XA[:, :10] += np.outer(zA, sep)                               # factor in A
    XB[:, :10] += np.outer(zB, sep * 3.0) + 5.0                   # factor in B: 3x stretch + offset (batch)
    XB[:, 10:] = XB[:, 10:] * 2.0 + 1.0                           # per-gene stretch on the rest too
    factors = discover_factors(XA, XB, genes)
    A2B = align_into_reference(XA, XB, genes, factors=factors)
    print(f"discovered {len(factors)} factor(s) {[len(f) for f in factors]}; aligned A -> B {A2B.shape}")
    print(f"factor-gene means:  A {XA[:, :10].mean():+.2f}  ->  aligned {A2B[:, :10].mean():+.2f}   "
          f"(target B {XB[:, :10].mean():+.2f})")
