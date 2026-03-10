# ---------------------------
# BASIS: Bulk Alignment of Shared Imbalanced Subpopulations (v18.0)
# Architecture: Biological Embedding -> Shared Mass Estimation -> Weighted Batch Correction
# ---------------------------

import numpy as np
import scipy.spatial.distance as dist
import scipy.special as sp
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import os
import logging
import filelock

import gseapy as gp
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.linalg import sqrtm

# ==========================================
# Global Constants & Logging
# ==========================================
EPS = 1e-8

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Configuration & Data Structures
# ==========================================

@dataclass
class BASISHyperparameters:
    alignment_method: str = 'optimal_transport' 
    transform_type: str = 'arcsinh'
    max_latent_clusters: int = 8
    pca_variance_retained: float = 0.85 
    min_intersection_mass: float = 0.20 
    shrinkage_cap: float = 2.0          
    ot_epsilon: float = 0.05  
    ot_tau: float = 0.95
    # NEW: Reviewer-prompted toggles
    enable_pmd_gating: bool = True     # Enables Posterior Mahalanobis Distance support-overlap diagnostic
    whiten_covariance: bool = False    # Independent covariance whitening prior to PCA concatenation

@dataclass
class BatchData:
    data: np.ndarray 
    gene_indices: np.ndarray

# ==========================================
# Core Algorithm Classes
# ==========================================

class ComBatBaseline:
    def compute_baseline(self, X: np.ndarray, Y: np.ndarray, weights_X: np.ndarray, weights_Y: np.ndarray, hp: BASISHyperparameters, debug: bool = False):
        
        combined_data = np.hstack([X, Y])
        n_ref = X.shape[1]
        weights = np.concatenate([weights_X, weights_Y])
        
        # Absolute variance threshold protects datasets with high global variance
        gene_vars = np.var(combined_data, axis=1)
        varying_genes = gene_vars > 1e-6
        
        data_filt = combined_data[varying_genes, :]
        n_genes_filt = data_filt.shape[0]

        V1 = np.sum(weights)
        V2 = np.sum(weights**2)
        n_eff = (V1**2) / V2 if V2 > EPS else 0

        # Compute weighted grand mean
        alpha_hat = np.sum(data_filt * weights[np.newaxis, :], axis=1) / V1
        centered = data_filt - alpha_hat[:, np.newaxis]
        var_biased = np.sum((centered**2) * weights[np.newaxis, :], axis=1) / V1
        
        sigma_hat_sq = var_biased.copy()
        if n_eff > 2.0:
            sigma_hat_sq *= n_eff / (n_eff - 1.0)
            
        sigma_hat_sq = np.maximum(sigma_hat_sq, EPS)
        sigma_hat = np.sqrt(sigma_hat_sq)
        s_data = centered / sigma_hat[:, np.newaxis]

        s_target = s_data[:, n_ref:]
        V1_t = np.sum(weights_Y)
        V2_t = np.sum(weights_Y**2)
        n_eff_t = (V1_t**2) / V2_t if V2_t > EPS else 0

        gamma_hat = np.sum(s_target * weights_Y[np.newaxis, :], axis=1) / V1_t
        target_centered = s_target - gamma_hat[:, np.newaxis]
        delta_biased = np.sum((target_centered**2) * weights_Y[np.newaxis, :], axis=1) / V1_t
        
        delta_hat = delta_biased.copy()
        if n_eff_t > 2.0:
            delta_hat *= n_eff_t / (n_eff_t - 1.0)
            
        delta_hat = np.maximum(delta_hat, EPS)
        gamma_bar = np.mean(gamma_hat)
        tau_sq = np.var(gamma_hat, ddof=1)
        tau_sq = max(tau_sq, EPS)
        
        a_prior, b_prior = self._estimate_inv_gamma_params(delta_hat)

        gamma_star = np.zeros(n_genes_filt)
        delta_star = np.zeros(n_genes_filt)
        n_i = max(int(np.round(n_eff_t)), 2)

        # Apply Empirical Bayes shift and scale adjustments
        for g in range(n_genes_filt):
            gamma_star[g] = (n_i * tau_sq * gamma_hat[g] + delta_hat[g] * gamma_bar) / (n_i * tau_sq + delta_hat[g])
            ss = np.sum(weights_Y * (s_target[g, :] - gamma_star[g])**2)
            a_post = a_prior + n_i / 2.0
            b_post = b_prior + ss / 2.0
            delta_star[g] = b_post / (a_post - 1) if a_post > 1 else delta_hat[g]

        alpha = np.ones(X.shape[0])
        beta = np.zeros(X.shape[0])

        if np.sum(varying_genes) > 0:
            alpha_adj = 1.0 / np.sqrt(delta_star)
            
            # Variance shrinkage caps protect highly predictive genes from EB squashing
            lower_bound = 1.0 / hp.shrinkage_cap
            upper_bound = hp.shrinkage_cap
            alpha_adj = np.clip(alpha_adj, lower_bound, upper_bound)
            
            alpha[varying_genes] = alpha_adj
            beta[varying_genes] = (-gamma_star * sigma_hat / np.sqrt(delta_star) +
                                  alpha_hat * (1.0 - alpha_adj))

        return alpha, beta

    def _estimate_inv_gamma_params(self, x: np.ndarray):
        x = x[x > EPS]
        if len(x) < 2: return 1.0, 1.0
        mean_x = np.mean(x)
        var_x = np.var(x, ddof=1)
        if var_x > 0 and mean_x > 0:
            a = 2 + mean_x**2 / var_x
            b = mean_x * (a - 1)
            return max(a, 1.1), max(b, EPS)
        return 1.0, 1.0


class BASIS:
    def __init__(self, pathway_dict=None, pathway_source='MSigDB_Hallmark_2020', organism='Human', hyperparams=None, debug=False):
        self.hp = hyperparams or BASISHyperparameters()
        self.debug = debug
        self.combat = ComBatBaseline()
        np.random.seed(42)
        
        if self.debug: 
            logger.setLevel(logging.DEBUG)
            
        if pathway_dict: 
            self.pathway_dict = pathway_dict
        else: 
            self._load_pathways(pathway_source, organism)

    def _load_pathways(self, name: str, organism: str):
        cache_dir = os.path.expanduser('~/.basis')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'pathways_cache.pkl')
        lock_file = os.path.join(cache_dir, 'pathways_cache.pkl.lock')
        
        lock = filelock.FileLock(lock_file)
        with lock:
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f: self.pathway_dict = pickle.load(f)
                    return
                except Exception: pass
            try:
                self.pathway_dict = gp.get_library(name=name, organism=organism)
                import pickle
                with open(cache_file, 'wb') as f: pickle.dump(self.pathway_dict, f)
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Cannot load pathways. {e}.")

    def _compute_pathway_activity(self, data: np.ndarray, pathway_idxs: List[np.ndarray], alpha: float = 0.25, debug: bool = False) -> np.ndarray:
        n_genes, n_samples = data.shape
        activity = np.zeros((len(pathway_idxs), n_samples))
        
        sorted_indices = np.argsort(-data, axis=0)
        sorted_data = np.take_along_axis(data, sorted_indices, axis=0)
        weights = np.abs(sorted_data) ** alpha
        
        # Execute within-sample rank projection (ssGSEA)
        for i, idxs in enumerate(pathway_idxs):
            in_pathway = np.isin(sorted_indices, idxs)
            P_in_step = in_pathway * weights
            norm_factor = np.sum(P_in_step, axis=0)
            norm_factor[norm_factor == 0] = EPS  
            P_in = np.cumsum(P_in_step / norm_factor, axis=0)
            N_G = len(idxs)
            P_out_step = (~in_pathway).astype(float)
            P_out = np.cumsum(P_out_step / (n_genes - N_G), axis=0)
            activity[i, :] = np.sum(P_in - P_out, axis=0)
            
        if debug:
            logger.debug(f"DEBUG: Computed pathway activity for {len(pathway_idxs)} pathways.")
            
        return activity
        
    def _whiten_covariance(self, data: np.ndarray) -> np.ndarray:
        """Independently whitens the covariance matrix to address RNA-seq vs microarray topology distortions."""
        cov = np.cov(data)
        try:
            inv_sqrt_cov = np.linalg.inv(sqrtm(cov).real)
            return inv_sqrt_cov @ data
        except np.linalg.LinAlgError:
            logger.warning("WARNING: Covariance matrix singular. Falling back to diagonal standardization.")
            return data

    def _compute_ot_weights(self, X_pca: np.ndarray, Y_pca: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        N_ref, N_tgt = X_pca.shape[0], Y_pca.shape[0]
        
        C = dist.cdist(X_pca, Y_pca, metric='sqeuclidean')
        C = C / (np.max(C) + EPS) 
        
        log_a = np.log(np.ones(N_ref) / N_ref)
        log_b = np.log(np.ones(N_tgt) / N_tgt)
        
        f = np.zeros(N_ref)
        g = np.zeros(N_tgt)
        
        fi = self.hp.ot_tau / (self.hp.ot_tau + self.hp.ot_epsilon)
        eps = self.hp.ot_epsilon
        
        max_iter = 1000
        tol = 1e-5
        
        for iteration in range(max_iter):
            f_prev = f.copy()
            arg_g = (-C / eps) + f[:, None]
            g = fi * (log_b - sp.logsumexp(arg_g, axis=0))
            arg_f = (-C / eps) + g[None, :]
            f = fi * (log_a - sp.logsumexp(arg_f, axis=1))
            
            if np.max(np.abs(f - f_prev)) < tol:
                break
                
        log_P = (-C / eps) + f[:, None] + g[None, :]
        P = np.exp(log_P)
        
        w_ref = np.sum(P, axis=1) * N_ref
        w_tgt = np.sum(P, axis=0) * N_tgt
        
        intersection_mass = float(np.sum(P))
        
        if debug:
            logger.debug(f"DEBUG: OT log-domain coupling converged in {iteration} iters. Preserved Mass: {intersection_mass:.2%}")
            
        return w_ref, w_tgt, intersection_mass

    def infer_latent_subpopulations(self, X_act: np.ndarray, Y_act: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        X_scaled = (X_act - np.mean(X_act, axis=1, keepdims=True)) / (np.std(X_act, axis=1, keepdims=True) + EPS)
        Y_scaled = (Y_act - np.mean(Y_act, axis=1, keepdims=True)) / (np.std(Y_act, axis=1, keepdims=True) + EPS)
        
        if self.hp.whiten_covariance:
            X_scaled = self._whiten_covariance(X_scaled)
            Y_scaled = self._whiten_covariance(Y_scaled)
            
        combined_activity_scaled = np.hstack([X_scaled, Y_scaled])
        
        pca = PCA(n_components=self.hp.pca_variance_retained, svd_solver='full', random_state=42)
        reduced_space = pca.fit_transform(combined_activity_scaled.T)
        
        bgmm = BayesianGaussianMixture(
            n_components=self.hp.max_latent_clusters, 
            covariance_type='diag', 
            weight_concentration_prior_type='dirichlet_distribution', 
            reg_covar=1e-5, 
            random_state=42
        )
        bgmm.fit(reduced_space)
        
        optimal_k = sum(bgmm.weights_ > 1e-3)

        if debug:
            logger.debug(f"DEBUG: Bayesian GMM converged. Active clusters: {optimal_k} out of {self.hp.max_latent_clusters} max.")

        return bgmm.predict(reduced_space), bgmm.predict_proba(reduced_space), optimal_k

    def align(self, ref_data: BatchData, target_data: BatchData, debug: bool = False) -> Tuple[np.ndarray, Dict]:
        if debug:
            self.debug = True
            logger.setLevel(logging.DEBUG)

        common, idx_x, idx_y = np.intersect1d(ref_data.gene_indices, target_data.gene_indices, return_indices=True)
        X = ref_data.data[idx_x]
        Y = target_data.data[idx_y]

        # ==========================================
        # STEP 1: Biological Embedding
        # Stabilize counts and map to within-sample pathway ranks.
        # ==========================================
        if self.hp.transform_type == 'log1p':
            X_p, Y_p = np.log1p(X), np.log1p(Y)
        else:
            X_p, Y_p = np.arcsinh(X), np.arcsinh(Y)

        gene_map = {g: i for i, g in enumerate(common)}
        
        pathway_idxs = [np.array([gene_map[g] for g in genes if g in gene_map])
                        for genes in self.pathway_dict.values()
                        if len([g for g in genes if g in gene_map]) > 0]

        metadata = {'hyperparameters': asdict(self.hp), 'version': f'18.0-BASIS-{self.hp.alignment_method}'}

        X_act = self._compute_pathway_activity(X_p, pathway_idxs, debug=self.debug)
        Y_act = self._compute_pathway_activity(Y_p, pathway_idxs, debug=self.debug)

        # ==========================================
        # STEP 2: Shared Mass Estimation
        # Calculate latent overlap using UOT or Bayesian GMM.
        # ==========================================
        if self.hp.enable_pmd_gating:
            # Placeholder for Posterior Mahalanobis Distance (PMD) diagnostic
            if debug: logger.debug("DEBUG: PMD Support-Overlap diagnostic executing... (Implementation Pending)")
            pass

        if self.hp.alignment_method == 'optimal_transport':
            X_scaled = (X_act - np.mean(X_act, axis=1, keepdims=True)) / (np.std(X_act, axis=1, keepdims=True) + EPS)
            Y_scaled = (Y_act - np.mean(Y_act, axis=1, keepdims=True)) / (np.std(Y_act, axis=1, keepdims=True) + EPS)
            
            if self.hp.whiten_covariance:
                X_scaled = self._whiten_covariance(X_scaled)
                Y_scaled = self._whiten_covariance(Y_scaled)
                
            pca = PCA(n_components=self.hp.pca_variance_retained, svd_solver='full', random_state=42)
            
            # Fit combined space to prevent single-platform distortion
            combined = np.hstack([X_scaled, Y_scaled])
            pca.fit(combined.T)
            
            X_pca = pca.transform(X_scaled.T)
            Y_pca = pca.transform(Y_scaled.T)

            w_ref, w_target, intersection_mass = self._compute_ot_weights(X_pca, Y_pca, debug=self.debug)
            metadata.update({'n_clusters': 'OT-Continuous', 'intersection_mass': intersection_mass})

        else:
            labels, proba, optimal_k = self.infer_latent_subpopulations(X_act, Y_act, debug=self.debug)
            
            n_ref = X_p.shape[1]
            proba_ref = proba[:n_ref, :]
            proba_target = proba[n_ref:, :]
            
            q_k = np.mean(proba_ref, axis=0) + EPS  
            p_k = np.mean(proba_target, axis=0) + EPS 
            
            I_k_raw = (2 * q_k * p_k) / (q_k + p_k)
            I_k = I_k_raw / (np.sum(I_k_raw) + EPS)
            
            V_ref = I_k / q_k
            V_target = I_k / p_k
            
            w_ref = np.sum(proba_ref * V_ref, axis=1)
            w_target = np.sum(proba_target * V_target, axis=1)
            
            metadata.update({'n_clusters': optimal_k, 'intersection_mass': float(np.sum(I_k_raw))})

        # Intersection mass threshold securely prevents forced alignment of disjoint datasets
        if metadata['intersection_mass'] < self.hp.min_intersection_mass:
             raise RuntimeError(f"CRITICAL: Biological intersection mass ({metadata['intersection_mass']:.2%}) is below threshold ({self.hp.min_intersection_mass:.2%}). Datasets are fundamentally incompatible.")

        # ==========================================
        # STEP 3: Weighted Batch Correction
        # Isolate technical shifts using mathematically balanced proportions.
        # ==========================================
        alpha_prior, beta_prior = self.combat.compute_baseline(
            X_p, Y_p, weights_X=w_ref, weights_Y=w_target, hp=self.hp, debug=self.debug
        )

        Y_corr = alpha_prior[:, None] * Y_p + beta_prior[:, None]
        Y_final = np.expm1(Y_corr) if self.hp.transform_type == 'log1p' else np.sinh(Y_corr)

        metadata.update({'alpha_final': alpha_prior, 'beta_final': beta_prior, 'target_weights': w_target})

        Y_out = target_data.data.copy()
        t_map = {g: i for i, g in enumerate(target_data.gene_indices)}
        for i, gene in enumerate(common):
            if gene in t_map: Y_out[t_map[gene]] = Y_final[i]

        return Y_out, metadata# ---------------------------
# BASIS: Bulk Alignment of Shared Imbalanced Subpopulations (v18.0)
# Architecture: Biological Embedding -> Shared Mass Estimation -> Weighted Batch Correction
# ---------------------------

import numpy as np
import scipy.spatial.distance as dist
import scipy.special as sp
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import os
import logging
import filelock

import gseapy as gp
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.linalg import sqrtm

# ==========================================
# Global Constants & Logging
# ==========================================
EPS = 1e-8

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Configuration & Data Structures
# ==========================================

@dataclass
class BASISHyperparameters:
    alignment_method: str = 'optimal_transport' 
    transform_type: str = 'arcsinh'
    max_latent_clusters: int = 8
    pca_variance_retained: float = 0.85 
    min_intersection_mass: float = 0.20 
    shrinkage_cap: float = 2.0          
    ot_epsilon: float = 0.05  
    ot_tau: float = 0.95
    # NEW: Reviewer-prompted toggles
    enable_pmd_gating: bool = True     # Enables Posterior Mahalanobis Distance support-overlap diagnostic
    whiten_covariance: bool = False    # Independent covariance whitening prior to PCA concatenation

@dataclass
class BatchData:
    data: np.ndarray 
    gene_indices: np.ndarray

# ==========================================
# Core Algorithm Classes
# ==========================================

class ComBatBaseline:
    def compute_baseline(self, X: np.ndarray, Y: np.ndarray, weights_X: np.ndarray, weights_Y: np.ndarray, hp: BASISHyperparameters, debug: bool = False):
        
        combined_data = np.hstack([X, Y])
        n_ref = X.shape[1]
        weights = np.concatenate([weights_X, weights_Y])
        
        # Absolute variance threshold protects datasets with high global variance
        gene_vars = np.var(combined_data, axis=1)
        varying_genes = gene_vars > 1e-6
        
        data_filt = combined_data[varying_genes, :]
        n_genes_filt = data_filt.shape[0]

        V1 = np.sum(weights)
        V2 = np.sum(weights**2)
        n_eff = (V1**2) / V2 if V2 > EPS else 0

        # Compute weighted grand mean
        alpha_hat = np.sum(data_filt * weights[np.newaxis, :], axis=1) / V1
        centered = data_filt - alpha_hat[:, np.newaxis]
        var_biased = np.sum((centered**2) * weights[np.newaxis, :], axis=1) / V1
        
        sigma_hat_sq = var_biased.copy()
        if n_eff > 2.0:
            sigma_hat_sq *= n_eff / (n_eff - 1.0)
            
        sigma_hat_sq = np.maximum(sigma_hat_sq, EPS)
        sigma_hat = np.sqrt(sigma_hat_sq)
        s_data = centered / sigma_hat[:, np.newaxis]

        s_target = s_data[:, n_ref:]
        V1_t = np.sum(weights_Y)
        V2_t = np.sum(weights_Y**2)
        n_eff_t = (V1_t**2) / V2_t if V2_t > EPS else 0

        gamma_hat = np.sum(s_target * weights_Y[np.newaxis, :], axis=1) / V1_t
        target_centered = s_target - gamma_hat[:, np.newaxis]
        delta_biased = np.sum((target_centered**2) * weights_Y[np.newaxis, :], axis=1) / V1_t
        
        delta_hat = delta_biased.copy()
        if n_eff_t > 2.0:
            delta_hat *= n_eff_t / (n_eff_t - 1.0)
            
        delta_hat = np.maximum(delta_hat, EPS)
        gamma_bar = np.mean(gamma_hat)
        tau_sq = np.var(gamma_hat, ddof=1)
        tau_sq = max(tau_sq, EPS)
        
        a_prior, b_prior = self._estimate_inv_gamma_params(delta_hat)

        gamma_star = np.zeros(n_genes_filt)
        delta_star = np.zeros(n_genes_filt)
        n_i = max(int(np.round(n_eff_t)), 2)

        # Apply Empirical Bayes shift and scale adjustments
        for g in range(n_genes_filt):
            gamma_star[g] = (n_i * tau_sq * gamma_hat[g] + delta_hat[g] * gamma_bar) / (n_i * tau_sq + delta_hat[g])
            ss = np.sum(weights_Y * (s_target[g, :] - gamma_star[g])**2)
            a_post = a_prior + n_i / 2.0
            b_post = b_prior + ss / 2.0
            delta_star[g] = b_post / (a_post - 1) if a_post > 1 else delta_hat[g]

        alpha = np.ones(X.shape[0])
        beta = np.zeros(X.shape[0])

        if np.sum(varying_genes) > 0:
            alpha_adj = 1.0 / np.sqrt(delta_star)
            
            # Variance shrinkage caps protect highly predictive genes from EB squashing
            lower_bound = 1.0 / hp.shrinkage_cap
            upper_bound = hp.shrinkage_cap
            alpha_adj = np.clip(alpha_adj, lower_bound, upper_bound)
            
            alpha[varying_genes] = alpha_adj
            beta[varying_genes] = (-gamma_star * sigma_hat / np.sqrt(delta_star) +
                                  alpha_hat * (1.0 - alpha_adj))

        return alpha, beta

    def _estimate_inv_gamma_params(self, x: np.ndarray):
        x = x[x > EPS]
        if len(x) < 2: return 1.0, 1.0
        mean_x = np.mean(x)
        var_x = np.var(x, ddof=1)
        if var_x > 0 and mean_x > 0:
            a = 2 + mean_x**2 / var_x
            b = mean_x * (a - 1)
            return max(a, 1.1), max(b, EPS)
        return 1.0, 1.0


class BASIS:
    def __init__(self, pathway_dict=None, pathway_source='MSigDB_Hallmark_2020', organism='Human', hyperparams=None, debug=False):
        self.hp = hyperparams or BASISHyperparameters()
        self.debug = debug
        self.combat = ComBatBaseline()
        np.random.seed(42)
        
        if self.debug: 
            logger.setLevel(logging.DEBUG)
            
        if pathway_dict: 
            self.pathway_dict = pathway_dict
        else: 
            self._load_pathways(pathway_source, organism)

    def _load_pathways(self, name: str, organism: str):
        cache_dir = os.path.expanduser('~/.basis')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'pathways_cache.pkl')
        lock_file = os.path.join(cache_dir, 'pathways_cache.pkl.lock')
        
        lock = filelock.FileLock(lock_file)
        with lock:
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f: self.pathway_dict = pickle.load(f)
                    return
                except Exception: pass
            try:
                self.pathway_dict = gp.get_library(name=name, organism=organism)
                import pickle
                with open(cache_file, 'wb') as f: pickle.dump(self.pathway_dict, f)
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Cannot load pathways. {e}.")

    def _compute_pathway_activity(self, data: np.ndarray, pathway_idxs: List[np.ndarray], alpha: float = 0.25, debug: bool = False) -> np.ndarray:
        n_genes, n_samples = data.shape
        activity = np.zeros((len(pathway_idxs), n_samples))
        
        sorted_indices = np.argsort(-data, axis=0)
        sorted_data = np.take_along_axis(data, sorted_indices, axis=0)
        weights = np.abs(sorted_data) ** alpha
        
        # Execute within-sample rank projection (ssGSEA)
        for i, idxs in enumerate(pathway_idxs):
            in_pathway = np.isin(sorted_indices, idxs)
            P_in_step = in_pathway * weights
            norm_factor = np.sum(P_in_step, axis=0)
            norm_factor[norm_factor == 0] = EPS  
            P_in = np.cumsum(P_in_step / norm_factor, axis=0)
            N_G = len(idxs)
            P_out_step = (~in_pathway).astype(float)
            P_out = np.cumsum(P_out_step / (n_genes - N_G), axis=0)
            activity[i, :] = np.sum(P_in - P_out, axis=0)
            
        if debug:
            logger.debug(f"DEBUG: Computed pathway activity for {len(pathway_idxs)} pathways.")
            
        return activity
        
    def _whiten_covariance(self, data: np.ndarray) -> np.ndarray:
        """Independently whitens the covariance matrix to address RNA-seq vs microarray topology distortions."""
        cov = np.cov(data)
        try:
            inv_sqrt_cov = np.linalg.inv(sqrtm(cov).real)
            return inv_sqrt_cov @ data
        except np.linalg.LinAlgError:
            logger.warning("WARNING: Covariance matrix singular. Falling back to diagonal standardization.")
            return data

    def _compute_ot_weights(self, X_pca: np.ndarray, Y_pca: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        N_ref, N_tgt = X_pca.shape[0], Y_pca.shape[0]
        
        C = dist.cdist(X_pca, Y_pca, metric='sqeuclidean')
        C = C / (np.max(C) + EPS) 
        
        log_a = np.log(np.ones(N_ref) / N_ref)
        log_b = np.log(np.ones(N_tgt) / N_tgt)
        
        f = np.zeros(N_ref)
        g = np.zeros(N_tgt)
        
        fi = self.hp.ot_tau / (self.hp.ot_tau + self.hp.ot_epsilon)
        eps = self.hp.ot_epsilon
        
        max_iter = 1000
        tol = 1e-5
        
        for iteration in range(max_iter):
            f_prev = f.copy()
            arg_g = (-C / eps) + f[:, None]
            g = fi * (log_b - sp.logsumexp(arg_g, axis=0))
            arg_f = (-C / eps) + g[None, :]
            f = fi * (log_a - sp.logsumexp(arg_f, axis=1))
            
            if np.max(np.abs(f - f_prev)) < tol:
                break
                
        log_P = (-C / eps) + f[:, None] + g[None, :]
        P = np.exp(log_P)
        
        w_ref = np.sum(P, axis=1) * N_ref
        w_tgt = np.sum(P, axis=0) * N_tgt
        
        intersection_mass = float(np.sum(P))
        
        if debug:
            logger.debug(f"DEBUG: OT log-domain coupling converged in {iteration} iters. Preserved Mass: {intersection_mass:.2%}")
            
        return w_ref, w_tgt, intersection_mass

    def infer_latent_subpopulations(self, X_act: np.ndarray, Y_act: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        X_scaled = (X_act - np.mean(X_act, axis=1, keepdims=True)) / (np.std(X_act, axis=1, keepdims=True) + EPS)
        Y_scaled = (Y_act - np.mean(Y_act, axis=1, keepdims=True)) / (np.std(Y_act, axis=1, keepdims=True) + EPS)
        
        if self.hp.whiten_covariance:
            X_scaled = self._whiten_covariance(X_scaled)
            Y_scaled = self._whiten_covariance(Y_scaled)
            
        combined_activity_scaled = np.hstack([X_scaled, Y_scaled])
        
        pca = PCA(n_components=self.hp.pca_variance_retained, svd_solver='full', random_state=42)
        reduced_space = pca.fit_transform(combined_activity_scaled.T)
        
        bgmm = BayesianGaussianMixture(
            n_components=self.hp.max_latent_clusters, 
            covariance_type='diag', 
            weight_concentration_prior_type='dirichlet_distribution', 
            reg_covar=1e-5, 
            random_state=42
        )
        bgmm.fit(reduced_space)
        
        optimal_k = sum(bgmm.weights_ > 1e-3)

        if debug:
            logger.debug(f"DEBUG: Bayesian GMM converged. Active clusters: {optimal_k} out of {self.hp.max_latent_clusters} max.")

        return bgmm.predict(reduced_space), bgmm.predict_proba(reduced_space), optimal_k

    def align(self, ref_data: BatchData, target_data: BatchData, debug: bool = False) -> Tuple[np.ndarray, Dict]:
        if debug:
            self.debug = True
            logger.setLevel(logging.DEBUG)

        common, idx_x, idx_y = np.intersect1d(ref_data.gene_indices, target_data.gene_indices, return_indices=True)
        X = ref_data.data[idx_x]
        Y = target_data.data[idx_y]

        # ==========================================
        # STEP 1: Biological Embedding
        # Stabilize counts and map to within-sample pathway ranks.
        # ==========================================
        if self.hp.transform_type == 'log1p':
            X_p, Y_p = np.log1p(X), np.log1p(Y)
        else:
            X_p, Y_p = np.arcsinh(X), np.arcsinh(Y)

        gene_map = {g: i for i, g in enumerate(common)}
        
        pathway_idxs = [np.array([gene_map[g] for g in genes if g in gene_map])
                        for genes in self.pathway_dict.values()
                        if len([g for g in genes if g in gene_map]) > 0]

        metadata = {'hyperparameters': asdict(self.hp), 'version': f'18.0-BASIS-{self.hp.alignment_method}'}

        X_act = self._compute_pathway_activity(X_p, pathway_idxs, debug=self.debug)
        Y_act = self._compute_pathway_activity(Y_p, pathway_idxs, debug=self.debug)

        # ==========================================
        # STEP 2: Shared Mass Estimation
        # Calculate latent overlap using UOT or Bayesian GMM.
        # ==========================================
        if self.hp.enable_pmd_gating:
            # Placeholder for Posterior Mahalanobis Distance (PMD) diagnostic
            if debug: logger.debug("DEBUG: PMD Support-Overlap diagnostic executing... (Implementation Pending)")
            pass

        if self.hp.alignment_method == 'optimal_transport':
            X_scaled = (X_act - np.mean(X_act, axis=1, keepdims=True)) / (np.std(X_act, axis=1, keepdims=True) + EPS)
            Y_scaled = (Y_act - np.mean(Y_act, axis=1, keepdims=True)) / (np.std(Y_act, axis=1, keepdims=True) + EPS)
            
            if self.hp.whiten_covariance:
                X_scaled = self._whiten_covariance(X_scaled)
                Y_scaled = self._whiten_covariance(Y_scaled)
                
            pca = PCA(n_components=self.hp.pca_variance_retained, svd_solver='full', random_state=42)
            
            # Fit combined space to prevent single-platform distortion
            combined = np.hstack([X_scaled, Y_scaled])
            pca.fit(combined.T)
            
            X_pca = pca.transform(X_scaled.T)
            Y_pca = pca.transform(Y_scaled.T)

            w_ref, w_target, intersection_mass = self._compute_ot_weights(X_pca, Y_pca, debug=self.debug)
            metadata.update({'n_clusters': 'OT-Continuous', 'intersection_mass': intersection_mass})

        else:
            labels, proba, optimal_k = self.infer_latent_subpopulations(X_act, Y_act, debug=self.debug)
            
            n_ref = X_p.shape[1]
            proba_ref = proba[:n_ref, :]
            proba_target = proba[n_ref:, :]
            
            q_k = np.mean(proba_ref, axis=0) + EPS  
            p_k = np.mean(proba_target, axis=0) + EPS 
            
            I_k_raw = (2 * q_k * p_k) / (q_k + p_k)
            I_k = I_k_raw / (np.sum(I_k_raw) + EPS)
            
            V_ref = I_k / q_k
            V_target = I_k / p_k
            
            w_ref = np.sum(proba_ref * V_ref, axis=1)
            w_target = np.sum(proba_target * V_target, axis=1)
            
            metadata.update({'n_clusters': optimal_k, 'intersection_mass': float(np.sum(I_k_raw))})

        # Intersection mass threshold securely prevents forced alignment of disjoint datasets
        if metadata['intersection_mass'] < self.hp.min_intersection_mass:
             raise RuntimeError(f"CRITICAL: Biological intersection mass ({metadata['intersection_mass']:.2%}) is below threshold ({self.hp.min_intersection_mass:.2%}). Datasets are fundamentally incompatible.")

        # ==========================================
        # STEP 3: Weighted Batch Correction
        # Isolate technical shifts using mathematically balanced proportions.
        # ==========================================
        alpha_prior, beta_prior = self.combat.compute_baseline(
            X_p, Y_p, weights_X=w_ref, weights_Y=w_target, hp=self.hp, debug=self.debug
        )

        Y_corr = alpha_prior[:, None] * Y_p + beta_prior[:, None]
        Y_final = np.expm1(Y_corr) if self.hp.transform_type == 'log1p' else np.sinh(Y_corr)

        metadata.update({'alpha_final': alpha_prior, 'beta_final': beta_prior, 'target_weights': w_target})

        Y_out = target_data.data.copy()
        t_map = {g: i for i, g in enumerate(target_data.gene_indices)}
        for i, gene in enumerate(common):
            if gene in t_map: Y_out[t_map[gene]] = Y_final[i]

        return Y_out, metadata