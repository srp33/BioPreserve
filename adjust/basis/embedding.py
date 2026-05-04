"""
BasisAtlas Orchestrator
Formalizes the Continuous-Ideal Translation Architecture into modular layers.
Supports Latent (AQN), Physical (APS), and Batch-Physical (B-APS) architectures.
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import median_abs_deviation, norm

# Import swappable modules
from basis.modules.input import global_ranking
from basis.modules.shape import (
    anchor_quantile_normalization, 
    individual_anchor_physical_spline,
    batch_anchor_physical_spline
)
from basis.modules.projection import int_pca_projection
from basis.modules.linear import bypass_correction
from basis.modules.diagnostic import soft_gmm_estimation
from basis.modules.reconstruction import gmm_semantic_blending

def _gini(weights):
    if len(weights) == 0: return 0
    w = np.sort(np.abs(weights))
    n = len(w)
    if n < 2 or w.sum() == 0: return 0
    cum = np.cumsum(w)
    return 1.0 - 2.0 * np.sum(cum) / (n * cum[-1]) + 1.0 / n

class BasisAtlas:
    """
    Orchestrator for the Continuous-Ideal Translation Architecture.
    Router Pattern: Supports 'latent_aqn', 'physical_aps', and 'batch_aps'.
    """
    def __init__(self, gene_sets, anchors):
        self.gene_sets = gene_sets
        self.anchors = anchors
        
        # Frozen parameters (The "Atlas")
        self.axis_params = {}   
        self.gene_mins = None
        self.global_gmm = None
        self.state_mappings = {} 
        self.ref_anchor_rank_profile = None     
        self.ref_anchor_physical_profile = None 

    def fit(self, expr_df):
        print("Fitting BasisAtlas (Modular Architecture) on reference cohort...")
        common_genes = expr_df.columns.tolist()
        self.gene_mins = expr_df.min(axis=0)
        
        # 1. Pure Global Ranking
        raw_ranks = global_ranking(expr_df, gene_mins=self.gene_mins, add_jitter=False)
        
        # 2. Learn Anchor Backbones
        valid_anchors = [g for g in self.anchors if g in raw_ranks.columns]
        self.ref_anchor_rank_profile = raw_ranks[valid_anchors].median(axis=0).sort_values()
        self.ref_anchor_physical_profile = expr_df[valid_anchors].median(axis=0).sort_values()

        # 3. Base Embedding Space (INT + PCA)
        std_df = pd.DataFrame(norm.ppf(np.clip(raw_ranks.values, 1e-6, 1 - 1e-6)),
                              index=raw_ranks.index, columns=raw_ranks.columns)

        embeddings = {}
        for axis_name, gene_weights in self.gene_sets.items():
            genes = [g for g in gene_weights if g in common_genes]
            if len(genes) < 3: continue
            X = std_df[genes].values
            hw = np.array([gene_weights[g] for g in genes])
            pca = PCA(n_components=1, random_state=42)
            scores = pca.fit_transform(X * hw).ravel()
            if np.corrcoef(scores, (X * hw).mean(axis=1))[0, 1] < 0:
                pca.components_[0] = -pca.components_[0]
                scores = -scores
            
            p_noise_mad_ref = median_abs_deviation(std_df[valid_anchors].values, axis=1)
            hub_mads_ref = median_abs_deviation(std_df[genes].values, axis=1)
            self.axis_params[axis_name] = {
                "pca": pca, "gini_sq": _gini(hw)**2, "ref_median": np.median(scores),
                "ref_mad": median_abs_deviation(scores), "ref_gain": np.median(hub_mads_ref / np.maximum(p_noise_mad_ref, 1e-12)),
                "genes": genes
            }
            embeddings[axis_name] = scores

        self.global_gmm = GaussianMixture(n_components=8, random_state=42)
        self.global_gmm.fit(pd.DataFrame(embeddings).values)
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/basis_atlas_summary.txt", "w") as f:
            f.write(f"Atlas Fitted with {self.global_gmm.n_components} states and {len(valid_anchors)} anchors.\n")

    def transform(self, expr_df, mode="batch_aps"):
        """Inference Router: Supports 'latent_aqn', 'physical_aps', and 'batch_aps'."""
        if mode == "latent_aqn":
            raw_ranks = global_ranking(expr_df, gene_mins=self.gene_mins)
            aligned_ranks = anchor_quantile_normalization(raw_ranks, self.ref_anchor_rank_profile)
            scores = {}
            for ax, p in self.axis_params.items():
                raw_score = int_pca_projection(aligned_ranks, ax, p, self.gene_sets)
                scores[ax] = bypass_correction(raw_score) * p["gini_sq"]
            return pd.DataFrame(scores, index=expr_df.index)

        elif mode == "physical_aps":
            # N=1 Physical Spline
            corrected_genes = individual_anchor_physical_spline(expr_df, self.ref_anchor_physical_profile, self.anchors)
            log_corrected = np.log(corrected_genes + 1.0)
            ranks = log_corrected.rank(axis=1, pct=True)
            std_df = pd.DataFrame(norm.ppf(np.clip(ranks.values, 1e-6, 1 - 1e-6)), index=ranks.index, columns=ranks.columns)
            scores = {}
            for ax, p in self.axis_params.items():
                scores[ax] = p["pca"].transform(std_df[p["genes"]].values * np.array([self.gene_sets[ax][g] for g in p["genes"]])).ravel() * p["gini_sq"]
            return pd.DataFrame(scores, index=expr_df.index)

        elif mode == "batch_aps":
            # Multi-Sample Physical Spline
            corrected_genes = batch_anchor_physical_spline(expr_df, self.ref_anchor_physical_profile, self.anchors)
            log_corrected = np.log(corrected_genes + 1.0)
            ranks = log_corrected.rank(axis=1, pct=True)
            std_df = pd.DataFrame(norm.ppf(np.clip(ranks.values, 1e-6, 1 - 1e-6)), index=ranks.index, columns=ranks.columns)
            scores = {}
            for ax, p in self.axis_params.items():
                scores[ax] = p["pca"].transform(std_df[p["genes"]].values * np.array([self.gene_sets[ax][g] for g in p["genes"]])).ravel() * p["gini_sq"]
            return pd.DataFrame(scores, index=expr_df.index)

    def correct_genes(self, expr_df, mode="batch_aps"):
        if mode == "physical_aps":
            return individual_anchor_physical_spline(expr_df, self.ref_anchor_physical_profile, self.anchors)
        elif mode == "batch_aps":
            return batch_anchor_physical_spline(expr_df, self.ref_anchor_physical_profile, self.anchors)
            
        embedded = self.transform(expr_df, mode="latent_aqn")
        posteriors = soft_gmm_estimation(embedded, self.global_gmm)
        return gmm_semantic_blending(expr_df, self.gene_mins, posteriors, self.state_mappings)

    def learn_translation(self, tgt_expr_df, ref_expr_df):
        X_ref = self.transform(ref_expr_df, mode="latent_aqn") 
        X_tgt = self.transform(tgt_expr_df, mode="latent_aqn")
        probs_ref = soft_gmm_estimation(X_ref, self.global_gmm)
        probs_tgt = soft_gmm_estimation(X_tgt, self.global_gmm)
        log_ref = np.log(np.maximum(ref_expr_df - self.gene_mins, 0) + 1.0)
        log_tgt = np.log(np.maximum(tgt_expr_df - self.gene_mins, 0) + 1.0)
        for k in range(self.global_gmm.n_components):
            mask_r, mask_t = probs_ref[:, k] > 0.5, probs_tgt[:, k] > 0.5
            if mask_r.sum() > 5 and mask_t.sum() > 5:
                mu_r, std_r = log_ref[mask_r].mean(axis=0), log_ref[mask_r].std(axis=0)
                mu_t, std_t = log_tgt[mask_t].mean(axis=0), log_tgt[mask_t].std(axis=0)
                scale = np.clip(std_r / np.maximum(std_t, 1e-6), 0.5, 1.5)
                self.state_mappings[k] = {"scale": scale, "shift": mu_r - (mu_t * scale)}
            else:
                self.state_mappings[k] = {"scale": pd.Series(1.0, index=log_ref.columns), "shift": pd.Series(0.0, index=log_ref.columns)}
