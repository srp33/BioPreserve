# Obsolete Scripts Capabilities Record

This document records the capabilities of the iterative debugging, ablation, and old workflow scripts before their consolidation and removal. These scripts played crucial roles in diagnosing cross-platform batch effects and mathematically defining the final $N=1$ pipeline, but their core logic has now been successfully integrated into the main `basis` package.

### 1. Old Embedding Methods
*   **`14_simple_mean_embedding.py`**: Computed Simple Mean embedding and visualized alignment quality. Demonstrated the baseline without sign-awareness. (Capability absorbed into `BasisAtlas`).
*   **`15_sign_aware_embedding.py`**: Computed Sign-Aware Simple Mean embedding. Identified negatively correlated genes within a community and flipped their sign before averaging. (Capability absorbed by the Push-Pull logic in `BasisAtlas`).
*   **`15_anchor_embedding.py`**: First implementation of Row-wise (Within-Sample) Standardization using an Invariant Anchor Set. Stripped technical batch effects while preserving biological asymmetry. (Capability absorbed by `anchor_only_rank` and `BasisAtlas.transform`).

### 2. Old Community Detection
*   **`04_consensus_leiden_hits.py`**: Previous iteration of community detection using a hard consensus threshold across multiple Leiden runs. (Fully replaced by the superior Directed Network Diffusion logic in `04_diffusion_communities.py`).

### 3. Iterative Debugging Scripts (The Diagnostic Trail)
*   **`16_test_asymmetry.py`**: Compared Anchor-Aware Standardization vs. Column-wise Standardization under biased cohort conditions (simulating 100% ER- target). Proved that Column-wise standardization paradoxically maps ER- and ER+ to the same technical baseline.
*   **`19_debug_side_by_side.py`**: The "Side-by-Side $N=1$ Trace". Deconstructed the `process_n1` function to print raw, log, and standardized values for a single ER+ patient from both platforms. Discovered the "Double-Logging" and "MAD Explosion" issues.
*   **`21_debug_last_mile.py`**: The "Microscope Trace". Implemented Gene-to-Rank traces, Sparsity vs. Noise-floor Density Checks, and Anchor-Hub Relativity Analysis. Discovered the Non-Linear "Digital Gain" Slope Discrepancy.
*   **`22_test_anchor_z_units.py`**: Calculated the Hub-to-Anchor Variance Ratio across platforms. Proved mathematically that RNA-seq hubs exhibit higher variance gain ($1.8\times$) than Microarray hubs ($1.3\times$). Evaluated Anchor-Z-Space.
*   **`24_residual_projection_plot.py`**: Visualized the individual contribution of every gene to the final Axis Score. Discovered the "Ghost Load" (+21.0 offset) where technical noise accumulated into a false positive signal. Led to the invention of Sentinel Zeroing.

### 4. Ablation Studies (Algorithm Verification)
*   **`26_ablation_study.py`**: The Comprehensive Pairwise Ablation Gauntlet. Evaluated 5 layers of normalization (Global Rank, Anchor-Only, Lamé Warp, Sentinel Tare, BGN, GMM) across 6 dataset pairs. Proved that Anchor-Only ranking with Continuous PC1 scores is the most robust architecture.
*   **`27_test_ot_weighting.py`**: Tested removing the Gini-squared component weighting before Optimal Transport. Proved that structural Gini weighting improves Intersection Mass and maintains biological AUC.
*   **`28_latent_interpolation_ablation.py`**: Tested Embedding-to-Gene batch correction strategies (Cohort OT, Global BGN, Hard GMM, Soft GMM). Proved that Soft GMM Latent Interpolation provides the best combination of high biological accuracy ($AUC > 0.92$) and perfect centroid superimposition.

### Final State
All these capabilities have been distilled and integrated into the robust, $N=1$ clinical-ready `basis` package (specifically `basis/embedding.py` and `basis/pipeline.py`), making these standalone exploratory scripts obsolete.
