# Batch Correction Methods

> **Navigation**: [← Main README](../../README.md) | [← Pipeline Documentation](../README.md) | [Evaluation Framework →](../evaluations/README.md) | [Data Preparation →](../prepdata/README.md)

Batch effect correction methods including GMM variants, deep learning approaches, and statistical methods.

## Method Categories

### Gaussian Mixture Model Suite (Primary Methods)

#### Standard GMM (gmm_adjust)
- **Files**: `gmm_adjust.R`, `gmm_adjust_python.py`
- **Description**: 1D 2-component GMM with Bayesian priors (Dirichlet for weights, Inverse-Gamma for variances)
- **Parameters**: `max_iter` (100), `tol` (1e-4), `weight_alpha`, `variance_alpha`

#### Nonlinear GMM
- **File**: `gmm_adjust_nonlinear.R`
- **Description**: Extended GMM for complex, non-linear batch structures

#### Global GMM
- **File**: `gmm_global_simple.R`
- **Description**: Simplified GMM optimized for large-scale corrections

### Deep Learning Approaches

#### AutoClass Framework
- **Files**: `autoclass.py`, `autoclass.sh`, `invert_autoclass.py`
- **Description**: Encoder-decoder with adversarial discriminator
- **Parameters**: `encoder_layer_size` ([128]), `adversarial_weight` (0.002), `epochs` (400), `lr` (15), `reg` (0.0001), `dropout_rate` (0.2)

#### Variational Autoencoders

**ICVAE (Information-Constrained VAE)**
- **Files**: `icvae.py`, `icvae.sh`, `run_icvae.py`
- **Description**: VAE with auxiliary classifier and mutual information penalty
- **Parameters**: `--latent-dim` (10), `--hidden-dim` (128), `--hidden-dim-aux` (64), `--epochs` (100), `--w-mi-penalty` (1.0)

**VFAE (Variational Fair Autoencoder)**
- **Files**: `vfae.py`, `vfae.sh`, `run_vfae.py`
- **Description**: VAE with MMD fairness constraints
- **Parameters**: `--latent-dim` (10), `--hidden-dim` (128), `--epochs` (100), `--w-mmd` (10.0), `--mmd-gamma` (1.0)

**Wasserstein Adversarial Training**
- **Files**: `wasserstein.py`, `wasserstein.sh`
- **Description**: Generator-critic architecture with Wasserstein loss and gradient penalty

### Statistical and Traditional Methods

#### R-Based Methods Integration
- **Files**: `adjust.R`, `adjustR_data.sh`, `adjustR_individual_prep.sh`, `adjustR_paired_datasets.sh`
- **Execution Modes**: Parallel (`ranked1`, `ranked2`, `ranked_batch`), Sequential (`mnn`, `liger`), Target (`combat`, `fairadapt`, `limma`)

#### Implemented Statistical Methods

**ComBat** - Empirical Bayes (`sva` package)
**Quantile Normalization** - Distribution-based (`preprocessCore`)
**MNN** - Mutual nearest neighbors (`batchelor`)
**Seurat Integration** - CCA and anchor-based (`Seurat`)
**LIGER** - Integrative NMF (`rliger`)
**Limma** - Linear modeling (`limma`)
**FairAdapt** - Causal fairness-aware (`fairadapt`)
**Ranked Methods** - Custom ranking-based normalization (ranked1, ranked2, ranked_batch)



## Method Selection and Usage

### Resource Requirements and Execution Strategy

**Memory-Intensive Methods** (Sequential Execution):
- **AutoClass**: GPU-accelerated, high memory usage, 400 epochs default
- **ICVAE**: Moderate memory, auxiliary classifier overhead
- **VFAE**: Moderate memory, MMD computation overhead  
- **Wasserstein**: High memory for critic network, gradient penalty computation

**Lightweight Methods** (Parallel Execution):
- **GMM variants (Recommended)**: Low memory, efficient EM algorithm, rigorously tested
- **Ranked methods**: Minimal memory, distribution-free
- **Statistical methods**: Variable memory based on method complexity

**Execution Control**:
- Parallel methods defined in `ADJUSTERS_PARALLEL` array
- Sequential methods defined in `ADJUSTERS_SEQUENTIAL` array
- Target methods defined in `ADJUSTERS_TARGET` array

### Detailed Usage Examples

#### Deep Learning Methods

```bash
# AutoClass with custom parameters
python /scripts/adjust/autoclass.py \
  -i /data/gold/gse49711/unadjusted.csv \
  -o /data/gold/gse49711/autoclass.csv \
  -b meta_Sex

# ICVAE with custom configuration
python /scripts/adjust/run_icvae.py \
  -i /data/gold/gse24080/unadjusted.csv \
  -o /data/gold/gse24080/icvae.csv \
  -b meta_batch \
  --latent-dim 20 \
  --epochs 400 \
  --w-mi-penalty 1.5

# VFAE with MMD penalty tuning
python /scripts/adjust/run_vfae.py \
  -i /data/gold/gse49711/unadjusted.csv \
  -o /data/gold/gse49711/vfae.csv \
  -b meta_Sex \
  --w-mmd 10.0 \
  --mmd-gamma 1.0

# Wasserstein adversarial training
python /scripts/adjust/wasserstein.py \
  --input /data/gold/dataset/unadjusted.csv \
  --output /data/gold/dataset/wasserstein.csv \
  --batch-col meta_batch
```

#### Statistical Methods

```bash
# Run all parallel R methods
bash /scripts/adjust/adjustR_data.sh

# Run specific R method with dataset
Rscript /scripts/adjust/adjust.R \
  --adjuster combat \
  --dataset gse49711 \
  --input-file /data/gold/gse49711/unadjusted.csv \
  --output-file /data/gold/gse49711/combat.csv

# Run GMM adjustment
Rscript /scripts/adjust/gmm_adjust.R \
  --input /data/gold/dataset/unadjusted.csv \
  --output /data/gold/dataset/gmm.csv \
  --max_iter 100 \
  --tol 1e-4
```

### Method Configuration Details

#### AutoClass Configuration
```python
# Core AutoClass parameters
BatchCorrectImpute(
    genes,                    # Gene expression matrix
    batches,                  # Batch labels
    cellwise_norm=False,      # Cell-wise normalization
    log1p=False,             # Log1p transformation
    verbose=True,            # Verbose output
    encoder_layer_size=[128], # Encoder architecture
    adversarial_weight=0.002, # Adversarial loss weight
    epochs=400,              # Training epochs
    lr=15,                   # Learning rate
    reg=0.0001,              # Regularization strength
    dropout_rate=0.2         # Dropout probability
)
```

#### ICVAE Configuration
```python
# ICVAE hyperparameters
--latent-dim 10              # Latent space dimensionality
--hidden-dim 128             # VAE hidden layer size
--hidden-dim-aux 64          # Auxiliary classifier hidden size
--epochs 100                 # Training epochs
--learning-rate 1e-3         # Optimizer learning rate
--batch-size 64              # Training batch size
--w-kl 1.0                   # KL divergence weight
--w-mi-penalty 1.0           # Mutual information penalty weight
```

#### VFAE Configuration
```python
# VFAE fairness parameters
--latent-dim 10              # Latent space dimensionality
--hidden-dim 128             # Hidden layer size
--epochs 100                 # Training epochs
--learning-rate 1e-3         # Optimizer learning rate
--batch-size 64              # Training batch size
--w-kl 1.0                   # KL divergence weight
--w-mmd 10.0                 # MMD penalty weight
--mmd-gamma 1.0              # RBF kernel gamma parameter
```

#### GMM Configuration
```r
# GMM Bayesian parameters
GaussianMixture1D(
  max_iter = 100,            # Maximum EM iterations
  tol = 1e-4,                # Convergence tolerance
  weight_alpha = NULL,       # Dirichlet prior for weights
  variance_alpha = NULL      # Inverse-Gamma prior for variances
)
```

#### R Statistical Methods Configuration
```r
# Environment configuration
Sys.setenv(OMP_NUM_THREADS = 1)  # Single-threaded OpenMP
ulimit -s unlimited               # Unlimited stack size

# Method-specific parameters
ADJUSTERS_PARALLEL <- c("ranked1", "ranked2", "ranked_batch")
ADJUSTERS_SEQUENTIAL <- c("mnn", "liger")  
ADJUSTERS_TARGET <- c("combat", "fairadapt", "limma")
```


## Performance Optimization

- **Execution**: Parallel for lightweight methods, sequential for memory-intensive
- **Caching**: Hash-based validation in `data/.cache/gmm_cache/`, `data/.cache/R/`
- **Monitoring**: Execution time and resource tracking

## Method Validation

### Validation Framework
- **Synthetic Data**: Controlled batch effects with known ground truth
- **Real Data**: Cross-study validation (GSE49711, GSE20194, GSE24080)
- **Performance**: Runtime, memory usage, scalability

### Quality Metrics
```bash
# Batch removal assessment
python /scripts/evaluations/classify_batch_bio_within_dataset/classify.py
python /scripts/evaluations/small_evals/mutual_info.py
python /scripts/evaluations/small_evals/mmd.py

# Biological signal preservation
python /scripts/evaluations/classify_er_mixed_datasets/classify.sh
python /scripts/evaluations/classifier_feature_importance/feature_importance_analysis.py
python /scripts/evaluations/small_evals/mse.py
```


## Adding New Methods

1. Implement in Python (`*.py`) or R (`*.R`)
2. Create shell wrapper (`*.sh`) for resource management
3. Update `all.sh` for pipeline integration
4. Add evaluation in `../evaluations/`

---

> **Navigation**: [← Main README](../../README.md) | [← Pipeline Documentation](../README.md) | [Evaluation Framework →](../evaluations/README.md) | [Data Preparation →](../prepdata/README.md)