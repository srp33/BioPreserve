# Deep Learning Batch Adjustment

This directory contains deep learning models for batch effect adjustment, using Hydra for configuration management and Snakemake for workflow orchestration.

## Architecture

- **Hydra**: Manages model hyperparameters and experiment configurations
- **Snakemake**: Orchestrates the workflow, handles dependencies, and parallelization
- **Python scripts**: The actual model implementations (in `../adjust/`)

## Directory Structure

```
adjust_deep_learning/
├── configs/
│   ├── config.yaml           # Main Hydra config
│   ├── model/                # Model-specific configs
│   │   ├── icvae.yaml
│   │   ├── vfae.yaml
│   │   ├── autoclass.yaml
│   │   └── wasserstein.yaml
│   ├── dataset/              # Dataset-specific configs
│   │   ├── gse49711.yaml
│   │   ├── gse24080.yaml
│   │   └── gse20194.yaml
│   └── training/             # Training configs
│       ├── default.yaml
│       ├── fast.yaml
│       └── thorough.yaml
├── run_model.py              # Hydra-enabled wrapper
├── Snakefile                 # Workflow orchestration
├── config.yaml               # Snakemake config
└── README.md                 # This file
```

## Usage

### 1. Run via Snakemake (Recommended for production)

```bash
# Run all models on all datasets
snakemake -j 4

# Run specific model/dataset combinations
snakemake /data/gold/gse49711/icvae.csv

# Dry run to see what would be executed
snakemake -n
```

### 2. Run via Hydra directly (For experimentation)

```bash
# Use default configs
python run_model.py

# Override model and dataset
python run_model.py model=vfae dataset=gse24080

# Use different training config
python run_model.py training=fast

# Override specific hyperparameters
python run_model.py model.latent_dim=20 training.epochs=500

# Run multiple experiments in parallel
python run_model.py --multirun model=icvae,vfae dataset=gse49711,gse24080
```

### 3. Hyperparameter Sweeps

```bash
# Run hyperparameter sweep via Snakemake
snakemake multirun/icvae_gse49711_sweep

# Or directly with Hydra
python run_model.py --multirun \
    model=icvae \
    dataset=gse49711 \
    model.latent_dim=5,10,20,50 \
    training.epochs=100,200,400
```

## Configuration

### Adding a New Model

1. Create config file: `configs/model/mymodel.yaml`
```yaml
name: mymodel
# Add model-specific hyperparameters
param1: value1
```

2. Update `run_model.py` to handle the new model's arguments

3. Add to `config.yaml`:
```yaml
models:
  - mymodel
```

### Adding a New Dataset

1. Create config file: `configs/dataset/mydataset.yaml`
```yaml
name: mydataset
batch_col: meta_batch
input_file: ${data_dir}/${dataset.name}/unadjusted.csv
output_file: ${data_dir}/${dataset.name}/${output_suffix}.csv
```

2. Add to `config.yaml`:
```yaml
datasets:
  - mydataset
```

## Benefits of This Approach

1. **Reproducibility**: All hyperparameters are version-controlled in YAML
2. **Flexibility**: Easy to override any parameter from command line
3. **Organization**: Configs are hierarchical and composable
4. **Automation**: Snakemake handles dependencies and parallelization
5. **Experimentation**: Hydra makes hyperparameter sweeps trivial

## Dependencies

```bash
pip install hydra-core omegaconf
```

## Examples

### Quick test with fast training
```bash
python run_model.py model=icvae dataset=gse49711 training=fast
```

### Production run with thorough training
```bash
snakemake -j 4  # Uses training=thorough by default
```

### Experiment with different latent dimensions
```bash
python run_model.py --multirun model.latent_dim=5,10,20,50
```

### Override data directory
```bash
python run_model.py data_dir=/path/to/data
```
