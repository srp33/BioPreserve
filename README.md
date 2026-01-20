# Batch Effect Correction Analysis Pipeline

Analysis pipeline for batch effect correction in gene expression data. Implements the gmm_adjust method alongside deep learning and statistical approaches, with comprehensive evaluation frameworks and containerization support.

## Overview

This codebase provides tools for correcting batch effects in gene expression data, with particular focus on the gmm_adjust method and rigorous testing across multiple datasets and evaluation metrics.

## Components

- **Batch Correction**: GMM-based methods (gmm_adjust, gmm_adjust_nonlinear), deep learning approaches (AutoClass, ICVAE, VFAE, Wasserstein), and statistical methods (ComBat, MNN, Seurat, LIGER)
- **Evaluation**: Classification metrics, statistical measures, and biological signal preservation assessment
- **Containerization**: Docker and Apptainer environments for reproducible execution
- **Data Processing**: Automated pipeline with caching
- **Dataset Support**: 16+ datasets for comprehensive method validation

## Quick Start

### Docker (Local Development)
```bash
./run_docker.sh
```

### Apptainer (HPC/Cluster Environments)
```bash
# Interactive shell
./run_in_apptainer.sh shell

# Run specific scripts
./run_in_apptainer.sh scripts/all.sh

# Submit to SLURM scheduler
./run_in_apptainer.sh --sbatch scripts/evaluations/robustifying/code/3_real_data_pipe.R
```

## Pipeline Overview

The analysis pipeline consists of four main phases orchestrated by `scripts/all.sh`:

1. **Data Preparation** - Dataset acquisition, processing, and combination generation
2. **Batch Effect Adjustment** - Apply correction methods including gmm_adjust and comparison methods
3. **Evaluation** - Assessment using classification metrics, statistical measures, and biological validation
4. **Visualization** - Generate plots and reports for method comparison

Individual phases can be enabled/disabled for targeted execution. The pipeline includes caching to avoid redundant computations.

See [scripts/README.md](scripts/README.md) for pipeline documentation and configuration options.

## Supported Data

16+ datasets from OSF, Google Drive, and Refinebio including cancer studies (GSE19615, GSE20194, METABRIC) and platform comparisons (GSE96058_HiSeq/NextSeq).

Processing includes download, format standardization, gene annotation, quality control, and pairwise dataset combination generation.

See [scripts/prepdata/README.md](scripts/prepdata/README.md) for data preparation details and [data/README.md](data/README.md) for data structure information.

## Container Environments

### Docker (Local Development)
```bash
# Run container
./run_docker.sh

# Interactive development
docker run -it --rm -v $(pwd):/workspace batch-effects-pipeline bash
```

Single-stage build with Python (Miniforge3), R (Bioconductor 3.21), and dependencies.

### Apptainer (HPC/Cluster Environments)
```bash
# Interactive shell
./run_in_apptainer.sh shell

# Run pipeline
./run_in_apptainer.sh scripts/all.sh

# Submit to SLURM scheduler
./run_in_apptainer.sh --sbatch scripts/evaluations/robustifying/code/3_real_data_pipe.R
```

Three-stage build system for HPC environments with SLURM integration and group permissions.

See [apptainer/README.md](apptainer/README.md) for Apptainer documentation including:
- Three-stage build system
- SLURM integration and job submission
- Group permissions setup (`grp_batch_effects`)
- Performance tuning and troubleshooting

## Documentation Navigation

## Documentation Structure

### Core Documentation
- [scripts/README.md](scripts/README.md) - Pipeline documentation, execution control, and configuration
- [data/README.md](data/README.md) - Data structure, organization, and management

### Method Documentation  
- [scripts/adjust/README.md](scripts/adjust/README.md) - Batch correction methods (deep learning, statistical, GMM)
- [scripts/evaluations/README.md](scripts/evaluations/README.md) - Evaluation framework and metrics
- [scripts/prepdata/README.md](scripts/prepdata/README.md) - Data preparation and processing pipeline

### Container Documentation
- [apptainer/README.md](apptainer/README.md) - Apptainer setup, SLURM integration, and HPC deployment

### Navigation
- Getting Started: Start here → [scripts/README.md](scripts/README.md) for pipeline details
- Method Selection: [scripts/adjust/README.md](scripts/adjust/README.md) for batch correction methods
- Results Analysis: [scripts/evaluations/README.md](scripts/evaluations/README.md) for evaluation metrics
- Data Management: [scripts/prepdata/README.md](scripts/prepdata/README.md) + [data/README.md](data/README.md)
- HPC Deployment: [apptainer/README.md](apptainer/README.md) for cluster setup

## Available Methods

### Batch Correction Methods
- **Gaussian Mixture Models**: gmm_adjust (primary method), gmm_adjust_nonlinear, gmm_global_simple
- **Deep Learning**: AutoClass, ICVAE, VFAE, Wasserstein adversarial approaches
- **Statistical**: ComBat, MNN, Seurat integration, LIGER, limma, quantile normalization

### Evaluation Metrics
- **Classification**: Batch vs. biological signal separation, ER status prediction
- **Statistical**: MMD, MSE, mutual information, feature importance analysis
- **Quality Control**: BatchQC, pathway analysis, dimensionality reduction

See [scripts/adjust/README.md](scripts/adjust/README.md) for method descriptions and [scripts/evaluations/README.md](scripts/evaluations/README.md) for evaluation framework details.

### Diagnostic Commands
```bash
# Check pipeline logs
tail -f outputs/prepdata.log

# Interactive container access
./run_in_apptainer.sh shell
```

See [apptainer/README.md](apptainer/README.md) for troubleshooting and optimization strategies.

## Development and Contributing

### Development Workflow

1. **Method Development**: Add new batch correction methods in `scripts/adjust/`
2. **Evaluation Extensions**: Extend metrics and validation in `scripts/evaluations/`
3. **Data Integration**: Add new data sources in `scripts/prepdata/`
4. **Container Updates**: Modify container definitions in `apptainer/`

### Contributing Guidelines
See method-specific documentation in:
- [scripts/adjust/README.md](scripts/adjust/README.md) - Adding new batch correction methods
- [scripts/evaluations/README.md](scripts/evaluations/README.md) - Extending evaluation frameworks
- [scripts/prepdata/README.md](scripts/prepdata/README.md) - Integrating new data sources
- [apptainer/README.md](apptainer/README.md) - Container development

### Support and Contact
- Issues: [GitHub Issues](https://github.com/srp33/confounded_analysis/issues) for bug reports and feature requests
- Research Contact: [Piccolo Lab](https://biology.byu.edu/piccolo-lab/contact) for research collaboration

