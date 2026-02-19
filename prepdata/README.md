# Data Preparation

> **Navigation**: [← Main README](../../README.md) | [← Pipeline Documentation](../README.md) | [← Batch Correction Methods](../adjust/README.md) | [← Evaluation Framework](../evaluations/README.md)

Dataset acquisition, processing, and combination generation for batch correction analysis.

## Pipeline

**Main Script**: `all.sh` orchestrates download, conversion, combination, and synthetic data generation
**Key Scripts**: `download_datasets.py`, `convert_raw_files.py`, `combine_all.py`, `generate_structured_synthetic.py`

## Data Sources

**OSF** - 13 cancer studies (GSE19615, GSE20194, GSE20271, GSE23720, GSE25055, GSE25065, GSE31448, GSE45255, GSE58644, GSE62944_Tumor, GSE76275, GSE81538, METABRIC) + platform studies (GSE96058_HiSeq/NextSeq)
**Google Drive** - Recent studies (GSE115577, GSE123845, GSE163882)
**Refinebio** - Pre-normalized datasets

**Downloaders**: `downloaders/base.py`, `downloaders/osf_downloader.py`, `downloaders/gdrive_downloader.py`
**Configuration**: `config.py`

## Download Usage

```bash
# Download all or specific sources
python download_datasets.py --source all
python download_datasets.py --source osf --project-id eky3p
python download_datasets.py --source gdrive --folder-id 1smhpktMRyP4yyFHKHSisxRd9jwb8kvrq
python download_datasets.py --source osf --datasets GSE20194,GSE24080,METABRIC

# Resume, verify, organize
python download_datasets.py --resume --source all
python download_datasets.py --verify --source all
python download_datasets.py --status
```

## Processing

**Conversion**: `convert_raw_files.py` (main), `convert_to_h5.py` (HDF5), `create_smaller_csv.R` (R formats), `transpose_matrix.py`
**Organization**: `organize_downloaded_files.py` for automated file structure

```bash
# Processing workflows
python convert_raw_files.py --input data/raw_download/ --output data/gold/ --full-pipeline
python convert_raw_files.py --dataset gse20194 --quality-check --gene-annotation
python convert_raw_files.py --parallel --max-workers 4 --datasets gse20194,gse24080

# Organization
python organize_downloaded_files.py --source-dir data/raw_download/ --target-dir data/raw_data/
```

**Pipeline Stages**: `raw_download/` → `raw_data/` → `gold/` → `paired_datasets/` + `synthetic/`

## Dataset Combinations

**Files**: `combine_datasets.py`, `combine_all.py`, `generate_all_combinations.py`

```bash
# Generate pairwise combinations
python generate_all_combinations.py --input data/gold/ --output data/paired_datasets/
python combine_datasets.py --dataset1 gse20194 --dataset2 gse24080 --output data/paired_datasets/
python combine_all.py --min_samples 50 --max_combinations 100
```

## Configuration

**File**: `config.py` - Download sources, processing parameters, quality thresholds
**Caching**: `data/.cache/gdown/`, `data/.cache/R/` with hash-based validation

## Adding New Sources

1. Create `downloaders/custom_downloader.py` extending `BaseDownloader`
2. Update `config.py` with source configuration
3. Integrate in `download_datasets.py`

---

> **Navigation**: [← Main README](../../README.md) | [← Pipeline Documentation](../README.md) | [← Batch Correction Methods](../adjust/README.md) | [← Evaluation Framework](../evaluations/README.md)