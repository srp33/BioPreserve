# Important Feature Selection Documentation

## Workflow
After running the prepdata files and creating gold/, this pipeline performs the following steps:
1. Combines all data, keeping all metadata columns
2. Subsets the combined dataset into target datasets, metabric and gse62944
3. Aligns metadata for target datasetes (metabric and gse62944), creating combined columns for er_status, etc.
4. Adjusts the datasets using each of the provided batch effect adjustment methods
5. Trains a classifier model on each adjusted dataset and saves the model and performance
6. Calculates feature importance values, either t-test or permutation importance
7. Plots heatmaps
8. Ranks and selects top genes

This workflow is also visible in the directory's Snakefile.

## Input Data
**Required Inputs**
1. Combined gene expression file (combined_file in config) 
    Rows = samples, columns = genes
2. GEO metadata (geo_metadata in config)
    Sample annotations (sample size used for MNN ordering)
3. Optional: Dataset list (dataset_list in config)
    List of datasets to include in subsetting.

Additionally, align_metadata.py must be customized to combine the columns and map values to binary. 

## Configuration
All paths, targets, adjusters, and parameters are controlled via config.yaml. Example entries:

```yaml 
out_dir: results
data_dir: data
combined_file: data/combined_expression.csv
geo_metadata: data/geo_metadata.csv
adjusters: [gmm, mnn, combat]
targets: [tumor_type, survival_status]
ttest_targets: [tumor_type]
test_sources: [metabric, gse62944]
top_k: 50
top_n: 10
threshold: 0.05
n_repeats: 5
n_jobs: 4
random_state: 42
```

Key configurable options:

`adjusters`: Methods for batch correction or adjustment.
`targets` / ttest_targets: Metadata columns to perform statistical tests on.
`top_k` / top_n: Number of genes to select for visualization.
`threshold`: Significance threshold for filtering.
`n_repeats`, `n_jobs`, `random_state`: Workflow reproducibility and parallelization.

## Workflow Steps
1. Subset datasets (rule subset)
    Extracts relevant samples per test source.
2.  metadata (rule align_metadata)
    Ensures that metadata matches subsetted expression data.
3. Adjust datasets (rule adjust)
    Applies batch corrections or other adjustments via R scripts.
4. Perform T-tests (rule ttest)
    Computes gene-level p-values per target.
5. Rank genes (rule rank_genes)
    Aggregates test statistics and creates gene matrices.
6. Select top genes (rule select_top_genes)
    Combines ranks across targets and selects the top-N genes.
7. Subset expression for selected genes (rule subset_expression)
    Produces train/test splits for downstream analysis.
8. Plot heatmaps (rule plot_heatmap)
    Visualizes gene × adjuster matrices, optionally filtered by p-value.

**Outputs**

- `subset/`: Subsetted expression and metadata files.
- `adjusted/{adjuster}/`: Adjusted expression matrices per adjuster.
- `ttest/{adjuster}/`: Gene-level p-values for each target.
- `ranked_genes/`: Ranked gene matrices per target and K.
- `selected_genes/`: Top-N genes selected for visualization.
- `plots/`: Heatmaps of top genes across adjusters.

## Usage
Run the workflow using Snakemake:
```bash
snakemake --configfile config.yaml --cores 4
```

Or run using 
```bash
sbatch run_snakemake.sh
```
to use Slurm and pixi environments.
