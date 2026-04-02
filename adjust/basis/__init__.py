"""
BASIS — Bulk Alignment of Shared Imbalanced Subpopulations.

BASIS is a biological alignment framework that discovers shared variance axes 
across disjoint datasets and performs weighted batch correction.

Python Usage:
    from basis import run_pipeline, load_combined
    from basis.config import BASISConfig, DatasetConfig

    # 1. Simple (2 datasets)
    results, gene_sets = run_pipeline(ref_path="ref.csv", tgt_path="target.csv",
                                      output_dir="out/", log_transform=True)

    # 2. Benchmarking Workflow (Combined CSV)
    # Automatically splits a single CSV into reference and target based on a study ID.
    ref_log, tgt_log, ref_meta, tgt_meta = load_combined("all_data.csv", test_source="GSE123")
    
    # 3. Multiple targets via Config
    cfg = BASISConfig(
        datasets=[
            DatasetConfig(path="ref.csv", log_transform=True, label="ref"),
            DatasetConfig(path="target1.csv", log_transform=True, label="rna_seq"),
            DatasetConfig(path="target2.csv", log_transform=False, label="microarray"),
        ], 
        output_dir="out/",
        keep_shared_only=True
    )
    results, gene_sets = run_pipeline(cfg=cfg)

Command Line Usage:
    # Standard 2-dataset alignment
    python -m basis --ref ref.csv --target target.csv --output-dir out/ --log-transform

    # Benchmarking workflow (Multi-study CSV)
    # Aligns 'GSE123' against all other studies in the file and saves a combined 'source' file.
    python -m basis --combined-path all.csv --test-source GSE123 --output-dir out/ \
                    --save-combined total_source.csv

    # Scaling experiments (Decoupled Dictionary)
    # Step A: Build and save the biological axes (Dictionary)
    python -m basis --combined-path all.csv --test-source GSE123 --output-dir out/ \
                    --save-dictionary axes.json --dictionary-only
    
    # Step B: Align using the pre-computed dictionary (much faster)
    python -m basis --combined-path subset.csv --test-source GSE123 --output-dir out/ \
                    --load-dictionary axes.json
"""

from basis.pipeline import preprocess, load_combined, build_dictionary, align, run_pipeline

__all__ = ["preprocess", "load_combined", "build_dictionary", "align", "run_pipeline"]
