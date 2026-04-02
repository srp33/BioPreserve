"""
BASIS — Bulk Alignment of Shared Imbalanced Subpopulations

Usage:
    from basis import preprocess, build_dictionary, align, run_pipeline
    from basis.config import BASISConfig, DatasetConfig

    # Simple (2 datasets)
    results, gene_sets = run_pipeline("ref.csv", "target.csv",
                                      output_dir="out/", log_transform=True)

    # Multiple targets
    cfg = BASISConfig(datasets=[
        DatasetConfig(path="ref.csv", log_transform=True, label="ref"),
        DatasetConfig(path="target1.csv", log_transform=True, label="rna_seq"),
        DatasetConfig(path="target2.csv", log_transform=False, label="microarray2"),
    ], output_dir="out/")
    results, gene_sets = run_pipeline(cfg=cfg)
"""

from basis.pipeline import preprocess, load_combined, build_dictionary, align, run_pipeline

__all__ = ["preprocess", "load_combined", "build_dictionary", "align", "run_pipeline"]
