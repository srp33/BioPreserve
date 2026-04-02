"""CLI entry point: python -m basis --ref REF.csv --target TGT.csv --output-dir outputs/"""

import argparse
import logging
import os
import json

from basis.config import BASISConfig, DatasetConfig
from basis.pipeline import (
    run_pipeline,
    load_combined,
    execute_pipeline
)

logger = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description="BASIS: cross-platform gene expression alignment")
    
    # Input modes
    parser.add_argument("--ref", help="Reference dataset CSV")
    parser.add_argument("--target", help="Target dataset CSV")
    parser.add_argument("--combined-path", help="Combined dataset CSV (contains ref + target)")
    parser.add_argument("--test-source", help="Study ID to treat as target in combined-path")
    
    # Output and parameters
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--log-transform", action="store_true", help="Apply log transform to all datasets")
    parser.add_argument("--log-transform-ref", action="store_true", help="Apply log transform to reference only")
    parser.add_argument("--log-transform-target", action="store_true", help="Apply log transform to target only")
    parser.add_argument("--ot-epsilon", type=float, default=0.01)
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization plots")
    
    # Dictionary management
    parser.add_argument("--save-dictionary", help="Path to save learned gene community dictionary (JSON)")
    parser.add_argument("--load-dictionary", help="Path to load pre-computed gene community dictionary (JSON)")
    parser.add_argument("--dictionary-only", action="store_true", help="Stop after building/saving dictionary")
    
    # Workflow integration
    parser.add_argument("--save-combined", help="Path to save single CSV with Reference + Adjusted Target")
    
    return parser.parse_args()


def run_combined_workflow(args, gene_sets):
    """Handle alignment for a single multi-study combined CSV."""
    if not args.test_source:
        raise ValueError("--test-source is required when using --combined-path")

    # 1. Load and split
    ref_log, tgt_log, ref_meta, tgt_meta = load_combined(
        args.combined_path, args.test_source, log_transform=args.log_transform
    )

    # 2. Config setup
    cfg = BASISConfig(
        output_dir=args.output_dir,
        ot_epsilon=args.ot_epsilon,
        ot_tau=args.ot_tau,
        viz=not args.no_viz,
        keep_shared_only=True
    )

    if args.dictionary_only:
        from basis.pipeline import build_dictionary
        gene_sets = build_dictionary([ref_log, tgt_log])
        if args.save_dictionary:
            with open(args.save_dictionary, "w") as f:
                json.dump(gene_sets, f, indent=2)
        return

    # 3. Execute
    execute_pipeline(ref_log, ref_meta, [(tgt_log, tgt_meta, args.test_source)], 
                     cfg, gene_sets=gene_sets, save_combined_path=args.save_combined)


def run_standard_workflow(args, gene_sets):
    """Handle alignment for individual reference and target files."""
    run_pipeline(ref_path=args.ref, 
                 tgt_path=args.target, 
                 output_dir=args.output_dir,
                 ot_epsilon=args.ot_epsilon,
                 ot_tau=args.ot_tau,
                 log_transform=args.log_transform,
                 viz=not args.no_viz,
                 gene_sets=gene_sets,
                 save_combined_path=args.save_combined)


def main():
    args = setup_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Pre-load dictionary if provided
    gene_sets = None
    if args.load_dictionary:
        with open(args.load_dictionary, "r") as f:
            gene_sets = json.load(f)

    # Dispatch to appropriate workflow
    if args.combined_path:
        run_combined_workflow(args, gene_sets)
    else:
        run_standard_workflow(args, gene_sets)


if __name__ == "__main__":
    main()
