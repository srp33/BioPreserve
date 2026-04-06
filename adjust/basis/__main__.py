"""CLI entry point: python -m basis --ref REF.csv --target TGT.csv --output-dir outputs/"""

import argparse
import logging
import os
import json

from basis.config import BASISConfig, DatasetConfig
from basis.pipeline import (
    run_pipeline,
    load_combined,
    execute_pipeline,
    preprocess
)

logger = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(
        description="BASIS: Bulk Alignment of Shared Imbalanced Subpopulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Input Modes ---
    group_input = parser.add_argument_group("Input Options")
    group_input.add_argument("--ref", help="Reference dataset CSV")
    group_input.add_argument("--target", action="append", help="Target dataset CSV (can be specified multiple times)")
    group_input.add_argument("--combined-path", help="Combined dataset CSV (contains ref + targets)")
    group_input.add_argument("--test-source", action="append", help="Study ID(s) to treat as targets in combined-path")
    group_input.add_argument("--ref-source", help="Study ID to treat as reference in combined-path (all others become targets)")
    
    # --- General Config ---
    group_gen = parser.add_argument_group("General Configuration")
    group_gen.add_argument("--output-dir", required=True, help="Output directory")
    group_gen.add_argument("--log-transform", action="store_true", help="Apply log transform to all datasets")
    group_gen.add_argument("--no-viz", action="store_true", help="Skip visualization plots")
    group_gen.add_argument("--meta-prefix", default="meta_", help="Prefix for metadata columns")
    group_gen.add_argument("--keep-shared-only", type=lambda x: (str(x).lower() == 'true'), default=True, 
                           help="Restrict output to shared genes (intersection)")

    # --- OT Alignment Parameters ---
    group_ot = parser.add_argument_group("Optimal Transport Parameters")
    group_ot.add_argument("--ot-epsilon", type=float, default=0.01, help="Entropy regularization")
    group_ot.add_argument("--ot-tau", type=float, default=0.1, help="Mass relaxation")

    # --- Dictionary Building Parameters ---
    group_dict = parser.add_argument_group("Dictionary Building")
    group_dict.add_argument("--save-dictionary", help="Path to save learned gene community dictionary (JSON)")
    group_dict.add_argument("--load-dictionary", help="Path to load pre-computed gene community dictionary (JSON)")
    group_dict.add_argument("--dictionary-only", action="store_true", help="Stop after building/saving dictionary")
    group_dict.add_argument("--dedup-threshold", type=float, default=0.999)
    group_dict.add_argument("--d-threshold", type=float, default=0.5)
    group_dict.add_argument("--w-floor", type=float, default=0.25)
    group_dict.add_argument("--top-k-edges", type=int, default=200)
    group_dict.add_argument("--corr-ceiling", type=float, default=0.99)
    group_dict.add_argument("--gp-n-calls", type=int, default=25)
    group_dict.add_argument("--res-range-min", type=float, default=1.0)
    group_dict.add_argument("--res-range-max", type=float, default=200.0)
    group_dict.add_argument("--n-runs", type=int, default=20)
    group_dict.add_argument("--consensus-threshold", type=float, default=0.7)
    group_dict.add_argument("--greedy-merge-threshold", type=float, default=0.7)
    group_dict.add_argument("--ghost-gene-floor", type=float, default=0.05)

    # --- Workflow Integration ---
    group_wf = parser.add_argument_group("Workflow Integration")
    group_wf.add_argument("--save-combined", help="Path to save single CSV with Reference + Adjusted Target(s)")
    group_wf.add_argument("--merge-order", help="JSON list or tree specifying merge order (e.g., '[\"S1\", \"S2\"]')")
    group_wf.add_argument("--auto-merge", action="store_true", help="Automatically determine optimal merge order based on mass")
    group_wf.add_argument("--progressive", action="store_true", help="Enable progressive reference expansion")
    group_wf.add_argument("--wls", action="store_true", help="Use OT-Barycentric Weighted Least Squares instead of ComBat")
    
    return parser.parse_args()


def main():
    args = setup_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Parse merge_order if provided as JSON
    merge_order = None
    if args.merge_order:
        try:
            merge_order = json.loads(args.merge_order)
        except Exception as e:
            # Fallback to comma-separated list if not valid JSON
            merge_order = [s.strip() for s in args.merge_order.split(",")]

    # 1. Map all CLI args to BASISConfig
    cfg = BASISConfig(
        output_dir=args.output_dir,
        viz=not args.no_viz,
        meta_prefix=args.meta_prefix,
        keep_shared_only=args.keep_shared_only,
        merge_order=merge_order,
        auto_merge=args.auto_merge,
        progressive=args.progressive,
        wls=args.wls,
        dedup_threshold=args.dedup_threshold,        d_threshold=args.d_threshold,
        w_floor=args.w_floor,
        top_k_edges=args.top_k_edges,
        corr_ceiling=args.corr_ceiling,
        gp_n_calls=args.gp_n_calls,
        res_range_min=args.res_range_min,
        res_range_max=args.res_range_max,
        n_runs=args.n_runs,
        consensus_threshold=args.consensus_threshold,
        greedy_merge_threshold=args.greedy_merge_threshold,
        ghost_gene_floor=args.ghost_gene_floor,
        ot_epsilon=args.ot_epsilon,
        ot_tau=args.ot_tau
    )

    # 2. Data Ingestion: Normalize into (ref_log, ref_meta) and list of (tgt_log, tgt_meta, label)
    targets = []
    
    if args.combined_path:
        # Benchmarking / Combined file mode
        ref_log, ref_meta, targets = load_combined(
            args.combined_path, 
            test_source=args.test_source, 
            ref_source=args.ref_source,
            log_transform=args.log_transform, 
            meta_prefix=cfg.meta_prefix
        )
    else:
        # Standard mode: Load separate files
        if not args.ref or not args.target:
            raise ValueError("Either --combined-path or BOTH --ref and --target must be provided.")
        
        ref_log, ref_meta, _ = preprocess(args.ref, log_transform=args.log_transform, meta_prefix=cfg.meta_prefix)
        for i, t_path in enumerate(args.target):
            t_log, t_meta, _ = preprocess(t_path, log_transform=args.log_transform, meta_prefix=cfg.meta_prefix)
            targets.append((t_log, t_meta, f"target_{i}"))

    # 3. Pre-load dictionary if provided
    gene_sets = None
    if args.load_dictionary:
        with open(args.load_dictionary, "r") as f:
            gene_sets = json.load(f)

    # 4. Dictionary-only Shortcut
    if args.dictionary_only:
        from basis.pipeline import build_dictionary
        gene_sets = build_dictionary([ref_log] + [t[0] for t in targets], config=cfg.dict_config())
        if args.save_dictionary:
            with open(args.save_dictionary, "w") as f:
                json.dump(gene_sets, f, indent=2)
        return

    # 5. Unified Execution
    execute_pipeline(ref_log, ref_meta, targets, cfg, 
                     gene_sets=gene_sets, 
                     save_combined_path=args.save_combined)


if __name__ == "__main__":
    main()
