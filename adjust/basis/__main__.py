"""CLI entry point: python -m basis --ref REF.csv --target TGT.csv --output-dir outputs/"""

import argparse
import logging

from basis import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="BASIS: cross-platform gene expression alignment")
    parser.add_argument("--ref", required=True, help="Reference dataset CSV")
    parser.add_argument("--target", required=True, help="Target dataset CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--log-transform", action="store_true", help="Apply log(x - min + 1) to both datasets")
    parser.add_argument("--log-transform-ref", action="store_true", help="Apply log transform to reference only")
    parser.add_argument("--log-transform-target", action="store_true", help="Apply log transform to target only")
    parser.add_argument("--ot-epsilon", type=float, default=0.01)
    parser.add_argument("--ot-tau", type=float, default=0.1)
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from basis.config import BASISConfig, DatasetConfig

    log_ref = args.log_transform or args.log_transform_ref
    log_tgt = args.log_transform or args.log_transform_target

    cfg = BASISConfig(
        ref=DatasetConfig(path=args.ref, log_transform=log_ref),
        target=DatasetConfig(path=args.target, log_transform=log_tgt),
        output_dir=args.output_dir,
        ot_epsilon=args.ot_epsilon,
        ot_tau=args.ot_tau,
        viz=not args.no_viz,
    )

    run_pipeline(cfg=cfg)


if __name__ == "__main__":
    main()
