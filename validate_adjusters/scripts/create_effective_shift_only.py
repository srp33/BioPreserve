#!/usr/bin/env python3
"""
Create "effective shift-only" adjusted data.

IMPORTANT DISTINCTION:
- "Raw shift-only": Apply only the raw shift parameter from Bayesian fitting
  Formula: y_adj = y - shift_param
  Problem: Shift was learned in context of scale, doesn't work alone
  
- "Effective shift-only": Apply a shift that achieves the same mean change
  as the full Bayesian shift+scale adjustment
  Formula: y_adj = y - effective_shift
  where effective_shift = mean(y_unadjusted) - mean(y_bayesian_adjusted)
  
This script implements "effective shift-only" which:
1. Preserves the mean alignment from Bayesian adjustment
2. Removes the variance scaling (keeps original variance)
3. Works well because it captures the location correction without distortion

Why this works better:
- Bayesian shift+scale: Changes both location AND scale
- Effective shift-only: Changes only location (preserves scale)
- For linear classifiers, location matters more than scale
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path


def compute_effective_shifts(unadjusted_df, adjusted_df):
    """
    Compute effective shifts from Bayesian shift+scale adjustment.
    
    Returns dictionary: {gene: effective_shift}
    """
    effective_shifts = {}
    
    for gene in unadjusted_df.columns:
        if gene in adjusted_df.columns:
            y_unadj = unadjusted_df[gene].to_numpy()
            y_adj = adjusted_df[gene].to_numpy()
            
            # Effective shift = change in mean
            effective_shift = np.nanmean(y_unadj) - np.nanmean(y_adj)
            effective_shifts[gene] = effective_shift
    
    return effective_shifts


def apply_effective_shifts(unadjusted_df, effective_shifts):
    """
    Apply effective shifts to unadjusted data.
    
    Formula: y_adjusted = y_unadjusted - effective_shift
    """
    adjusted = {}
    
    for gene, shift in effective_shifts.items():
        if gene in unadjusted_df.columns:
            y = unadjusted_df[gene].to_numpy()
            adjusted[gene] = y - shift
    
    return pl.DataFrame(adjusted)


def main():
    parser = argparse.ArgumentParser(
        description="Create effective shift-only adjusted data from Bayesian results"
    )
    parser.add_argument("--unadjusted", required=True,
                       help="Path to unadjusted test gene expression")
    parser.add_argument("--bayesian-adjusted", required=True,
                       help="Path to Bayesian shift+scale adjusted data")
    parser.add_argument("--output-test", required=True,
                       help="Output path for effective shift-only adjusted test data")
    parser.add_argument("--output-train", required=True,
                       help="Output path for train data (copied from Bayesian)")
    parser.add_argument("--train-source", required=True,
                       help="Source train data to copy")
    parser.add_argument("--output-shifts", required=True,
                       help="Output path for effective shifts CSV")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Effective Shift-Only Adjusted Data")
    print("="*80)
    print("\nIMPORTANT: This uses 'effective shifts' not 'raw shifts'")
    print("Effective shift = mean(unadjusted) - mean(bayesian_adjusted)")
    print("This preserves the location correction while removing variance scaling")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    unadjusted_df = pl.read_csv(args.unadjusted)
    bayesian_adjusted_df = pl.read_csv(args.bayesian_adjusted)
    
    print(f"  Unadjusted genes: {len(unadjusted_df.columns)}")
    print(f"  Bayesian adjusted genes: {len(bayesian_adjusted_df.columns)}")
    
    # Compute effective shifts
    print("\nComputing effective shifts...")
    effective_shifts = compute_effective_shifts(unadjusted_df, bayesian_adjusted_df)
    
    # Show some examples
    print("\nExample effective shifts (first 5 genes):")
    for i, (gene, shift) in enumerate(list(effective_shifts.items())[:5]):
        print(f"  {gene}: {shift:.3f}")
    
    # Apply effective shifts
    print("\nApplying effective shifts...")
    adjusted_df = apply_effective_shifts(unadjusted_df, effective_shifts)
    
    # Save adjusted test data
    print(f"\nSaving adjusted test data to: {args.output_test}")
    adjusted_df.write_csv(args.output_test)
    
    # Copy train data (same as Bayesian since we only adjust test)
    print(f"Copying train data to: {args.output_train}")
    train_df = pl.read_csv(args.train_source)
    train_df.write_csv(args.output_train)
    
    # Save effective shifts for reference
    print(f"Saving effective shifts to: {args.output_shifts}")
    shifts_df = pl.DataFrame({
        'gene': list(effective_shifts.keys()),
        'effective_shift': list(effective_shifts.values())
    })
    shifts_df.write_csv(args.output_shifts)
    
    print("\n" + "="*80)
    print("SUCCESS")
    print(f"  Adjusted {len(adjusted_df.columns)} genes")
    print(f"  Mean absolute effective shift: {np.mean(np.abs(list(effective_shifts.values()))):.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
