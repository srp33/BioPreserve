#!/usr/bin/env python3
"""Verify what the Bayesian parameters actually do."""

import numpy as np
import polars as pl

# Load data
test_unadjusted = pl.read_csv(
    "../bayesian_shift_scale_adjuster/outputs/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv"
)
test_adjusted = pl.read_csv(
    "../bayesian_shift_scale_adjuster/outputs/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv"
)

# Load parameters
params_df = pl.read_csv(
    "../bayesian_shift_scale_adjuster/outputs/precision_matrix.csv"
)

genes = ['ERBB2', 'STARD3', 'PGAP3']

print("Bayesian Parameters and Effects:")
print("="*80)
for gene in genes:
    param_row = params_df.filter(pl.col('gene') == gene)
    scale = float(param_row['test_mean_slope'][0])
    shift = float(param_row['test_mean_intercept'][0])
    
    y_unadj = test_unadjusted[gene].to_numpy()
    y_adj = test_adjusted[gene].to_numpy()
    
    # What the formula should give
    y_expected = (y_unadj - shift) / scale
    
    # Check if it matches
    diff = np.nanmean(np.abs(y_expected - y_adj))
    
    # Effective shift (change in mean)
    mean_unadj = np.nanmean(y_unadj)
    mean_adj = np.nanmean(y_adj)
    effective_shift = mean_unadj - mean_adj
    
    print(f"\n{gene}:")
    print(f"  Raw parameters: shift={shift:.3f}, scale={scale:.3f}")
    print(f"  Formula: (y - {shift:.3f}) / {scale:.3f}")
    print(f"  Mean unadjusted: {mean_unadj:.3f}")
    print(f"  Mean adjusted (shift+scale): {mean_adj:.3f}")
    print(f"  Effective shift (shift+scale): {effective_shift:.3f}")
    print(f"  Formula matches actual? {diff < 0.001}")
    
    # Now compute shift-only
    y_shift_only = y_unadj - shift
    mean_shift_only = np.nanmean(y_shift_only)
    effective_shift_only = mean_unadj - mean_shift_only
    
    print(f"  Mean adjusted (shift-only): {mean_shift_only:.3f}")
    print(f"  Effective shift (shift-only): {effective_shift_only:.3f}")
