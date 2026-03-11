#!/usr/bin/env python3
"""
Investigate why shift+scale helps ER but shift-only helps HER2.

Hypotheses to test:
1. Which genes are selected for ER vs HER2?
2. How does scaling affect those genes?
3. Is there a variance pattern difference?
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_selected_features_and_coefs(X_train, X_test, y_train, y_test, C=0.05):
    """Get selected features and their coefficients."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_train_filt = X_train[mask_train]
    y_train_filt = y_train[mask_train]
    X_test_filt = X_test[mask_test]
    y_test_filt = y_test[mask_test]
    
    if len(np.unique(y_train_filt)) < 2:
        return None, None, np.nan
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_filt)
    X_test_sc = scaler.transform(X_test_filt)
    
    clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                            random_state=42, max_iter=1000)
    clf.fit(X_train_sc, y_train_filt)
    y_pred = clf.predict(X_test_sc)
    
    score = matthews_corrcoef(y_test_filt, y_pred)
    
    # Get selected features
    coefs = clf.coef_[0]
    selected_mask = np.abs(coefs) > 1e-5
    
    return selected_mask, coefs, score


def main():
    print("="*80)
    print("Investigating Shift+Scale vs Shift-Only Trade-off")
    print("="*80)
    
    # Load data
    train_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadj_df = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bay_ss_df = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bay_so_df = pl.read_csv("adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv")
    
    train_meta = pl.read_csv("metadata/metadata_train.csv")
    test_meta = pl.read_csv("metadata/metadata_test.csv")
    
    common_genes = [g for g in train_df.columns 
                   if g in test_unadj_df.columns 
                   and g in test_bay_ss_df.columns
                   and g in test_bay_so_df.columns]
    
    print(f"\nGenes: {len(common_genes)}")
    
    X_train = train_df.select(common_genes).to_numpy()
    X_test_unadj = test_unadj_df.select(common_genes).to_numpy()
    X_test_bay_ss = test_bay_ss_df.select(common_genes).to_numpy()
    X_test_bay_so = test_bay_so_df.select(common_genes).to_numpy()
    
    # Prepare labels
    labels_to_test = ['meta_er_status', 'meta_her2_status']
    
    all_y_train = {}
    all_y_test = {}
    
    for label in labels_to_test:
        y_tr = train_meta[label].to_numpy()
        y_te = test_meta[label].to_numpy()
        
        le = LabelEncoder()
        all_vals = np.concatenate([
            y_tr[~pl.Series(y_tr).is_null()],
            y_te[~pl.Series(y_te).is_null()]
        ])
        if len(all_vals) > 0:
            le.fit(all_vals)
            y_tr = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_tr])
            y_te = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_te])
            all_y_train[label] = y_tr
            all_y_test[label] = y_te
    
    # Analyze feature selection for each label and method
    print("\n" + "="*80)
    print("Feature Selection Analysis (C=0.05)")
    print("="*80)
    
    results = {}
    
    for label in ['meta_er_status', 'meta_her2_status']:
        print(f"\n{label}:")
        results[label] = {}
        
        for method_name, X_test in [
            ('shift+scale', X_test_bay_ss),
            ('shift-only', X_test_bay_so)
        ]:
            y_tr = all_y_train[label]
            y_te = all_y_test[label]
            
            selected_mask, coefs, score = get_selected_features_and_coefs(
                X_train, X_test, y_tr, y_te, C=0.05
            )
            
            if selected_mask is not None:
                selected_genes = [common_genes[i] for i in range(len(common_genes)) 
                                 if selected_mask[i]]
                selected_coefs = coefs[selected_mask]
                
                results[label][method_name] = {
                    'genes': selected_genes,
                    'coefs': selected_coefs,
                    'score': score
                }
                
                print(f"\n  {method_name}: {score:.3f} ({len(selected_genes)} features)")
                print(f"    Top 5 genes: {selected_genes[:5]}")
    
    # Compare selected genes
    print("\n" + "="*80)
    print("Gene Selection Comparison")
    print("="*80)
    
    er_ss_genes = set(results['meta_er_status']['shift+scale']['genes'])
    er_so_genes = set(results['meta_er_status']['shift-only']['genes'])
    her2_ss_genes = set(results['meta_her2_status']['shift+scale']['genes'])
    her2_so_genes = set(results['meta_her2_status']['shift-only']['genes'])
    
    print(f"\nER status:")
    print(f"  Shift+scale only: {er_ss_genes - er_so_genes}")
    print(f"  Shift-only only: {er_so_genes - er_ss_genes}")
    print(f"  Shared: {er_ss_genes & er_so_genes}")
    
    print(f"\nHER2 status:")
    print(f"  Shift+scale only: {her2_ss_genes - her2_so_genes}")
    print(f"  Shift-only only: {her2_so_genes - her2_ss_genes}")
    print(f"  Shared: {her2_ss_genes & her2_so_genes}")
    
    # Analyze variance changes
    print("\n" + "="*80)
    print("Variance Analysis")
    print("="*80)
    
    # Compute variance for each gene in each dataset
    var_unadj = np.nanvar(X_test_unadj, axis=0)
    var_ss = np.nanvar(X_test_bay_ss, axis=0)
    var_so = np.nanvar(X_test_bay_so, axis=0)
    
    # Variance ratios
    var_ratio_ss = var_ss / var_unadj
    var_ratio_so = var_so / var_unadj
    
    print(f"\nOverall variance changes:")
    print(f"  Shift+scale: mean ratio = {np.mean(var_ratio_ss):.3f}")
    print(f"  Shift-only: mean ratio = {np.mean(var_ratio_so):.3f}")
    
    # Variance for selected genes
    for label in ['meta_er_status', 'meta_her2_status']:
        print(f"\n{label}:")
        
        for method_name in ['shift+scale', 'shift-only']:
            selected_genes = results[label][method_name]['genes']
            selected_indices = [common_genes.index(g) for g in selected_genes]
            
            var_unadj_sel = var_unadj[selected_indices]
            var_ss_sel = var_ss[selected_indices]
            var_so_sel = var_so[selected_indices]
            
            print(f"\n  {method_name} selected genes:")
            print(f"    Unadjusted variance: mean={np.mean(var_unadj_sel):.3f}")
            print(f"    Shift+scale variance: mean={np.mean(var_ss_sel):.3f}, ratio={np.mean(var_ss_sel/var_unadj_sel):.3f}")
            print(f"    Shift-only variance: mean={np.mean(var_so_sel):.3f}, ratio={np.mean(var_so_sel/var_unadj_sel):.3f}")
    
    # Create visualizations
    output_dir = Path("outputs/shift_scale_tradeoff_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Variance ratios for selected genes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, label in enumerate(['meta_er_status', 'meta_her2_status']):
        label_name = 'ER' if 'er' in label else 'HER2'
        
        for jdx, method_name in enumerate(['shift+scale', 'shift-only']):
            ax = axes[idx, jdx]
            
            selected_genes = results[label][method_name]['genes']
            selected_indices = [common_genes.index(g) for g in selected_genes]
            
            # Variance ratios for selected genes
            var_ratios_ss = var_ratio_ss[selected_indices]
            var_ratios_so = var_ratio_so[selected_indices]
            
            x = np.arange(len(selected_genes))
            width = 0.35
            
            ax.bar(x - width/2, var_ratios_ss, width, label='Shift+scale', alpha=0.7)
            ax.bar(x + width/2, var_ratios_so, width, label='Shift-only', alpha=0.7)
            
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='No change')
            ax.set_xlabel('Gene')
            ax.set_ylabel('Variance Ratio (adjusted/unadjusted)')
            ax.set_title(f'{label_name} - {method_name} selected genes')
            ax.set_xticks(x)
            ax.set_xticklabels(selected_genes, rotation=90, fontsize=8)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "variance_ratios_by_selection.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Coefficient comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, label in enumerate(['meta_er_status', 'meta_her2_status']):
        ax = axes[idx]
        label_name = 'ER' if 'er' in label else 'HER2'
        
        # Get all genes selected by either method
        all_selected = sorted(set(results[label]['shift+scale']['genes']) | 
                             set(results[label]['shift-only']['genes']))
        
        coefs_ss = []
        coefs_so = []
        
        for gene in all_selected:
            # Shift+scale coef
            if gene in results[label]['shift+scale']['genes']:
                gene_idx = results[label]['shift+scale']['genes'].index(gene)
                coefs_ss.append(results[label]['shift+scale']['coefs'][gene_idx])
            else:
                coefs_ss.append(0)
            
            # Shift-only coef
            if gene in results[label]['shift-only']['genes']:
                gene_idx = results[label]['shift-only']['genes'].index(gene)
                coefs_so.append(results[label]['shift-only']['coefs'][gene_idx])
            else:
                coefs_so.append(0)
        
        x = np.arange(len(all_selected))
        width = 0.35
        
        ax.bar(x - width/2, coefs_ss, width, label='Shift+scale', alpha=0.7)
        ax.bar(x + width/2, coefs_so, width, label='Shift-only', alpha=0.7)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Gene')
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{label_name} - Coefficient Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_selected, rotation=90, fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "coefficient_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    analysis_data = []
    for label in ['meta_er_status', 'meta_her2_status']:
        for method_name in ['shift+scale', 'shift-only']:
            selected_genes = results[label][method_name]['genes']
            selected_coefs = results[label][method_name]['coefs']
            selected_indices = [common_genes.index(g) for g in selected_genes]
            
            for i, gene in enumerate(selected_genes):
                gene_idx = selected_indices[i]
                analysis_data.append({
                    'label': label,
                    'method': method_name,
                    'gene': gene,
                    'coefficient': selected_coefs[i],
                    'var_unadjusted': var_unadj[gene_idx],
                    'var_shift_scale': var_ss[gene_idx],
                    'var_shift_only': var_so[gene_idx],
                    'var_ratio_ss': var_ratio_ss[gene_idx],
                    'var_ratio_so': var_ratio_so[gene_idx]
                })
    
    analysis_df = pl.DataFrame(analysis_data)
    analysis_df.write_csv(output_dir / "detailed_analysis.csv")
    
    print("\n" + "="*80)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
