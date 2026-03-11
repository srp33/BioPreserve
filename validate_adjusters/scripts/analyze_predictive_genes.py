#!/usr/bin/env python3
"""
Analyze top predictive genes for HER2 and chemo status.

For each metadata label:
1. Find top 10 most predictive genes
2. Visualize distributions conditioned on label
3. Compare patterns across datasets
"""

import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from pathlib import Path


def get_top_predictive_genes(X, y, gene_names, n_top=10):
    """
    Find top predictive genes using Random Forest feature importance.
    
    Returns: list of (gene_name, importance_score) tuples
    """
    # Filter out samples with missing labels
    mask = ~np.isnan(y)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # Skip if too few samples or only one class
    if len(y_filtered) < 10 or len(np.unique(y_filtered)) < 2:
        return []
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_filtered, y_filtered)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Sort and get top N
    top_indices = np.argsort(importances)[-n_top:][::-1]
    top_genes = [(gene_names[i], importances[i]) for i in top_indices]
    
    return top_genes


def plot_gene_distributions_split_violin(gene_data_dict, gene_name, label_name, output_path):
    """
    Create split violin plot showing gene distribution by label for each dataset.
    
    gene_data_dict: {
        'train': {'X': array, 'y': array, 'dataset_name': str},
        'test': {'X': array, 'y': array, 'dataset_name': str}
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (split_name, data) in enumerate(gene_data_dict.items()):
        ax = axes[idx]
        
        # Prepare data for plotting
        mask = ~np.isnan(data['y'])
        gene_values = data['X'][mask]
        labels = data['y'][mask]
        
        if len(np.unique(labels)) < 2:
            ax.text(0.5, 0.5, f'Only one class in {split_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create DataFrame for seaborn
        plot_df = pl.DataFrame({
            'expression': gene_values,
            'label': labels.astype(str)
        }).to_pandas()
        
        # Split violin plot
        sns.violinplot(data=plot_df, x='label', y='expression', 
                      ax=ax, inner='box', cut=0)
        
        # Add individual points with jitter
        sns.stripplot(data=plot_df, x='label', y='expression',
                     ax=ax, color='black', alpha=0.3, size=2)
        
        ax.set_title(f'{data["dataset_name"]} (n={len(gene_values)})')
        ax.set_xlabel(label_name)
        ax.set_ylabel(f'{gene_name} expression')
        
        # Add statistics
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            group1 = gene_values[labels == unique_labels[0]]
            group2 = gene_values[labels == unique_labels[1]]
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
            cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else 0
            
            ax.text(0.02, 0.98, f"Cohen's d = {cohens_d:.3f}", 
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{gene_name} distribution by {label_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_genes_heatmap(gene_data_dict, top_genes, label_name, output_path):
    """
    Create heatmap showing mean expression of top genes by label and dataset.
    
    Rows: genes
    Columns: dataset_label combinations (e.g., train_positive, train_negative, test_positive, test_negative)
    """
    results = []
    
    for gene_name, importance in top_genes:
        row = {'gene': gene_name, 'importance': importance}
        
        for split_name, data in gene_data_dict.items():
            mask = ~np.isnan(data['y'])
            gene_idx = list(data['gene_names']).index(gene_name)
            gene_values = data['X'][mask, gene_idx]
            labels = data['y'][mask]
            
            for label_val in np.unique(labels):
                label_mask = labels == label_val
                mean_expr = np.mean(gene_values[label_mask])
                col_name = f"{split_name}_{label_val}"
                row[col_name] = mean_expr
        
        results.append(row)
    
    # Create DataFrame
    df = pl.DataFrame(results).to_pandas()
    
    # Prepare for heatmap (genes as rows, conditions as columns)
    heatmap_data = df.set_index('gene').drop(columns=['importance'])
    
    # Normalize each gene (row) to see relative differences
    heatmap_data_norm = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0).div(heatmap_data.std(axis=1), axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_genes) * 0.5)))
    sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, cbar_kws={'label': 'Z-score'}, ax=ax)
    ax.set_title(f'Top {len(top_genes)} genes for {label_name}\n(normalized within gene)', 
                fontsize=12)
    ax.set_xlabel('Dataset and Label')
    ax.set_ylabel('Gene')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_comparison(gene_data_dict, top_genes, label_name, output_path):
    """
    PCA plot of top genes, colored by label, with separate panels for each dataset.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (split_name, data) in enumerate(gene_data_dict.items()):
        ax = axes[idx]
        
        # Get top genes
        gene_indices = [list(data['gene_names']).index(g[0]) for g in top_genes]
        X_top = data['X'][:, gene_indices]
        
        # Filter out missing labels
        mask = ~np.isnan(data['y'])
        X_filtered = X_top[mask]
        y_filtered = data['y'][mask]
        
        if len(np.unique(y_filtered)) < 2:
            ax.text(0.5, 0.5, f'Only one class in {split_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        unique_labels = np.unique(y_filtered)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label_val, color in zip(unique_labels, colors):
            mask_label = y_filtered == label_val
            ax.scatter(X_pca[mask_label, 0], X_pca[mask_label, 1],
                      c=[color], label=f'{label_val}', alpha=0.6, s=30)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'{data["dataset_name"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PCA of top {len(top_genes)} genes for {label_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dataset_informativeness(gene_data_dict, top_genes, label_name, output_path):
    """
    Compare how informative each dataset is for predicting the label.
    
    Shows:
    1. Per-gene predictive power in each dataset
    2. Overall separation quality (silhouette score)
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Per-gene effect sizes
    ax1 = axes[0]
    
    effect_sizes = {'gene': [], 'train': [], 'test': []}
    
    for gene_name, _ in top_genes:
        effect_sizes['gene'].append(gene_name)
        
        for split_name, data in gene_data_dict.items():
            mask = ~np.isnan(data['y'])
            gene_idx = list(data['gene_names']).index(gene_name)
            gene_values = data['X'][mask, gene_idx]
            labels = data['y'][mask]
            
            # Calculate effect size (Cohen's d)
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                group1 = gene_values[labels == unique_labels[0]]
                group2 = gene_values[labels == unique_labels[1]]
                pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
                cohens_d = abs((np.mean(group2) - np.mean(group1)) / pooled_std) if pooled_std > 0 else 0
            else:
                cohens_d = 0
            
            effect_sizes[split_name].append(cohens_d)
    
    # Plot grouped bar chart
    x = np.arange(len(effect_sizes['gene']))
    width = 0.35
    
    ax1.bar(x - width/2, effect_sizes['train'], width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, effect_sizes['test'], width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Gene')
    ax1.set_ylabel("Effect Size (|Cohen's d|)")
    ax1.set_title(f'Per-gene discriminative power for {label_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(effect_sizes['gene'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium effect')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
    
    # Panel 2: Overall separation quality
    ax2 = axes[1]
    
    metrics = {'dataset': [], 'silhouette': [], 'n_samples': [], 'class_balance': []}
    
    for split_name, data in gene_data_dict.items():
        # Get top genes
        gene_indices = [list(data['gene_names']).index(g[0]) for g in top_genes]
        X_top = data['X'][:, gene_indices]
        
        # Filter out missing labels
        mask = ~np.isnan(data['y'])
        X_filtered = X_top[mask]
        y_filtered = data['y'][mask]
        
        if len(np.unique(y_filtered)) < 2:
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Calculate silhouette score (measures cluster separation)
        sil_score = silhouette_score(X_scaled, y_filtered)
        
        # Class balance
        unique, counts = np.unique(y_filtered, return_counts=True)
        balance = min(counts) / max(counts)
        
        metrics['dataset'].append(data['dataset_name'])
        metrics['silhouette'].append(sil_score)
        metrics['n_samples'].append(len(y_filtered))
        metrics['class_balance'].append(balance)
    
    # Plot metrics
    x_pos = np.arange(len(metrics['dataset']))
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x_pos - 0.2, metrics['silhouette'], 0.4, 
                    label='Silhouette Score', alpha=0.8, color='steelblue')
    bars2 = ax2_twin.bar(x_pos + 0.2, metrics['class_balance'], 0.4,
                         label='Class Balance', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Silhouette Score (higher = better separation)', color='steelblue')
    ax2_twin.set_ylabel('Class Balance (1.0 = perfect balance)', color='coral')
    ax2.set_title(f'Overall dataset informativeness for {label_name}')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics['dataset'])
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add sample size annotations
    for i, (x, n) in enumerate(zip(x_pos, metrics['n_samples'])):
        ax2.text(x, -0.05, f'n={n}', ha='center', va='top', fontsize=9)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze top predictive genes for metadata labels"
    )
    parser.add_argument("--train-genes", required=True, help="Train gene expression CSV")
    parser.add_argument("--test-genes", required=True, help="Test gene expression CSV")
    parser.add_argument("--train-metadata", required=True, help="Train metadata CSV")
    parser.add_argument("--test-metadata", required=True, help="Test metadata CSV")
    parser.add_argument("--metadata-labels", nargs="+", required=True,
                       help="Metadata labels to analyze (e.g., meta_her2_status meta_chemotherapy)")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--n-top-genes", type=int, default=10, help="Number of top genes to analyze")
    
    args = parser.parse_args()
    
    # Load data
    train_genes_df = pl.read_csv(args.train_genes)
    test_genes_df = pl.read_csv(args.test_genes)
    train_meta_df = pl.read_csv(args.train_metadata)
    test_meta_df = pl.read_csv(args.test_metadata)
    
    # Get common genes
    common_genes = [g for g in train_genes_df.columns if g in test_genes_df.columns]
    
    train_X = train_genes_df.select(common_genes).to_numpy()
    test_X = test_genes_df.select(common_genes).to_numpy()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each metadata label
    for meta_label in args.metadata_labels:
        print(f"\n{'='*60}")
        print(f"Analyzing: {meta_label}")
        print(f"{'='*60}")
        
        # Get labels
        train_y = train_meta_df[meta_label].to_numpy()
        test_y = test_meta_df[meta_label].to_numpy()
        
        # Encode string labels to numeric
        if train_y.dtype == object or test_y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            all_labels = np.concatenate([train_y[~pl.Series(train_y).is_null()], 
                                        test_y[~pl.Series(test_y).is_null()]])
            le.fit(all_labels)
            train_y = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
            test_y = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
        
        # Find top predictive genes (using train data)
        print(f"Finding top {args.n_top_genes} predictive genes...")
        top_genes = get_top_predictive_genes(train_X, train_y, common_genes, args.n_top_genes)
        
        if not top_genes:
            print(f"  Skipping {meta_label}: insufficient data")
            continue
        
        print(f"  Top genes: {[g[0] for g in top_genes]}")
        
        # Prepare data dict
        gene_data_dict = {
            'train': {
                'X': train_X,
                'y': train_y,
                'gene_names': common_genes,
                'dataset_name': 'Train (GSE)'
            },
            'test': {
                'X': test_X,
                'y': test_y,
                'gene_names': common_genes,
                'dataset_name': 'Test (METABRIC)'
            }
        }
        
        # Create visualizations
        label_clean = meta_label.replace('meta_', '')
        
        print("  Creating heatmap...")
        plot_top_genes_heatmap(
            gene_data_dict, top_genes, meta_label,
            output_dir / f"{label_clean}_heatmap.png"
        )
        
        print("  Creating PCA plot...")
        plot_pca_comparison(
            gene_data_dict, top_genes, meta_label,
            output_dir / f"{label_clean}_pca.png"
        )
        
        print("  Creating informativeness comparison...")
        plot_dataset_informativeness(
            gene_data_dict, top_genes, meta_label,
            output_dir / f"{label_clean}_informativeness.png"
        )
        
        print("  Creating individual gene violin plots...")
        for gene_name, importance in top_genes[:5]:  # Top 5 genes only
            gene_idx = common_genes.index(gene_name)
            gene_data_for_plot = {
                'train': {
                    'X': train_X[:, gene_idx],
                    'y': train_y,
                    'dataset_name': 'Train (GSE)'
                },
                'test': {
                    'X': test_X[:, gene_idx],
                    'y': test_y,
                    'dataset_name': 'Test (METABRIC)'
                }
            }
            plot_gene_distributions_split_violin(
                gene_data_for_plot, gene_name, meta_label,
                output_dir / f"{label_clean}_{gene_name}_violin.png"
            )
        
        print(f"  ✓ Plots saved to {output_dir}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
