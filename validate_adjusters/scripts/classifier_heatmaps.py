#!/usr/bin/env python3
"""
Generate heatmaps comparing different classifiers across adjusters and metadata labels.

For each classifier:
- Rows: adjusters (unadjusted, bayesian, etc.)
- Columns: metadata labels (ER status, HER2, etc.)
- Values: MCC scores
- Include CV ceiling row for reference
"""

import argparse
import numpy as np
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path


def get_classifier(classifier_name, is_continuous=False):
    """Return classifier/regressor instance based on name."""
    if classifier_name == 'Logistic (L2)':
        if is_continuous:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=42)
        return LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif classifier_name.startswith('ElasticNet'):
        if is_continuous:
            from sklearn.linear_model import ElasticNet
            l1_ratio = float(classifier_name.split('=')[1].rstrip(')'))
            return ElasticNet(alpha=0.0001, l1_ratio=l1_ratio, random_state=42, max_iter=1000)
        l1_ratio = float(classifier_name.split('=')[1].rstrip(')'))
        return SGDClassifier(
            loss='log_loss', penalty='elasticnet',
            alpha=0.0001, l1_ratio=l1_ratio,
            max_iter=1000, random_state=42, tol=1e-3
        )
    elif classifier_name == 'Gradient Boosting':
        if is_continuous:
            return HistGradientBoostingRegressor(random_state=42)
        return HistGradientBoostingClassifier(random_state=42)
    elif classifier_name == 'Random Forest':
        if is_continuous:
            return RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


def needs_scaling(classifier_name):
    """Return True if classifier needs feature scaling."""
    return classifier_name.startswith('Logistic') or classifier_name.startswith('ElasticNet')


def evaluate_cross_dataset(X_train, X_test, y_train, y_test, classifier_name, is_continuous=False):
    """
    Evaluate cross-dataset performance with specified classifier.
    Returns (score_train_to_test, score_test_to_train, mean_score)
    """
    # Filter out missing labels
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    # Check if we have enough data
    if is_continuous:
        if len(y_train_filtered) < 10 or len(y_test_filtered) < 10:
            return np.nan, np.nan, np.nan
    else:
        if len(np.unique(y_train_filtered)) < 2 or len(np.unique(y_test_filtered)) < 2:
            return np.nan, np.nan, np.nan
    
    # Scale if needed
    if needs_scaling(classifier_name):
        scaler_fwd = StandardScaler()
        X_train_scaled = scaler_fwd.fit_transform(X_train_filtered)
        X_test_scaled = scaler_fwd.transform(X_test_filtered)
        
        scaler_rev = StandardScaler()
        X_test_scaled_rev = scaler_rev.fit_transform(X_test_filtered)
        X_train_scaled_rev = scaler_rev.transform(X_train_filtered)
    else:
        X_train_scaled = X_train_filtered
        X_test_scaled = X_test_filtered
        X_train_scaled_rev = X_train_filtered
        X_test_scaled_rev = X_test_filtered
    
    # Get classifier
    clf_fwd = get_classifier(classifier_name, is_continuous)
    clf_rev = get_classifier(classifier_name, is_continuous)
    
    # Train -> Test
    clf_fwd.fit(X_train_scaled, y_train_filtered)
    y_pred_fwd = clf_fwd.predict(X_test_scaled)
    
    if is_continuous:
        score_fwd = r2_score(y_test_filtered, y_pred_fwd)
    else:
        score_fwd = matthews_corrcoef(y_test_filtered, y_pred_fwd)
    
    # Test -> Train
    clf_rev.fit(X_test_scaled_rev, y_test_filtered)
    y_pred_rev = clf_rev.predict(X_train_scaled_rev)
    
    if is_continuous:
        score_rev = r2_score(y_train_filtered, y_pred_rev)
    else:
        score_rev = matthews_corrcoef(y_train_filtered, y_pred_rev)
    
    return score_fwd, score_rev, (score_fwd + score_rev) / 2


def compute_cv_ceiling(X_test, y_test, classifier_name, is_continuous=False):
    """
    Compute CV ceiling on test set only.
    """
    # Filter out missing labels
    test_mask = ~np.isnan(y_test)
    X_filtered = X_test[test_mask]
    y_filtered = y_test[test_mask]
    
    if is_continuous:
        if len(y_filtered) < 10:
            return np.nan
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        if len(np.unique(y_filtered)) < 2:
            return np.nan
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in kfold.split(X_filtered, y_filtered if not is_continuous else None):
        X_tr, X_te = X_filtered[train_idx], X_filtered[test_idx]
        y_tr, y_te = y_filtered[train_idx], y_filtered[test_idx]
        
        # Scale if needed
        if needs_scaling(classifier_name):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        
        clf = get_classifier(classifier_name, is_continuous)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
    
    if is_continuous:
        return r2_score(all_y_true, all_y_pred)
    else:
        return matthews_corrcoef(all_y_true, all_y_pred)


def generate_heatmap_for_classifier(classifier_name, X_train_meta, X_test_meta,
                                    adjuster_data, continuous_metadata, output_path):
    """
    Generate heatmap for a single classifier across all adjusters and metadata.
    """
    print(f"\nGenerating heatmap for: {classifier_name}")
    
    # Get metadata columns
    metadata_cols = [col for col in X_train_meta.columns if col.startswith('meta_')]
    
    # Label mapping
    label_map = {
        "meta_er_status": "ER status",
        "meta_menopause_status": "Menopause",
        "meta_sex": "Sex",
        "meta_age_at_diagnosis": "Age (R²)",
        "meta_chemotherapy": "Chemo",
        "meta_histological_type": "Hist. type",
        "meta_her2_status": "HER2",
        "meta_age_at_diagnosis_combined_lt50": "Age <50",
        "meta_age_at_diagnosis_combined_50_69": "Age 50-69",
        "meta_age_at_diagnosis_combined_ge70": "Age ≥70",
    }
    
    results = []
    
    # Compute CV ceiling
    print("  Computing CV ceiling...")
    for meta_col in metadata_cols:
        if meta_col == 'meta_source':
            continue
        
        y_test = X_test_meta[meta_col].to_numpy()
        
        # Encode if needed
        if y_test.dtype == object:
            le = LabelEncoder()
            non_null = y_test[~pl.Series(y_test).is_null()]
            if len(non_null) > 0:
                le.fit(non_null)
                y_test = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_test])
        
        is_continuous = meta_col in continuous_metadata
        
        # Get test gene data (use unadjusted for CV ceiling)
        X_test_genes = adjuster_data['unadjusted']['test']
        
        cv_score = compute_cv_ceiling(X_test_genes, y_test, classifier_name, is_continuous)
        
        results.append({
            'adjuster': 'CV Ceiling (test set)',
            'metadata_label': meta_col,
            'score_mean': cv_score
        })
    
    # Evaluate each adjuster
    for adjuster_name, data in adjuster_data.items():
        print(f"  Evaluating: {adjuster_name}")
        
        X_train_genes = data['train']
        X_test_genes = data['test']
        
        for meta_col in metadata_cols:
            if meta_col == 'meta_source':
                continue
            
            # Get labels
            y_train = X_train_meta[meta_col].to_numpy()
            y_test = X_test_meta[meta_col].to_numpy()
            
            # Encode if needed
            if y_train.dtype == object or y_test.dtype == object:
                le = LabelEncoder()
                all_labels = np.concatenate([
                    y_train[~pl.Series(y_train).is_null()],
                    y_test[~pl.Series(y_test).is_null()]
                ])
                if len(all_labels) > 0:
                    le.fit(all_labels)
                    y_train = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_train])
                    y_test = np.array([le.transform([v])[0] if v is not None else np.nan for v in y_test])
            
            is_continuous = meta_col in continuous_metadata
            
            score_fwd, score_rev, score_mean = evaluate_cross_dataset(
                X_train_genes, X_test_genes, y_train, y_test,
                classifier_name, is_continuous
            )
            
            results.append({
                'adjuster': adjuster_name,
                'metadata_label': meta_col,
                'score_mean': score_mean
            })
    
    # Create DataFrame and pivot
    results_df = pl.DataFrame(results)
    pivot = results_df.pivot(values="score_mean", index="adjuster", on="metadata_label")
    
    # Convert to pandas for plotting
    heatmap_df = pivot.to_pandas().set_index("adjuster")
    heatmap_df.columns = [label_map.get(c, c) for c in heatmap_df.columns]
    
    # Reorder: CV ceiling first, then sort others by ER status
    if 'CV Ceiling (test set)' in heatmap_df.index:
        cv_row = heatmap_df.loc[['CV Ceiling (test set)']]
        other_rows = heatmap_df.drop('CV Ceiling (test set)')
        
        if 'ER status' in other_rows.columns:
            other_rows = other_rows.sort_values('ER status', ascending=False)
        
        import pandas as pd
        heatmap_df = pd.concat([cv_row, other_rows])
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(heatmap_df.columns) * 1.1), 
                                    max(3, len(heatmap_df) * 1.2 + 1.5)))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    
    ax.set_title(
        f"Cross-Dataset Classification Performance: {classifier_name}\n"
        f"(MCC, except Age which uses R²)\nCV Ceiling = k-fold CV on test set only",
        pad=15,
    )
    ax.set_xlabel("Metadata Label")
    ax.set_ylabel("Adjuster")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved to: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for each classifier"
    )
    parser.add_argument("--X-train-metadata", required=True)
    parser.add_argument("--X-test-metadata", required=True)
    parser.add_argument("--adjusters", nargs="+", required=True,
                       help="name=train_path:test_path")
    parser.add_argument("--continuous-metadata", nargs="*", default=[],
                       help="Metadata columns to treat as continuous")
    parser.add_argument("--classifiers", nargs="+", required=True,
                       help="Classifier names to evaluate")
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Generating Classifier Heatmaps")
    print("="*60)
    
    # Load metadata
    print("\nLoading metadata...")
    X_train_meta = pl.read_csv(args.X_train_metadata)
    X_test_meta = pl.read_csv(args.X_test_metadata)
    
    # Load adjuster data
    print("Loading adjuster data...")
    adjuster_data = {}
    for pair in args.adjusters:
        name, paths = pair.split("=", 1)
        train_path, test_path = paths.split(":", 1)
        
        train_df = pl.read_csv(train_path)
        test_df = pl.read_csv(test_path)
        
        # Get common genes (exclude metadata columns and non-numeric columns)
        common_genes = []
        for col in train_df.columns:
            if col in test_df.columns and not col.startswith('meta_'):
                # Check if column is numeric in both dataframes
                if train_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    if test_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                        common_genes.append(col)
        
        adjuster_data[name] = {
            'train': train_df.select(common_genes).to_numpy(),
            'test': test_df.select(common_genes).to_numpy()
        }
        
        print(f"  {name}: {len(common_genes)} genes")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap for each classifier
    all_results = {}
    for classifier_name in args.classifiers:
        safe_name = classifier_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        output_path = output_dir / f"heatmap_{safe_name}.png"
        
        results_df = generate_heatmap_for_classifier(
            classifier_name,
            X_train_meta,
            X_test_meta,
            adjuster_data,
            args.continuous_metadata,
            output_path
        )
        
        all_results[classifier_name] = results_df
    
    print("\n" + "="*60)
    print("All heatmaps generated!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
