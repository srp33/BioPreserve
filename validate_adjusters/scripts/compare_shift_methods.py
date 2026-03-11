#!/usr/bin/env python3
"""
Compare three shift-learning approaches:
1. Effective shifts (from full shift+scale model)
2. Shift-only model (Bayesian model with only shift parameter)
3. DBA shifts (direct optimization for classification)
"""

import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler


def evaluate(X_train, X_test, y_train, y_test):
    """Evaluate classification performance."""
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[mask_train])
    X_test_scaled = scaler.transform(X_test[mask_test])
    
    clf = SGDClassifier(
        loss='log_loss', penalty='elasticnet',
        alpha=0.0001, l1_ratio=0.15,
        max_iter=1000, random_state=42, tol=1e-3
    )
    
    clf.fit(X_train_scaled, y_train[mask_train])
    y_pred = clf.predict(X_test_scaled)
    
    return matthews_corrcoef(y_test[mask_test], y_pred)


def main():
    print("="*80)
    print("Comparing Shift-Learning Approaches")
    print("="*80)
    
    # Load data
    train_genes = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_train_selected_genes.csv")
    test_unadj = pl.read_csv("adjusters/unadjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayes_full = pl.read_csv("adjusters/adjusted_bayesian_shift_scale_2studies_test_metabric_test_selected_genes.csv")
    test_bayes_eff = pl.read_csv("adjusters/adjusted_bayesian_effective_shift_only_2studies_test_metabric_test_selected_genes.csv")
    test_dba = pl.read_csv("outputs/dba_analysis_shift_only/test_genes_dba_direct.csv")
    
    train_meta = pl.read_csv("metadata/metadata_train.csv")
    test_meta = pl.read_csv("metadata/metadata_test.csv")
    
    # Get labels
    train_y = train_meta['meta_her2_status'].to_numpy()
    test_y = test_meta['meta_her2_status'].to_numpy()
    le = LabelEncoder()
    all_labels = np.concatenate([
        train_y[~pl.Series(train_y).is_null()],
        test_y[~pl.Series(test_y).is_null()]
    ])
    le.fit(all_labels)
    train_y_enc = np.array([le.transform([v])[0] if v is not None else np.nan for v in train_y])
    test_y_enc = np.array([le.transform([v])[0] if v is not None else np.nan for v in test_y])
    
    # Top 3 genes
    genes = ['ERBB2', 'STARD3', 'PGAP3']
    X_train = train_genes.select(genes).to_numpy()
    X_test_unadj = test_unadj.select(genes).to_numpy()
    X_test_bayes_full = test_bayes_full.select(genes).to_numpy()
    X_test_bayes_eff = test_bayes_eff.select(genes).to_numpy()
    X_test_dba = test_dba.select(genes).to_numpy()
    
    # Compute shifts
    eff_shifts = np.nanmean(X_test_unadj, axis=0) - np.nanmean(X_test_bayes_eff, axis=0)
    dba_shifts = np.nanmean(X_test_unadj, axis=0) - np.nanmean(X_test_dba, axis=0)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n1. Unadjusted")
    score = evaluate(X_train, X_test_unadj, train_y_enc, test_y_enc)
    print(f"   MCC: {score:.3f}")
    
    print("\n2. Bayesian shift+scale (original)")
    score = evaluate(X_train, X_test_bayes_full, train_y_enc, test_y_enc)
    print(f"   MCC: {score:.3f}")
    
    print("\n3. Bayesian effective shifts (from full model)")
    score = evaluate(X_train, X_test_bayes_eff, train_y_enc, test_y_enc)
    print(f"   MCC: {score:.3f}")
    print(f"   Shifts: {eff_shifts}")
    
    print("\n4. DBA shifts (direct optimization)")
    score = evaluate(X_train, X_test_dba, train_y_enc, test_y_enc)
    print(f"   MCC: {score:.3f}")
    print(f"   Shifts: {dba_shifts}")
    
    # Check if shift-only model exists
    try:
        test_shift_model = pl.read_csv("../bayesian_shift_scale_adjuster/outputs/adjusted_bayesian_shift_only_model_2studies_test_metabric_test_selected_genes.csv")
        X_test_shift_model = test_shift_model.select(genes).to_numpy()
        shift_model_shifts = np.nanmean(X_test_unadj, axis=0) - np.nanmean(X_test_shift_model, axis=0)
        
        print("\n5. Bayesian shift-only model (NEW)")
        score = evaluate(X_train, X_test_shift_model, train_y_enc, test_y_enc)
        print(f"   MCC: {score:.3f}")
        print(f"   Shifts: {shift_model_shifts}")
        
        print("\n" + "="*80)
        print("SHIFT COMPARISON")
        print("="*80)
        for i, gene in enumerate(genes):
            print(f"\n{gene}:")
            print(f"  Effective (from full): {eff_shifts[i]:7.3f}")
            print(f"  Shift-only model:      {shift_model_shifts[i]:7.3f}")
            print(f"  DBA (optimized):       {dba_shifts[i]:7.3f}")
            print(f"  Diff (eff vs model):   {abs(eff_shifts[i] - shift_model_shifts[i]):7.3f}")
            print(f"  Diff (eff vs DBA):     {abs(eff_shifts[i] - dba_shifts[i]):7.3f}")
    
    except FileNotFoundError:
        print("\n5. Bayesian shift-only model: NOT YET RUN")
        print("   Run: pixi run snakemake fit_and_shift_shift_only_model")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
