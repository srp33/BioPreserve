import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Ensure the parent directories are in the path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import BASIS
from adjust.basis import BASIS, BASISHyperparameters, BatchData

# Import batch simulation tools
from simulate_batches.src.simulation.shift import ShiftTransform
from simulate_batches.src.simulation.scale import ScaleTransform
from simulate_batches.src.simulation.composite import CompositeTransform


def load_real_dataset(dataset_name, data_dir="../data/gold"):
    """
    Loads real breast cancer expression and metadata.
    """
    expr_path = os.path.join(data_dir, dataset_name, "expression.csv")
    meta_path = os.path.join(data_dir, dataset_name, "metadata.csv")
    
    if not os.path.exists(expr_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find data for {dataset_name} at {data_dir}")

    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    er_raw = meta_df['ER_Status'].astype(str).str.upper()
    er_labels = np.where(er_raw.isin(['1', 'POS', 'POSITIVE', 'TRUE']), 1, 0)
    
    return expr_df.T.values, expr_df.columns.values, er_labels


def calculate_bssi(true_Z, corrected_Y, er_labels):
    """Biological Signal Suppression Index (BSSI)"""
    er_pos_idx = np.where(er_labels == 1)[0]
    er_neg_idx = np.where(er_labels == 0)[0]
    
    if len(er_pos_idx) == 0 or len(er_neg_idx) == 0:
        return 0.0 
        
    true_centroid_pos = np.mean(true_Z[:, er_pos_idx], axis=1)
    true_centroid_neg = np.mean(true_Z[:, er_neg_idx], axis=1)
    true_dist = np.linalg.norm(true_centroid_pos - true_centroid_neg)
    
    corr_centroid_pos = np.mean(corrected_Y[:, er_pos_idx], axis=1)
    corr_centroid_neg = np.mean(corrected_Y[:, er_neg_idx], axis=1)
    corr_dist = np.linalg.norm(corr_centroid_pos - corr_centroid_neg)
    
    return min(corr_dist / (true_dist + 1e-8), 1.0)


def calculate_classifier_metrics(ref_data, tgt_corrected, ref_labels, tgt_labels):
    """Trains a classifier on the Reference, evaluates on the Corrected Target."""
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(ref_data.T, ref_labels)
    
    preds = clf.predict(tgt_corrected.T)
    probs = clf.predict_proba(tgt_corrected.T)[:, 1]
    
    accuracy = accuracy_score(tgt_labels, preds)
    try:
        auc = roc_auc_score(tgt_labels, probs)
    except ValueError:
        auc = 0.5 
        
    return accuracy, auc


def run_evaluation(args):
    # 1. Load Real Data
    X_ref_raw, genes_ref, er_ref = load_real_dataset(args.ref, args.data_dir)
    Y_tgt_raw, genes_tgt, er_tgt = load_real_dataset(args.tgt, args.data_dir)
    
    # 2. Simulate Batch Effects on Target
    batch_injector = CompositeTransform([
        ScaleTransform(scale_mean=1.5, scale_std=0.2), 
        ShiftTransform(shift_mean=2.0, shift_std=1.0)
    ])
    Y_tgt_corrupted = batch_injector.transform(Y_tgt_raw.T).T
    
    # 3. Setup BASIS objects with parsed arguments
    ref_batch = BatchData(data=X_ref_raw, gene_indices=genes_ref)
    tgt_batch = BatchData(data=Y_tgt_corrupted, gene_indices=genes_tgt)
    
    hp_ot = BASISHyperparameters(
        alignment_method=args.alignment_method, 
        transform_type=args.transform_type,
        shrinkage_cap=args.shrinkage_cap,
        pca_variance_retained=args.pca_variance_retained,
        enable_pmd_gating=args.enable_pmd_gating.lower() == 'true',
        whiten_covariance=args.whiten_covariance.lower() == 'true'
    )
    
    basis = BASIS(hyperparams=hp_ot, debug=False)
    
    # 4. Align
    try:
        Y_tgt_corrected, meta = basis.align(ref_batch, tgt_batch)
        intersection_mass = meta.get('intersection_mass', 0.0)
        n_clusters = meta.get('n_clusters', 'N/A')
        status = "Success"
        
        # 5. Evaluate Metrics
        bssi_score = calculate_bssi(true_Z=Y_tgt_raw, corrected_Y=Y_tgt_corrected, er_labels=er_tgt)
        acc, auc = calculate_classifier_metrics(X_ref_raw, Y_tgt_corrected, er_ref, er_tgt)
        
    except Exception as e:
        status = f"Failed: {str(e)}"
        intersection_mass, bssi_score, acc, auc, n_clusters = 0.0, 0.0, 0.5, 0.5, 'Error'

    # 6. Save results to JSON
    results = {
        "ref_dataset": args.ref,
        "tgt_dataset": args.tgt,
        "alignment_method": args.alignment_method,
        "transform_type": args.transform_type,
        "whiten_covariance": args.whiten_covariance,
        "shrinkage_cap": args.shrinkage_cap,
        "pca_variance_retained": args.pca_variance_retained,
        "status": status,
        "intersection_mass": intersection_mass,
        "n_clusters": n_clusters,
        "bssi": bssi_score,
        "accuracy": acc,
        "roc_auc": auc
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BASIS harmonization over simulated batch effects.")
    
    # I/O Args
    parser.add_argument("--ref", type=str, required=True, help="Reference dataset name (e.g., GSE20194)")
    parser.add_argument("--tgt", type=str, required=True, help="Target dataset name (e.g., GSE25066)")
    parser.add_argument("--data_dir", type=str, default="../data/gold", help="Path to gold standard datasets")
    parser.add_argument("--out", type=str, required=True, help="Path to save output JSON results")
    
    # Hyperparameter Sweeps
    parser.add_argument("--alignment_method", type=str, default="optimal_transport", choices=["optimal_transport", "harmonic_gmm"])
    parser.add_argument("--transform_type", type=str, default="arcsinh", choices=["arcsinh", "log1p"])
    parser.add_argument("--whiten_covariance", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--enable_pmd_gating", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--shrinkage_cap", type=float, default=2.0)
    parser.add_argument("--pca_variance_retained", type=float, default=0.85)
    
    args = parser.parse_args()
    run_evaluation(args)