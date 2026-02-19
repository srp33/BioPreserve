from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from icvae import ICVAE, AuxiliaryClassifier, AuxiliaryForestClassifier, AuxiliaryBoostedClassifier, AuxiliaryBoostedLogitClassifier
import sys
import torch.optim as optim
from jaxtyping import Float, Int
from torch import Tensor
from beartype import beartype

parser = ArgumentParser(description="ICVAE for learning fair representations using a Mutual Information penalty.")
parser.add_argument("-i", "--input-file", help="Path to input CSV file.", required=True)
parser.add_argument("-o", "--output-file", help="Path to output CSV file for fair reconstructions.", required=True)
parser.add_argument("-b", "--batch-col", help="Column name for the sensitive attribute (batch).", required=True)
parser.add_argument("-l", "--latent-dim", type=int, default=10, help="Dimensionality of the latent space.")
parser.add_argument("-hd", "--hidden-dim", type=int, default=128, help="Dimensionality of hidden layers for ICVAE.")
parser.add_argument("-hda", "--hidden-dim-aux", type=int, default=64, help="Dimensionality of hidden layers for Auxiliary Classifier.")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizers.")
parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size for training.")
parser.add_argument("--w-kl", type=float, default=1.0, help="Weight for the KL divergence loss term (beta).")
parser.add_argument("--w-mi-penalty", type=float, default=1.0, help="Weight for the Mutual Information penalty term (gamma).")
args = parser.parse_args()

def print_now(*args, **kwargs):
    print(*args, flush=True, **kwargs)

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_now(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
print_now(f"Loading data from '{args.input_file}'...")
try:
    df = pd.read_csv(args.input_file)
except Exception as e:
    print_now(f"Error reading the input file: {e}", file=sys.stderr)
    sys.exit(1)

if args.batch_col not in df.columns:
    print_now(f"Error: Batch column '{args.batch_col}' not found.", file=sys.stderr)
    sys.exit(1)

# Separate feature columns from metadata and sensitive attribute
feature_cols = [col for col in df.columns if not col.startswith('meta_') and col != args.batch_col]
print_now(f"Found {len(feature_cols)} feature columns.")

features = df[feature_cols].copy()
features_min = features.min()
features_max = features.max()

# Normalize data to [-1, 1] range for stability with MSE loss
features_normalized = (2 * (features - features_min) / (features_max - features_min)) - 1
features_normalized = features_normalized.fillna(0)  # Handle columns with no variance

batch_series = df[args.batch_col]
batch_codes, unique_batches = pd.factorize(batch_series)
num_batches = len(unique_batches)
print_now(f"Found batch attribute '{args.batch_col}' with {num_batches} categories: {unique_batches.tolist()}")

features_tensor = torch.tensor(features_normalized.values, dtype=torch.float32)
batch_tensor = torch.tensor(batch_codes, dtype=torch.long)

dataset = TensorDataset(features_tensor, batch_tensor)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# --- Model Initialization ---
input_dim = len(feature_cols)
icvae_model = ICVAE(input_dim, num_batches, args.latent_dim, args.hidden_dim).to(device)
aux_classifier = AuxiliaryClassifier(args.latent_dim, num_batches, args.hidden_dim_aux).to(device)

# 🔥 COMPILE THE MODELS! 🔥
print_now("Compiling models for optimized performance...")
icvae_model = torch.compile(icvae_model)
aux_classifier = torch.compile(aux_classifier)

opt_icvae = optim.Adam(icvae_model.parameters(), lr=args.learning_rate)
opt_aux = optim.Adam(aux_classifier.parameters(), lr=args.learning_rate)

# --- Training Loop ---
print_now("\nStarting training...")

first_quarter = args.epochs // 4
second_quarter = args.epochs // 2

def determinant_based_mutual_information(
    prob_s: Float[Tensor, "batch num_batches"],
    s_batch: Int[Tensor, "batch 1"],
    num_batches: int
) -> Float[Tensor, ""]:
    s_batch_one_hot: Float[Tensor, "batch num_batches"] = F.one_hot(s_batch, num_classes=num_batches).float()

    contingency_matrix = s_batch_one_hot.t() @ prob_s
    normalized_contingency_matrix = contingency_matrix / contingency_matrix.sum()

    det_mi = torch.abs(torch.det(normalized_contingency_matrix + torch.eye(num_batches, device=prob_s.device) * 1e-10))
    return det_mi



for epoch in range(args.epochs):
    # Initialize accumulators for losses and accuracy
    total_recon_loss, total_kl_loss, total_mi_penalty, total_aux_loss = 0, 0, 0, 0
    total_aux_correct = 0

    for x_batch, s_batch in train_loader:
        x_batch = x_batch.to(device)
        s_batch = s_batch.to(device)

        # --- ICVAE forward pass ---
        x_recon, mu, logvar, z = icvae_model(x_batch, s_batch)
        
        # --- Update Auxiliary Classifier: q(s|z) ---
        # Train classifier to predict `s` from `z` (gradients do not flow to encoder)
        s_log_probs_aux = aux_classifier(z.detach())

        # --- Calculate Auxiliary Accuracy ---
        # Get the predicted class labels by finding the index of the max log-probability
        pred_s_aux = torch.argmax(s_log_probs_aux, dim=1)
        # Sum up the number of correct predictions in the batch
        total_aux_correct += (pred_s_aux == s_batch).sum().item()
        
        det_mi = determinant_based_mutual_information(
            prob_s=torch.exp(s_log_probs_aux),
            s_batch=s_batch,
            num_batches=num_batches
        )
        loss_aux = -torch.log(det_mi + 1e-30)  # Add small value to avoid log(0)
        loss_aux += F.nll_loss(s_log_probs_aux, s_batch, reduction='mean')  # Add NLL loss for better training stability
        opt_aux.zero_grad()
        loss_aux.backward()
        opt_aux.step()

        # --- Update ICVAE (Encoder + Decoder) ---
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(x_recon, x_batch, reduction='mean')
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. Mutual Information Penalty
        log_probs_s: Float[Tensor, "batch num_batches"] = aux_classifier(z)
        prob_s: Float[Tensor, "batch num_batches"] = torch.exp(log_probs_s)

        det_mi = determinant_based_mutual_information(
            prob_s=prob_s,
            s_batch=s_batch,
            num_batches=num_batches
        )
        mi_penalty = torch.log(det_mi + 1e-30)

        if epoch < first_quarter:
            mi_penalty = -mi_penalty
            
        total_icvae_loss = recon_loss + args.w_kl * kl_loss + args.w_mi_penalty * mi_penalty
        opt_icvae.zero_grad()
        total_icvae_loss.backward()
        opt_icvae.step()
        
        # Accumulate losses for logging
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_mi_penalty += mi_penalty.item()
        total_aux_loss += loss_aux.item()

    # --- Logging ---
    num_samples = len(dataset)
    avg_recon = total_recon_loss / num_samples
    avg_kl = total_kl_loss / num_samples
    avg_mi = total_mi_penalty / len(train_loader)
    avg_aux_loss = total_aux_loss / len(train_loader)
    avg_aux_acc = total_aux_correct / num_samples
    
    # Print epoch statistics including auxiliary classifier accuracy
    print_now(f"Epoch {epoch+1}/{args.epochs} | Recon: {avg_recon:.6f} | KL: {avg_kl:.6f} | MI Pen: {avg_mi:.4f} | Aux Loss: {avg_aux_loss:.4f} | Aux Acc: {avg_aux_acc:.4f}")

print_now("\nTraining complete.")

# --- Generate and Save Fair Reconstructions ---
print_now(f"Generating fair reconstructions and saving to '{args.output_file}'...")
icvae_model.eval()
with torch.no_grad():
    recon_all = []
    for batch_val in range(1): # Just replicate the first batch for now
        # Batch values here starts from 0, since we used pd.factorize
        batch_indices = torch.full(
            (len(features_tensor),),
            batch_val,
            dtype=torch.long,
            device=device
        )

        # Pass the raw indices to the model. The model will perform the one-hot encoding.
        x_recon, _, _, _ = icvae_model(features_tensor.to(device), batch_indices)

        recon_all.append(x_recon.cpu().numpy())
    
    recon_all_cpu = np.mean(recon_all, axis=0)

# De-normalize the data back to its original scale
recon_denormalized = (recon_all_cpu + 1) / 2 * (features_max.values - features_min.values) + features_min.values

# Create a new DataFrame for the reconstructed data
recon_df = pd.DataFrame(recon_denormalized, columns=feature_cols, index=df.index)

# Combine with metadata if it exists
meta_df = df[[col for col in df.columns if col.startswith('meta_') or col == args.batch_col]]
output_df = pd.concat([meta_df, recon_df], axis=1)

try:
    output_df.to_csv(args.output_file, index=False)
    print_now("Successfully saved reconstructed data.")
except Exception as e:
    print_now(f"Error saving the output file: {e}", file=sys.stderr)
    sys.exit(1)

