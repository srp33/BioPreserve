from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from vfae import VFAE, mmd_rbf_loss
import sys
import torch.optim as optim

# --- Argument Parsing ---
parser = ArgumentParser(description="VFAE for learning fair representations using a Maximum Mean Discrepancy (MMD) penalty.")
parser.add_argument("-i", "--input-file", help="Path to input CSV file.", required=True)
parser.add_argument("-o", "--output-file", help="Path to output CSV file for fair reconstructions.", required=True)
parser.add_argument("-b", "--batch-col", help="Column name for the sensitive attribute (batch).", required=True)
parser.add_argument("-l", "--latent-dim", type=int, default=10, help="Dimensionality of the latent space.")
parser.add_argument("-hd", "--hidden-dim", type=int, default=128, help="Dimensionality of hidden layers for VFAE.")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size for training.")
parser.add_argument("--w-kl", type=float, default=1.0, help="Weight for the KL divergence loss term (beta).")
parser.add_argument("--w-mmd", type=float, default=10.0, help="Weight for the MMD penalty term (gamma).")
parser.add_argument("--mmd-gamma", type=float, default=1.0, help="Gamma parameter for the RBF kernel in MMD.")
args = parser.parse_args()

def print_now(*args, **kwargs):
    """Helper function to print with flush."""
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
print_now("Normalizing data to [-1, 1] range.")
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
model = VFAE(
    input_dim=input_dim,
    num_s_categories=num_batches,
    latent_dim=args.latent_dim,
    hidden_dim_enc=args.hidden_dim,
    hidden_dim_dec=args.hidden_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# --- Training Loop ---
print_now("\nStarting training...")
for epoch in range(args.epochs):
    model.train()
    total_recon_loss, total_kl_loss, total_mmd_loss = 0, 0, 0

    for x_batch, s_batch in train_loader:
        x_batch = x_batch.to(device)
        s_batch = s_batch.to(device)

        optimizer.zero_grad()
        
        # --- VFAE forward pass ---
        x_recon, mu, logvar, z = model(x_batch, s_batch)

        # --- Calculate Loss Components ---
        # 1. Reconstruction Loss (MSE), ensures the model can reconstruct the input
        recon_loss = F.mse_loss(x_recon, x_batch, reduction='mean')

        # 2. KL Divergence, regularizes the latent space
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. MMD Loss, encourages the latent space to be independent of the sensitive attribute
        # Compare the aggregated posterior q(z) to the prior p(z)=N(0,1)
        # This encourages the latent space to be independent of s
        prior_samples = torch.randn_like(z)
        mmd_loss = mmd_rbf_loss(z, prior_samples, gamma=args.mmd_gamma)
            
        # --- Combine and backpropagate losses ---
        total_loss = recon_loss + (args.w_kl * kl_loss / x_batch.size(0)) + (args.w_mmd * mmd_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses for logging
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_mmd_loss += mmd_loss.item()

    # --- Logging ---
    num_samples = len(dataset)
    # Note: recon_loss is already a mean over the batch, so we average the means.
    avg_recon = total_recon_loss / len(train_loader) 
    avg_kl = total_kl_loss / num_samples
    avg_mmd = total_mmd_loss / len(train_loader)
    
    print_now(f"Epoch {epoch+1}/{args.epochs} | Recon: {avg_recon:.6f} | KL: {avg_kl:.6f} | MMD Loss: {avg_mmd:.4f}")

print_now("\nTraining complete.")

# --- Generate and Save Fair Reconstructions ---
print_now(f"Generating fair reconstructions and saving to '{args.output_file}'...")
model.eval()
with torch.no_grad():
    recon_all = []
    for batch_val in range(num_batches):
        batch_indices = torch.full(
            (len(features_tensor),),
            batch_val,
            dtype=torch.long,
            device=device
        )
        
        # Pass raw indices; the model handles one-hot encoding
        x_recon, _, _, _ = model(features_tensor.to(device), batch_indices)
        recon_all.append(x_recon.cpu().numpy())
    
    # Average the reconstructions across all sensitive attribute categories
    recon_all_cpu = np.mean(recon_all, axis=0)

# De-normalize the data back to its original scale from [-1, 1]
recon_denormalized = (recon_all_cpu + 1) / 2 * (features_max.values - features_min.values) + features_min.values

# Create a new DataFrame for the reconstructed data
recon_df = pd.DataFrame(recon_denormalized, columns=feature_cols, index=df.index)

# Combine with metadata and the original sensitive attribute
meta_df = df[[col for col in df.columns if col.startswith('meta_') or col == args.batch_col]]
output_df = pd.concat([meta_df, recon_df], axis=1)

try:
    output_df.to_csv(args.output_file, index=False)
    print_now("Successfully saved reconstructed data.")
except Exception as e:
    print_now(f"Error saving the output file: {e}", file=sys.stderr)
    sys.exit(1)