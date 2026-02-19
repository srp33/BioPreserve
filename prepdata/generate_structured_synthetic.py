import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def generate_structured_gene_data(
    output_dir,
    n_samples_per_group=50,
    n_genes=12000,
    n_modules=100,
    n_bio_only_modules=10,
    n_batch_only_modules=10,
    n_shared_modules=10,
    mean_bio=2.5,
    mean_batch=2.5,
    std_dev_noise=0.5,
    master_gene_std_dev=0.25,
    debug=False
):
    """
    Generates a high-dimensional dataset with a modular correlation structure
    to simulate gene expression data for testing batch correction.

    Args:
        output_dir (pathlib.Path): The directory to save the output file in.
        n_samples_per_group (int): Samples per each of the 4 groups.
        n_genes (int): Total number of genes (columns).
        n_modules (int): Number of gene modules to partition genes into.
        n_bio_only_modules (int): Number of modules affected only by the biological signal.
        n_batch_only_modules (int): Number of modules affected only by the batch signal.
        n_shared_modules (int): Number of modules affected by BOTH bio and batch signals.
        mean_bio (float): Magnitude of the biological effect shift.
        mean_batch (float): Magnitude of the batch effect shift.
        std_dev_noise (float): Std dev of noise added to individual genes.
        master_gene_std_dev (float): Std dev for weights connecting master genes
                                     to other genes in their module.
        debug (bool): If True, prints debugging information.
    """
    if debug:
        print("DEBUG: Starting structured gene data generation.")
        print(f"DEBUG: Output directory: {output_dir}")

    if n_genes % n_modules != 0:
        raise ValueError("n_genes must be perfectly divisible by n_modules.")
    
    genes_per_module = n_genes // n_modules
    total_samples = n_samples_per_group * 4

    # --- Define which modules are affected ---
    # New logic to allow for bio-only, batch-only, and shared effect modules.
    total_affected_modules = n_bio_only_modules + n_batch_only_modules + n_shared_modules
    if total_affected_modules > n_modules:
        raise ValueError("The sum of bio, batch, and shared modules cannot exceed the total number of modules.")
    
    module_indices = np.arange(n_modules)
    np.random.shuffle(module_indices)
    
    # Slice the shuffled indices to get our three groups
    bio_only_idx = module_indices[:n_bio_only_modules]
    batch_only_idx = module_indices[n_bio_only_modules : n_bio_only_modules + n_batch_only_modules]
    shared_idx = module_indices[n_bio_only_modules + n_batch_only_modules : total_affected_modules]
    
    # The final sets of affected modules are unions of the specific groups
    bio_module_idx = np.concatenate([bio_only_idx, shared_idx])
    batch_module_idx = np.concatenate([batch_only_idx, shared_idx])

    if debug:
        print(f"DEBUG: {genes_per_module} genes per module.")
        print(f"DEBUG: Bio-only effect on modules: {bio_only_idx}")
        print(f"DEBUG: Batch-only effect on modules: {batch_only_idx}")
        print(f"DEBUG: Shared effect on modules: {shared_idx}")
        print(f"DEBUG: All bio-affected modules: {bio_module_idx}")
        print(f"DEBUG: All batch-affected modules: {batch_module_idx}")


    # --- Vectorized Data Generation ---
    
    # 1. Create group assignments for all samples at once
    bio_group_vec = np.repeat([0, 0, 1, 1], n_samples_per_group)
    batch_group_vec = np.repeat([0, 1, 0, 1], n_samples_per_group)

    # 2. Generate latent "metagene" expression for all modules and samples
    metagene_expression = np.random.normal(loc=0, scale=1, size=(total_samples, n_modules))

    # 3. Apply biological and batch effects vectorially
    bio_effect = bio_group_vec[:, np.newaxis] * mean_bio
    batch_effect = batch_group_vec[:, np.newaxis] * mean_batch
    
    metagene_expression[:, bio_module_idx] += bio_effect
    metagene_expression[:, batch_module_idx] += batch_effect

    # 4. Generate individual gene expression from metagenes vectorially
    gene_expression_base = np.repeat(metagene_expression, genes_per_module, axis=1)
    noise = np.random.normal(loc=0, scale=std_dev_noise, size=(total_samples, n_genes))
    gene_expression = gene_expression_base + noise

    # --- 5. Add master gene intra-module correlation ---
    # This section adds a more complex correlation structure within each module.
    if debug:
        print("DEBUG: Adding master gene correlations within modules.")
    
    for i in range(n_modules):
        # Define the column indices for the current module
        start_idx = i * genes_per_module
        end_idx = (i + 1) * genes_per_module
        
        # The first gene in the module is the "master regulator"
        master_gene_idx = start_idx
        # The other genes are the targets
        other_genes_indices = np.arange(start_idx + 1, end_idx)
        
        if len(other_genes_indices) > 0:
            # Get the expression of the master gene across all samples
            master_gene_expr = gene_expression[:, master_gene_idx]
            
            # Generate random weights for the other genes
            num_other_genes = len(other_genes_indices)
            weights = np.random.normal(loc=0, scale=master_gene_std_dev, size=num_other_genes)
            
            # Add the weighted master gene expression to the other genes
            # Reshape master_gene_expr for broadcasting
            influence = master_gene_expr[:, np.newaxis] * weights
            gene_expression[:, other_genes_indices] += influence

    # 6. Create final DataFrame
    gene_cols = [f'gene_{i}' for i in range(n_genes)]
    final_df = pd.DataFrame(gene_expression, columns=gene_cols)
    final_df['meta_bio'] = bio_group_vec
    final_df['meta_batch'] = batch_group_vec
    # Reorder so meta columns are first
    final_df = final_df[['meta_bio', 'meta_batch'] + gene_cols]

    # --- File Saving ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "unadjusted.csv"
    final_df.to_csv(output_path, index=False)
    
    if debug:
        print(f"DEBUG: Saved data to {output_path}")
        print("DEBUG: Data generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a high-dimensional structured gene expression dataset."
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True,
        help="The directory to save the output file in."
    )
    parser.add_argument(
        "--n_genes", type=int, default=12000, help="Total number of genes."
    )
    parser.add_argument(
        "--n_modules", type=int, default=100, help="Number of gene modules."
    )
    parser.add_argument(
        "--n_bio_only_modules", type=int, default=20, help="Number of modules with only a biological effect."
    )
    parser.add_argument(
        "--n_batch_only_modules", type=int, default=20, help="Number of modules with only a batch effect."
    )
    parser.add_argument(
        "--n_shared_modules", type=int, default=30, help="Number of modules with both a biological and batch effect."
    )
    parser.add_argument(
        "--master_gene_std_dev", type=float, default=0.25,
        help="Std dev for master gene weights. Default is 0.25."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug printing."
    )
    args = parser.parse_args()

    print("Generating structured gene expression dataset...")
    generate_structured_gene_data(
        output_dir=args.output_dir,
        n_genes=args.n_genes,
        n_modules=args.n_modules,
        n_bio_only_modules=args.n_bio_only_modules,
        n_batch_only_modules=args.n_batch_only_modules,
        n_shared_modules=args.n_shared_modules,
        master_gene_std_dev=args.master_gene_std_dev,
        debug=args.debug
    )
    print("Done.")
