#!/usr/bin/env python3
"""
Hydra-enabled wrapper for running deep learning batch adjustment models.

Usage:
    # Use default configs
    python run_model.py
    
    # Override specific configs
    python run_model.py model=vfae dataset=gse24080 training=thorough
    
    # Override individual parameters
    python run_model.py model.latent_dim=20 training.epochs=500
    
    # Run multiple experiments
    python run_model.py --multirun model=icvae,vfae dataset=gse49711,gse24080
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess
import sys
from pathlib import Path

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run batch adjustment model with Hydra configuration."""
    
    print("="*80)
    print(f"Running {cfg.model.name} on {cfg.dataset.name}")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Resolve all interpolations
    OmegaConf.resolve(cfg)
    
    # Map model names to their Python scripts
    model_scripts = {
        "icvae": "run_icvae.py",
        "vfae": "run_vfae.py",
        "autoclass": "autoclass.py",
        "wasserstein": "wasserstein.py",
    }
    
    script_path = model_scripts.get(cfg.model.name)
    if not script_path:
        print(f"Error: Unknown model '{cfg.model.name}'", file=sys.stderr)
        sys.exit(1)
    
    # Check if input file needs preprocessing (e.g., wasserstein needs combat)
    input_file = cfg.dataset.input_file
    if hasattr(cfg.model, 'requires_preprocessing'):
        preprocess = cfg.model.requires_preprocessing
        input_file = input_file.replace('unadjusted.csv', f'{preprocess}.csv')
        print(f"Note: Using preprocessed input: {input_file}")
    
    # Build command based on model type
    cmd = ["python", script_path]
    
    # Common arguments
    cmd.extend(["-i", input_file])
    cmd.extend(["-o", cfg.dataset.output_file])
    cmd.extend(["-b", cfg.dataset.batch_col])
    
    # Training arguments
    if hasattr(cfg.training, 'epochs'):
        cmd.extend(["-e", str(cfg.training.epochs)])
    if hasattr(cfg.training, 'learning_rate'):
        cmd.extend(["-lr", str(cfg.training.learning_rate)])
    if hasattr(cfg.training, 'batch_size'):
        cmd.extend(["-bs", str(cfg.training.batch_size)])
    
    # Model-specific arguments
    if cfg.model.name == "icvae":
        cmd.extend(["-l", str(cfg.model.latent_dim)])
        cmd.extend(["-hd", str(cfg.model.hidden_dim)])
        cmd.extend(["-hda", str(cfg.model.hidden_dim_aux)])
        cmd.extend(["--w-kl", str(cfg.model.w_kl)])
        cmd.extend(["--w-mi-penalty", str(cfg.model.w_mi_penalty)])
    elif cfg.model.name == "vfae":
        cmd.extend(["-l", str(cfg.model.latent_dim)])
        cmd.extend(["-hd", str(cfg.model.hidden_dim)])
    elif cfg.model.name == "wasserstein":
        if cfg.model.verbose:
            cmd.append("-v")
    
    # Run the command
    print(f"\nExecuting: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nError: Model execution failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    
    print(f"\n✓ Successfully created: {cfg.dataset.output_file}")

if __name__ == "__main__":
    main()
