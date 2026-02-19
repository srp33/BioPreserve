import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def generate_sanity_data(
    base_dir,
    n_dims=2,
    n_samples_per_group=250,
    mean_bio=1,
    mean_batch=1,
    std_dev=0.1,
    debug=False
):
    """
    Generates and saves datasets for four sanity-check permutations
    to test batch correction algorithms.

    Args:
        base_dir (pathlib.Path): The root directory to save the data folders in.
        n_dims (int): The number of dimensions for the feature data.
        n_samples_per_group (int): Number of samples for each of the 4 combinations.
        mean_bio (float): The magnitude of the biological effect shift.
        mean_batch (float): The magnitude of the batch effect shift.
        std_dev (float): The standard deviation of the Gaussian noise.
        debug (bool): If True, prints debugging information.
    """
    if debug:
        print("DEBUG: Starting data generation process.")
        print(f"DEBUG: Output directory: {base_dir}")
        print(f"DEBUG: Number of dimensions: {n_dims}")

    # Define the four scenarios
    scenarios = {
        "yes_bio_yes_batch": {"bio_effect": True, "batch_effect": True},
        "yes_bio_no_batch": {"bio_effect": True, "batch_effect": False},
        "no_bio_yes_batch": {"bio_effect": False, "batch_effect": True},
        "no_bio_no_batch": {"bio_effect": False, "batch_effect": False},
    }

    for name, config in scenarios.items():
        if debug:
            print(f"DEBUG: Generating data for scenario: {name}")

        # Create a list to hold data for the four subgroups
        dfs = []

        # Iterate through the four possible combinations of bio and batch groups
        for bio_group in [0, 1]:
            for batch_group in [0, 1]:
                
                # --- Generalize for n-dimensions ---
                # Base mean is [-1, 1, 0, ..., 0]
                base_mean = np.zeros(n_dims)
                base_mean[0] = -1
                if n_dims > 1:
                    base_mean[1] = 1

                # Shift is applied only to the first two dimensions
                shift = np.zeros(n_dims)
                
                # Biological effect
                if config["bio_effect"]:
                    shift[0] += bio_group * mean_bio
                    if n_dims > 1:
                        shift[1] += bio_group * -mean_bio
                
                # Batch effect
                if config["batch_effect"]:
                    shift[0] += batch_group * mean_batch
                    if n_dims > 1:
                        shift[1] += batch_group * -mean_batch
                
                # Final mean for the multivariate normal distribution
                loc = base_mean + shift
                
                # Generate random data
                data = np.random.normal(loc=loc, scale=std_dev, size=(n_samples_per_group, n_dims))

                # Create a DataFrame for the current subgroup
                df = pd.DataFrame(data, columns=[f'dim_{i}' for i in range(n_dims)])
                df['meta_bio'] = bio_group
                df['meta_batch'] = batch_group
                dfs.append(df)

        # Combine the subgroups into one final DataFrame
        final_df = pd.concat(dfs, ignore_index=True)

        # --- File Saving ---
        # Create a flat directory name combining dimension and scenario
        dir_name = f"{n_dims}_dims_{name}"
        output_dir = base_dir / dir_name
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file path
        output_path = output_dir / "unadjusted.csv"
        
        # Save the DataFrame to a CSV file
        final_df.to_csv(output_path, index=False)
        if debug:
            print(f"DEBUG: Saved data to {output_path}")

    if debug:
        print("DEBUG: Data generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate sanity check datasets for batch correction algorithms."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="The base directory to save the data folders in. (default: 'data')"
    )
    parser.add_argument(
        "-d",
        "--n_dims",
        type=int,
        default=2,
        help="Number of dimensions for the data. (default: 2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug printing."
    )
    args = parser.parse_args()

    print("Generating sanity check datasets...")
    generate_sanity_data(
        base_dir=args.output_dir,
        n_dims=args.n_dims,
        debug=args.debug
    )
    print("Done.")
