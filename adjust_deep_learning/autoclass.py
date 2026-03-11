import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from argparse import ArgumentParser

from invert_autoclass import BatchCorrectImpute, take_norm

# Parse command line args --------------------------
parser = ArgumentParser(description="AutoClass Imputation Example")
parser.add_argument("-i", "--input-file", help="Path to input CSV file", required=True)
parser.add_argument("-o", "--output-file", help="Path to output CSV file", required=True)
parser.add_argument("-b", "--batch-column", help="Column name for batch information", required=True)
args = parser.parse_args()

# Load the dataset --------------------------------
input_file = args.input_file
if not input_file.endswith('.csv'):
    raise ValueError("Input file must be a CSV file.")

try:
    df = pd.read_csv(input_file)
except Exception as e:
    raise ValueError(f"AutoClass Error reading the input file: {input_file}\n{e}")

# Perform the adjustments --------------------------

# Check for negative values. Assuming that microarray data is mean 0, while RNA-seq data is counts
# Make sure to filter out the metadata columns if they exist. Metadata columns start with 'meta_'
meta_cols = [col for col in df.columns if col.startswith('meta_')]
print(f"Found metadata columms: {meta_cols}")
gene_cols = [col for col in df.columns if col not in meta_cols]

genes = df[gene_cols]
normalized = genes.min().min() < 0


# Normalize and logp1-transform the data if necessary
if not normalized:
    print("Data is not normalized. Normalizing and log1p-transforming the data.")
    X_norm = take_norm(genes)

batches = df[args.batch_column].values
res = BatchCorrectImpute(genes,batches,cellwise_norm=False,log1p=False,verbose=True,encoder_layer_size=[128],
                         adversarial_weight=0.002,epochs=400,lr=15,reg=0.0001,dropout_rate=0.2)
result = res['imp']

# Convert the result to a DataFrame
result = pd.DataFrame(result, columns=gene_cols)

# Add the metadata columns back to the result
if meta_cols:
    result = pd.concat([df[meta_cols], result], axis=1)


# Save the imputed data to a CSV file --------------------
output_file = args.output_file
try:
    result.to_csv(output_file, index=False)
    print(f"Imputed data saved to {output_file}")
except Exception as e:
    raise ValueError(f"AutoClass Error saving the output file: {output_file}\n{e}")

