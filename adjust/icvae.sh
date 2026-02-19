 #!/bin/bash

set -e

printf "\033[0;32mAdjusting the data with ICVAE\033[0m\n"

# parser.add_argument("-i", "--input-file", help="Path to input CSV file.", required=True)
# parser.add_argument("-o", "--output-file", help="Path to output CSV file for fair reconstructions.", required=True)
# parser.add_argument("-s", "--sensitive-col", help="Column name for the sensitive attribute.", required=True)
# parser.add_argument("-l", "--latent-dim", type=int, default=10, help="Dimensionality of the latent space.")
# parser.add_argument("-hd", "--hidden-dim", type=int, default=128, help="Dimensionality of hidden layers for VFAE.")
# parser.add_argument("-hda", "--hidden-dim-aux", type=int, default=64, help="Dimensionality of hidden layers for Auxiliary Classifier.")
# parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs.")
# parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizers.")
# parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size for training.")
# parser.add_argument("--w-kl", type=float, default=1.0, help="Weight for the KL divergence loss term (beta).")
# parser.add_argument("--w-mi-penalty", type=float, default=1.0, help="Weight for the Mutual Information penalty term (gamma).")

# python /scripts/adjust/icvae.py -i /data/gold/gse20194/unadjusted.csv -b meta_batch

# python /scripts/adjust/run_icvae.py -i /data/gold/gse20194/unadjusted.csv -o /data/gold/gse20194/icvae.csv -b meta_batch -l 20 -e 400 
python /scripts/adjust/run_icvae.py -i /data/gold/gse24080/unadjusted.csv -o /data/gold/gse24080/icvae.csv -b meta_batch -e 400
python /scripts/adjust/run_icvae.py -i /data/gold/gse49711/unadjusted.csv -o /data/gold/gse49711/icvae.csv -b meta_Sex -e 400
