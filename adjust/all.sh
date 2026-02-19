#!/bin/bash

set -e

# bash /scripts/adjust/autoclass.sh
# bash /scripts/adjust/icvae.sh
# bash /scripts/adjust/vfae.sh
# bash /scripts/adjust/wasserstein.sh

bash /scripts/adjust/adjustR_data.sh
# bash /scripts/adjust/adjustR_individual_prep.sh

# bash /scripts/adjust/adjustR_paired_datasets.sh


