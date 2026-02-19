 #!/bin/bash

set -e

printf "\033[0;32mAdjusting the data with VFAE\033[0m\n"

# python /scripts/adjust/run_vfae.py -i /data/gold/gse20194/unadjusted.csv -o /data/gold/gse20194/vfae.csv -b meta_batch
python /scripts/adjust/run_vfae.py -i /data/gold/gse24080/unadjusted.csv -o /data/gold/gse24080/vfae.csv -b meta_batch
python /scripts/adjust/run_vfae.py -i /data/gold/gse49711/unadjusted.csv -o /data/gold/gse49711/vfae.csv -b meta_Sex
