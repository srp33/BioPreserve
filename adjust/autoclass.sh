#!/bin/bash

set -e

printf "\033[0;32mAdjusting the data with AutoClass\033[0m\n"

# python /scripts/adjust/invert_autoclass.py

python /scripts/adjust/autoclass.py -i /data/gold/gse49711/unadjusted.csv -o /data/gold/gse49711/autoclass.csv -b meta_Sex
# python /scripts/adjust/autoclass.py -i /data/gold/gse20194/unadjusted.csv -o /data/gold/gse20194/autoclass.csv -b meta_batch
# python /scripts/adjust/autoclass.py -i /data/gold/gse24080/unadjusted.csv -o /data/gold/gse24080/autoclass.csv -b meta_batch

# python /scripts/adjust/autoclass.py -i /data/gold/simple2d/unadjusted.csv -o /data/gold/simple2d/autoclass.csv -b meta_batch

wait
