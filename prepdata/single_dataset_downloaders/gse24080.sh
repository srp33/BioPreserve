#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mPreparing the GSE24080 dataset\033[0m\n"

# This file is large, so it was failing when I tried downloading it with GEOquery.
#wget -O /tmp/GSE24080_RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE24nnn/GSE24080/suppl/GSE24080_RAW.tar
#cd /tmp
#mkdir -p GSE24080
#mv GSE24080_RAW.tar GSE24080/
#cd GSE24080/
#tar -xvf GSE24080_RAW.tar
#rm GSE24080_RAW.tar

cd ${thisDir}
Rscript gse24080.R
