#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mDownloading the TCGA dataset\033[0m\n"

full=/data/gold/tcga/unadjusted.csv
medium=/data/gold/tcga_medium/unadjusted.csv
small=/data/gold/tcga_small/unadjusted.csv
rnaseq="/tmp/rnaseq.tsv.gz"
mutations="/tmp/mutations.tsv.gz"
labels="/tmp/labels.tsv.gz"

## This gets the TCGA RNA-Seq dataset
wget -O /tmp/GSE62944_RAW.tar "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE62944&format=file"
cd /tmp
tar -xvf GSE62944_RAW.tar
rm GSE62944_RAW.tar
mv GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz rnaseq.tsv.gz
python $thisDir/transpose_matrix.py rnaseq.tsv.gz rnaseq2.tsv.gz
python $thisDir/tcga_expression.py rnaseq2.tsv.gz $rnaseq

wget https://osf.io/na3rp/download -O $mutations
wget https://osf.io/frxv6/download -O $labels

printf "\033[0;32mTidying the TCGA dataset\033[0m\n"

Rscript --vanilla $thisDir/tcga.R $rnaseq $mutations $labels $full

Rscript --vanilla $thisDir/tcga_sampling.R
