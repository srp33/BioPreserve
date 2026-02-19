#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mDownloading the GSE37199 dataset\033[0m\n"

raw_loc="/tmp/raw.tar.gz"
expression_tsv="/tmp/GSE37199_Expression_BatchUnadjusted.txt.gz"
clinical_tsv="/tmp/GSE37199_Clinical.txt"
out_csv="/data/gold/gse37199/unadjusted.csv"

wget https://osf.io/av3yt/download -O $raw_loc
tar -zxvf $raw_loc -C /tmp

printf "\033[0;32mTidying the GSE37199 dataset\033[0m\n"

python $thisDir/gse37199.py $expression_tsv $clinical_tsv $out_csv
