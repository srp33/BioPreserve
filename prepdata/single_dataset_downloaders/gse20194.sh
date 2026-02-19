#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mPreparing the GSE20194 dataset\033[0m\n"

Rscript ${thisDir}/gse20194.R
