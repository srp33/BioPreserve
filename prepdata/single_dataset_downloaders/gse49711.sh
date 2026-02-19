#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mPreparing the GSE49711 dataset\033[0m\n"

Rscript ${thisDir}/gse49711.R
