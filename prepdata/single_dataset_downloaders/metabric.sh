#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mPreparing the METABRIC dataset (raw, non-z-scored expression)\033[0m\n"

Rscript ${thisDir}/metabric.R
