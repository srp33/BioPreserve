#!/bin/bash

set -e

thisDir=$(dirname $0)

printf "\033[0;32mDownloading and tidying the bladderbatch dataset\033[0m\n"

Rscript $thisDir/bladderbatch.R /data/gold/bladderbatch/unadjusted.csv
