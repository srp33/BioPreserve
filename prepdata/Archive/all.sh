#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#set -e

osf_datasets="GSE19615,GSE20194,GSE20271,GSE23720,GSE25055,GSE25065,GSE31448,"\
"GSE45255,GSE58644,GSE62944_Tumor,GSE76275,GSE81538,GSE96058_HiSeq,GSE96058_NextSeq,METABRIC"

gdrive_datasets="GSE115577,GSE123845,GSE163882"

rm /outputs/prepdata.log

# echo "Downloading OSF datasets..."
# python3 /scripts/prepdata/download_datasets.py \
#     --source osf \
#     --project-id eky3p \
#     --raw-download-dir /data/raw_download \
#     --raw-data-dir /data/raw_data \
#     --verbose \
#     --datasets "$osf_datasets"

# echo "Downloading Google Drive datasets..."
# python3 /scripts/prepdata/download_datasets.py \
#     --source gdrive \
#     --folder-id 1smhpktMRyP4yyFHKHSisxRd9jwb8kvrq \
#     --raw-download-dir /data/raw_download \
#     --raw-data-dir /data/raw_data \
#     --verbose \
#     --datasets "$gdrive_datasets"


# echo "Organizing..."
python /scripts/prepdata/organize_downloaded_files.py \
    --raw-dir /data/raw_download \
    --target-dir /data/raw_data \


echo "🔧 Converting files, and fixing if needed..."
python3 /scripts/prepdata/convert_raw_files.py \
    --raw-dir /data/raw_data \
    --target-dir /data/gold \
    --debug 2>&1 | tee /outputs/prepdata.log

echo "🔗 Generating all dataset combinations with caching (only the unadjusted files)..."
python3 /scripts/prepdata/generate_all_combinations.py --csv-files unadjusted.csv --debug --parallel 10 |& tee /outputs/prepdata2.log 


# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20194
# https://pubmed.ncbi.nlm.nih.gov/20064235/
# bash /scripts/prepdata/gse20194.sh

# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080
# https://pubmed.ncbi.nlm.nih.gov/20064235/
# bash /scripts/prepdata/gse24080.sh

# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49711
# https://pubmed.ncbi.nlm.nih.gov/25150839/
# bash /scripts/prepdata/gse49711.sh

#Other possibilities:
#bash /scripts/prepdata/bladderbatch.sh
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE47792 (SEQC superseries)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49711 (same as GSE49711 but uses Agilent microarrays)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54275 (specifically, the samples for GPL15932)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25507 (Affymetrix Human Genome U133 Plus 2.0)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65204 (Agilent)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58979 (Affymetrix PrimeView)
#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19750 (Affymetrix Human Genome U133 Plus 2.0)
#bash /scripts/prepdata/tcga.sh

# Rscript /scripts/prepdata/generate_hgu133a_annotation.R
# python /scripts/prepdata/generate_hgu133a_annotation.py
# Rscript /scripts/prepdata/generate_entrez_map.R
# Rscript /scripts/prepdata/gse62944.R

# python /scripts/prepdata/combine_datasets.py \
#     --input1 /data/gold/gse20194/unadjusted.csv \
#     --annot1 /data/gold/annotations/entrez_to_symbol_map.csv \
#     --map_type1 'entrez' \
#     --input2 /data/gold/gse62944/unadjusted.csv \
#     --output /data/gold/gse_20194_62944/unadjusted.csv \
#     --debug

# python /scripts/prepdata/generate_sanity_permutations.py -o /data/gold/ -d 2 --debug
# python /scripts/prepdata/generate_sanity_permutations.py -o /data/gold/ -d 400 --debug
# python /scripts/prepdata/generate_sanity_permutations.py -o /data/gold/ -d 1000 --debug
# python /scripts/prepdata/generate_structured_synthetic.py -o /data/gold/structured_synthetic --debug



# https://www.refine.bio/compendia/normalized
# bash /scripts/prepdata/refinebio.sh
