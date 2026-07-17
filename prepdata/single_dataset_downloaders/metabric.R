suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(tibble)
})

# METABRIC acquisition — RAW (non-z-scored) expression, fetched directly from cBioPortal's public
# brca_metabric datahub (https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric).
#
# WHY THIS SCRIPT EXISTS: the dataset previously resolved through the shared OSF mirror (project eky3p,
# dataset name METABRIC) was `data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt` — cBioPortal's
# Z-SCORE release, already standardized per-gene (mean 0, std 1) relative to a diploid reference. Every
# OTHER dataset in this pipeline is on its native expression scale, so METABRIC's pre-normalization was a
# platform-inconsistency artifact baked into the corpus, not a real property of the data: any cross-cohort
# technical-batch estimate involving METABRIC was silently comparing a standardized cohort against raw-scale
# ones. This script instead pulls `data_mrna_illumina_microarray.txt`, the RAW log2 microarray intensity
# file cBioPortal publishes alongside the z-score one — real platform variation, not an artifact of it.
#
# Clinical metadata is fetched directly from the same datahub (data_clinical_patient.txt) rather than via
# the OSF metadata mirror, so this script is self-contained and reproducible from the primary source. The
# INTCLUST column is dropped and remaining columns renamed to match this pipeline's established METABRIC
# meta_ schema (gold/metabric/unadjusted.csv, pre-existing consumers: combine_all.py, convert_raw_files.py).

BASE_URL <- "https://github.com/cBioPortal/datahub/raw/refs/heads/master/public/brca_metabric"

# Compute nodes on this cluster have no internet egress (only the login node does), so this script first
# looks for pre-staged local copies (env vars METABRIC_EXPR_LOCAL / METABRIC_CLIN_LOCAL) before falling back
# to downloading directly -- lets the (large, ~660MB) fetch happen once on the login node and the actual
# transpose/join compute run on an allocated node via sbatch.
tmp_expr <- Sys.getenv("METABRIC_EXPR_LOCAL", unset = NA)
tmp_clin <- Sys.getenv("METABRIC_CLIN_LOCAL", unset = NA)

if (is.na(tmp_expr) || !file.exists(tmp_expr)) {
    tmp_expr <- tempfile(fileext = ".txt")
    cat("Downloading METABRIC raw expression (data_mrna_illumina_microarray.txt, ~660MB)...\n")
    download.file(paste0(BASE_URL, "/data_mrna_illumina_microarray.txt"), tmp_expr, quiet = FALSE)
} else {
    cat("Using pre-staged local expression file:", tmp_expr, "\n")
}

if (is.na(tmp_clin) || !file.exists(tmp_clin)) {
    tmp_clin <- tempfile(fileext = ".txt")
    cat("Downloading METABRIC clinical metadata (data_clinical_patient.txt)...\n")
    download.file(paste0(BASE_URL, "/data_clinical_patient.txt"), tmp_clin, quiet = FALSE)
} else {
    cat("Using pre-staged local clinical file:", tmp_clin, "\n")
}

# Expression: genes as ROWS (Hugo_Symbol, Entrez_Gene_Id, then one column per MB-#### sample). Transpose to
# samples-as-rows, genes-as-columns. Duplicate Hugo_Symbol values (multiple probes per gene) are preserved
# AS DUPLICATE COLUMN NAMES (.name_repair="minimal") to match the prior file's structure, where downstream
# tooling already tolerates repeated gene-symbol columns (e.g. ABCF1 appearing 7x) -- a plain
# as.data.frame() would silently rename these to ABCF1.1/.2/... via make.names(), which is NOT what the
# existing corpus/tooling expects.
cat("Reading + transposing expression matrix...\n")
expr_raw <- read_tsv(tmp_expr, show_col_types = FALSE, progress = FALSE)
gene_names <- expr_raw$Hugo_Symbol
sample_ids <- setdiff(colnames(expr_raw), c("Hugo_Symbol", "Entrez_Gene_Id"))
mat_t <- t(as.matrix(expr_raw[, sample_ids]))
colnames(mat_t) <- gene_names
rownames(mat_t) <- sample_ids
# dplyr verbs (mutate/inner_join/rename_with) refuse to touch a frame with duplicate column names, so the
# duplicate-gene-column join is done in BASE R below; dplyr is used only on pData, which has none.

# Clinical: skip the 4 cBioPortal header/description rows (lines starting with '#').
clin_raw <- read_tsv(tmp_clin, comment = "#", show_col_types = FALSE,
                     col_names = c("PATIENT_ID", "LYMPH_NODES_EXAMINED_POSITIVE", "NPI", "CELLULARITY",
                                   "CHEMOTHERAPY", "COHORT", "ER_IHC", "HER2_SNP6", "HORMONE_THERAPY",
                                   "INFERRED_MENOPAUSAL_STATE", "SEX", "INTCLUST", "AGE_AT_DIAGNOSIS",
                                   "OS_MONTHS", "OS_STATUS", "CLAUDIN_SUBTYPE", "THREEGENE", "VITAL_STATUS",
                                   "LATERALITY", "RADIO_THERAPY", "HISTOLOGICAL_SUBTYPE", "BREAST_SURGERY",
                                   "RFS_MONTHS", "RFS_STATUS"))

pData <- clin_raw %>%
    dplyr::rename(Sample_ID = PATIENT_ID, er_status = ER_IHC, her2_status = HER2_SNP6) %>%
    dplyr::mutate(Dataset_ID = "METABRIC", Platform_ID = "GPL6947") %>%
    dplyr::select(-INTCLUST) %>%
    dplyr::select(Sample_ID, Dataset_ID, Platform_ID, LYMPH_NODES_EXAMINED_POSITIVE, NPI, CELLULARITY,
                  CHEMOTHERAPY, COHORT, er_status, her2_status, HORMONE_THERAPY, INFERRED_MENOPAUSAL_STATE,
                  SEX, AGE_AT_DIAGNOSIS, OS_MONTHS, OS_STATUS, CLAUDIN_SUBTYPE, THREEGENE, VITAL_STATUS,
                  LATERALITY, RADIO_THERAPY, HISTOLOGICAL_SUBTYPE, BREAST_SURGERY, RFS_STATUS, RFS_MONTHS)

# `/data/gold` matches the OTHER single_dataset_downloaders/ scripts' convention (a Docker/Apptainer mount
# in the confounded_analysis container). On a bare-metal run (e.g. BioPreserve via sbatch, no /data mount),
# override with GOLD_DIR_LOCAL to point at the real shared-storage path.
gold_dir <- Sys.getenv("GOLD_DIR_LOCAL", unset = "/data/gold")
out_dir <- file.path(gold_dir, "metabric")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# INNER JOIN in base R (keep only samples present in both pData and the expression matrix), preserving
# duplicate gene-symbol columns exactly as in the source file.
keep <- intersect(pData$Sample_ID, rownames(mat_t))
cat("Samples with both clinical + expression data:", length(keep), "/", nrow(pData), "\n")
pData_m <- pData[match(keep, pData$Sample_ID), ]
colnames(pData_m) <- paste0("meta_", colnames(pData_m))
expr_m <- mat_t[match(keep, rownames(mat_t)), , drop = FALSE]

out <- cbind(as.data.frame(pData_m, check.names = FALSE),
            as.data.frame(expr_m, check.names = FALSE))
write_csv(out, file.path(out_dir, "unadjusted.csv"))

cat("Wrote", file.path(out_dir, "unadjusted.csv"), "\n")
