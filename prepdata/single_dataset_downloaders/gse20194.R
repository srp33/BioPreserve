suppressPackageStartupMessages({
    library(doParallel)
    library(dplyr)
    library(readr)
    library(readxl)
    library(SCAN.UPC)
    library(stringr)
    library(tibble)
})

registerDoParallel(cores=20)

eSet = SCAN("GSE20194", probeSummaryPackage="hgu133ahsentrezgprobe")

eData = t(data.matrix(exprs(eSet))) %>%
  as.data.frame() %>%
  rownames_to_column("CEL_file") %>%
  as_tibble() %>%
  mutate(CEL_file = str_replace_all(CEL_file, "\\.gz", "")) %>%
  mutate(CEL_file = str_replace_all(CEL_file, "^GSM\\d+_", ""))

download.file("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE20194&format=file&file=GSE20194%5FMDACC%5FSample%5FInfo%2Exls%2Egz", "/tmp/gse20194_meta.xls.gz")
gunzip("/tmp/gse20194_meta.xls.gz", overwrite = TRUE)

pData = read_excel("/tmp/gse20194_meta.xls") %>%
    filter(description != "MAQC_Distribution_Status: MDA_R -- Not used") %>%
    filter(description != "MAQC_Distribution_Status: MAQC_Q -- Not used") %>%
    dplyr::rename(meta_Sample = `Sample name`) %>%
    dplyr::rename(meta_CEL_file = `CEL file`) %>%
    dplyr::rename(meta_batch = description) %>%
    dplyr::mutate(meta_batch = str_replace_all(meta_batch, "MAQC_Distribution_Status: MAQC_T -- Training", "1")) %>%
    dplyr::mutate(meta_batch = str_replace_all(meta_batch, "MAQC_Distribution_Status: MAQC_V -- Validation", "2")) %>%
    dplyr::rename(meta_treatment_response = `characteristics: pCR_vs_RD`) %>%
    dplyr::rename(meta_age = `characteristics: age`) %>%
    dplyr::rename(meta_race = `characteristics: race`) %>%
    dplyr::rename(meta_er_status = `characteristics: ER_status`) %>%
    dplyr::rename(meta_pr_status = `characteristics: PR_status`) %>%
    dplyr::rename(meta_her2_status = `HER2 Status`) %>%
    dplyr::rename(meta_histology = `Histology`) %>%
    dplyr::rename(meta_treatment_code = `Treatment Code`) %>%
    dplyr::rename(meta_bmn_grade = `BMNgrd`) %>%
    dplyr::select(-`title`, -`source name`, -`organism`, -`molecule`, -`label`, -`platform`, -`Additional information`, -`Tbefore`, -`Nbefore`, -`ER`, -`Her2 IHC`, -`Her2 FISH`, -`Treatments Comments`)

if (!dir.exists("/data/gold/gse20194"))
    dir.create("/data/gold/gse20194")

inner_join(eData, pData) %>%
    dplyr::select(-CEL_file) %>%
    dplyr::select(Sample, age, race, er_status, treatment_response, pr_status, batch, bmn_grade, her2_status, histology, treatment_code, matches("^\\d.+")) %>%
    write_csv("/data/gold/gse20194/unadjusted.csv")
