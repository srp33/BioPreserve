out_file_path = "/data/gold/gse49711/unadjusted.csv"

suppressPackageStartupMessages({
    library(dplyr)
    library(GEOquery)
    library(readr)
    library(stringr)
    library(tibble)
})

download.file("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE49711&format=file&file=GSE49711%5FSEQC%5FNB%5FTAV%5FG%5Flog2%2Efinal%2Etxt%2Egz", "/tmp/gse49711.expr.tsv.gz")

expr_data = read_tsv("/tmp/gse49711.expr.tsv.gz") %>%
    filter(Gene_set == "Gene_AceView") %>%
    filter(RefSeq_transcript_ID != ".") %>%
    select(Gene, starts_with("SEQC_")) %>%
    as.data.frame()

rownames(expr_data) = pull(expr_data, Gene)

expr_data = data.matrix(expr_data[,2:ncol(expr_data)])

# Remove genes with lots of zero values
num_zero = apply(expr_data, 1, function(x) { sum(x==0) })
expr_data = expr_data[-which(num_zero > (ncol(expr_data) / 2)),]

# This returned zero.
#print(sum(is.na(expr_data)))

expr_data = t(expr_data) %>%
    as.data.frame() %>%
    rownames_to_column(var = "meta_Sample_ID")

meta_tibble = as_tibble(as.data.frame(getGEO("GSE49711")))

varying_tibble <- meta_tibble %>%
  select(where(~ n_distinct(.) > 1))
print("Varying")
print(varying_tibble)
print("Varying Columns")
print(colnames(varying_tibble))

slim_tibble <- varying_tibble %>%
  select(where(~ n_distinct(.) < 10))

print("Slim")
print(slim_tibble)
print("Slim Columns")
print(colnames(slim_tibble))
print("Unique Values (not how many, but what they are for each column)")
print(sapply(slim_tibble, unique))



metadata = as_tibble(as.data.frame(getGEO("GSE49711"))) %>%
    dplyr::rename(meta_Dataset = `GSE49711_series_matrix.txt.gz.dataset.ch1`) %>%
    dplyr::rename(meta_Sample_ID = `GSE49711_series_matrix.txt.gz.title`) %>%
    dplyr::rename(meta_Class = `GSE49711_series_matrix.txt.gz.class.label.ch1`) %>%
    dplyr::rename(meta_Age_at_Diagnosis = `GSE49711_series_matrix.txt.gz.age.at.diagnosis.ch1`) %>%
    dplyr::rename(meta_Death_from_Disease = `GSE49711_series_matrix.txt.gz.death.from.disease.ch1`) %>%
    dplyr::rename(meta_High_Risk = `GSE49711_series_matrix.txt.gz.high.risk.ch1`) %>%
    dplyr::rename(meta_INSS_Stage = `GSE49711_series_matrix.txt.gz.inss.stage.ch1`) %>%
    dplyr::mutate(meta_INSS_Stage_Split_1_2 = ifelse(meta_INSS_Stage %in% c("1", "2"), "1", "2")) %>%
    dplyr::mutate(meta_INSS_Stage_Split_2_3 = ifelse(meta_INSS_Stage %in% c("1", "2", "3"), "1", "2")) %>%
    dplyr::mutate(meta_INSS_Stage_Split_3_4 = ifelse(meta_INSS_Stage %in% c("1", "2", "3", "4"), "1", "2")) %>%
    dplyr::rename(meta_MYCN_Status = `GSE49711_series_matrix.txt.gz.mycn.status.ch1`) %>%
    dplyr::rename(meta_Progression = `GSE49711_series_matrix.txt.gz.progression.ch1`) %>%
    dplyr::rename(meta_Sex = `GSE49711_series_matrix.txt.gz.Sex.ch1`) %>%
    select(!starts_with("GSE49711_"))

data = inner_join(metadata, expr_data, by="meta_Sample_ID")

#print(table(data$Sex))
#print(table(data$MYCN_Status))

if (!dir.exists("/data/gold/gse49711"))
    dir.create("/data/gold/gse49711")

write_csv(data, out_file_path)
