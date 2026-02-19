suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
})

rnaseq_file_path = commandArgs()[7]
mutation_file_path = commandArgs()[8]
cancer_types_file_path = commandArgs()[9]
out_file_path = commandArgs()[10]

# Not sure why I have to do this...
# https://github.com/tidyverse/vroom/issues/364
Sys.setenv(VROOM_CONNECTION_SIZE = 500072)

rnaseq = read_tsv(rnaseq_file_path)
mutations = read_tsv(mutation_file_path)
cancer_types = read_tsv(cancer_types_file_path)

unique_sample_ids = intersect(intersect(pull(rnaseq, Sample), pull(mutations, Sample)), pull(cancer_types, Sample))

rnaseq = filter(rnaseq, Sample %in% unique_sample_ids)
mutations = filter(mutations, Sample %in% unique_sample_ids)
cancer_types = filter(cancer_types, Sample %in% unique_sample_ids)

mutated_sample_ids = filter(mutations, MutatedGene == "TP53") %>%
    pull(Sample)

mutations = tibble(Sample = unique_sample_ids, Mutated = as.integer(unique_sample_ids %in% mutated_sample_ids))

data = inner_join(cancer_types, mutations, by = "Sample") %>%
    inner_join(rnaseq, by = "Sample") %>%
    dplyr::rename(TP53_Mutated = Mutated)

write_csv(data, out_file_path)
