suppressPackageStartupMessages(library(readr))

in_file_path = commandArgs(trailingOnly=TRUE)[1]
num_columns = as.integer(commandArgs(trailingOnly=TRUE)[2])
out_file_path = commandArgs(trailingOnly=TRUE)[3]

data = read_csv(in_file_path)
data = data[,1:num_columns]
write_csv(data, out_file_path)
