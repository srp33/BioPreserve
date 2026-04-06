#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
ranked_csv <- args[1]
target_genes <- args[2]
outdir <- args[3]

# Load adjusted matrix
ranked <- read_csv(ranked_csv)
ranked <- as_tibble(ranked)

# Load target genes
genes <- read_csv(target_genes, show_col_types = FALSE)$Gene

# Keep target genes + meta_source
filtered <- ranked %>% select(any_of(c(genes, "meta_source", "meta_er_status")))

# Run PCA
numeric_data <- filtered %>% 
    select(where(is.numeric)) %>%
    select(-starts_with("meta_"))  
pca_res <- prcomp(numeric_data, scale. = TRUE, center = TRUE)
scores <- as_tibble(pca_res$x) 

# Add meta_source and meta_er_status for grouping
scores <- scores %>% 
    mutate(source = filtered$meta_source) %>%
    mutate(er_status = as.factor(filtered$meta_er_status))

# Variance explained
var_exp <- (pca_res$sdev^2) / sum(pca_res$sdev^2)

# Plot PCA
p1 <- ggplot(scores, aes(x = PC1, y = PC2, color = er_status, shape = source)) +
     geom_point(size=2, alpha=0.5) +
     labs(
         x = paste0("PC1 (", round(var_exp[1]*100,1), "%)"),
         y = paste0("PC2 (", round(var_exp[2]*100,1), "%)")
     ) +
     theme_minimal()

p2  <- ggplot(scores, aes(x = PC1, y = PC2, color = source, shape = er_status)) +
     geom_point(size=2, alpha=0.5) +
     labs(
         x = paste0("PC1 (", round(var_exp[1]*100,1), "%)"),
         y = paste0("PC2 (", round(var_exp[2]*100,1), "%)")
     ) +
     theme_minimal()

p <- p1 / p2
# Save
if(!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
ggsave(file.path(outdir, "stack_pca_plot.png"), plot = p, width=6, height=4, dpi=300)