#!/usr/bin/env Rscript

# plot_performance.R
# Generates scaling performance plots and delta metrics across datasets.

# --- Load Libraries ---
library(ggplot2)
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(patchwork)
library(argparse)
library(tidytext)

# --- Helper Functions ---

# Clean metrics dataframe
clean_metrics <- function(df) {
  df %>%
    mutate(
      n_studies = as.numeric(str_extract(subset_file, "(\\d+)(?=studies)")),
      test_source = tolower(test_source)
    )
}

# Clean metadata dataframe
clean_metadata <- function(df) {
  df %>% mutate(gse_id = tolower(trimws(gse_id)),
                technology = factor(technology))
}

# Calculate delta vs unadjusted
compute_delta <- function(df_adj, df_unadj, metric_col) {
  adj <- df_adj %>%
    group_by(adjuster, n_studies, test_source) %>%
    summarise(Adj = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  
  unadj <- df_unadj %>%
    group_by(n_studies, test_source) %>%
    summarise(Unadj = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  
  full_join(adj, unadj, by = c("n_studies", "test_source")) %>%
    mutate(Mean_Metric = Adj - Unadj)
}

# Read all order files once
read_order_files <- function(order_folder, test_sources) {
  orders <- lapply(test_sources, function(src) {
    file <- file.path(order_folder, paste0(src, "_order.csv"))
    if (!file.exists(file)) stop(paste("Order file not found:", file))
    read_csv(file, show_col_types = FALSE)
  })
  names(orders) <- test_sources
  return(orders)
}

# Prepare plot dataframe with train ordering and labels
prepare_plot_data <- function(metrics_df, metric_col, gse_metadata, order_list) {
  
  df_plot <- metrics_df %>%
    rowwise() %>%
    mutate(train_source = if (n_studies == 0) NA_character_ else {
      order_list[[test_source]]$train_source[n_studies]
    }) %>%
    ungroup() %>%
    filter(!is.na(train_source)) %>%
    left_join(gse_metadata, by = c("train_source" = "gse_id")) %>%
    group_by(test_source, adjuster, n_studies, train_source) %>%
    summarise(
      mean_val = mean(.data[[metric_col]], na.rm = TRUE),
      se_val = sd(.data[[metric_col]], na.rm = TRUE) / sqrt(n()),
      technology = first(technology),
      .groups = "drop"
    )
  
  # Create consistent dataset labels
  all_datasets <- unique(c(df_plot$train_source, df_plot$test_source))
  dataset_labels <- setNames(LETTERS[seq_along(all_datasets)], all_datasets)
  df_plot <- df_plot %>% mutate(train_label = dataset_labels[train_source])
  
  # Facet labels
  facet_labels <- setNames(
    paste0(names(dataset_labels), " (", dataset_labels, ")"),
    names(dataset_labels)
  )
  
  list(df_plot = df_plot, dataset_labels = dataset_labels, facet_labels = facet_labels)
}

# Generate ggplot
generate_plot <- function(df_plot, metric_col, facet_labels, cv_value = NULL) {
  
  p <- ggplot(
    df_plot,
    aes(
      x = reorder_within(train_source, n_studies, test_source),
      y = mean_val,
      color = adjuster,
      shape = technology,
      group = adjuster
    )
  ) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
                  width = 0.2) +
    scale_x_reordered(labels = function(x) {
      gsub("__.*$", "", x) %>% { df_plot$train_label[.] }
    }) +
    facet_wrap(~ test_source, scales = "free_x", labeller = labeller(test_source = facet_labels)) +
    labs(
      x = "Training dataset added (ordered)",
      y = metric_col,
      color = "Adjuster",
      shape = "Technology"
    ) +
    theme_minimal(base_size = 14) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_text(hjust = 1))
  
  # Add CV line if provided
  if (!is.null(cv_value)) {
    p <- p + geom_hline(yintercept = cv_value, linetype = "dashed", color = "black") +
      annotate("text", x = 1, y = cv_value, label = paste0("cv = ", cv_value),
               vjust = -0.5, hjust = 0)
  }
  
  return(p)
}

# Get CV value for a metric
get_cv_value <- function(cv_data, test_source, metric) {
  if (is.null(cv_data)) return(NULL)
  row <- cv_data %>% filter(test_source == !!test_source)
  if (nrow(row) == 0 || !(metric %in% colnames(row))) return(NULL)
  row[[metric]][1]
}

# --- Main Processing ---
main <- function(metrics_file, metadata_file, order_folder, figures_dir, cv_file = NULL) {
  
  dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
  
  all_metrics <- read_csv(metrics_file, show_col_types = FALSE) %>% clean_metrics()
  gse_metadata <- read_csv(metadata_file, show_col_types = FALSE) %>% clean_metadata()
  df_unadj <- all_metrics %>% filter(adjuster == "log_transformed")
  all_adjusters <- all_metrics %>% filter(adjuster != "log_transformed")
  
  cv_data <- if (!is.null(cv_file)) read_csv(cv_file, col_types = cols()) else NULL
  
  # Read all order files once
  test_sources <- unique(all_metrics$test_source)
  order_list <- read_order_files(order_folder, test_sources)
  
  results <- data.frame()
  
  for (metric in c("ROC_AUC", "MCC")) {
    # Compute delta vs unadjusted
    delta_df <- compute_delta(all_adjusters, df_unadj, metric) %>% mutate(Metric = metric)
    results <- bind_rows(results, delta_df)
    
    # Prepare absolute metrics for plotting
    plot_metrics <- all_metrics %>%
      group_by(adjuster, n_studies, test_source) %>%
      summarise(
        mean_metric = mean(.data[[metric]], na.rm = TRUE),
        .groups = "drop"
      )
    
    # Use first test_source for CV line
    cv_value <- get_cv_value(cv_data, unique(plot_metrics$test_source)[1], metric)
    
    plot_prep <- prepare_plot_data(plot_metrics, "mean_metric", gse_metadata, order_list)
    p <- generate_plot(plot_prep$df_plot, metric, plot_prep$facet_labels, cv_value)
    
    ggsave(file.path(figures_dir, paste0("scaling_", metric, ".png")), p, width = 12, height = 8)
  }
  
  write_csv(results, file.path(figures_dir, "scaling_comparison_results.csv"))
  cat("✅ Scaling plots generated successfully in:", figures_dir, "\n")
}

# --- Argument Parsing ---
parser <- ArgumentParser(description = "Generate performance plots from aggregated data.")
parser$add_argument('--metrics_file', required = TRUE, help = "Path to CSV containing aggregated metrics")
parser$add_argument('--figures_dir', required = TRUE, help = "Directory to save figures")
parser$add_argument('--metadata_file', required = TRUE, help = "Path to CSV containing dataset metadata")
parser$add_argument('--cv_file', required = FALSE, default = NULL, help = "Path to CSV containing CV metrics")
parser$add_argument('--order_folder', required = TRUE, help = "Folder containing training order files")
args <- parser$parse_args()

# --- Run main ---
main(args$metrics_file, args$metadata_file, args$order_folder, args$figures_dir, args$cv_file)