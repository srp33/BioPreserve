# complex_heatmap.R
#
# This script creates 2x2 heatmaps of AUC scores and Matthews Correlation Coefficient (MCC)
# for dataset combinations from the ER classification results CSV file.
# Usage: Rscript train_test_heatmap.R 
# Example: Rscript train_test_heatmap.R 

# Load libraries
library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ComplexHeatmap)
library(circlize)
library(tibble)
library(purrr)

# --- Configuration ---
CONFIG <- list(
  metrics_dir = "/outputs/metrics",
  figures_dir = "/outputs/figures",
  metadata_file = "/scripts/evaluations/geo_metadata.csv",
  pattern = "^er_classification_.*\\.csv$",
  adjuster_pattern = "^er_classification_(.+)\\.csv$",  # Pattern to extract adjuster names
  unadjusted_file = "er_classification_unadjusted.csv"
)

# --- Auto-detect Available Adjusters ---
discover_adjusters <- function(config) {
  valid_adjusters <- list.files(config$metrics_dir, pattern = config$pattern, full.names = TRUE) %>%
    .[sapply(., has_meaningful_data)] %>%
    basename() %>%
    gsub(config$adjuster_pattern, "\\1", .)
  
  if (length(valid_adjusters) == 0) {
    stop("No adjusters found with meaningful data.", call. = FALSE)
  }
  
  cat("Found adjusters:", paste(valid_adjusters, collapse = ", "), "\n")
  return(valid_adjusters)
}

# Helper function to construct adjuster file paths
get_adjuster_file_path <- function(adjuster, config) {
  file.path(config$metrics_dir, paste0("er_classification_", adjuster, ".csv"))
}

# Function to check if a CSV file has meaningful data
has_meaningful_data <- function(csv_file) {
  tryCatch({
    data <- read_csv(csv_file, show_col_types = FALSE)
    return(nrow(data) > 0)
  }, error = function(e) {
    return(FALSE)
  })
}

# Initialize configuration
adjusters <- discover_adjusters(CONFIG)
FIG_DIR <- CONFIG$figures_dir

# Load platform metadata
platform_df <- read.csv(CONFIG$metadata_file)
platform_df$platform <- trimws(platform_df$platform)
dataset_to_platform <- setNames(platform_df$platform, platform_df$GSE_ID)

# --- Helper Functions ---

# Function to calculate Matthews Correlation Coefficient (vectorized)
calculate_mcc <- function(tp, tn, fp, fn) {
  numerator <- (tp * tn) - (fp * fn)
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  # Handle division by zero (vectorized)
  result <- ifelse(denominator == 0, 0, numerator / denominator)
  
  return(result)
}

prepare_metric_matrix <- function(metric_data, metric_col) {
  # Check if metric_data is empty
  if (nrow(metric_data) == 0) {
    cat("Warning: No data available for", metric_col, "\n")
    return(NULL)
  }
  
  all_datasets <- sort(union(metric_data$Train, metric_data$Test))
  
  # Check if we have any datasets
  if (length(all_datasets) == 0) {
    cat("Warning: No datasets found for", metric_col, "\n")
    return(NULL)
  }
  
  # Ensure metric_data is unique for (Train, Test)
  metric_data_unique <- metric_data %>%
    group_by(Train, Test) %>%
    summarise(Mean_Metric = mean(Mean_Metric, na.rm = TRUE), .groups = "drop")
  
  metric_matrix <- expand.grid(Train = all_datasets, Test = all_datasets) %>%
    left_join(metric_data_unique, by = c("Train", "Test")) %>%
    pivot_wider(names_from = Test, values_from = Mean_Metric) %>%
    column_to_rownames("Train") %>%
    as.matrix()
  
  # Check if matrix is valid
  if (nrow(metric_matrix) == 0 || ncol(metric_matrix) == 0) {
    cat("Warning: Empty matrix created for", metric_col, "\n")
    return(NULL)
  }
  
  cat("Created matrix for", metric_col, "with dimensions:", nrow(metric_matrix), "x", ncol(metric_matrix), "\n")
  return(metric_matrix)
}

# Function to read and prepare data
read_and_prepare_data <- function(csv_file) {
  cat("Reading data from:", csv_file, "\n")
  input_data <- tryCatch({
    read_csv(csv_file, show_col_types = FALSE)
  }, error = function(e) {
    message(paste("Error reading file:", csv_file))
    message("Please ensure the CSV file exists in the current directory.")
    return(NULL)
  })
  
  if (is.null(input_data)) {
    stop("Aborting script due to missing or invalid input file.", call. = FALSE)
  }

  # Check if data is empty or only contains headers
  if (nrow(input_data) == 0) {
    warning(paste("Input CSV file", csv_file, "is empty. No data to process."))
    return(data.frame())
  }

  input_data <- as.data.frame(input_data)

  # --- 🔧 Add MCC calculation ---
  input_data$MCC <- calculate_mcc(
    tp = input_data$`True Positive`,
    tn = input_data$`True Negative`,
    fp = input_data$`False Positive`,
    fn = input_data$`False Negative`
  )

  return(input_data)
}

# Process metric data for both regular and diagonal delta
process_metric_data <- function(df_adj, df_unadj = NULL, metric_col, is_diagonal_delta = FALSE) {
  if (is_diagonal_delta) {
    return(prepare_diagonal_delta_metric_data(df_adj, df_unadj, metric_col))
  } else if (!is.null(df_unadj)) {
    return(prepare_delta_metric_data(df_adj, df_unadj, metric_col))
  } else {
    return(prepare_metric_data(df_adj, metric_col))
  }
}

# Create and draw heatmaps
create_heatmap_for_metric <- function(df_adj, df_unadj, metric_col, adjuster, train_combined, is_diagonal_delta = FALSE) {
  metric_data <- process_metric_data(df_adj, df_unadj, metric_col, is_diagonal_delta)
  metric_matrix <- prepare_metric_matrix(metric_data, metric_col)
  
  if (!is.null(metric_matrix)) {
    is_difference <- !is.null(df_unadj) || is_diagonal_delta
    return(draw_heatmap(metric_matrix, metric_col, adjuster, train_combined, is_difference, is_diagonal_delta))
  }
  return(NULL)
}

# Function to filter data (common filtering logic)
filter_datasets <- function(input_data, train_combined) {
  if (train_combined) {
    result <- input_data %>%
      # Use metrics for models trained on both datasets, plus the diagonal
      filter(str_detect(Train, ";") | (Train == Test)) %>%
      filter(!str_detect(Test, ";")) %>%
      mutate(
        Train = map2_chr(Train, Test, function(train_val, test_val) {
          if (str_detect(train_val, ";")) {
            train_parts <- str_split(train_val, ";")[[1]]
            # Return the part that's NOT the test value
            train_parts[train_parts != test_val][1]
          } else {
            train_val
          }
        })
      )
    return(result)
  } else {
    # Use cross-training data
    return(input_data %>%
      filter(!str_detect(Train, ";")) %>%
      filter(!str_detect(Test, ";")))
  }
}

# Function to prepare metric data
prepare_metric_data <- function(df, metric_col) {
  df %>%
    group_by(Train, Test) %>%
    summarise(Mean_Metric = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
}

prepare_delta_metric_data <- function(df_adj, df_unadj, metric_col) {
  data_adj <- prepare_metric_data(df_adj, metric_col) %>% rename(Adj = Mean_Metric)
  data_unadj <- prepare_metric_data(df_unadj, metric_col) %>% rename(Unadj = Mean_Metric)

  full_join(data_adj, data_unadj, by = c("Train", "Test")) %>%
    mutate(Mean_Metric = Adj - Unadj)
}

prepare_diagonal_delta_metric_data <- function(df_adj, df_unadj, metric_col) {
  data_adj <- df_adj %>%
    group_by(Train, Test) %>%
    summarise(Adj = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")

  # Extract diagonal values from unadjusted data (where Train == Test)
  diagonal <- df_unadj %>%
    group_by(Train, Test) %>%
    summarise(Unadj = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop") %>%
    filter(Train == Test) %>%
    select(Test, Diagonal = Unadj)
  
  # Join diagonal values and calculate difference
  data_adj %>%
    left_join(diagonal, by = "Test") %>%
    mutate(Mean_Metric = Adj - Diagonal) %>%
    select(Train, Test, Mean_Metric)
}

get_platform_annotations <- function(datasets) {
  platforms <- dataset_to_platform[datasets]
  platforms[is.na(platforms)] <- "Unknown"

  platform_type_map <- c(
    "Affymetrix Human Genome U133 Plus 2.0 Array" = "Microarray",
    "Affymetrix Human Genome U133A Array" = "Microarray",
    "Affymetrix Human Gene 1.0 ST Array [transcript (gene) version]" = "Microarray",
    "Affymetrix Human Transcriptome Array 2.0 [probe set (exon) version] / Custom" = "Microarray",
    "Illumina HumanHT-12 V3.0 expression beadchip" = "Microarray",
    "Illumina Genome Analyzer (Homo sapiens)" = "RNAseq",
    "Illumina HiSeq 2000 (Homo sapiens)" = "RNAseq",
    "Illumina NextSeq 500 (Homo sapiens)" = "RNAseq",
    "Illumina HiSeq 2500 (Homo sapiens)" = "RNAseq"
  )

  types <- platform_type_map[platforms]
  types[is.na(types)] <- "Unknown"

  splits <- list(
    factor(types, levels = c("Microarray", "RNAseq", "Unknown")),
    factor(platforms, levels = unique(platforms))
  )

  colors <- c(
    "Affymetrix Human Genome U133 Plus 2.0 Array" = "#56B4E9FF",
    "Affymetrix Human Genome U133A Array" = "#009E73FF",
    "Illumina HiSeq 2000 (Homo sapiens)" = "#E69F00FF",
    "Illumina NextSeq 500 (Homo sapiens)" = "#D55E00FF",
    "Illumina HiSeq 2500 (Homo sapiens)" = "#CC79A7FF",
    "Illumina Genome Analyzer (Homo sapiens)" = "#F0E442FF",
    "Affymetrix Human Gene 1.0 ST Array [transcript (gene) version]" = "#0072B2FF",
    "Affymetrix Human Transcriptome Array 2.0 [probe set (exon) version] / Custom" = "#4682B4FF",
    "Illumina HumanHT-12 V3.0 expression beadchip" = "#6A9FB5FF",
    "Unknown" = "#000000"
  )

  annotation <- list(
    splits = splits,
    colors = colors,
    platforms = platforms
  )

  return(annotation)
}

draw_heatmap <- function(data_matrix, metric_col, adjuster, train_combined, is_difference = FALSE, is_diagonal_delta = FALSE) {
  # Check if matrix is NULL or empty
  if (is.null(data_matrix) || nrow(data_matrix) == 0 || ncol(data_matrix) == 0) {
    cat("Skipping heatmap for", metric_col, "- no valid data matrix\n")
    return(NULL)
  }
  
  row_anno_info <- get_platform_annotations(rownames(data_matrix))
  col_anno_info <- get_platform_annotations(colnames(data_matrix))

  row_ha <- rowAnnotation(
    Platform = row_anno_info$platforms,
    col = list(Platform = row_anno_info$colors),
    show_annotation_name = FALSE
  )

  col_ha <- HeatmapAnnotation(
    Platform = col_anno_info$platforms,
    col = list(Platform = col_anno_info$colors),
    show_annotation_name = FALSE
  )

  # Fix scale to [-1, 1] for all heatmaps
  col_fun <- circlize::colorRamp2(c(-1, 0, 1), c("#D62728", "#FFFFFF", "#2CA02C"))

  legend_title <- if (is_difference) {
    paste0("Diff ", metric_col)
  } else {
    metric_col
  }

  train_text <- if (train_combined) {
    " (Trained on Combined)"
  } else {
    " (Cross-Trained)"
  }

  title_text <- if (is_difference) {
    if (is_diagonal_delta) {
      paste0("Diff ", metric_col, ": ", adjuster, " - diagonal", train_text)
    } else {
      paste0("Diff ", metric_col, ": ", adjuster, " - unadjusted", train_text)
    }
  } else {
    paste0(metric_col, ": Dataset Combinations (", adjuster, ")", train_text)
  }

  # Optional: If metric is AUC, rescale the matrix before plotting
  if (metric_col == "ROC AUC" && !is_difference) {
    # Rescale AUC from [0,1] to [-1,1]
    data_matrix <- 2 * (data_matrix - 0.5)
  }

  row_title <- if (train_combined) {
    "Training Dataset, in Combination with Test"
  } else {
    "Train Dataset"
  }

  ht <- Heatmap(data_matrix,
                name = legend_title,
                col = col_fun,
                na_col = "white",
                row_split = row_anno_info$splits,
                column_split = col_anno_info$splits,
                top_annotation = col_ha,
                left_annotation = row_ha,
                cluster_rows = FALSE,
                cluster_columns = FALSE,
                show_row_names = TRUE,
                row_names_gp = gpar(fontsize = 9),
                show_column_names = TRUE,
                column_names_gp = gpar(fontsize = 9),
                column_names_rot = 45,
                column_title = title_text,
                column_title_gp = gpar(fontsize = 14, fontface = "bold"),
                row_title = row_title,
                row_title_gp = gpar(fontsize = 12),
                heatmap_legend_param = list(title = legend_title),
                heatmap_width = unit(1, "npc"),
                heatmap_height = unit(1, "npc"),
                cell_fun = function(j, i, x, y, width, height, fill) {
                  val <- data_matrix[i, j]
                  if (!is.na(val)) {
                    grid.text(sprintf("%.2f", val), x, y,
                              gp = gpar(fontsize = 8, col = ifelse(abs(val) > 0.5, "white", "black")))
                  }
                })

  return(ht)
}

# Function to calculate percentage of times each adjuster performs best
calculate_best_performer_percentages <- function(all_diff_data) {
  # Exclude diagonal entries (where Train == Test) for percentage calculation
  off_diagonal_data <- all_diff_data %>%
    filter(Train != Test)
  
  # For each dataset combination (Train, Test) and metric, find which adjuster performs best
  best_performers <- off_diagonal_data %>%
    group_by(Train, Test, Metric) %>%
    filter(Mean_Metric == max(Mean_Metric, na.rm = TRUE)) %>%
    ungroup()
  
  # Calculate percentage of times each adjuster is best (excluding diagonal)
  total_combinations <- off_diagonal_data %>%
    select(Train, Test, Metric) %>%
    distinct() %>%
    nrow()
  
  percentages <- best_performers %>%
    count(Adjuster, Metric) %>%
    group_by(Metric) %>%
    mutate(
      total_for_metric = sum(n),
      percentage = round((n / total_for_metric) * 100, 1)
    ) %>%
    ungroup() %>%
    select(Adjuster, Metric, percentage)
  
  return(percentages)
}

# Unified boxplot function (formerly jitter plot)
generate_jitter_plot <- function(all_diff_data, fig_dir, cross, plot_type = "regular") {
  # Filter out rows with missing values
  all_diff_data <- all_diff_data %>%
    filter(!is.na(Mean_Metric), is.finite(Mean_Metric))
  
  if (nrow(all_diff_data) == 0) {
    cat("No valid data for", plot_type, "boxplot\n")
    return()
  }
  
  # Calculate percentages of best performance
  percentages <- calculate_best_performer_percentages(all_diff_data)
  
  # Configure plot based on type
  if (plot_type == "diagonal") {
    title <- "Distribution of Diagonal Delta Metrics for Adjusters"
    subtitle <- "Difference from diagonal (Train == Test) performance"
    y_label <- "Difference from Diagonal Performance"
    filename_prefix <- "diagonal_delta_boxplot"
  } else {
    title <- "Distribution of Metric Differences for Adjusters"
    subtitle <- NULL
    y_label <- "Difference in Metric (Adjusted - Unadjusted)"
    filename_prefix <- "boxplot"
  }
  
  # Create the main boxplot
  p <- ggplot(all_diff_data, aes(x = Adjuster, y = Mean_Metric, color = Metric)) +
    geom_boxplot(outlier.shape = 16, outlier.size = 2, outlier.alpha = 0.7, 
                 alpha = 0.7, width = 0.6) +
    scale_color_manual(values = c("MCC" = "skyblue", "AUC" = "orange")) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.3, alpha = 0.8) +
    theme_minimal() +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Adjuster",
      y = y_label,
      color = "Metric"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      text = element_text(size = 12)
    ) +
    facet_wrap(~ Metric, scales = "free_y")
  
  # Add percentage labels on top of boxplots
  if (nrow(percentages) > 0) {
    # Get y-axis limits for positioning labels
    y_limits <- all_diff_data %>%
      group_by(Metric) %>%
      summarise(
        max_y = max(Mean_Metric, na.rm = TRUE),
        min_y = min(Mean_Metric, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(label_y = max_y + (max_y - min_y) * 0.1)
    
    # Merge percentages with y-axis positioning
    label_data <- percentages %>%
      left_join(y_limits, by = "Metric") %>%
      mutate(label_text = paste0(percentage, "%"))
    
    p <- p + 
      geom_text(data = label_data, 
                aes(x = Adjuster, y = label_y, label = label_text, color = Metric),
                size = 3, fontface = "bold", vjust = 0)
  }

  cross_suffix = ifelse(cross, "_cross", "")
  file_path <- file.path(fig_dir, paste0(filename_prefix, cross_suffix, ".png"))
  ggsave(file_path, plot = p, width = 10, height = 6)
  cat(paste(plot_type, "boxplot saved to:"), file_path, "\n")
  
  # Print percentage summary
  cat("\nPercentage of times each adjuster performed best (excluding diagonal Train==Test):\n")
  print(percentages)
}

generate_all_heatmaps_to_pdf <- function(adjuster, train_combined, fig_dir = "/outputs/figures") {
  file_adjusted <- get_adjuster_file_path(adjuster, CONFIG)
  file_unadjusted <- file.path(CONFIG$metrics_dir, CONFIG$unadjusted_file)

  df_adj <- read_and_prepare_data(file_adjusted) %>% filter_datasets(train_combined)
  df_unadj <- read_and_prepare_data(file_unadjusted) %>% filter_datasets(train_combined)

  # Check if we have data after filtering
  if (nrow(df_adj) == 0 || nrow(df_unadj) == 0) {
    cat("Skipping", adjuster, "- no data after filtering for train_combined =", train_combined, "\n")
    return()
  }

  # Define heatmap configurations
  heatmap_configs <- list(
    list(name = "delta_mcc", metric = "MCC", use_unadj = TRUE, diagonal = FALSE),
    list(name = "delta_auc", metric = "ROC AUC", use_unadj = TRUE, diagonal = FALSE),
    list(name = "diag_delta_mcc", metric = "MCC", use_unadj = FALSE, diagonal = TRUE),
    list(name = "diag_delta_auc", metric = "ROC AUC", use_unadj = FALSE, diagonal = TRUE),
    list(name = "mcc", metric = "MCC", use_unadj = FALSE, diagonal = FALSE),
    list(name = "auc", metric = "ROC AUC", use_unadj = FALSE, diagonal = FALSE)
  )

  # Generate heatmaps using unified function
  heatmap_list <- list()
  for (config in heatmap_configs) {
    # Diagonal delta calculations always need unadjusted data
    df_unadj_param <- if (config$use_unadj || config$diagonal) df_unadj else NULL
    heatmap_list[[config$name]] <- create_heatmap_for_metric(
      df_adj, df_unadj_param, config$metric, adjuster, train_combined, config$diagonal
    )
  }

  # Filter out NULL heatmaps
  heatmap_list <- heatmap_list[!sapply(heatmap_list, is.null)]
  
  if (length(heatmap_list) == 0) {
    cat("No valid heatmaps to save for", adjuster, "with train_combined =", train_combined, "\n")
    return()
  }

  ## Save all to one PDF
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
  combined_suffix = ifelse(train_combined, "_train_combined", "")
  pdf_file <- file.path(fig_dir, paste0("combined_heatmaps_", adjuster, combined_suffix, ".pdf"))
  pdf(pdf_file, width = 14, height = 8)

  for (ht in heatmap_list) {
    if (!is.null(ht)) {
      draw(ht, padding = unit(c(10, 10, 10, 10), "mm"),
           merge_legend = TRUE,
           heatmap_legend_side = "right",
           annotation_legend_side = "right")
      grid.text("Test Dataset", x = unit(0.3, "npc"), y = unit(0.02, "npc"), gp = gpar(fontsize = 14))
      grid.newpage()
    }
  }

  dev.off()
  cat("All heatmaps saved to:", pdf_file, "\n")
}

all_adjuster_diffs <- data.frame()
all_adjuster_diffs_cross <- data.frame()
all_diag_diffs <- data.frame()
all_diag_diffs_cross <- data.frame()

# Function to process data for a single adjuster and train_combined setting
process_adjuster_data <- function(adjuster, train_combined, file_unadjusted) {
  file_adjusted <- get_adjuster_file_path(adjuster, CONFIG)
  
  df_adj <- read_and_prepare_data(file_adjusted) %>% filter_datasets(train_combined)
  df_unadj <- read_and_prepare_data(file_unadjusted) %>% filter_datasets(train_combined)

  cat("After filtering - df_adj rows:", nrow(df_adj), ", df_unadj rows:", nrow(df_unadj), "\n")

  if (nrow(df_adj) == 0 || nrow(df_unadj) == 0) {
    cat("Skipping due to empty data after filtering\n")
    return(list(regular_diffs = NULL, diag_diffs = NULL))
  }

  # Prepare metric data for both MCC and AUC
  metrics <- c("MCC", "ROC AUC")
  metric_labels <- c("MCC", "AUC")
  
  regular_diffs <- map2_dfr(metrics, metric_labels, function(metric, label) {
    prepare_delta_metric_data(df_adj, df_unadj, metric) %>%
      mutate(Adjuster = adjuster, Metric = label)
  })
  
  diag_diffs <- map2_dfr(metrics, metric_labels, function(metric, label) {
    prepare_diagonal_delta_metric_data(df_adj, df_unadj, metric) %>%
      mutate(Adjuster = adjuster, Metric = label)
  })

  return(list(regular_diffs = regular_diffs, diag_diffs = diag_diffs))
}

# Main processing function
process_all_adjusters <- function(adjusters, config) {
  file_unadjusted <- file.path(config$metrics_dir, config$unadjusted_file)
  
  # Initialize result containers
  results <- list(
    combined = list(regular = data.frame(), diagonal = data.frame()),
    cross = list(regular = data.frame(), diagonal = data.frame())
  )
  
  for (adjuster in adjusters) {
    cat("\n=== Processing adjuster:", adjuster, "===\n")
    
    for (train_combined in c(TRUE, FALSE)) {
      cat("Processing train_combined =", train_combined, "\n")
      
      # Process data
      result <- process_adjuster_data(adjuster, train_combined, file_unadjusted)
      
      # Accumulate results
      if (!is.null(result$regular_diffs)) {
        key <- if (train_combined) "combined" else "cross"
        results[[key]]$regular <- bind_rows(results[[key]]$regular, result$regular_diffs)
        results[[key]]$diagonal <- bind_rows(results[[key]]$diagonal, result$diag_diffs)
      }
      
      # Generate heatmaps
      tryCatch({
        generate_all_heatmaps_to_pdf(adjuster, train_combined, config$figures_dir)
      }, error = function(e) {
        cat("Error generating heatmaps for", adjuster, ":", e$message, "\n")
      })
    }
  }
  
  return(results)
}

# Execute main processing
results <- process_all_adjusters(adjusters, CONFIG)
all_adjuster_diffs <- results$combined$regular
all_diag_diffs <- results$combined$diagonal
all_adjuster_diffs_cross <- results$cross$regular
all_diag_diffs_cross <- results$cross$diagonal
 
# Generate all boxplots
boxplot_configs <- list(
  list(data = all_adjuster_diffs, cross = FALSE, type = "regular"),
  list(data = all_adjuster_diffs_cross, cross = TRUE, type = "regular"),
  list(data = all_diag_diffs, cross = FALSE, type = "diagonal"),
  list(data = all_diag_diffs_cross, cross = TRUE, type = "diagonal")
)

for (config in boxplot_configs) {
  generate_jitter_plot(config$data, FIG_DIR, config$cross, config$type)
}

cat("All heatmaps and boxplots generated successfully.\n")