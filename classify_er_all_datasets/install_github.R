if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

if  (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

pkgs <- list(
  polyester = "alyssafrazee/polyester"
)

# Bioconductor-only packages (install if missing)
bioc_needed <- c("GenomeInfoDbData", "fairadapt")
for (b in bioc_needed) {
  if (!requireNamespace(b, quietly = TRUE)) {
    message("Installing Bioconductor package: ", b)
    tryCatch(
      BiocManager::install(b, update = FALSE, ask = FALSE),
      error = function(e) message("Bioc install failed for ", b, ": ", e$message)
    )
  } else {
    message("Bioconductor package ", b, " already installed")
  }
}

# GitHub packages (install only if missing)
for (pkg in names(pkgs)) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing GitHub package ", pkg, " from ", pkgs[[pkg]])
    tryCatch(
      remotes::install_github(pkgs[[pkg]], upgrade = "never"),
      error = function(e) message("Failed to install ", pkg, ": ", e$message)
    )
  } else {
    message("Package ", pkg, " already installed")
  }
}
