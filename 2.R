# Start a new project
rm(list = ls())  # Clear environment
cat("\014")      # Clear console
if (!is.null(dev.list())) dev.off()  # Clear all plots

# Set global options
options(scipen = 999)
options(stringsAsFactors = FALSE)

# Set working directory
setwd("C:/ALY6110/DPRS")

# Install catboost from the official CRAN-like repository
install.packages('catboost', repos = 'https://cloud.r-project.org/')


# Load the library
library(catboost)
