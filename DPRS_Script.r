# Start a new project
rm(list = ls())  # Clear environment
cat("\014")      # Clear console
if (!is.null(dev.list())) dev.off()  # Clear all plots

# Set global options
options(scipen = 999)
options(stringsAsFactors = FALSE)

# Set working directory
setwd("C:/ALY6110/DPRS")

# Load required libraries (install if missing)
required_packages <- c("dplyr", "ggplot2", "readr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
lapply(required_packages, library, character.only = TRUE)

# Load the games dataset
games_data <- read_csv("games_v1.csv")

# Preview the data
head(games_data)

# Set seed for reproducibility
set.seed(123)

########
# Split the dataset into free and paid games
free_games <- games_data %>% filter(price == 0)
paid_games <- games_data %>% filter(price > 0)

# Preview the split datasets
head(free_games)
head(paid_games)

# Save the split datasets into separate CSV files
write_csv(free_games, "C:/ALY6110/DPRS/free_games.csv")
write_csv(paid_games, "C:/ALY6110/DPRS/paid_games.csv")

# Confirmation message
cat("Datasets have been split and saved as 'free_games.csv' and 'paid_games.csv'.\n")


################################################
# Load required library for one-hot encoding
library(dplyr)

# Function for one-hot encoding and normalization
process_data <- function(data) {
  # One-Hot Encode the categorical columns: 'supported_languages', 'windows', 'mac', 'linux'
  data_encoded <- data %>%
    mutate(supported_languages = as.factor(supported_languages),
           windows = as.numeric(windows),
           mac = as.numeric(mac),
           linux = as.numeric(linux)) %>%
    # Create one-hot encoding for 'supported_languages'
    mutate(across(where(is.factor), ~ as.numeric(as.factor(.))))
  
  return(data_encoded)
}

# Apply to free games and paid games datasets
free_games_encoded <- process_data(free_games)
paid_games_encoded <- process_data(paid_games)

# Save the updated datasets with encoding and normalization
write_csv(free_games_encoded, "C:/ALY6110/DPRS/free_games_encoded.csv")
write_csv(paid_games_encoded, "C:/ALY6110/DPRS/paid_games_encoded.csv")

# Confirmation message
cat("Free and paid datasets have been one-hot encoded and saved.\n")

