# Start a new project
rm(list = ls())  # Clear environment
cat("\014")      # Clear console
if (!is.null(dev.list())) dev.off()  # Clear all plots

# Set global options
options(scipen = 999)
options(stringsAsFactors = FALSE)

# Set working directory to the folder where the dataset is located
setwd("C:/ALY6110/DPRS")  # Adjust this path if necessary

# Load required libraries
library(catboost)
library(caret)
library(dplyr)

# Load and prepare the dataset
print("Loading paid_games_encoded dataset...")
df <- read.csv("C:/ALY6110/DPRS/paid_games_encoded.csv")  # Load the dataset

# Check the dataset
print("Dataset loaded. Summary:")
print(dim(df))  # Print the dimensions of the dataset
print(head(df)) # Show the first few rows to verify the data

# Select the target (price) and features for training
target <- df$price
# Use column names instead of the data itself
feature_columns <- c('genres', 'categories', 'tags', 'developers', 
                     'supported_languages', 'achievements', 
                     'windows', 'mac', 'linux', 
                     'release_month', 'release_year', 'publishers')

# Categorical feature indices for CatBoost
categorical_features <- c('genres', 'categories', 'tags', 
                          'developers', 'supported_languages', 'publishers')

# Convert categorical columns to factors
df[categorical_features] <- lapply(df[categorical_features], as.factor)

# Split the data into training (70%), validation (15%), and testing (15%)
set.seed(42)

# First, split into training (70%) and remaining (30%)
train_index <- createDataPartition(target, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
remaining_data <- df[-train_index, ]

# Now split the remaining 30% into validation (15%) and test (15%)
val_index <- createDataPartition(remaining_data$price, p = 0.5, list = FALSE)
validation_data <- remaining_data[val_index, ]
test_data <- remaining_data[-val_index, ]

# Check if data was split correctly
print("Train data size:")
print(dim(train_data))
print("Validation data size:")
print(dim(validation_data))
print("Test data size:")
print(dim(test_data))

# Create CatBoost pools for training, validation, and testing
train_pool <- catboost.load_pool(data = train_data[, feature_columns], 
                                 label = train_data$price)

validation_pool <- catboost.load_pool(data = validation_data[, feature_columns], 
                                      label = validation_data$price)

test_pool <- catboost.load_pool(data = test_data[, feature_columns], 
                                label = test_data$price)

# Train the CatBoost model
print("Training the CatBoost model...")
params <- list(
  loss_function = 'RMSE',
  iterations = 1000,
  depth = 6,
  learning_rate = 0.1
)

model <- catboost.train(train_pool, params = params, test_pool = validation_pool)

# Make predictions and evaluate the model
print("Evaluating the model...")
preds <- catboost.predict(model, test_pool)

# Calculate RMSE
rmse <- sqrt(mean((preds - test_data$price)^2))
cat("RMSE:", rmse, "\n")

# Calculate R²
r_squared <- 1 - (sum((test_data$price - preds)^2) / sum((test_data$price - mean(test_data$price))^2))
cat("R²:", r_squared, "\n")

# Save the model for later use
