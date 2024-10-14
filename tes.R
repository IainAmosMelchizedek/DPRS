# Start a new project
rm(list = ls())  # Clear environment
cat("\014")      # Clear console
if (!is.null(dev.list())) dev.off()  # Clear all plots

# Set global options
options(scipen = 999)
options(stringsAsFactors = FALSE)

# Set working directory
setwd("C:/ALY6110/DPRS")  # Set the working directory to your project folder

# Load required libraries
install.packages(catboost)

library(catboost)
library(caret)
library(dplyr)

# Function to load and prepare the data
load_and_prepare_data <- function(file_path) {
  df <- read.csv(file_path)  # Loading the dataset from the precise location
  
  # Select the target and feature columns
  target <- df$price
  features <- df[, c('genres', 'categories', 'tags', 'developers', 'supported_languages', 
                     'achievements', 'windows', 'mac', 'linux', 
                     'release_month', 'release_year', 'publishers')]
  
  return(list(df = df, target = target, features = features))
}

# Function to split the data into training, validation, and test sets
split_data <- function(df, target, features, seed = 42) {
  set.seed(seed)
  
  # 70% training, 15% validation, 15% testing
  train_index <- createDataPartition(target, p = 0.7, list = FALSE)
  train_data <- df[train_index, ]
  remaining_data <- df[-train_index, ]
  
  # 50% of remaining data for validation, 50% for testing
  val_index <- createDataPartition(remaining_data$price, p = 0.5, list = FALSE)
  validation_data <- remaining_data[val_index, ]
  test_data <- remaining_data[-val_index, ]
  
  return(list(train_data = train_data, validation_data = validation_data, test_data = test_data))
}

# Function to create CatBoost pools for training, validation, and testing
create_catboost_pools <- function(train_data, validation_data, test_data, features, categorical_features) {
  train_pool <- catboost.load_pool(data = train_data[, features], label = train_data$price, cat_features = categorical_features)
  validation_pool <- catboost.load_pool(data = validation_data[, features], label = validation_data$price, cat_features = categorical_features)
  test_pool <- catboost.load_pool(data = test_data[, features], label = test_data$price, cat_features = categorical_features)
  
  return(list(train_pool = train_pool, validation_pool = validation_pool, test_pool = test_pool))
}

# Function to train the CatBoost model
train_catboost_model <- function(train_pool, validation_pool, categorical_features) {
  params <- list(
    loss_function = 'RMSE',
    eval_metric = 'RMSE',
    iterations = 1000,
    depth = 6,
    learning_rate = 0.1,
    cat_features = categorical_features
  )
  
  model <- catboost.train(train_pool, params = params, test_pool = validation_pool)
  return(model)
}

# Function to evaluate the model
evaluate_model <- function(model, test_pool, test_data) {
  # Make predictions
  y_pred <- catboost.predict(model, test_pool)
  
  # Calculate RMSE
  rmse <- sqrt(mean((y_pred - test_data$price)^2))
  cat("RMSE:", rmse, "\n")
  
  # Calculate R²
  r_squared <- 1 - (sum((test_data$price - y_pred)^2) / sum((test_data$price - mean(test_data$price))^2))
  cat("R²:", r_squared, "\n")
}

# Main function to execute the workflow
run_catboost_pipeline <- function() {
  # Load and prepare data
  data <- load_and_prepare_data("C:/ALY6110/DPRS/paid_games_encoded.csv")  # Loading from the precise dataset location
  df <- data$df
  target <- data$target
  features <- data$features
  
  # Categorical feature indices for CatBoost
  categorical_features <- c('genres', 'categories', 'tags', 'developers', 'supported_languages', 'publishers')
  
  # Split the data into training, validation, and test sets
  split <- split_data(df, target, features)
  train_data <- split$train_data
  validation_data <- split$validation_data
  test_data <- split$test_data
  
  # Create CatBoost pools
  pools <- create_catboost_pools(train_data, validation_data, test_data, features, categorical_features)
  
  # Train the model
  model <- train_catboost_model(pools$train_pool, pools$validation_pool, categorical_features)
  
  # Evaluate the model
  evaluate_model(model, pools$test_pool, test_data)
  
  # Save the model
  catboost.save_model(model, "catboost_price_prediction_model.bin")
}

# Execute the pipeline
run_catboost_pipeline()
