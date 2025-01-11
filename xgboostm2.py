# -*- coding: utf-8 -*-
"""
Optimized Code for Predicting Game Prices Using XGBoost
Created on Thu Oct 24 13:00:22 2024

@author: Iaina (Optimized by Assistant)
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the pre-encoded dataset
data_path = 'C:/ALY6110/DPRS/paid_games_encoded.csv'  # Ensure this path is correct for your environment
data = pd.read_csv(data_path)

# Check for missing values and handle them (e.g., imputation or removal)
if data.isnull().any().any():
    print("Dataset contains missing values. Please handle them before proceeding.")
    # Example: data.fillna(0, inplace=True)  # Simple imputation

# Automatically identify categorical columns (object or category types)
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

# Convert these columns to category dtype (if not already done)
for col in categorical_cols:
    data[col] = data[col].astype('category')

# Define the features and target
X = data.drop(columns=['price'])  # Assuming 'price' is the target column
y = data['price']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=None)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=None)

# Initialize the XGBoost model with improvements
xgboost_model = xgb.XGBRegressor(
    n_estimators=1000,         # High number of trees
    learning_rate=0.1,         # Moderate learning rate
    max_depth=6,               # Depth of each tree
    subsample=0.8,             # Fraction of data to sample for each tree
    colsample_bytree=0.8,      # Fraction of features to sample for each tree
    enable_categorical=True,   # Enable categorical feature support
    n_jobs=-1,                 # Utilize all CPU cores for faster training
    random_state=42            # Ensure reproducibility
)

# Train the model with early stopping
xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # Stop training if no improvement in 50 rounds
    verbose=True
)

# Make predictions on the test set
y_pred = xgboost_model.predict(X_test)

# Calculate performance metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Output the performance metrics
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

# Save the trained model in binary format for better performance
xgboost_model.save_model('xgboost_model.bin')