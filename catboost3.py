# Optimized Code for Game Price Prediction Using CatBoost
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Load and inspect the dataset
file_path = 'C:/ALY6110/DPRS/paid_games.csv'
df = pd.read_csv(file_path)

# Handle missing values
df.fillna({'price': df['price'].median(), **{col: 'unknown' for col in df.select_dtypes('object').columns}}, inplace=True)

# Select features and target
features = df.columns.difference(['price'])
X = df[features]
y = df['price']

# Split data (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 2: Prepare CatBoost Pools
train_pool = Pool(X_train, y_train, cat_features=features)
val_pool = Pool(X_val, y_val, cat_features=features)

# Step 3: Train CatBoost model
model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=100)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

# Step 5: Save the model
model.save_model('catboost_game_price_model_v1.cbm')

# Step 6: Optional Feature Importance
feature_importance = model.get_feature_importance()
plt.barh(features, feature_importance)
plt.title('Feature Importance')
plt.show()
