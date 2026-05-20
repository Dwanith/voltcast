
# src/modelling/train_xgboost.py

import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# --- 1. SETUP: Dynamic Paths and Parameters ---

# CRITICAL FIX: Define paths relative to the project root for Dockerization/Deployment
# Assumes this script is in src/modelling/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define Features/Target (Must match pre-processing - 9 total features)
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]
TARGET = "demand_mwh"

# --- 2. Data Loading (Full 2D Data) ---

print("--- 1. Data Loading ---")
try:
    df_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "final_train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test_scaled.csv"))

    # Load Scalers for inverse transformation (metrics interpretation)
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required files. Please check paths and re-run preprocessing.py. Error: {e}")
    exit()

# CRITICAL FIX: Use the full 2D data. DO NOT trim with SEQUENCE_LENGTH.
X_train = df_train[ALL_FEATURES].values
Y_train = df_train[TARGET].values

X_test_scaled = df_test[ALL_FEATURES].values
Y_test_scaled = df_test[TARGET].values
# We also save the 'time' column for visualization, if present in test_scaled.csv
time_index = df_test['time'] if 'time' in df_test.columns else None

print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test_scaled.shape}")


# --- 3. Evaluation Function ---

def evaluate_and_invert(Y_true_scaled, Y_pred_scaled, model_name, scaler_y):
    """Calculates metrics and inverts the scaled demand_mwh back to MW."""

    # Reshape and inverse transform
    Y_true = scaler_y.inverse_transform(Y_true_scaled.reshape(-1, 1)).flatten()
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate Metrics (on unscaled data)
    mae = mean_absolute_error(Y_true, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    r2 = r2_score(Y_true, Y_pred)

    print(f"\n--- {model_name} Test Metrics (Unscaled MW) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f} MW")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f} MW")
    print(f"  R-squared (R2):                 {r2:.4f}")

    return Y_true, Y_pred, mae, rmse, r2


# --- 4. XGBoost Model Training ---

print("\n--- 2. Training XGBoost Regressor (Final Model) ---")

# Define Hyperparameters (Reasonable defaults for a structured ML project)
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1, # Use all available cores
    'early_stopping_rounds': 50 # Stops if validation score doesn't improve
}

start_time = time.time()
xgb_model = XGBRegressor(**XGBOOST_PARAMS)

# Fit the model. Note: We use the full training set (X_train, Y_train)
xgb_model.fit(
    X_train, Y_train,
    # Use a small portion of the training data as an internal validation set
    # to monitor performance and trigger early stopping (if desired)
    eval_set=[(X_test_scaled, Y_test_scaled)],
    verbose=False
)
training_time_xgb = time.time() - start_time

print(f"XGBoost Training Time: {training_time_xgb:.2f} seconds")


# --- 5. Prediction and Evaluation ---

print("\n--- 3. Prediction and Evaluation ---")
Y_pred_xgb_scaled = xgb_model.predict(X_test_scaled)

# Evaluate metrics using the unscaled values
Y_true_xgb, Y_pred_xgb, mae_xgb, rmse_xgb, r2_xgb = evaluate_and_invert(
    Y_test_scaled, Y_pred_xgb_scaled, "XGBoost", scaler_y
)


# --- 6. Model Saving ---

model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
joblib.dump(xgb_model, model_path)
print(f"\nSUCCESS: Final XGBoost model saved to {model_path}")

print("\n--- Training Script Finished ---")