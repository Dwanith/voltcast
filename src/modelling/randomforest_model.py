import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import time  # For tracking training time






# --- 1. SETUP: Paths and Parameters ---

PROJECT_ROOT = os.environ.get(
    "APP_HOME",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
models_dir = os.path.join(PROJECT_ROOT, "models")
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

# --- HYPERPARAMETERS ---
SEQUENCE_LENGTH = 366
LSTM_EPOCHS = 75
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 128
RANDOM_FOREST_N_ESTIMATORS = 200

# Define Features/Target (Must match pre-processing - 9 total features)
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]
TARGET = "demand_mwh"
N_FEATURES = len(ALL_FEATURES)

# --- 2. Data Loading and Sequence Function ---
print("--- 1. Data Loading ---")
try:
    df_train = pd.read_csv(os.path.join(data_processed_dir, "final_train.csv"))
    df_val = pd.read_csv(os.path.join(data_processed_dir, "final_val.csv"))
    df_test = pd.read_csv(os.path.join(data_processed_dir, "test_scaled.csv"))
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required CSV files. Please re-run preprocessing.py. Error: {e}")
    exit()

# Separate features (X) and target (Y)
X_train_2d = df_train[ALL_FEATURES].values
Y_train_2d = df_train[TARGET].values

X_val_2d = df_val[ALL_FEATURES].values
Y_val_2d = df_val[TARGET].values

X_test_2d = df_test[ALL_FEATURES].values
Y_test_2d = df_test[TARGET].values

# Load Scalers for inverse transformation (metrics interpretation)
scaler_y = joblib.load(os.path.join(models_dir, "scaler_y.pkl"))


def create_sequences(X, Y, seq_len):
    """Converts 2D data into 3D sequences for LSTM input."""
    X_seq, Y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len, :])
        Y_seq.append(Y[i + seq_len])
    return np.array(X_seq), np.array(Y_seq)


# --- 3. Sequence Generation ---
print("--- 2. Sequence Generation (Preparing 3D Data for Random Forest) ---")
# Only sequence the data for the LSTM model
X_train_seq, Y_train_seq = create_sequences(X_train_2d, Y_train_2d, SEQUENCE_LENGTH)
X_val_seq, Y_val_seq = create_sequences(X_val_2d, Y_val_2d, SEQUENCE_LENGTH)
X_test_seq, Y_test_seq = create_sequences(X_test_2d, Y_test_2d, SEQUENCE_LENGTH)


# --- 4. Model Training and Evaluation Functions ---

def evaluate_and_invert(Y_true_scaled, Y_pred_scaled, model_name):
    """Calculates metrics and inverts the scaled demand_mwh back to MW."""

    # Inverse transform
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












#1. Random Forest model
# --- 5. Baseline Model: Random Forest Regressor ---
print("\n--- 3. Training Random Forest Regressor (Baseline) ---")

# RF is a non-sequential model, so it uses the 2D data, but we must align it with the sequence data
# We drop the first 24 hours (SEQUENCE_LENGTH) of 2D data to match the length of the sequenced data
X_train_rf = X_train_2d[SEQUENCE_LENGTH:]
Y_train_rf = Y_train_2d[SEQUENCE_LENGTH:]
X_test_rf = X_test_2d[SEQUENCE_LENGTH:]
Y_test_rf = Y_test_2d[SEQUENCE_LENGTH:]

start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=RANDOM_FOREST_N_ESTIMATORS, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, Y_train_rf)
training_time_rf = time.time() - start_time

print(f"Random Forest Training Time: {training_time_rf:.2f} seconds")

# Predict and Evaluate
Y_pred_rf = rf_model.predict(X_test_rf)
Y_true_rf, Y_pred_rf, mae_rf, rmse_rf, r2_rf = evaluate_and_invert(Y_test_rf, Y_pred_rf, "Random Forest")

# Save model and predictions
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.pkl"))





















































