# src/modelling/gru_model.py

import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Suppress warnings if needed
import warnings

warnings.filterwarnings('ignore')

# --- 1. SETUP: Dynamic Paths and Parameters ---
# Assumes this script is in src/modelling/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define Features/Target (Must match pre-processing - NOW 12 total features)
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]
TARGET = "demand_mwh"

# Sequence Length (Use the last 24 hours of data to predict the next hour)
SEQUENCE_LENGTH = 24

# --- 2. Data Loading ---

print("--- 1. Data Loading ---")
try:
    # Load Training and Validation data (used for training the model)
    df_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "final_train.csv"))
    df_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "final_val.csv"))

    # Load the FINAL API TEST SET for final evaluation
    df_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test_scaled.csv"))

    # Load Scaler for inverse transformation (metrics interpretation)
    scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required files. Please ensure preprocess.py has run successfully. Error: {e}")
    exit()

# Extract 2D arrays
X_train_2d = df_train[ALL_FEATURES].values
Y_train_2d = df_train[TARGET].values

X_val_2d = df_val[ALL_FEATURES].values
Y_val_2d = df_val[TARGET].values

# Extract 2D Test arrays (API Data)
X_test_2d = df_test[ALL_FEATURES].values
Y_test_2d = df_test[TARGET].values

print(f"2D Train set shape: {X_train_2d.shape}")
print(f"2D Validation set shape: {X_val_2d.shape}")
print(f"2D Test set shape: {X_test_2d.shape}")


# --- 3. Sequence Creation Function (The Deep Learning Requirement) ---

def create_sequences(X, Y, seq_len):
    """Converts 2D feature/target arrays into 3D sequences for GRU/LSTM."""
    X_seq, Y_seq = [], []
    # Loop stops early to ensure a full sequence can be created
    for i in range(len(X) - seq_len):
        # Input sequence is of length seq_len (e.g., 24 hours)
        X_seq.append(X[i:i + seq_len])
        # Target is the next single hour's value
        Y_seq.append(Y[i + seq_len])
    return np.array(X_seq), np.array(Y_seq)


# Transform 2D data into 3D sequences for training
X_train_3d, Y_train_3d = create_sequences(X_train_2d, Y_train_2d, SEQUENCE_LENGTH)
X_val_3d, Y_val_3d = create_sequences(X_val_2d, Y_val_2d, SEQUENCE_LENGTH)

print(f"3D Train set shape: {X_train_3d.shape} (samples, seq_len, features)")
print(f"3D Validation set shape: {X_val_3d.shape}")

# --- 4. GRU Model Definition and Training ---

print("\n--- 2. Training GRU Model ---")

# Define the GRU Model Architecture
# Input shape is (SEQUENCE_LENGTH, N_FEATURES)
input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
gru_model = Sequential([
    GRU(units=64, return_sequences=False, input_shape=input_shape),
    Dense(units=1)  # Single output (demand at t+1)
])

gru_model.compile(optimizer='adam', loss='mse')

# Callbacks for robust training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join(MODELS_DIR, 'gru_best_model.h5'),
                    monitor='val_loss', save_best_only=True)
]

start_time = time.time()
gru_model.fit(
    X_train_3d,
    Y_train_3d,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_3d, Y_val_3d),
    callbacks=callbacks,
    verbose=1  # Show training progress
)
training_time_gru = time.time() - start_time

print(f"GRU Training Time: {training_time_gru:.2f} seconds")

# --- 5. Evaluation and Saving (Using the API Test Set) ---

print("\n--- 3. Final Prediction and Evaluation on API Test Set ---")

# Load the best model weights
gru_model.load_weights(os.path.join(MODELS_DIR, 'gru_best_model.h5'))

# Create sequences for the Test Set (API Data)
X_test_3d, Y_test_3d = create_sequences(X_test_2d, Y_test_2d, SEQUENCE_LENGTH)
print(f"3D Test set shape: {X_test_3d.shape}")

# Predict on Test data
Y_pred_scaled = gru_model.predict(X_test_3d).flatten()

# Reshape and inverse transform
Y_true_scaled = Y_test_3d
Y_true = scaler_y.inverse_transform(Y_true_scaled.reshape(-1, 1)).flatten()
Y_pred = scaler_y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate Metrics (on unscaled data)
mae = mean_absolute_error(Y_true, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
r2 = r2_score(Y_true, Y_pred)

print(f"\n--- GRU FINAL API TEST Metrics (Unscaled MW) ---")
print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f} MW")
print(f"  Mean Absolute Error (MAE):      {mae:,.2f} MW")
print(f"  R-squared (R2):                 {r2:.4f}")

# Save the final model architecture and weights
gru_model.save(os.path.join(MODELS_DIR, "gru_final_model.h5"))
print(f"\nSUCCESS: Final GRU model saved to {os.path.join(MODELS_DIR, 'gru_final_model.h5')}")

print("\n--- GRU Training Script Finished ---")