import pandas as pd
import os
import pytz
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

# --- CRITICAL PATH CORRECTION START ---
# Use the APP_HOME environment variable defined in the Dockerfile ("APP_HOME=/app")
# This defines the project root dynamically, essential for containerization.
PROJECT_ROOT = os.environ.get("APP_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- CRITICAL PATH CORRECTION END ---


#1. Load and clean historical downloaded EIA data
# --- 1. Define Paths and File Names (Manual Specification) ---
# base_dir = "/Users/dwanith/Desktop/Personal Projects/voltcast"  <-- REMOVED
# *** REPLACED ALL base_dir USES WITH PROJECT_ROOT ***
data_raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")

file_2022 = os.path.join(data_raw_dir, "pjm_load_act_hr_2022.csv")
file_2023 = os.path.join(data_raw_dir, "pjm_load_act_hr_2023.csv")
file_2024 = os.path.join(data_raw_dir, "pjm_load_act_hr_2024.csv")

# --- 2. Define Required Column Names and Settings ---
time_column_name = "UTC Timestamp (Interval Ending)"
load_column_name = "PJM Total Actual Load (MW)"
header_row_index = 3  # Fix for the EIA metadata lines


# --- 3. Function to Load and Clean a Single File ---
def load_and_clean_eia_yearly(file_path):
    """Loads one yearly EIA file, fixes the header issue, and cleans columns."""

    # Load with the header fix and only the required columns
    df = pd.read_csv(file_path,
                     header=header_row_index,
                     usecols=[time_column_name, load_column_name])

    # Rename columns to a simple standard
    df = df.rename(columns={time_column_name: "time", load_column_name: "demand_mwh"})

    # Convert time column to a datetime object
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    return df.dropna(subset=['time', 'demand_mwh'])


# --- 4. Load Each File Manually ---
df_2022 = load_and_clean_eia_yearly(file_2022)
df_2023 = load_and_clean_eia_yearly(file_2023)
df_2024 = load_and_clean_eia_yearly(file_2024)

# --- 5. Concatenate All Three DataFrames ---
list_of_dfs = [df_2022, df_2023, df_2024]
historical_eia_data_combined = pd.concat(list_of_dfs, ignore_index=True)

# Sort chronologically (oldest to newest)
historical_eia_data_combined = historical_eia_data_combined.sort_values("time").reset_index(drop=True)

# --- 6. Save the Combined Data to the Specified Path ---
output_file_path = os.path.join(data_processed_dir, "historical_EIAdata_combined.csv")

os.makedirs(data_processed_dir, exist_ok=True)
historical_eia_data_combined.to_csv(output_file_path, index=False)

# --- Verification Output ---
print(f"1. Historical EIA Data Combined: {historical_eia_data_combined.shape[0]} rows saved.")
print(f"File saved to: {output_file_path}")


#2. Merging the Meteo and EIA data (Train and Validation) (CORRECTED)

# --- SETUP (Corrected Paths) ---
# base_dir = "/Users/dwanith/Desktop/Personal Projects/voltcast" <-- REMOVED
# *** REPLACED ALL base_dir USES WITH PROJECT_ROOT ***
data_raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
models_dir = os.path.join(PROJECT_ROOT, "models")

# Define ALL Features (NOW 12 features, including Lags)
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]
TARGET = "demand_mwh"

# --- 1. Load Historical Data (Same as before) ---
historical_demand_df = pd.read_csv(os.path.join(data_processed_dir, "historical_EIAdata_combined.csv"))
historical_demand_df['time'] = pd.to_datetime(historical_demand_df['time']).dt.tz_localize('UTC')

meteo_raw_file = os.path.join(data_raw_dir, "meteo_weather_raw.csv")
df_meteo = pd.read_csv(meteo_raw_file)

# FIX: Rename Time column in Meteo data (based on your earlier fix)
df_meteo = df_meteo.rename(columns={'Time': 'time'}, errors='ignore')
meteo_keep_cols = [col for col in df_meteo.columns if col not in ['time', 'Time']]
meteo_clean = df_meteo[['time'] + meteo_keep_cols].copy()

# Set Meteo data to UTC
meteo_clean["time"] = pd.to_datetime(meteo_clean["time"], errors="coerce").dt.floor('h')
meteo_clean['time'] = meteo_clean['time'].dt.tz_localize('UTC')
meteo_clean = meteo_clean.dropna(subset=['time']).drop_duplicates(subset=["time"]).sort_values("time")


# --- 2. Final Merge and Scope Definition ---
merged_historical_df = pd.merge(meteo_clean, historical_demand_df, on="time", how="inner")
merged_historical_df = merged_historical_df.sort_values("time").reset_index(drop=True)

# Remove timezone information for ML model training
merged_historical_df['time'] = merged_historical_df['time'].dt.tz_localize(None)


# STEP 2.5 - ADD AUTOREGRESSIVE (LAGGED) FEATURES HERE
# 1. Add Lag 1 Hour (most critical feature)
merged_historical_df['lag_1h'] = merged_historical_df['demand_mwh'].shift(1)
# 2. Add Lag 24 Hours (Daily Seasonality)
merged_historical_df['lag_24h'] = merged_historical_df['demand_mwh'].shift(24)
# 3. Add Lag 168 Hours (Weekly Seasonality)
merged_historical_df['lag_168h'] = merged_historical_df['demand_mwh'].shift(168)
# After adding lags, we must drop the rows that now have NaN values at the beginning.
# The maximum lag is 168, so we drop the first 168 rows.
merged_historical_df = merged_historical_df.dropna().reset_index(drop=True)


# --- 3. FEATURE ENGINEERING (MOVED UP) ---
merged_historical_df['Hour'] = merged_historical_df['time'].dt.hour
merged_historical_df['DayOfWeek'] = merged_historical_df['time'].dt.dayofweek
merged_historical_df['DayOfYear'] = merged_historical_df['time'].dt.dayofyear
merged_historical_df['IsWeekend'] = (merged_historical_df['DayOfWeek'] >= 5).astype(int)

# Drop the original 'time' column before splitting and scaling
merged_historical_df = merged_historical_df.drop(columns=['time'])


# --- 4. Chronological Train/Validation Split (75%/25%) ---
split_date_index = int(len(merged_historical_df) * 0.75)

# Split the data based on index, as 'time' column is dropped
df_train = merged_historical_df.iloc[:split_date_index].copy()
df_val = merged_historical_df.iloc[split_date_index:].copy()


# --- 5. Scaling and Saving (Using all 12 features) ---
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_processed_dir, exist_ok=True)

# Initialize Scalers and fit ONLY on the Training Data (12 features + target)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and Transform Training Data
x_train_scaled = scaler_x.fit_transform(df_train[ALL_FEATURES])
y_train_scaled = scaler_y.fit_transform(df_train[[TARGET]])

# Transform Validation Data
x_val_scaled = scaler_x.transform(df_val[ALL_FEATURES])
y_val_scaled = scaler_y.transform(df_val[[TARGET]])

# Save Scalers (12 features now saved)
joblib.dump(scaler_x, os.path.join(models_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(models_dir, "scaler_y.pkl"))

# Save Scaled DataFrames
df_train_scaled = pd.DataFrame(x_train_scaled, columns=ALL_FEATURES)
df_train_scaled[TARGET] = y_train_scaled
df_train_scaled.to_csv(os.path.join(data_processed_dir, "final_train.csv"), index=False)

df_val_scaled = pd.DataFrame(x_val_scaled, columns=ALL_FEATURES)
df_val_scaled[TARGET] = y_val_scaled
df_val_scaled.to_csv(os.path.join(data_processed_dir, "final_val.csv"), index=False)


print(f"\n2. FINAL TRAIN/VALIDATION PREPROCESSING SUCCESS (LOGIC CORRECTED): {merged_historical_df.shape[0]} total historical rows.")
print(f"Train Rows: {df_train_scaled.shape[0]}. Saved to final_train.csv (12 features).")
print(f"Validation Rows: {df_val_scaled.shape[0]}. Saved to final_val.csv (12 features).")


#4. Merging the Meteo and EIA data (Test)

print("----------------4. Merging the Meteo and EIA data (Test)----------------------")

# --- SETUP ---
# base_dir = "/Users/dwanith/Desktop/Personal Projects/voltcast" <-- REMOVED
# *** REPLACED ALL base_dir USES WITH PROJECT_ROOT ***
data_raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")

# Define necessary columns and parameters
api_time_col = "period"
api_load_col = "value"
demand_type = "D"
TARGET = "demand_mwh"

# --- 1. Load and Process EIA API Test Data (Filter 'D' Rows) ---
eia_api_file = os.path.join(data_raw_dir, "eia_demand_raw.csv")
df_eia_raw = pd.read_csv(eia_api_file)

# Filter for only Demand ('D') type rows
df_demand_test = df_eia_raw[df_eia_raw["type"] == demand_type].copy()

# Rename columns and clean/localize time (UTC)
df_demand_test = df_demand_test.rename(columns={api_time_col: "time", api_load_col: TARGET})
df_demand_test["time"] = pd.to_datetime(df_demand_test["time"], errors="coerce").dt.floor('h')
df_demand_test['time'] = df_demand_test['time'].dt.tz_localize('UTC')
df_demand_test = df_demand_test[['time', TARGET]].drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

print(f"Cleaned EIA Test Rows (Demand Only): {df_demand_test.shape[0]}")


# --- 2. Load and Process Meteo Data ---
meteo_raw_file = os.path.join(data_raw_dir, "meteo_weather_raw.csv")
df_meteo = pd.read_csv(meteo_raw_file)

# --- FIX APPLIED (Ensure Meteo 'Time' column is renamed) ---
df_meteo = df_meteo.rename(columns={'Time': 'time'}, errors='ignore')

# We only need the weather features from the Meteo file
meteo_keep_cols = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature"
]
df_meteo_test = df_meteo[['time'] + meteo_keep_cols].copy()

# Clean time and localize to UTC
df_meteo_test["time"] = pd.to_datetime(df_meteo_test["time"], errors="coerce").dt.floor('h')
df_meteo_test['time'] = df_meteo_test['time'].dt.tz_localize('UTC')
df_meteo_test = df_meteo_test.dropna(subset=['time']).drop_duplicates(subset=["time"]).sort_values("time")


# --- 3. Final Merge for Test Set ---
df_test = pd.merge(df_meteo_test, df_demand_test, on="time", how="inner")
df_test = df_test.sort_values("time").reset_index(drop=True)

# Remove timezone information before feature engineering
df_test['time'] = df_test['time'].dt.tz_localize(None)

print(f"Final Merged Test Rows: {df_test.shape[0]}")


# --------------------------------------------------------------------------------------------------
# CRITICAL FIX: STEP 4.3 - AUTOREGRESSIVE (LAGGED) FEATURES INSERTED HERE (Test Set)
# --------------------------------------------------------------------------------------------------
# 1. Add Lag 1 Hour (most critical feature)
df_test['lag_1h'] = df_test['demand_mwh'].shift(1)
# 2. Add Lag 24 Hours (Daily Seasonality)
df_test['lag_24h'] = df_test['demand_mwh'].shift(24)
# 3. Add Lag 168 Hours (Weekly Seasonality)
df_test['lag_168h'] = df_test['demand_mwh'].shift(168)
# Drop the rows with NaN values introduced by the shift
df_test = df_test.dropna().reset_index(drop=True)


# --- 4. Feature Engineering for Test Set (CREATES df_test_data) ---
print("\n--- STEP 4.4: FEATURE ENGINEERING ---")

# Add time features
df_test['Hour'] = df_test['time'].dt.hour
df_test['DayOfWeek'] = df_test['time'].dt.dayofweek
df_test['DayOfYear'] = df_test['time'].dt.dayofyear
df_test['IsWeekend'] = (df_test['DayOfWeek'] >= 5).astype(int)

# Separate features/target and remove the 'time' column before scaling
df_test_data = df_test.drop(columns=['time'])

print("Test Set df_test and df_test_data created successfully.")


#5. Scaling and test set preparation

# --- SETUP (Corrected Paths and Features) ---
# base_dir = "/Users/dwanith/Desktop/Personal Projects/voltcast" <-- REMOVED
# *** REPLACED ALL base_dir USES WITH PROJECT_ROOT ***
data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
models_dir = os.path.join(PROJECT_ROOT, "models")

# Define ALL Features and Target (12 columns)
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]
TARGET = "demand_mwh"

# Assuming df_test and df_test_data are still in memory from Section 4
# If not, you may need to re-run the final feature engineering steps from Section 4 before this block.

# --- 2. Load Scalers and Scale Data ---
print("\n--- STEP 3.2: FINAL SCALING AND SAVING ---")

# Paths will now be correct: /voltcast/models/scaler_X.pkl
scaler_x_path = os.path.join(models_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(models_dir, "scaler_y.pkl")

scaler_x = joblib.load(scaler_x_path)
scaler_y = joblib.load(scaler_y_path)

# Transform (DO NOT FIT) the Test Set data
X_test_scaled = scaler_x.transform(df_test_data[ALL_FEATURES])
Y_test_scaled = scaler_y.transform(df_test_data[[TARGET]])

# --- 3. Save Final Scaled Test Set ---
df_test_scaled = pd.DataFrame(X_test_scaled, columns=ALL_FEATURES)
df_test_scaled[TARGET] = Y_test_scaled

# Re-add the original 'time' column for reference
# NOTE: This assumes df_test is still in memory with the 'time' column!
df_test_scaled['time'] = df_test['time'].reset_index(drop=True)

# Save the Test Set
df_test_scaled.to_csv(os.path.join(data_processed_dir, "test_scaled.csv"), index=False)


print("\n--- TEST SET PREPARATION COMPLETE 🚀 ---")
print(f"Final Test Rows Saved: {df_test_scaled.shape[0]}")
print(f"File saved to: {os.path.join(data_processed_dir, 'test_scaled.csv')}")