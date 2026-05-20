

# src/api/routers/predict.py - FINAL, FULLY CORRECTED VERSION

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

router = APIRouter()

PROJECT_ROOT = os.environ.get(
    "APP_HOME",
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")



print(f"DEBUG: PROJECT_ROOT={PROJECT_ROOT}")
print(f"DEBUG: MODELS_DIR={MODELS_DIR}")

SEQUENCE_LENGTH = 24
ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "apparent_temperature",
    "lag_1h", "lag_24h", "lag_168h",
    "Hour", "DayOfWeek", "DayOfYear", "IsWeekend"
]

MODELS = {}
SCALERS = {}

def load_all_artifacts():
    """Loads all saved model artifacts (XGboost, GRU, RF, Scalers) into memory."""
    try:
        # Load Scalers
        SCALERS['X'] = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
        SCALERS["Y"] = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.pkl'))

        # Load the XGBoost model
        MODELS['XGBoost'] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

        # Load Random forest model
        MODELS['RandomForest'] = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))

        # Load the GRU model
        MODELS['GRU'] = load_model(os.path.join(MODELS_DIR, "gru_final_model.h5"), compile=False)

        print("INFO: All models and scalers are loaded successfully into API memory.")
        return True
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FATAL ERROR DURING MODEL LOADING: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return False

# Load models
if not load_all_artifacts():
    raise RuntimeError("Models failed to load. Check Docker logs.")

# Prediction Input Schema
class PredictionInput(BaseModel):
    """Defines the structured input data required for the forecast"""
    temperature_2m: float = Field(..., description="Current Air Temperature (°C)")
    relative_humidity_2m: float = Field(..., description="Current Relative Humidity (%)")
    dew_point_2m: float = Field(..., description="Current dew point temperature in (°C)")
    shortwave_radiation: float = Field(..., description="Current solar radiation (W/m²)")
    apparent_temperature: float = Field(..., description="Current apparent temperature (°C)")
    lag_1h: float = Field(..., description="Observed demand 1 hr ago (MWh)")
    lag_24h: float = Field(..., description="Observed demand 24 hrs ago (MWh)")
    lag_168h: float = Field(..., description="Observed demand 168 hrs ago")
    Hour: int = Field(..., description="Hour of the day 0-23")
    DayOfWeek: int = Field(..., description="Day of the week (0 - 6) 0 = Monday, 6 = Sunday")
    DayOfYear: int = Field(..., description="Day of the year (1 - 365)")
    IsWeekend: int = Field(..., description="1 = Yes if weekend, 0 = No otherwise")
    model_choice: str = Field("RandomForest", description="Model to use for prediction: 'RandomForest', 'XGBoost', or 'GRU'")

def create_sequences(X_2d, seq_len):
    """Create proper 3D sequence for GRU from current input"""
    sequence = np.tile(X_2d, (1, seq_len, 1))
    return sequence

# Prediction endpoint
@router.post("/forecast", summary="Predicts the next hour's electricity demand (MW)")
async def predict_demand(input_data: PredictionInput):
    """
    Accepts current weather, time, and lagged demand values and returns the predicted demand
    (MWh) for the next hour.
    """
    # Data Preparation and Scaling
    input_dict = input_data.dict(exclude={"model_choice"})
    df_input = pd.DataFrame([input_dict], columns=ALL_FEATURES)

    # Apply the trained X-scaler
    X_scaled = SCALERS['X'].transform(df_input)

    # Model selection and prediction
    model_choice = input_data.model_choice

    if model_choice == "XGBoost" or model_choice == 'RandomForest':
        model = MODELS[model_choice]
        Y_pred_scaled = model.predict(X_scaled)[0]
    elif model_choice == 'GRU':
        model = MODELS['GRU']
        X_3d = create_sequences(X_scaled, SEQUENCE_LENGTH)
        Y_pred_scaled = model.predict(X_3d)[0][0]
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice. Must be 'RandomForest', 'XGBoost' or 'GRU'")

    Y_pred_unscaled = SCALERS['Y'].inverse_transform(np.array(Y_pred_scaled).reshape(-1, 1))[0][0]

    return {
        "model_used": model_choice,
        "forecast_time_ahead": "1 Hour",
        "Predicted_demand_mw": float(round(Y_pred_unscaled, 2))
    }