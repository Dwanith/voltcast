# src/api/routers/monitor.py

from fastapi import APIRouter
from datetime import datetime
import time

router = APIRouter()

# Define the start time globally for calculating uptime
start_time = time.time()

# --- Health Check Endpoint ---
@router.get("/health", summary="Performs a health check and verifies application status.")
async def get_health():
    """
    Returns the current status of the API service.
    """
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "uptime": f"{time.time() - start_time:.2f} seconds"
    }


# --- Metrics Endpoint ---
@router.get("/metrics", summary="Returns key performance metrics of the API.")
async def get_metrics():
    """
    Provides key service metrics, including model version and latency.
    """
    return {
        "metrics_status": "Metrics data available",
        "current_model_version": "v1.0 (RF/XGBoost/GRU)",
        "model_loaded_timestamp": None,
        "request_count_24h": "To be implemented"
    }