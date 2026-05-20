from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

#latest final
# -------------------------------------------------
# Resolve project root (works locally + in Docker)
# -------------------------------------------------
PROJECT_ROOT = os.environ.get(
    "APP_HOME",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Allow absolute imports from src/
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import routers
from src.api.routers import predict, monitor

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Voltcast: Electricity Demand Forecast Engine",
    description="Real-time PJM electricity demand prediction using ML models.",
    version="1.0.0",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Dashboard (STATIC FILES)
# -------------------------------------------------
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "src", "dashboard")
app.mount(
    "/dashboard",
    StaticFiles(directory=DASHBOARD_PATH, html=True),
    name="dashboard"
)

# -------------------------------------------------
# API routers
# -------------------------------------------------
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(monitor.router, prefix="/monitor", tags=["Monitoring"])

# -------------------------------------------------
# Root endpoint
# -------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Voltcast API running. Visit /dashboard for UI or /docs for API."
    }
