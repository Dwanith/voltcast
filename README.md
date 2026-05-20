# Voltcast

Short-term electricity demand forecasting powered by weather signals and ensemble machine learning.

## Overview

Voltcast is an end-to-end energy demand forecasting platform that ingests live electricity consumption and meteorological data, trains multiple machine-learning models, and serves real-time forecasts through a REST API and a browser dashboard. The full stack is containerized and runs in production on AWS EC2.

## Features

- **Live data ingestion** from the U.S. Energy Information Administration (EIA) and Open-Meteo weather APIs
- **Multi-model forecasting** — Random Forest, XGBoost, GRU, and LSTM trained and benchmarked on the same feature set
- **Automated preprocessing pipeline** for feature engineering, scaling, and missing-value handling
- **REST API** built on FastAPI with `/predict` and `/monitor` endpoints
- **Lightweight dashboard** for visualizing forecasts vs. actuals in real time
- **Containerized deployment** with Docker, running on AWS EC2

## Architecture

```
EIA / Open-Meteo  ─►  Preprocessing  ─►  Model Training (RF / XGB / GRU )
                                                │
                                                ▼
                                        FastAPI Service
                                       (/predict, /monitor)
                                                │
                                                ▼
                                          Web Dashboard
```

## Tech Stack

- **Language:** Python 3
- **API:** FastAPI, Uvicorn
- **ML:** scikit-learn, XGBoost, TensorFlow/Keras
- **Data:** Pandas, NumPy
- **Infrastructure:** Docker, AWS EC2
- **Frontend:** HTML, JavaScript

## Project Structure

```
voltcast/
├── src/
│   ├── api/                 # FastAPI service and routers
│   ├── dashboard/           # Browser dashboard
│   ├── data_ingestation/    # EIA + weather data fetchers
│   ├── preprocessing/       # Feature engineering pipeline
│   └── modelling/           # Model training scripts
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Local setup
```bash
git clone git@github.com:Dwanith/voltcast.git
cd voltcast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t voltcast .
docker run -p 8000:8000 voltcast
```

## API

| Endpoint   | Method | Description                                |
|------------|--------|--------------------------------------------|
| `/predict` | POST   | Returns demand forecast for given inputs   |
| `/monitor` | GET    | Service health and model status            |

## Deployment

Voltcast is containerized with Docker and runs on AWS EC2. The image bundles the trained models, the FastAPI service, and the dashboard so a single container serves the full stack.

## Licenses

Proprietary — all rights reserved.
