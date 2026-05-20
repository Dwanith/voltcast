# src/data_ingestion/meteo_fetcher.py (Updated)

import requests
import pandas as pd
import os
import sys # Include sys for robust path logic, though os.environ is primary

# --- PATH CORRECTION START ---
# Use the APP_HOME environment variable defined in the Dockerfile
# This resolves to "/app" inside the container.
# Fallback is used for local development when APP_HOME is not set.
PROJECT_ROOT = os.environ.get("APP_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "meteo_weather_raw.csv")
# --- PATH CORRECTION END ---

# --- DMV (Washington D.C. coordinates) ---
LATITUDE = 38.9072
LONGITUDE = -77.0369
START_DATE = '2020-01-01'
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')


def fetch_meteo_data(lat: float, lon: float, start: str,
                     end: str) -> pd.DataFrame or None:

    # Define weather variables
    hourly_vars = [
        'temperature_2m',
        'relative_humidity_2m',
        'dew_point_2m',
        'shortwave_radiation',
        # --- CRITICAL ADDITION: APPARENT TEMPERATURE (for better thermal load proxy) ---
        'apparent_temperature'
    ]

    # Define API endpoint URL
    url = "https://archive-api.open-meteo.com/v1/era5"

    # Define request parameters
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": hourly_vars,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        # --- CRITICAL FIX: REQUEST DATA IN UTC ---
        "timezone": "UTC"
    }

    print(f"Fetching weather data for Lat {lat}, Lon {lon} in UTC...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            print("ERROR: Open-Meteo API returned no hourly data.")
            return None

        df = pd.DataFrame(data['hourly'])

        return df

    except requests.exceptions.RequestException as e:
        print(f"ERROR during API request: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- Main execution block ---

def main():
    weather_df = fetch_meteo_data(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    if weather_df is not None and not weather_df.empty:
        # RAW_DATA_PATH is now dynamically defined
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

        weather_df.to_csv(RAW_DATA_PATH, index=False)
        print(f"\nSUCCESS: Weather data fetched and saved to {RAW_DATA_PATH}")
        print(weather_df.head())
        print(f"Shape: {weather_df.shape}")

    else:
        print("Weather data fetch failed or returned an empty dataset.")


# Python entry point
if __name__ == "__main__":
    main()

    print("\n--- FINAL DATA FRESHNESS CHECK (TAIL) ---")

    try:
        verification_df = pd.read_csv(RAW_DATA_PATH)
        # Note the addition of 'apparent_temperature'
        print(verification_df.tail(5)[['time', 'temperature_2m', 'apparent_temperature']])

    except Exception as e:
        print(f"Could not perform final verification: {e}")