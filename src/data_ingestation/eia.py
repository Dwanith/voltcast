
import requests
import pandas as pd
import datetime
import os

PROJECT_ROOT = os.environ.get("APP_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the save path relative to the dynamic PROJECT_ROOT
SAVE_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "eia_demand_raw.csv")



url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

end_date = datetime.date.today().strftime("%Y-%m-%d")

params = {
    "api_key": "5jejlr8VpdMSfsSGCQwei3zDhLFuieDM6eS0Uqmh",
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "PJM",
    "start": "2020-01-01",
    "end": end_date
}


r = requests.get(url, params=params)
r.raise_for_status()
data = r.json()
df = pd.DataFrame(data["response"]["data"])

print(df.head(100))
print(df.shape)
print(df.tail(100))


# Use the corrected, dynamic path for saving
os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
df.to_csv(SAVE_FILE, index=False)
print(f"\nSaved to {SAVE_FILE}")