import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

os.makedirs("data", exist_ok=True)

BASE_URL = "https://a.windbornesystems.com/treasure/"

for hour in range(24):
    url = f"{BASE_URL}{hour:02d}.json"
    out_path = f"data/{hour:02d}.json"
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(out_path, "w") as f:
            f.write(r.text)
        print(f"✅ Saved {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to fetch hour {hour:02d}: {e}")

DATA_DIR = "data"

history = []
cache_session = requests_cache.CachedSession(backend='memory', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.1)
openmeteo = openmeteo_requests.Client(session=retry_session)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

@lru_cache(maxsize=2000)
def get_weather(lat, lon, hour):
    if np.isnan(lat) or np.isnan(lon):
        return {}
    # round coordinates to reduce API calls
    lat_r, lon_r = round(lat, 2), round(lon, 2)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat_r,
        "longitude": lon_r,
        "past_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover,pressure_msl",
        "timezone": "UTC"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(5).ValuesAsNumpy()
    
    hourly_data = {"date": pd.date_range(
    	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = hourly.Interval()),
    	inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["pressure_msl"] = hourly_pressure_msl

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe.loc[hour]

def fetch_hour_data(hour, data):
    """Fetch weather data for all balloons in one hour concurrently."""
    results = []
    coords = []

    # prepare coordinate list
    for i, triple in enumerate(data):  # remove [:10] later for full dataset
        if not triple or len(triple) != 3:
            results.append({
                "balloon_index": i,
                "latitude": np.nan,
                "longitude": np.nan,
                "altitude": np.nan,
                "hour": hour,
            })
        else:
            lat, lon, alt = (safe_float(x) for x in triple)
            coords.append((i, lat, lon, alt))

    # concurrent API calls
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(get_weather, lat, lon, hour): (i, lat, lon, alt)
            for (i, lat, lon, alt) in coords
        }

        for future in as_completed(future_map):
            i, lat, lon, alt = future_map[future]
            try:
                weather = future.result()
            except Exception as e:
                weather = {}

            results.append({
                "balloon_index": i,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "hour": hour,
                "temperature_2m": weather.get("temperature_2m", np.nan),
                "relative_humidity_2m": weather.get("relative_humidity_2m", np.nan),
                "wind_speed_10m": weather.get("wind_speed_10m", np.nan),
                "wind_direction_10m": weather.get("wind_direction_10m", np.nan),
                "cloud_cover": weather.get("cloud_cover", np.nan),
                "pressure_msl": weather.get("pressure_msl", np.nan)
            })
    return results

for hour in range(24):
    file_path = os.path.join(DATA_DIR, f"{hour:02d}.json")
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        continue

    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            continue

    hour_results = fetch_hour_data(hour, data)
    history.extend(hour_results)


df = pd.DataFrame(history)
# --- Derive u/v wind components from speed + direction ---
if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
    rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = df["wind_speed_10m"] * np.sin(rad)   # +east
    df["wind_v"] = df["wind_speed_10m"] * np.cos(rad)   # +north

df = df.sort_values(["balloon_index", "hour"])

# --- Replace out of range values with NaN ---
df.loc[~df["latitude"].between(-90, 90), "latitude"] = np.nan
df.loc[~df["longitude"].between(-180, 180), "longitude"] = np.nan
df.loc[~df["altitude"].between(0, 50), "altitude"] = np.nan

# --- Handle extreme jumps ---
df["lat_diff"] = df.groupby("balloon_index")["latitude"].diff().abs()
df["lon_diff"] = df.groupby("balloon_index")["longitude"].diff().abs()
df["alt_diff"] = df.groupby("balloon_index")["altitude"].diff().abs()

LAT_LON_THRESHOLD = 30
ALT_THRESHOLD = 15

df.loc[df["lat_diff"] > LAT_LON_THRESHOLD, "latitude"] = np.nan
df.loc[df["lon_diff"] > LAT_LON_THRESHOLD, "longitude"] = np.nan
df.loc[df["alt_diff"] > ALT_THRESHOLD, "altitude"] = np.nan

# --- Interpolate missing data ---
for col in ["latitude", "longitude", "altitude","temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "cloud_cover", "pressure_msl", "wind_u", "wind_v"]:
    if col in df.columns:
        df[col] = (
            df.groupby("balloon_index")[col]
            .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
            .reset_index(level=0, drop=True)
        )

# Save a copy before pivoting
flight_df = df.copy()

# --- Plot sample balloon trajectories ---
#plt.figure(figsize=(10,6))
#for b in flight_df["balloon_index"].unique()[:20]:  # plot a few random balloons
    #subset = flight_df[flight_df["balloon_index"] == b]
    #plt.plot(subset["longitude"], subset["latitude"], alpha=0.6, label=f"Balloon {b}")

#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.title("Sample Balloon Trajectories (24h)")
#plt.legend()
#plt.show()

# --- Pivot table 
pivot_cols = [
    "latitude", "longitude", "altitude",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "pressure_msl",
    "wind_u", "wind_v"
]

df = df.pivot(index="balloon_index", columns="hour", values=pivot_cols)
df.columns = [f"{col[0]}_h{col[1]}" for col in df.columns]
df.reset_index(inplace=True)

output_path = "processed_balloon_data.csv"
df.to_csv(output_path, index=False)