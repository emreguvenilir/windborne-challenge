import os
import json
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests
import time

os.makedirs("data", exist_ok=True)
os.makedirs("cache", exist_ok=True)

BASE_URL = "https://a.windbornesystems.com/treasure/"

# Fetch data for all 24 hours
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
        print(f"Failed to fetch hour {hour:02d}: {e}")

DATA_DIR = "data"
CACHE_FILE = "cache/weather_cache.pkl"

# Load existing cache
if os.path.exists(CACHE_FILE):
    weather_cache = pd.read_pickle(CACHE_FILE)
    print(f"📦 Loaded {len(weather_cache)} cached weather entries")
else:
    weather_cache = {}

# Setup API client
cache_session = requests_cache.CachedSession(backend='memory', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.1)
openmeteo = openmeteo_requests.Client(session=retry_session)

api_call_count = 0
cache_hits = 0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def get_weather_batch(coords_list, hour):
    """
    Fetch weather for multiple coordinates in a SINGLE API call
    coords_list: list of (balloon_index, lat, lon, alt) tuples
    Returns: dict mapping balloon_index to weather data
    """
    global api_call_count, cache_hits
    
    if not coords_list:
        return {}
    
    results = {}
    uncached_coords = []
    coord_to_balloon = {}  # Map rounded coords to balloon indices
    
    # Check cache first
    for balloon_idx, lat, lon, alt in coords_list:
        if np.isnan(lat) or np.isnan(lon):
            results[balloon_idx] = {}
            continue
        
        lat_r, lon_r = round(lat, 0), round(lon, 0)
        cache_key = (lat_r, lon_r, hour)
        
        if cache_key in weather_cache:
            cache_hits += 1
            results[balloon_idx] = weather_cache[cache_key]
        else:
            # Need to fetch this one
            coord_key = (lat_r, lon_r)
            if coord_key not in coord_to_balloon:
                coord_to_balloon[coord_key] = []
                uncached_coords.append((lat_r, lon_r))
            coord_to_balloon[coord_key].append(balloon_idx)
    
    # If everything was cached, return
    if not uncached_coords:
        return results
    
    # Batch fetch uncached coordinates
    # With our parameters: weight = nLocations * (2/14) * (6/10) ≈ nLocations * 0.086
    # To stay under 10k limit: use smaller batches
    # ~1000 unique coords/hour * 24 hours = ~24000 coords total
    # With batch size 50: weight ≈ 50 * 0.086 = 4.3 per batch
    # Total: ~200 batches * 4.3 = ~860 weighted calls (well under 10k!)
    BATCH_SIZE = 50
    
    for batch_start in range(0, len(uncached_coords), BATCH_SIZE):
        batch = uncached_coords[batch_start:batch_start + BATCH_SIZE]
        
        if batch_start > 0:
            time.sleep(0.5)  # Rate limiting

        # Prepare batch request
        lats = [coord[0] for coord in batch]
        lons = [coord[1] for coord in batch]
        
        api_call_count += 1
        
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lats,  
                "longitude": lons, 
                "past_days": 1,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover,pressure_msl",
                "timezone": "UTC"
            }
            
            responses = openmeteo.weather_api(url, params=params)
            
            # Process each location in the batch
            for i, response in enumerate(responses):
                lat_r, lon_r = batch[i]
                
                hourly = response.Hourly()
                
                weather_data = {
                    "temperature_2m": float(hourly.Variables(0).ValuesAsNumpy()[hour]),
                    "relative_humidity_2m": float(hourly.Variables(1).ValuesAsNumpy()[hour]),
                    "wind_speed_10m": float(hourly.Variables(2).ValuesAsNumpy()[hour]),
                    "wind_direction_10m": float(hourly.Variables(3).ValuesAsNumpy()[hour]),
                    "cloud_cover": float(hourly.Variables(4).ValuesAsNumpy()[hour]),
                    "pressure_msl": float(hourly.Variables(5).ValuesAsNumpy()[hour])
                }
                
                # Cache it
                cache_key = (lat_r, lon_r, hour)
                weather_cache[cache_key] = weather_data
                
                # Assign to all balloons at this coordinate
                for balloon_idx in coord_to_balloon[(lat_r, lon_r)]:
                    results[balloon_idx] = weather_data
        
        except Exception as e:
            print(f"⚠️ Batch API error: {e}")
            # Assign empty dicts for failed balloons
            for lat_r, lon_r in batch:
                for balloon_idx in coord_to_balloon[(lat_r, lon_r)]:
                    results[balloon_idx] = {}
    
    return results

def process_hour(hour, data):
    """Process all balloons for one hour using batch API"""
    results = []
    coords = []
    
    # Collect all coordinates for this hour
    for i, triple in enumerate(data):
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
    
    # Batch fetch weather for all coordinates
    weather_results = get_weather_batch(coords, hour)
    
    # Build result rows
    for i, lat, lon, alt in coords:
        weather = weather_results.get(i, {})
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

# Process all hours
history = []

for hour in range(24):
    file_path = os.path.join(DATA_DIR, f"{hour:02d}.json")
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        continue
    
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            continue
    
    print(f"Hour {hour:02d}: API calls={api_call_count}, Cache hits={cache_hits}")
    hour_results = process_hour(hour, data)
    history.extend(hour_results)
    
    # Save cache every 6 hours
    if hour % 6 == 0:
        pd.to_pickle(weather_cache, CACHE_FILE)

# Final save
pd.to_pickle(weather_cache, CACHE_FILE)

print(f"\n Data collected")
print(f"Batch API calls: {api_call_count}")
print(f"(Weighted calls ≈ {api_call_count * 50 * 0.086:.0f} toward 10k limit)")
print(f"Cache hits: {cache_hits}")
print(f"Total cached: {len(weather_cache)}")

# Create DataFrame
df = pd.DataFrame(history)
df = df.sort_values(["balloon_index", "hour"])

# Validate positions
df.loc[~df["latitude"].between(-90, 90), "latitude"] = np.nan
df.loc[~df["longitude"].between(-180, 180), "longitude"] = np.nan
df.loc[~df["altitude"].between(0, 50), "altitude"] = np.nan

# Interpolate missing data
weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
                "wind_direction_10m", "cloud_cover", "pressure_msl"]
pos_cols = ["latitude", "longitude", "altitude"]

# Weather: forward-fill, back-fill, then interpolate
df[weather_cols] = (
    df.groupby("balloon_index")[weather_cols]
    .apply(lambda g: g.ffill().bfill().interpolate(method="linear", limit_direction="both"))
    .reset_index(level=0, drop=True)
)

# Position: interpolate, then fill edges
df[pos_cols] = (
    df.groupby("balloon_index")[pos_cols]
    .apply(lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill())
    .reset_index(level=0, drop=True)
)

# Calculate wind components AFTER interpolation
rad = np.deg2rad(df["wind_direction_10m"])
df["wind_u"] = df["wind_speed_10m"] * np.sin(rad)
df["wind_v"] = df["wind_speed_10m"] * np.cos(rad)

# Pivot to wide format
pivot_cols = ["latitude", "longitude", "altitude", "temperature_2m", "relative_humidity_2m",
              "wind_speed_10m", "wind_direction_10m", "cloud_cover", "pressure_msl", "wind_u", "wind_v"]

df_wide = df.pivot(index="balloon_index", columns="hour", values=pivot_cols)
df_wide.columns = [f"{col[0]}_h{col[1]}" for col in df_wide.columns]
df_wide.reset_index(inplace=True)

# Save
output_path = "processed_balloon_data.csv"
df_wide.to_csv(output_path, index=False)

# Quality report
weather_check = df_wide[[f'temperature_2m_h{i}' for i in range(24)]].notna().any(axis=1)
print(f"\nData Quality:")
print(f"   Total balloons: {len(df_wide)}")
print(f"   With weather: {weather_check.sum()} ({100*weather_check.sum()/len(df_wide):.1f}%)")
print(f"   Missing weather: {(~weather_check).sum()} ({100*(~weather_check).sum()/len(df_wide):.1f}%)")