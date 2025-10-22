# ============= Imports ============= 
import os
import json
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
import requests
import logging
import psutil

# ============= Memory Checks =============

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB RSS")

# ============= Declarations and Setup ============= 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)

# API endpoints
WINDBORNE_BASE_URL = "https://a.windbornesystems.com/treasure/"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# Keyframe hours to fetch (9 hours instead of 24)
KEYFRAME_HOURS = [0, 1, 2, 3, 6, 10, 14, 18, 23]

# Weather parameters to fetch
WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "pressure_msl"
]

BATCH_SIZE = 50
MAX_WORKERS = 4
RATE_LIMIT_DELAY = 0.15

# ============= API Session Setup =============

retry_session = retry(requests.Session(), retries=3, backoff_factor=0.1)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ============= Global State =============
#Data collection
history = []

# Weather cache: key = "lat_lon_hour_date", value = weather dict
weather_cache = {}

# API call tracking
api_call_count = 0
api_lock = Lock()
last_call_time = 0
cache_hits = 0
cache_misses = 0

# Rate limit tracking
minute_start = None
hour_start = None
calls_this_minute = 0
calls_this_hour = 0
# ============= Helper Functions =============

def safe_float(x):
    """Safely convert to float, return NaN on error."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def round_coords(lat, lon):
    """Round coordinates to grid resolution for caching."""
    return round(lat, 1), round(lon, 1)

def make_cache_key(lat, lon, hour):
    """Create cache key for weather data."""
    lat_r, lon_r = round_coords(lat, lon)
    return f"{lat_r}_{lon_r}_{hour}"

def group_balloons_by_grid(balloons):
    """
    Group balloons by grid cell for deduplication.
    Returns: dict of {grid_key: [list of balloons]}
    """
    grid_groups = {}
    
    for balloon in balloons:
        lat_r, lon_r = round_coords(balloon["lat"], balloon["lon"])
        grid_key = f"{lat_r}_{lon_r}"
        
        if grid_key not in grid_groups:
            grid_groups[grid_key] = []
        
        grid_groups[grid_key].append(balloon)
    
    return grid_groups

def fetch_weather_batch(grid_keys, hour):
    """
    Fetch weather for multiple grid locations in one API call.
    
    Args:
        grid_keys: list of grid keys like ["45.1_10.2", "78.4_-23.1", ...]
        hour: which hour to fetch (0-23)
    
    Returns:
        dict of {grid_key: weather_data}
    """
    global api_call_count, last_call_time, cache_hits, cache_misses
    global minute_start, hour_start, calls_this_minute, calls_this_hour
    
    if not grid_keys:
        return {}
    
    log_memory_usage(f"[fetch_weather_batch start] hour={hour}, batch_size={len(grid_keys)}")

    # Step 1: Check cache first, separate uncached locations
    results = {}
    needs_fetch = []  # (lat, lon, grid_key) tuples
    
    for grid_key in grid_keys:
        # Parse grid_key back to coordinates
        lat_str, lon_str = grid_key.split("_")
        lat, lon = float(lat_str), float(lon_str)
        
        cache_key = make_cache_key(lat, lon, hour)
        
        if cache_key in weather_cache:
            results[grid_key] = weather_cache[cache_key]
            cache_hits += 1
        else:
            needs_fetch.append((lat, lon, grid_key))
    
    # All cached - early return
    if not needs_fetch:
        logger.info(f"Batch of {len(grid_keys)} locations - all cached")
        return results
    
    logger.info(f"Fetching weather for {len(needs_fetch)} uncached locations (batch size: {len(grid_keys)})")
    
    # Step 2: Rate limiting
    with api_lock:
        current_time = time.time()

        # Initialize on first API call
        if minute_start is None:
            minute_start = current_time
        if hour_start is None:
            hour_start = current_time
        
        # Reset counters if windows expired
        if current_time - minute_start >= 60:
            calls_this_minute = 0
            minute_start = current_time
        
        if current_time - hour_start >= 3600:
            calls_this_hour = 0
            hour_start = current_time
        
        # Check if we're at limits
        if calls_this_minute >= 600:
            wait_time = (60 - (current_time - minute_start)) + 5
            logger.warning(f"Minute limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            calls_this_minute = 0
            minute_start = time.time()
        
        if calls_this_hour >= 5000:
            wait_time = 3600 - ((current_time - hour_start)) +5
            logger.warning(f"Hour limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            calls_this_hour = 0
            hour_start = time.time()
        
        # Standard delay between calls
        elapsed = current_time - last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        
        last_call_time = time.time()
        calls_this_minute += len(needs_fetch)
        calls_this_hour += len(needs_fetch)
    
    # Step 3: Prepare batch API call
    lats = [loc[0] for loc in needs_fetch]
    lons = [loc[1] for loc in needs_fetch]
    
    params = {
        "latitude": lats,
        "longitude": lons,
        "past_days": 1,
        "forecast_days": 0,
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone":"UTC"
    }
    
    try:
        # Step 4: Make API call
        responses = openmeteo.weather_api(OPENMETEO_URL, params=params)
        
        # Step 5: Process each location's response
        for idx, response in enumerate(responses):
            lat, lon, grid_key = needs_fetch[idx]
            
            hourly = response.Hourly()
            
            # Extract weather data for this specific hour
            weather_data = {
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy()[hour],
                "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy()[hour],
                "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy()[hour],
                "wind_direction_10m": hourly.Variables(3).ValuesAsNumpy()[hour],
                "cloud_cover": hourly.Variables(4).ValuesAsNumpy()[hour],
                "pressure_msl": hourly.Variables(5).ValuesAsNumpy()[hour]
            }
            
            # Store in results
            results[grid_key] = weather_data
            
            # Cache for future use
            cache_key = make_cache_key(lat, lon, hour)
            weather_cache[cache_key] = weather_data
        
        # Step 6: Track API usage
        api_call_count += len(needs_fetch)
        cache_misses += len(needs_fetch)
        
        if api_call_count % 100 == 0:
            logger.info(f"API calls: {api_call_count} | Cache hits: {cache_hits} | Misses: {cache_misses}")

        log_memory_usage(f"[fetch_weather_batch end] hour={hour}, processed={len(needs_fetch)}")
        
    except Exception as e:
        logger.error(f"Batch weather fetch failed for hour {hour}: {e}")
        
        # Return NaN for all failed locations (will be interpolated later)
        for lat, lon, grid_key in needs_fetch:
            results[grid_key] = {
                "temperature_2m": np.nan,
                "relative_humidity_2m": np.nan,
                "wind_speed_10m": np.nan,
                "wind_direction_10m": np.nan,
                "cloud_cover": np.nan,
                "pressure_msl": np.nan
            }
    
    return results

# ============= Main Script Start =============

logger.info("Starting balloon weather data pipeline")
logger.info(f"Keyframe hours: {KEYFRAME_HOURS}")
logger.info(f"Batch size: {BATCH_SIZE} balloons")
logger.info("="*60)

log_memory_usage("Startup")

logger.info("Starting balloon data download")

for hour in range(24):
    url = f"{WINDBORNE_BASE_URL}{hour:02d}.json"
    output_path = f"data/{hour:02d}.json"
    
    max_retries = 2 if hour == 0 else 1  # Retry hour 00 once more
    retry_delay = 300 if hour == 0 else 0  # 5 minutes for hour 00

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(output_path, "w") as f:
                f.write(response.text)
            logger.info(f"Downloaded hour {hour:02d}")
            break  
        except requests.exceptions.RequestException as e:
            if hour == 0 and attempt < max_retries-1:
                logger.warning(f"Hour 00 not available, retrying in 5 minutes...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download hour {hour:02d}: {e}")
                with open(output_path, "w") as f:
                    f.write("[]")
                break
log_memory_usage("After all downloads")        
# ============= Load and Process Balloon Data =============

logger.info("Processing all 24 hours of balloon data")

for hour in range(24):
    file_path = f"data/{hour:02d}.json"
    
    # Skip if file doesn't exist or is empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.warning(f"Skipping hour {hour:02d} - file missing or empty")
        continue
    
    # Load balloon positions from JSON
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hour {hour:02d}: {e}")
            continue
    
    logger.info(f"Processing hour {hour:02d} - {len(data)} balloons")
    
    # Parse balloon positions into structured format
    balloons = []
    for balloon_idx, triple in enumerate(data):
        if not triple or len(triple) != 3:
            continue
        
        lat, lon, alt = (safe_float(x) for x in triple)
        
        # Skip invalid positions
        if np.isnan(lat) or np.isnan(lon) or np.isnan(alt):
            continue
        
        balloons.append({
            "idx": balloon_idx,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "hour": hour
        })
    
    logger.info(f"Valid balloons: {len(balloons)}")
    
    # Check if this is a keyframe hour
    if hour in KEYFRAME_HOURS:
        logger.info(f"Keyframe hour {hour:02d} - fetching weather")
        
        # Group balloons by grid (deduplication)
        grid_groups = group_balloons_by_grid(balloons)
        logger.info(f"Unique grid cells: {len(grid_groups)} (reduced from {len(balloons)} balloons)")
        
        # Extract unique grid locations
        unique_grids = list(grid_groups.keys())
        
        # Batch unique grids into groups of BATCH_SIZE
        batches = [unique_grids[i:i+BATCH_SIZE] for i in range(0, len(unique_grids), BATCH_SIZE)]
        
        # Fetch weather for each batch
        weather_by_grid = {}
        for batch in batches:
            batch_results = fetch_weather_batch(batch, hour)
            weather_by_grid.update(batch_results)
            log_memory_usage(f"After weather fetch batch (hour {hour:02d}, size {len(batch)})")
        
        # === Map weather back to all balloons (preserving original coords) ===
        for grid_key, balloon_list in grid_groups.items():
            # Get weather for this grid cell; may be NaN-filled dict if failed
            weather = weather_by_grid.get(grid_key, {
                "temperature_2m": np.nan,
                "relative_humidity_2m": np.nan,
                "wind_speed_10m": np.nan,
                "wind_direction_10m": np.nan,
                "cloud_cover": np.nan,
                "pressure_msl": np.nan
            })
            
            # Append one record per balloon in this grid cell
            for balloon in balloon_list:
                history.append({
                    "balloon_index": balloon["idx"],
                    "hour": hour,
                    "latitude": balloon["lat"],
                    "longitude": balloon["lon"],
                    "altitude": balloon["alt"],
                    "temperature_2m": weather["temperature_2m"],
                    "relative_humidity_2m": weather["relative_humidity_2m"],
                    "wind_speed_10m": weather["wind_speed_10m"],
                    "wind_direction_10m": weather["wind_direction_10m"],
                    "cloud_cover": weather["cloud_cover"],
                    "pressure_msl": weather["pressure_msl"]
                })
    else:
        # Non-keyframe hour - store positions with NaN weather
        logger.info(f"Non-keyframe hour {hour:02d} - storing positions with NaN weather")
        
        for balloon in balloons:
            history.append({
                "balloon_index": balloon["idx"],
                "hour": hour,
                "latitude": balloon["lat"],
                "longitude": balloon["lon"],
                "altitude": balloon["alt"],
                "temperature_2m": np.nan,
                "relative_humidity_2m": np.nan,
                "wind_speed_10m": np.nan,
                "wind_direction_10m": np.nan,
                "cloud_cover": np.nan,
                "pressure_msl": np.nan
            })
    log_memory_usage(f"After processing hour {hour:02d}")    

logger.info(f"Data collection complete. Total records: {len(history)}")

# ============= Conversion to DF, Handle Missing Data, and Pivoting =============

df = pd.DataFrame(history)
log_memory_usage("After creating DataFrame from history")
df = df.sort_values(["balloon_index", "hour"])

logger.info(f"Starting processing with {len(df)} balloon records")

# Replace out of range values with NaN
df.loc[~df["latitude"].between(-90, 90), "latitude"] = np.nan
df.loc[~df["longitude"].between(-180, 180), "longitude"] = np.nan
df.loc[~df["altitude"].between(0, 50), "altitude"] = np.nan

invalid_count = df[["latitude", "longitude", "altitude"]].isna().sum().sum()
if invalid_count > 0:
    logger.warning(f"Found {invalid_count} out-of-range position values, set to NaN")

# Derive u/v wind components BEFORE interpolation
if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
    rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = df["wind_speed_10m"] * np.cos(rad)

# Columns to interpolate
weather_cols = [
    "temperature_2m", "relative_humidity_2m", "cloud_cover", 
    "pressure_msl", "wind_u", "wind_v"
]
pos_cols = ["latitude", "longitude", "altitude"]

# Create MultiIndex with all balloon_index × hour combinations
df = df.set_index(["balloon_index", "hour"])
full_index = pd.MultiIndex.from_product(
    [df.index.get_level_values("balloon_index").unique(), range(24)],
    names=["balloon_index", "hour"]
)
df = df.reindex(full_index).reset_index()

logger.info(f"After reindexing: {len(df)} records (should be {len(df['balloon_index'].unique())} × 24)")

# ============= INTERPOLATE WEATHER DATA =============

# Interpolate weather columns (linear within each balloon)
df[weather_cols] = (
    df.groupby("balloon_index")[weather_cols]
      .transform(lambda g: g.interpolate(method="linear", limit_direction="both"))
)

# Forward/backward fill any remaining NaNs at edges
df[weather_cols] = (
    df.groupby("balloon_index")[weather_cols]
      .transform(lambda g: g.ffill().bfill())
)
log_memory_usage("After weather interpolation steps")
# ============= INTERPOLATE POSITION DATA =============

# Interpolate position columns (linear within each balloon)
df[pos_cols] = (
    df.groupby("balloon_index")[pos_cols]
      .transform(lambda g: g.interpolate(method="linear", limit_direction="both"))
)

# Forward/backward fill any remaining NaNs
df[pos_cols] = (
    df.groupby("balloon_index")[pos_cols]
      .transform(lambda g: g.ffill().bfill())
)
log_memory_usage("After all interpolation steps")
# Reconstruct wind speed/direction from interpolated u/v
df["wind_speed_10m"] = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
df["wind_direction_10m"] = np.rad2deg(np.arctan2(df["wind_u"], df["wind_v"])) % 360

# Pivot table
pivot_cols = [
    "latitude", "longitude", "altitude",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "pressure_msl",
    "wind_u", "wind_v"
]

# Handle remaining NaNs
initial_count = len(df["balloon_index"].unique())
df = df.dropna(subset=pivot_cols)
final_count = len(df["balloon_index"].unique())

if initial_count != final_count:
    logger.warning(f"Dropped {initial_count - final_count} balloons due to incomplete data")

# --- Interpolation summary ---
logger.info(f"Final interpolated dataset shape: {df.shape}")
logger.info("Weather interpolation coverage:")
for col in weather_cols:
    pct = df[col].notna().mean() * 100
    logger.info(f"  {col}: {pct:.1f}% non-NaN values")

# Save a copy before pivoting
flight_df = df.copy()

df = df.pivot(index="balloon_index", columns="hour", values=pivot_cols)
df.columns = [f"{col[0]}_h{col[1]}" for col in df.columns]
df.reset_index(inplace=True)

output_path = "processed_balloon_data.csv"
df.to_csv(output_path, index=False)

log_memory_usage("After saving CSV")

logger.info(f"Processing complete. Total balloons: {len(df)}")

# ============= Upload to R2 (Optional - scheduler handles this) =============
# Uncomment if you want pipeline.py to upload independently
try:
    from utils.r2_helper import upload_file
    logger.info("Uploading processed CSV to R2...")
    upload_file(output_path, output_path)
except Exception as e:
    logger.error(f"Failed to upload to R2: {e}")