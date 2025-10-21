from flask import Flask, jsonify, render_template
import json, os, time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import requests
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths - use persistent disk if available, fallback to local
DATA_FILE = "/mnt/data/current_balloon_data.json" if os.path.exists("/mnt/data") else "current_balloon_data.json"
CSV_FILE = "processed_balloon_data.csv"
MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"
METRICS_FILE = "model_metrics.json"
WINDBORNE_URL = "https://a.windbornesystems.com/treasure/" 

logger.info("Loading model...")

# Load model and scalers
model = load_model(MODEL_FILE, compile=False)
scaler_obj = joblib.load(SCALER_FILE)
input_scaler = scaler_obj["scaler"]
output_scaler = scaler_obj["output_scaler"]
feature_cols = scaler_obj["features"]

# ==================== FUNCTIONS FOR CRON JOB ====================

def fetch_latest_full_dataset():
    """Fetch all 24 hours of balloon data - called by cron job"""
    logger.info("🔄 Fetching full 24-hour dataset from Windborne...")
    all_data = {}
    
    for hour in range(24):
        url = f"{WINDBORNE_URL}{hour:02d}.json"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            all_data[hour] = data
            logger.info(f"  ✓ Fetched hour {hour:02d} ({len(data)} balloons)")
        except Exception as e:
            logger.warning(f"  ✗ Failed hour {hour:02d}: {e}")
            all_data[hour] = []
    
    return all_data

def update_csv_with_new_positions(all_data):
    """Update processed_balloon_data.csv with fresh balloon positions"""
    if not os.path.exists(CSV_FILE):
        logger.error("CSV file missing!")
        return False
    
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Updating CSV with {len(df)} balloons...")
    
    # Update position columns only
    for hour in range(24):
        lat_col = f"latitude_h{hour}"
        lon_col = f"longitude_h{hour}"
        alt_col = f"altitude_h{hour}"
        
        if hour in all_data:
            for balloon_id in range(len(df)):
                if balloon_id < len(all_data[hour]) and all_data[hour][balloon_id]:
                    try:
                        lat, lon, alt = map(float, all_data[hour][balloon_id])
                        df.at[balloon_id, lat_col] = lat
                        df.at[balloon_id, lon_col] = lon
                        df.at[balloon_id, alt_col] = alt
                    except:
                        continue
    
    df.to_csv(CSV_FILE, index=False)
    logger.info("✅ CSV updated with new positions")
    return True

def build_predictions_batch(batch_size=200):
    """Generate predictions with next-hour delta predictions in memory-safe batches"""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("processed_balloon_data.csv missing!")

    logger.info("📦 Building prediction snapshot (batched inference)...")
    df = pd.read_csv(CSV_FILE)

    n = len(df)
    preds_all = []

    # Process in chunks to avoid OOM
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]

        data = batch[feature_cols].values.reshape(len(batch), 24, 11)
        X_pred = data[:, :10, :][:, ::-1, :]

        # scale inputs
        X_pred_scaled = np.array([input_scaler.transform(x) for x in X_pred])

        # predict deltas
        preds_scaled = model.predict(X_pred_scaled, verbose=0)
        preds_all.append(output_scaler.inverse_transform(preds_scaled))

        logger.info(f"  → processed {end}/{n} rows")

    # Concatenate all predictions
    deltas = np.vstack(preds_all)

    # Compute next-hour positions
    last_lat = df["latitude_h0"].values
    last_lon = df["longitude_h0"].values
    last_alt = df["altitude_h0"].values

    df["pred_latitude"]  = np.clip(last_lat + deltas[:, 0], -90, 90)
    df["pred_longitude"] = np.clip(last_lon + deltas[:, 1], -180, 180)
    df["pred_altitude"]  = np.clip(last_alt + deltas[:, 2], 0, None)

    # Build compact snapshot list
    snapshot = [
        {
            "balloon_id": int(row["balloon_index"]),
            "latitude": row["latitude_h23"],
            "longitude": row["longitude_h23"],
            "altitude": row["altitude_h23"],
            "pred_latitude": row["pred_latitude"],
            "pred_longitude": row["pred_longitude"],
            "pred_altitude": row["pred_altitude"],
            "temperature_2m": row["temperature_2m_h23"],
            "relative_humidity_2m": row["relative_humidity_2m_h23"],
            "wind_speed_10m": row["wind_speed_10m_h23"],
            "wind_direction_10m": row["wind_direction_10m_h23"],
            "cloud_cover": row["cloud_cover_h23"],
            "pressure_msl": row["pressure_msl_h23"],
            "wind_u": row["wind_u_h23"],
            "wind_v": row["wind_v_h23"]
        }
        for _, row in df.iterrows()
    ]

    logger.info(f"✅ Snapshot complete — {len(df)} balloons")
    return snapshot

# ==================== ROUTES ====================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/balloons")
def get_balloons():
    """Serve pre-computed predictions from persistent disk (updated by cron job)"""
    try:
        if not os.path.exists(DATA_FILE):
            logger.error(f"{DATA_FILE} not found - waiting for cron job to generate predictions")
            return jsonify({"error": "Predictions not yet available. Please wait for hourly update."}), 503
        
        with open(DATA_FILE) as f:
            data = json.load(f)
        
        logger.info(f"Served {len(data)} balloons from cache")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Failed to serve balloons: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model_metrics")
def get_model_metrics():
    """Serve model performance metrics"""
    if not os.path.exists(METRICS_FILE):
        return jsonify({"error": "model_metrics.json not found"}), 404
    
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    
    # Add last modified time for "Last Updated" field
    metrics["last_updated"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", 
        time.localtime(os.path.getmtime(METRICS_FILE))
    )
    return jsonify(metrics)

@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "predictions_available": os.path.exists(DATA_FILE),
        "last_csv_update": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(os.path.getmtime(CSV_FILE))
        ) if os.path.exists(CSV_FILE) else None,
        "last_prediction_update": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(os.path.getmtime(DATA_FILE))
        ) if os.path.exists(DATA_FILE) else None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)