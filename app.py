from flask import Flask, jsonify, render_template
import json, os, time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import requests
import logging
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = "current_balloon_data.json"
CSV_FILE = "processed_balloon_data.csv"
MODEL_FILE = "model.keras"
SCALER_FILE = "scaler.pkl"
METRICS_FILE = "model_metrics.json"
WINDBORNE_URL = "https://a.windbornesystems.com/treasure/" 
LOSS_CURVE = "loss_curve.png"
updating = False

if not os.path.exists(CSV_FILE):
    logger.info("CSV not found - fetching data...")
    os.system("python pipeline.py")
    logger.info("Data fetch complete!")

if not os.path.exists(MODEL_FILE):
    logger.info("Model not found - training from CSV...")
    os.system("python model1.py")
    logger.info("Model training complete!")
else:
    logger.info("Loading existing model...")

# Load model and scalers
model = load_model(MODEL_FILE, compile=False)
scaler_obj = joblib.load(SCALER_FILE)
input_scaler = scaler_obj["scaler"]
output_scaler = scaler_obj["output_scaler"]
feature_cols = scaler_obj["features"]

def build_snapshot():
    """Generate snapshot with next-hour delta predictions"""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("processed_balloon_data.csv missing!")

    df = pd.read_csv(CSV_FILE)

    # Prepare input
    data = df[feature_cols].values.reshape(len(df), 24, 11)
    X_pred = data[:, :10, :][:, ::-1, :]
    X_pred_scaled = np.array([input_scaler.transform(x) for x in X_pred])

    # Predict next-hour delta for lat/lon/alt
    preds_scaled = model.predict(X_pred_scaled, verbose=0)
    deltas = output_scaler.inverse_transform(preds_scaled)  # Δlat, Δlon, Δalt

    # Get last known actual coordinates (hour 0)
    last_lat = df["latitude_h0"].values
    last_lon = df["longitude_h0"].values
    last_alt = df["altitude_h0"].values

    # Compute predicted next-hour positions
    pred_lat = last_lat + deltas[:, 0]
    pred_lon = last_lon + deltas[:, 1]
    pred_alt = np.clip(last_alt + deltas[:, 2], 0, None)  # alt >= 0

    # Keep coordinates within bounds
    pred_lat = np.clip(pred_lat, -90, 90)
    pred_lon = np.clip(pred_lon, -180, 180)

    # Store predictions in dataframe
    df["pred_latitude"] = pred_lat
    df["pred_longitude"] = pred_lon
    df["pred_altitude"] = pred_alt

    snapshot = []
    for _, row in df.iterrows():
        snapshot.append({
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
        })

    with open(DATA_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)

def update_positions_from_windborne():
    """
    Refresh just balloon position data (lat/lon/alt) in processed_balloon_data.csv
    using latest Windborne JSONs. Keeps weather data untouched.
    """
    if not os.path.exists(CSV_FILE):
        logger.warning("processed_balloon_data.csv missing; skipping position update.")
        return

    logger.info("🔄 Fetching latest Windborne position data...")
    # Load existing dataset
    df = pd.read_csv(CSV_FILE)

    # Fetch latest JSONs (fast)
    latest_positions = {}
    for hour in range(24):
        url = f"{WINDBORNE_URL}{hour:02d}.json"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            for i, triple in enumerate(data):
                if not triple or len(triple) != 3:
                    continue
                lat, lon, alt = map(float, triple)
                # Store latest seen position per balloon
                latest_positions.setdefault(i, {})[hour] = (lat, lon, alt)
        except Exception as e:
            logger.warning(f"Failed to fetch hour {hour:02d}: {e}")
            continue

    # Overwrite only latitude/longitude/altitude columns
    for hour in range(24):
        lat_col = f"latitude_h{hour}"
        lon_col = f"longitude_h{hour}"
        alt_col = f"altitude_h{hour}"

        if lat_col in df.columns and lon_col in df.columns and alt_col in df.columns:
            for idx in range(len(df)):
                balloon_id = df.loc[idx, "balloon_index"]
                if balloon_id in latest_positions and hour in latest_positions[balloon_id]:
                    lat, lon, alt = latest_positions[balloon_id][hour]
                    df.at[idx, lat_col] = lat
                    df.at[idx, lon_col] = lon
                    df.at[idx, alt_col] = alt

    df.to_csv(CSV_FILE, index=False)
    logger.info("✅ Positions updated")

# ==================== SCHEDULED BACKGROUND TASK ====================

def full_data_update():
    """
    Runs every 8 hours: fetch weather + retrain model
    Runs in background thread - doesn't interrupt users
    """
    global updating
    updating = True
    logger.info("=" * 60)
    logger.info("🔄 Starting scheduled full update...")
    logger.info("=" * 60)
    
    try:
        # Step 1: Fetch balloon + weather data
        logger.info("Step 1/2: Fetching data (pipeline.py)...")
        start_time = time.time()
        exit_code = os.system("python pipeline.py")
        
        if exit_code != 0:
            logger.error(f"❌ pipeline.py failed with exit code {exit_code}")
            return
        
        logger.info(f"✅ Data fetched in {time.time() - start_time:.1f}s")
        
        # Step 2: Retrain model
        logger.info("Step 2/2: Retraining model (model1.py)...")
        start_time = time.time()
        exit_code = os.system("python model1.py")
        
        if exit_code != 0:
            logger.error(f"❌ model1.py failed with exit code {exit_code}")
            return
        
        logger.info(f"✅ Model retrained in {time.time() - start_time:.1f}s")
        
        # Step 3: Reload model in memory
        logger.info("Step 3/3: Reloading model into memory...")
        global model, scaler_obj, input_scaler, output_scaler, feature_cols
        model = load_model(MODEL_FILE, compile=False)
        scaler_obj = joblib.load(SCALER_FILE)
        input_scaler = scaler_obj["scaler"]
        output_scaler = scaler_obj["output_scaler"]
        feature_cols = scaler_obj["features"]
        
        logger.info("✅ Model reloaded")
        logger.info("=" * 60)
        logger.info("🎉 FULL UPDATE COMPLETE!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Full update failed: {e}", exc_info=True)
    finally:
        updating = False

# Setup scheduler
scheduler = BackgroundScheduler()

# Schedule full update every 8 hours
scheduler.add_job(
    func=full_data_update,
    trigger='interval',
    hours=24,
    id='full_update',
    name='Full data + model update',
    replace_existing=True
)

# ==================== ROUTES ====================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/balloons")
def get_balloons():
    global updating
    # If updating, return stale cached data (don't rebuild)
    if updating:
        if os.path.exists(DATA_FILE):
            logger.info("Serving cached data during update")
            with open(DATA_FILE) as f:
                return jsonify(json.load(f))
    
    # Normal path when not updating
    try:
        update_positions_from_windborne()
        build_snapshot()
        with open(DATA_FILE) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed: {e}")
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
    """Health check endpoint - monitor scheduler status"""
    next_run = scheduler.get_job('full_update').next_run_time if scheduler.get_job('full_update') else None
    
    return jsonify({
        "status": "healthy",
        "scheduler_running": scheduler.running,
        "next_full_update": next_run.isoformat() if next_run else None,
        "last_csv_update": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(os.path.getmtime(CSV_FILE))
        ) if os.path.exists(CSV_FILE) else None
    })

if __name__ == "__main__":
    scheduler.start()
    logger.info("Scheduler started")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)