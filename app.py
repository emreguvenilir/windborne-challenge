from flask import Flask, jsonify, render_template
import json, os, time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import requests

app = Flask(__name__)

DATA_FILE = "current_balloon_data.json"
CSV_FILE = "processed_balloon_data.csv"
MODEL_FILE = "model.keras"
SCALER_FILE = "scaler.pkl"
METRICS_FILE = "model_metrics.json"
WINDBORNE_URL = "https://a.windbornesystems.com/treasure/"

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

    # --- Prepare input ---
    data = df[feature_cols].values.reshape(len(df), 24, 11)
    X_pred = data[:, -10:, :]
    X_pred_scaled = np.array([input_scaler.transform(x) for x in X_pred])

    # --- Predict next-hour delta for lat/lon/alt ---
    preds_scaled = model.predict(X_pred_scaled, verbose=0)
    deltas = output_scaler.inverse_transform(preds_scaled)  # Δlat, Δlon, Δalt

    # --- Get last known actual coordinates (hour 23) ---
    last_lat = df["latitude_h23"].values
    last_lon = df["longitude_h23"].values
    last_alt = df["altitude_h23"].values

    # --- Compute predicted next-hour positions ---
    pred_lat = last_lat + deltas[:, 0]
    pred_lon = last_lon + deltas[:, 1]
    pred_alt = np.clip(last_alt + deltas[:, 2], 0, None)  # alt >= 0

    # --- Keep coordinates within bounds ---
    pred_lat = np.clip(pred_lat, -90, 90)
    pred_lon = np.clip(pred_lon, -180, 180)

    # --- Store predictions ---
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
        print("⚠️ processed_balloon_data.csv missing; skipping position update.")
        return

    print("🔄 Fetching latest Windborne position data...")
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
            print(f"⚠️ Failed to fetch hour {hour:02d}: {e}")
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

# ---------------- ROUTES ----------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/balloons")
def get_balloons():
    """Serve latest balloon snapshot, refreshing positions from Windborne if stale"""
    try:
        # Always refresh positions before building snapshot
        update_positions_from_windborne()
    except Exception as e:
        print(f"⚠️ Position update failed: {e}")

    build_snapshot()

    with open(DATA_FILE) as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/api/model_metrics")
def get_model_metrics():
    """Serve model performance metrics"""
    if not os.path.exists(METRICS_FILE):
        return jsonify({"error": "model_metrics.json not found"}), 404
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    # Add last modified time for "Last Updated" field
    metrics["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(METRICS_FILE)))
    return jsonify(metrics)

#reload FULL data for processed_balloon_data.csv
@app.route("/api/update_data", methods=["POST"])
def update_data():
    """Rebuild processed_balloon_data.csv by calling workingV1.py"""
    try:
        os.system("python3 workingV1.py")
        return jsonify({"status": "success", "message": "Data updated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
#retrain model by calling model1.py, only when processed_balloon_data.csv is updated
@app.route("/api/retrain_model", methods=["POST"])
def retrain_model():
    """Retrain LSTM model using the latest processed data"""
    try:
        os.system("python3 model1.py")
        build_snapshot()
        return jsonify({"status": "success", "message": "Model retrained and snapshot rebuilt"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
