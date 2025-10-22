import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import logging

logger = logging.getLogger(__name__)

CSV_FILE = "processed_balloon_data.csv"
MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"

logger.info("Loading model and scalers...")

# Load model and scalers
model = load_model(MODEL_FILE, compile=False)
scaler_obj = joblib.load(SCALER_FILE)
input_scaler = scaler_obj["scaler"]
output_scaler = scaler_obj["output_scaler"]
feature_cols = scaler_obj["features"]
logger.info("Model loaded successfully")

def update_csv_with_new_positions(all_data):
    """Update processed_balloon_data.csv with fresh balloon positions"""
    if not os.path.exists(CSV_FILE):
        logger.error("CSV file missing!")
        return False
    
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Updating CSV with {len(df)} balloons...")
    
    # Update position columns only (keep weather data from training)
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

        # Scale inputs
        X_pred_scaled = np.array([input_scaler.transform(x) for x in X_pred])

        # Predict deltas
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
            "latitude": row["latitude_h0"],
            "longitude": row["longitude_h0"],
            "altitude": row["altitude_h0"],
            "pred_latitude": row["pred_latitude"],
            "pred_longitude": row["pred_longitude"],
            "pred_altitude": row["pred_altitude"],
            "temperature_2m": row["temperature_2m_h0"],
            "relative_humidity_2m": row["relative_humidity_2m_h0"],
            "wind_speed_10m": row["wind_speed_10m_h0"],
            "wind_direction_10m": row["wind_direction_10m_h0"],
            "cloud_cover": row["cloud_cover_h0"],
            "pressure_msl": row["pressure_msl_h0"],
            "wind_u": row["wind_u_h0"],
            "wind_v": row["wind_v_h0"]
        }
        for _, row in df.iterrows()
    ]

    logger.info(f"✅ Snapshot complete — {len(df)} balloons")
    return snapshot