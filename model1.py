import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

df = pd.read_csv("processed_balloon_data.csv")
features_per_hour = 11
hours = 24

# Drop index and reshape into 3D array 
feature_cols = [c for c in df.columns if c != "balloon_index"]
data = df[feature_cols].values

X_full = data.reshape(len(df), hours, features_per_hour)

# --- Create sequences (10-hour sliding windows) per balloon ---
sequence_length = 10
X, y = [], []
for sample in X_full:
    for i in range(hours - sequence_length):
        X.append(sample[i:i+sequence_length])
        y.append(sample[i+sequence_length][:3] - sample[i+sequence_length-1][:3])

X, y = np.array(X), np.array(y)

# --- Split before scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit scaler per feature (column-wise normalization) ---
scaler = MinMaxScaler()
flat_train = X_train.reshape(-1, features_per_hour)
scaler.fit(flat_train)

output_scaler = MinMaxScaler()
output_scaler.fit(y_train)

# Save both scaler and feature order for app.py
joblib.dump({"scaler": scaler, "features": feature_cols, "output_scaler": output_scaler}, "scaler.pkl")

# --- Transform train/test sets ---
X_train_scaled = np.array([scaler.transform(x) for x in X_train])
X_test_scaled  = np.array([scaler.transform(x) for x in X_test])
y_train_scaled = output_scaler.transform(y_train)
y_test_scaled  = output_scaler.transform(y_test)


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, features_per_hour)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(3)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32,
                    validation_data=(X_test_scaled, y_test_scaled),verbose=0)

model.save("model.keras")

# Evaluation
predictions_scaled = model.predict(X_test_scaled)
predictions_inv = output_scaler.inverse_transform(predictions_scaled)
y_test_inv = output_scaler.inverse_transform(y_test_scaled)

#Performance metrics saved to a json dump
mse = mean_squared_error(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)
overall_corr = np.corrcoef(y_test_inv.flatten(), predictions_inv.flatten())[0, 1]
#print(f"Overall MSE: {mse:.4f}")
#print(f"Overall MAE: {mae:.4f}")
#print(f"Overall Corr: {overall_corr:.4f}")
 
features = ["latitude", "longitude", "altitude"]

per_feature = {}
for i, feat in enumerate(features):
    mse_i = mean_squared_error(y_test_inv[:, i], predictions_inv[:, i])
    mae_i = mean_absolute_error(y_test_inv[:, i], predictions_inv[:, i])
    corr_i = np.corrcoef(y_test_inv[:, i], predictions_inv[:, i])[0, 1]
    per_feature[feat] = {"mse": float(mse_i), "mae": float(mae_i), "corr": float(corr_i)}
    #print(f"{feat:15s} → MSE: {mse_i:.4f}, MAE: {mae_i:.4f}, corr: {corr_i:.3f}")

# Loss curve saved to png

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()

loss_curve_path = os.path.join("static", "loss_curve.png")
plt.savefig(loss_curve_path, dpi=200)
plt.close()

# --- Store summarized metrics ---
metrics_summary = {
    "overall": {"mse": float(mse), "mae": float(mae), "corr": float(overall_corr)},
    "key_features": {
        feat: per_feature[feat] for feat in ["latitude", "longitude", "altitude"]
    },
    "correlations": {
        feat: per_feature[feat]["corr"] for feat in ["latitude", "longitude", "altitude"]
    },
    "loss_curve_path": "/static/loss_curve.png"
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

