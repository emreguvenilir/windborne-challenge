import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import psutil

# ====== Memory Checks ========
def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[MEMORY] {stage}: {mem_mb:.2f} MB RSS")

df = pd.read_csv("processed_balloon_data.csv")
log_memory("After loading CSV")
features_per_hour = 11
hours = 24

# Drop index and reshape into 3D array 
feature_cols = [c for c in df.columns if c != "balloon_index"]
data = df[feature_cols].values

X_full = data.reshape(len(df), hours, features_per_hour)
log_memory("After reshaping into 3D array")

# Split balloons before creating sequences
balloon_ids = np.arange(len(df))
train_ids, test_ids = train_test_split(balloon_ids, test_size=0.2, random_state=42)
log_memory("After train/test split")

X_full_train = X_full[train_ids]
X_full_test = X_full[test_ids]

sequence_length = 10

def create_sequences(data):
    X, y = [], []
    for sample in data:
        sample = sample[::-1]
        for i in range(hours - sequence_length):
            X.append(sample[i:i+sequence_length])
            y.append(sample[i+sequence_length][:3] - sample[i+sequence_length-1][:3]) #Prediction displacement
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(X_full_train)
X_test, y_test = create_sequences(X_full_test)
log_memory("After creating sequences")

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Fit scalers
X_all_flat = X_full.reshape(-1, features_per_hour)
scaler = MinMaxScaler()
scaler.fit(X_all_flat)

output_scaler = MinMaxScaler()
output_scaler.fit(y_train) 

joblib.dump({"scaler": scaler, "features": feature_cols, "output_scaler": output_scaler}, "scaler.pkl")

# Transform train/test sets
X_train_scaled = np.array([scaler.transform(x) for x in X_train])
X_test_scaled  = np.array([scaler.transform(x) for x in X_test])
y_train_scaled = output_scaler.transform(y_train)
y_test_scaled  = output_scaler.transform(y_test)
log_memory("After scaling data")

# Model definition
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, features_per_hour)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(3)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
log_memory("After model compilation")

# Callbacks for early stopping and model checkpointing
os.makedirs("static", exist_ok=True)
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]
# Training
log_memory("Before training")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=75,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    callbacks=callbacks,
    verbose=1
)
log_memory("After training")
# Save final model
model.save("model.h5")

# Evaluation
predictions_scaled = model.predict(X_test_scaled, verbose=0)
log_memory("After prediction step")
predictions_inv = output_scaler.inverse_transform(predictions_scaled)
y_test_inv = output_scaler.inverse_transform(y_test_scaled)

# Overall metrics
mse = mean_squared_error(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)
overall_corr = np.corrcoef(y_test_inv.flatten(), predictions_inv.flatten())[0, 1]

# Per-feature metrics
features = ["latitude", "longitude", "altitude"]
per_feature = {}

for i, feat in enumerate(features):
    mse_i = mean_squared_error(y_test_inv[:, i], predictions_inv[:, i])
    mae_i = mean_absolute_error(y_test_inv[:, i], predictions_inv[:, i])
    corr_i = np.corrcoef(y_test_inv[:, i], predictions_inv[:, i])[0, 1]
    per_feature[feat] = {"mse": float(mse_i), "mae": float(mae_i), "corr": float(corr_i)}

# Loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_curve_path = os.path.join("static", "loss_curve.png")
plt.savefig(loss_curve_path, dpi=200)
plt.close()

# Save metrics summary to JSON
metrics_summary = {
    "overall": {
        "mse": float(mse),
        "mae": float(mae),
        "corr": float(overall_corr)
    },
    "key_features": {
        feat: per_feature[feat] for feat in features
    },
    "correlations": {
        feat: per_feature[feat]["corr"] for feat in features
    },
    "visualizations": {
        "loss_curve": "/static/loss_curve.png",
    }
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print(f"Overall MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {overall_corr:.4f}")
print(f"Files saved: model.h5, scaler.pkl, model_metrics.json, static/loss_curve.png")

# ============= Upload to R2 (Optional - scheduler handles this) =============
# Uncomment if you want model1.py to upload independently
try:
    from utils.r2_helper import upload_file
    print("Uploading model files to R2...")
    upload_file('model.h5', 'model.h5')
    upload_file('scaler.pkl', 'scaler.pkl')
    upload_file('model_metrics.json', 'model_metrics.json')
    upload_file('static/loss_curve.png', 'static/loss_curve.png')
except Exception as e:
    print(f"Failed to upload to R2: {e}")

log_memory("Before exit")
