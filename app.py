from flask import Flask, jsonify, render_template
import json, os, time
import requests
import logging
from utils.r2_helper import download_file, download_json

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
CSV_FILE = "processed_balloon_data.csv"
METRICS_FILE = "model_metrics.json"
PREDICTIONS_FILE = "current_balloon_data.json"
WINDBORNE_URL = "https://a.windbornesystems.com/treasure/"

# Cache settings
CACHE_TTL = 300  # 5 minutes
prediction_cache = {"data": None, "timestamp": 0}
metrics_cache = {"data": None, "timestamp": 0}

# ==================== HELPER FUNCTIONS (used by scheduler_update.py) ====================

def fetch_latest_full_dataset():
    """Fetch all 24 hours of balloon data - used by schedulers"""
    logger.info("🔄 Fetching full 24-hour dataset from Windborne...")
    all_data = {}
    
    for hour in range(24):
        url = f"{WINDBORNE_URL}{hour:02d}.json"
        
        max_retries = 2 if hour == 0 else 1
        retry_delay = 300 if hour == 0 else 0  # 5 min for hour 00
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                all_data[hour] = data
                logger.info(f"  ✓ Fetched hour {hour:02d} ({len(data)} balloons)")
                break
            except requests.exceptions.RequestException as e:
                if hour == 0 and attempt < max_retries - 1:
                    logger.warning(f"Hour 00 not available (attempt {attempt+1}), retrying in 5 minutes...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"  ✗ Failed hour {hour:02d}: {e}")
                    all_data[hour] = []
                    break
    
    return all_data

# ==================== FLASK ROUTES ====================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/balloons")
def get_balloons():
    """Serve pre-computed predictions from R2 with local caching"""
    try:
        current_time = time.time()
        
        # Check local cache first
        if (prediction_cache["data"] is not None and 
            current_time - prediction_cache["timestamp"] < CACHE_TTL):
            logger.info(f"Serving {len(prediction_cache['data'])} balloons from local cache")
            return jsonify(prediction_cache["data"])
        
        # Cache miss - download from R2
        logger.info("Cache miss - downloading predictions from R2...")
        
        # Try downloading from R2
        data = download_json('current_balloon_data.json')
        
        if data is None:
            # Fallback to local file if R2 fails
            if os.path.exists(PREDICTIONS_FILE):
                logger.warning("R2 download failed, using local file")
                with open(PREDICTIONS_FILE) as f:
                    data = json.load(f)
            else:
                return jsonify({"error": "Predictions not yet available. Please wait for scheduler to generate predictions."}), 503
        
        # Update cache
        prediction_cache["data"] = data
        prediction_cache["timestamp"] = current_time
        
        logger.info(f"Served {len(data)} balloons from R2")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Failed to serve balloons: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/model_metrics")
def get_model_metrics():
    """Serve model performance metrics with caching"""
    try:
        current_time = time.time()
        
        # Check cache (6 hour TTL for metrics)
        if (metrics_cache["data"] is not None and 
            current_time - metrics_cache["timestamp"] < 21600):
            return jsonify(metrics_cache["data"])
        
        # Cache miss - download from R2
        logger.info("Downloading metrics from R2...")
        metrics = download_json('model_metrics.json')
        
        if metrics is None:
            # Fallback to local file
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE) as f:
                    metrics = json.load(f)
            else:
                return jsonify({"error": "model_metrics.json not found"}), 404
        
        # Add last modified time
        metrics["last_updated"] = time.strftime(
            "%Y-%m-%d %H:%M:%S UTC", 
            time.gmtime()
        )
        
        # Update cache
        metrics_cache["data"] = metrics
        metrics_cache["timestamp"] = current_time
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Failed to serve metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    """Health check endpoint with R2 connectivity status"""
    try:
        from utils.r2_helper import file_exists
        r2_status = "connected" if file_exists('current_balloon_data.json') else "disconnected"
    except:
        r2_status = "error"
    
    return jsonify({
        "status": "healthy",
        "r2_connection": r2_status,
        "cache_status": {
            "predictions_cached": prediction_cache["data"] is not None,
            "metrics_cached": metrics_cache["data"] is not None
        },
        "last_prediction_cache_update": time.strftime(
            "%Y-%m-%d %H:%M:%S UTC",
            time.gmtime(prediction_cache["timestamp"])
        ) if prediction_cache["timestamp"] > 0 else None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)