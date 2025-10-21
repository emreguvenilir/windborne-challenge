import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import json
import time
from datetime import datetime, timedelta
import requests
import psutil

OUTPUT_PATH = "/mnt/data" if os.path.exists("/mnt/data") else "."
DATA_FILE = f"{OUTPUT_PATH}/current_balloon_data.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from app import (
    update_csv_with_new_positions, 
    build_predictions_batch,
    WINDBORNE_URL
)

def log_resources(stage):
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss / (1024 ** 2)
    cpu = psutil.cpu_percent(interval=0.3)
    logger.info(f"[📊 {stage}] CPU={cpu:.1f}% | RAM={mem:.1f} MB")

def fetch_latest_full_dataset():
    """Fetch all 24 hours of balloon data with retry logic for hour 00"""
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

def rebuild_predictions():
    """Execute hourly prediction rebuild"""
    logger.info("🕐 Hourly prediction rebuild started")
    
    try:
        all_data = fetch_latest_full_dataset()
        log_resources("After fetch")
        update_csv_with_new_positions(all_data)
        predictions = build_predictions_batch()
        log_resources("After predictions")

        with open(DATA_FILE, "w") as f:
            json.dump(predictions, f)
        log_resources("After write")
        logger.info(f"✅ Predictions written to {DATA_FILE}")
        
    except Exception as e:
        logger.error(f"❌ Rebuild failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Background worker started - rebuilding predictions hourly")
    
    last_run_hour = -1
    
    while True:
        now = datetime.utcnow()
        current_hour = now.hour
        current_minute = now.minute
        
        # Run at the top of each hour (minute 0)
        if current_minute == 0 and current_hour != last_run_hour:
            logger.info(f"⏰ Top of hour {current_hour:02d}:00 UTC - triggering rebuild")
            rebuild_predictions()
            last_run_hour = current_hour
            time.sleep(60)  # Sleep 60s to avoid double-trigger
        else:
            # Sleep until the start of the next minute
            logger.info("Sleeping until start of next minute")
            next_minute = (datetime.utcnow().replace(second=0, microsecond=0) + timedelta(minutes=1))
            sleep_seconds = max(1,(next_minute - datetime.utcnow()).total_seconds())
            time.sleep(sleep_seconds)