import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import json

# Local testing: use current directory instead of /mnt/data
OUTPUT_PATH = "/mnt/data" if os.path.exists("/mnt/data") else "."
DATA_FILE = f"{OUTPUT_PATH}/current_balloon_data.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app import (
    fetch_latest_full_dataset, 
    update_csv_with_new_positions, 
    build_predictions_batch
)

if __name__ == "__main__":
    logger.info("🕐 Hourly prediction rebuild started")
    
    # Fetch all 24 hours
    all_data = fetch_latest_full_dataset()
    
    # Update CSV
    update_csv_with_new_positions(all_data)
    
    # Rebuild predictions
    predictions = build_predictions_batch()
    
    # Write to disk (will be current dir locally, /mnt/data on Render)
    with open(DATA_FILE, "w") as f:
        json.dump(predictions, f)
    
    logger.info(f"✅ Predictions written to {DATA_FILE}")