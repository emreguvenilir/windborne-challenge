import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from utils.r2_helper import download_file, upload_json, file_exists
import time
import psutil
try:
    import resource
except ImportError:
    resource = None

# ===== Memory tracking helpers =====
def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB RSS")

def log_peak_memory():
    if resource:
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage_mb = usage_kb / 1024 if os.name != "nt" else usage_kb / (1024 * 1024)
        print(f"[PEAK MEMORY] Maximum RSS: {usage_mb:.2f} MB")
    else:
        print("[PEAK MEMORY] Not available on this OS")

# ===== Logging setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [UPDATE] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def update_predictions():
    """Lightweight job: fetch new balloon positions → update predictions → upload to R2"""
    logger.info("🔄 Starting prediction update")
    log_memory("At job start")

    try:
        # Step 1: Download required files from R2 (cached locally)
        if not os.path.exists('processed_balloon_data.csv'):
            logger.info("Downloading CSV from R2...")
            download_file('processed_balloon_data.csv', 'processed_balloon_data.csv')
            log_memory("After CSV download")

        if not os.path.exists('model.h5'):
            logger.info("Downloading model from R2...")
            download_file('model.h5', 'model.h5')
            download_file('scaler.pkl', 'scaler.pkl')
            log_memory("After model/scaler download")

        # Step 2: Fetch latest balloon positions and update predictions
        from app import fetch_latest_full_dataset
        from predictor import update_csv_with_new_positions, build_predictions_batch
        
        all_data = fetch_latest_full_dataset()
        log_memory("After fetching latest dataset")

        update_csv_with_new_positions(all_data)
        
        predictions = build_predictions_batch()
        log_memory("After generating predictions")

        # Step 3: Upload updated predictions to R2
        upload_json(predictions, 'current_balloon_data.json')
        log_memory("After uploading predictions")

        logger.info(f"✅ Updated {len(predictions)} balloon predictions")
        
    except Exception as e:
        logger.error(f"❌ Update failed: {e}")

    log_peak_memory()
    logger.info("🧠 Memory stats logged for this run")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # Run every 5 minutes
    scheduler.add_job(
        update_predictions,
        IntervalTrigger(minutes=15),
        id='update_predictions',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30
    )
    
    logger.info("🚀 Update scheduler started (runs every 15 minutes)")
    
    # Run once immediately on startup
    update_predictions()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")