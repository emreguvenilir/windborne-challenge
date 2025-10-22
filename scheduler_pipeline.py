import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
import sys
from utils.r2_helper import upload_file, upload_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PIPELINE] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run full pipeline: data fetch → train model → upload to R2"""
    logger.info("🔄 Starting daily pipeline job")
    
    try:
        # Step 1: Fetch weather data and build CSV
        logger.info("Running pipeline.py...")
        result = subprocess.run(['python', 'pipeline.py'], check=True, capture_output=True, text=True)
        logger.info(f"pipeline.py output:\n{result.stdout}")
        
        # Step 2: Train model
        logger.info("Training model...")
        result = subprocess.run(['python', 'model1.py'], check=True, capture_output=True, text=True)
        logger.info(f"model1.py output:\n{result.stdout}")
        
        # Step 3: Upload outputs to R2
        logger.info("Uploading to R2...")
        upload_file('processed_balloon_data.csv', 'processed_balloon_data.csv')
        upload_file('model.h5', 'model.h5')
        upload_file('scaler.pkl', 'scaler.pkl')
        upload_file('model_metrics.json', 'model_metrics.json')
        upload_file('static/loss_curve.png', 'static/loss_curve.png')
        
        # Step 4: Generate initial predictions
        logger.info("Generating initial predictions...")
        from app import build_predictions_batch, fetch_latest_full_dataset, update_csv_with_new_positions
        import json
        
        all_data = fetch_latest_full_dataset()
        update_csv_with_new_positions(all_data)
        predictions = build_predictions_batch()
        
        upload_json(predictions, 'current_balloon_data.json')
        
        logger.info("✅ Pipeline complete")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error(f"stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()

    # Run daily at 6AM EST (11AM UTC)
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour=11, minute=0),
        id='daily_pipeline',
        max_instances=1,
        coalesce=True
    )

    logger.info("🚀 Pipeline scheduler started (runs daily at 6AM EST / 11AM UTC)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")