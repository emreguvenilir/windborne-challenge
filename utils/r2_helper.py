# utils/r2_helper.py
import os
import json
import boto3
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# === Initialize S3 client for Cloudflare R2 ===
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",  # R2 uses 'auto'
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=30,
            connect_timeout=10,
            tcp_keepalive=True,
        ),
    )
    BUCKET = os.environ["R2_BUCKET_NAME"]
except KeyError as e:
    raise RuntimeError(f"Missing required R2 environment variable: {e}")

# === Core R2 helper functions ===

def upload_file(local_path: str, r2_key: str) -> bool:
    """Upload a file to R2."""
    try:
        s3_client.upload_file(local_path, BUCKET, r2_key)
        logger.info(f"✅ Uploaded {local_path} → r2://{BUCKET}/{r2_key}")
        return True
    except ClientError as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def download_file(r2_key: str, local_path: str) -> bool:
    """Download a file from R2."""
    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        s3_client.download_file(BUCKET, r2_key, local_path)
        logger.info(f"✅ Downloaded r2://{BUCKET}/{r2_key} → {local_path}")
        return True
    except ClientError as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def file_exists(r2_key: str) -> bool:
    """Check if a file exists in R2."""
    try:
        s3_client.head_object(Bucket=BUCKET, Key=r2_key)
        return True
    except ClientError:
        return False


def upload_json(data, r2_key: str) -> bool:
    """Upload a JSON object directly to R2."""
    try:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=r2_key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"✅ Uploaded JSON → r2://{BUCKET}/{r2_key}")
        return True
    except ClientError as e:
        logger.error(f"❌ JSON upload failed: {e}")
        return False


def download_json(r2_key: str):
    """Download and parse a JSON object from R2."""
    try:
        response = s3_client.get_object(Bucket=BUCKET, Key=r2_key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        logger.error(f"❌ JSON download failed: {e}")
        return None


def get_last_modified(r2_key: str):
    """Return last modified timestamp for a given R2 object (ISO format)."""
    try:
        metadata = s3_client.head_object(Bucket=BUCKET, Key=r2_key)
        return metadata["LastModified"].isoformat()
    except ClientError:
        return None
