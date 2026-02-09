import logging
import os
import sys
from pathlib import Path
import json
import polars as pl
from datetime import datetime, date
import random
import numpy as np
import torch
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "stock-prediction-data-chirag")
ENABLE_S3 = os.getenv("ENABLE_S3", "False").lower() == "true"

def download_from_s3(s3_key: str, local_path: str):
    """
    Safely attempts to download from S3.
    If S3 is disabled or fails, it checks if we have a local copy.
    If no local copy exists, THEN it raises an error.
    """
    if os.path.exists(local_path):
        logger.info(f"✅ Found local file: {local_path} (Skipping S3 download)")
        return

    if not ENABLE_S3:
        logger.warning(f"⚠️ S3 is disabled. File {local_path} is missing locally, and we can't fetch it.")
        raise FileNotFoundError(f"File {local_path} not found and S3 is disabled.")

    s3 = boto3.client('s3')
    try:
        logger.info(f"⬇️ Attempting download from {BUCKET_NAME}...")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        logger.info("✅ Download complete.")
    except Exception as e:
        logger.error(f"❌ S3 Download failed: {e}")
        raise e

def upload_to_s3(local_path: str, s3_key: str):
    """
    Safely attempts to upload to S3.
    If it fails (bucket deleted, no wifi), it LOGS a warning but DOES NOT CRASH.
    """
    if not ENABLE_S3:
        logger.info("ℹ️ S3 is disabled. Skipping upload.")
        return

    s3 = boto3.client('s3')
    try:
        logger.info(f"⬆️ Uploading to {BUCKET_NAME}...")
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        logger.info("✅ Upload complete.")
    except Exception as e:
        logger.warning(f"⚠️ Upload failed (Bucket might be deleted or network down). Continuing... Error: {e}")

def list_s3_files(prefix: str = "") -> List[str]:
    """
    Returns a list of filenames in the S3 bucket.
    """
    if not ENABLE_S3:
        return []

    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        # Extract filenames from the response
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents'] if not obj['Key'].endswith('/')]
        else:
            return []
    except Exception as e:
        logger.error(f"❌ Failed to list S3 files: {e}")
        return []

def set_global_seed(seed: int):
    """
    Enforces best-effort determinism across Python, NumPy and PyTorch.
    Call ONCE at program start.
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Disable nondeterministic optimizations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_logging(log_dir="logs"):
    """
    Configures the ROOT logger. 
    Run this exactly once at the start of your main script.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate filename once
    session_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = os.path.join(log_dir, session_filename)

    # Get the Root Logger (no name provided)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clean up any existing handlers (prevents double printing if re-run in notebooks)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create Formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler 1: Console
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    root_logger.addHandler(c_handler)

    # Handler 2: File
    f_handler = logging.FileHandler(log_path, mode='a')
    f_handler.setFormatter(formatter)
    root_logger.addHandler(f_handler)
    
    # Log that we started (so we know which file is the main one)
    root_logger.info(f"Logging setup complete. Saving to: {log_path}")
    
    return log_path

# 2. Define the Accessor Function (Call this everywhere else)
def get_logger(name):
    """
    Just returns a logger. It assumes setup_logging() was called in main.
    """
    # Simply return the logger. It will inherit handlers from the Root Logger.
    return logging.getLogger(name)

logger = get_logger(__name__)


def save_json(path, data):

    def default_converter(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, default=default_converter)
        logger.info(f"JSON saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise e

def load_json(path):
    if not os.path.exists(path):
        logger.error(f"JSON file not found: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"JSON loaded from {path}")
    return data

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_files_in_dir(dir_path: str):
    file_names = [f.name for f in Path(dir_path).iterdir() if f.is_file()]
    return file_names

def is_parquet_file(file_path: str):
    try:
        pl.read_parquet(file_path)
        return True
    except Exception:
        return False