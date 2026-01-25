import logging
import os
import sys
import json
from datetime import datetime

LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

SESSION_LOG_FILE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
SESSION_LOG_PATH = os.path.join(LOG_DIR, SESSION_LOG_FILE)

def get_logger(name):
    """
    Creates a logger instance that writes to both:
      1. Console (stdout)
      2. File (The shared SESSION_LOG_FILE)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(SESSION_LOG_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger(__name__)


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
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