"""Single place for constants, paths and tunable settings."""
import os
from pathlib import Path

IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 / 1 (sigmoid output = P(PNEUMONIA))
BATCH_SIZE = 32
SEED = 42
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
UPLOAD_DIR = ROOT / "data" / "uploads"
MODEL_PATH = ROOT / "models" / "pneumonia_model.h5"
METRICS_PATH = ROOT / "models" / "metrics.json"
ONNX_PATH = ROOT / "models" / "pneumonia_model.onnx"

MIN_RETRAIN_IMAGES = int(os.getenv("MIN_RETRAIN_IMAGES", "20"))        # retraining on 1 image is meaningless
AUTO_RETRAIN_THRESHOLD = int(os.getenv("AUTO_RETRAIN_THRESHOLD", "50"))  # automatic trigger
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH = 32
MAX_ZIP_BYTES = 300 * 1024 * 1024
MAX_ZIP_FILES = 5000
