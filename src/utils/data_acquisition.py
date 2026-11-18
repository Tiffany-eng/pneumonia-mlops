"""Data acquisition: download the Kaggle chest X-ray dataset into data/train and data/test."""
import shutil
from pathlib import Path

from src.utils.config import ALLOWED_EXT, CLASS_NAMES, TEST_DIR, TRAIN_DIR


def download_dataset():
    """Needs Kaggle credentials (~/.kaggle/kaggle.json), or run it on Kaggle."""
    import kagglehub

    path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    src = next(path.rglob("chest_xray"), path)
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            from_dir = src / split / cls
            if not from_dir.exists():
                continue
            # the official val split has only 16 images -> merge it into train and re-split later
            to_dir = (TEST_DIR if split == "test" else TRAIN_DIR) / cls
            to_dir.mkdir(parents=True, exist_ok=True)
            for f in from_dir.iterdir():
                if f.suffix.lower() in ALLOWED_EXT:
                    shutil.copy(f, to_dir / f"{split}_{f.name}")
    print("Dataset ready in", TRAIN_DIR, "and", TEST_DIR)
