"""Dataset insights shown in the UI (class balance, brightness, contrast, image size)."""
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import ALLOWED_EXT, CLASS_NAMES, TRAIN_DIR
from src.utils.validation import count_images

_cache = {}


def compute_insights(sample_per_class=150):
    if "data" in _cache:
        return _cache["data"]
    rng = random.Random(42)
    stats = {}
    for cls in CLASS_NAMES:
        files = [p for p in (Path(TRAIN_DIR) / cls).glob("*") if p.suffix.lower() in ALLOWED_EXT]
        rng.shuffle(files)
        rows = []
        for f in files[:sample_per_class]:
            with Image.open(f) as im:
                w, h = im.size
                a = np.asarray(im.convert("L"), dtype="float32")
            rows.append({"brightness": float(a.mean()), "contrast": float(a.std()), "width": w, "height": h})
        stats[cls] = rows
    data = {"class_counts": count_images(TRAIN_DIR), "features": stats}
    _cache["data"] = data
    return data
