"""Image <-> array/base64 conversions."""
import base64
import io

import numpy as np
from PIL import Image

from src.utils.config import IMG_SIZE


def bytes_to_array(data: bytes) -> np.ndarray:
    """Raw image bytes -> (1, 224, 224, 3) float32 array in [0, 255].
    Scaling happens inside the model (Rescaling layer), so training and inference match exactly."""
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype="float32"), 0)


def thumbnail_b64(path, size=256) -> str:
    """Small JPEG thumbnail as base64 text, for showing sample images in the UI."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
