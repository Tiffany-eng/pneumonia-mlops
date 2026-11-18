"""Image validation and dataset counting helpers."""
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.utils.config import ALLOWED_EXT, CLASS_NAMES, MAX_IMAGE_BYTES


def is_valid_image(data: bytes) -> bool:
    """Reject corrupt, non-image or oversized files before they reach the database."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        return False
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def has_images(directory) -> bool:
    d = Path(directory)
    return d.exists() and any(p.suffix.lower() in ALLOWED_EXT for p in d.rglob("*"))


def count_images(directory) -> dict:
    d = Path(directory)
    return {c: sum(1 for p in (d / c).glob("*") if p.suffix.lower() in ALLOWED_EXT)
            if (d / c).exists() else 0 for c in CLASS_NAMES}
