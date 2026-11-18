"""Bulk upload: safely unpack a zip of labelled images (NORMAL/ and PNEUMONIA/ folders)."""
import io
import uuid
import zipfile
from pathlib import PurePosixPath

from src import database as db
from src.utils.config import ALLOWED_EXT, CLASS_NAMES, MAX_ZIP_FILES, UPLOAD_DIR
from src.utils.validation import is_valid_image


def label_from_path(name: str):
    """The label is the nearest parent folder named NORMAL or PNEUMONIA (case-insensitive)."""
    for part in reversed(PurePosixPath(name).parts[:-1]):
        if part.upper() in CLASS_NAMES:
            return part.upper()
    return None


def ingest_zip(data: bytes) -> dict:
    """Validate every file, save it under data/uploads/<label>/ and register it in SQLite."""
    saved = {c: 0 for c in CLASS_NAMES}
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_ZIP_FILES:
            raise ValueError(f"Zip has too many files (max {MAX_ZIP_FILES})")
        for m in members:
            name = m.filename.replace("\\", "/")
            ext = PurePosixPath(name).suffix.lower()
            label = label_from_path(name)
            if ".." in PurePosixPath(name).parts or name.startswith("/") \
                    or ext not in ALLOWED_EXT or label is None or "__MACOSX" in name:
                skipped += 1
                continue
            content = zf.read(m)
            if not is_valid_image(content):
                skipped += 1
                continue
            folder = UPLOAD_DIR / label
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{uuid.uuid4().hex}{ext}"
            path.write_bytes(content)
            db.add_upload(PurePosixPath(name).name, label, path)
            saved[label] += 1
    return {"saved": saved, "total_saved": sum(saved.values()), "skipped": skipped}
