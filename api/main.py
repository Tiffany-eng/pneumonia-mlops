"""FastAPI service: predict (single + batch), upload (files / zip), train, retrain, monitor."""
import json
import os
import time
import uuid
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import database as db
from src.insights import compute_insights
from src.prediction import classifier
from src.retrain import get_status, is_running, retrain, train_from_scratch
from src.utils.archive import ingest_zip
from src.utils.config import (ALLOWED_EXT, AUTO_RETRAIN_THRESHOLD, CLASS_NAMES, MAX_BATCH, MAX_ZIP_BYTES,
                              METRICS_PATH, MIN_RETRAIN_IMAGES, MODEL_PATH, TRAIN_DIR, UPLOAD_DIR)
from src.utils.image_io import thumbnail_b64
from src.utils.validation import is_valid_image

app = FastAPI(title="Pneumonia X-ray Classifier API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
STATS = {"requests": 0, "predictions": 0, "errors": 0, "latency_sum": 0.0}


@app.on_event("startup")
def startup():
    db.init_db()
    if MODEL_PATH.exists():
        classifier.load()


@app.middleware("http")
async def track(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise
    STATS["requests"] += 1
    STATS["latency_sum"] += time.perf_counter() - t0
    if response.status_code >= 500:
        STATS["errors"] += 1
    return response


def _require_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No trained model yet. Train one first (POST /train).")


# ----------------------------------------------------------------- health / monitoring
@app.get("/health")
def health():
    return {"status": "up", "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_loaded": MODEL_PATH.exists(), "retraining": is_running()}


@app.get("/metrics")
def metrics():
    model_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    n = max(STATS["requests"], 1)
    return {"model": model_metrics,
            "service": {**STATS, "avg_latency_ms": round(1000 * STATS["latency_sum"] / n, 2),
                        "uptime_seconds": round(time.time() - START_TIME, 1)}}


# ----------------------------------------------------------------- prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict ONE chest X-ray."""
    data = await file.read()
    if not is_valid_image(data):
        raise HTTPException(400, "Upload a valid JPG/PNG chest X-ray under 10 MB")
    _require_model()
    STATS["predictions"] += 1
    return classifier.predict(data)


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict several X-rays in one request."""
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"At most {MAX_BATCH} images per batch")
    _require_model()
    names, images = [], []
    for f in files:
        data = await f.read()
        if not is_valid_image(data):
            raise HTTPException(400, f"{f.filename} is not a valid JPG/PNG under 10 MB")
        names.append(f.filename)
        images.append(data)
    results = classifier.predict_batch(images)
    STATS["predictions"] += len(images)
    return {"count": len(results),
            "results": [{"filename": n, **r} for n, r in zip(names, results)]}


# ----------------------------------------------------------------- bulk upload
def _maybe_auto_retrain(background: BackgroundTasks):
    pending = db.pending_count()
    auto = pending >= AUTO_RETRAIN_THRESHOLD and not is_running() and MODEL_PATH.exists()
    if auto:  # automatic retraining trigger
        background.add_task(retrain)
    return pending, auto


@app.post("/upload/zip")
async def upload_zip(background: BackgroundTasks, file: UploadFile = File(...)):
    """Bulk upload: a .zip with NORMAL/ and PNEUMONIA/ folders of images."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file")
    data = await file.read()
    if len(data) > MAX_ZIP_BYTES:
        raise HTTPException(400, "Zip is too large (max 300 MB)")
    try:
        result = ingest_zip(data)
    except Exception as e:
        raise HTTPException(400, f"Could not read the zip: {e}")
    if result["total_saved"] == 0:
        raise HTTPException(400, "No valid images found. Use folders named NORMAL and PNEUMONIA.")
    pending, auto = _maybe_auto_retrain(background)
    return {**result, "pending_for_training": pending, "min_to_retrain": MIN_RETRAIN_IMAGES,
            "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD, "auto_retrain_started": auto}


@app.post("/upload")
async def upload(background: BackgroundTasks, label: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload several loose images that all share one label."""
    label = label.upper()
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for f in files:
        data = await f.read()
        if not is_valid_image(data):
            rejected.append(f.filename)
            continue
        ext = os.path.splitext(f.filename or "")[1].lower()
        path = folder / f"{uuid.uuid4().hex}{ext if ext in ALLOWED_EXT else '.jpg'}"
        path.write_bytes(data)
        db.add_upload(f.filename, label, path)
        saved += 1
    pending, auto = _maybe_auto_retrain(background)
    return {"saved": saved, "rejected": rejected, "pending_for_training": pending,
            "min_to_retrain": MIN_RETRAIN_IMAGES, "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD,
            "auto_retrain_started": auto}


# ----------------------------------------------------------------- train / retrain
@app.post("/train")
def train_model(background: BackgroundTasks):
    """Train from scratch on data/train (replaces the current model)."""
    if is_running():
        raise HTTPException(409, "A training job is already running")
    background.add_task(train_from_scratch)
    return {"started": True}


@app.post("/retrain")
def trigger_retrain(background: BackgroundTasks):
    """Manual retraining trigger: fine-tunes the saved model on the uploaded data."""
    if is_running():
        raise HTTPException(409, "A training job is already running")
    _require_model()
    pending = db.pending_count()
    if pending < MIN_RETRAIN_IMAGES:
        raise HTTPException(400, f"Upload at least {MIN_RETRAIN_IMAGES} new images first (pending: {pending})")
    background.add_task(retrain)
    return {"started": True, "pending": pending}


@app.get("/retrain/status")
def retrain_status():
    return {**get_status(), "pending_uploads": db.pending_count(),
            "min_to_retrain": MIN_RETRAIN_IMAGES, "history": db.run_history()}


# ----------------------------------------------------------------- data views
@app.get("/insights")
def insights():
    return compute_insights()


@app.get("/samples")
def samples(n: int = 4):
    """A few labelled training images (base64 thumbnails) for the UI gallery."""
    n = max(1, min(n, 8))
    out = {}
    for cls in CLASS_NAMES:
        files = sorted(p for p in (TRAIN_DIR / cls).glob("*") if p.suffix.lower() in ALLOWED_EXT)
        step = max(len(files) // n, 1)
        out[cls] = [thumbnail_b64(f) for f in files[::step][:n]]
    return out
