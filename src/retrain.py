"""Train from scratch and retrain. Retraining fine-tunes the *saved* model on uploads + original data."""
import json
import shutil
import threading
from datetime import datetime

import tensorflow as tf

from src import database as db
from src.model import callbacks, compile_model, evaluate, train
from src.preprocessing import (METRICS_PATH, MIN_RETRAIN_IMAGES, MODEL_PATH, TRAIN_DIR, UPLOAD_DIR,
                               compute_class_weights, make_datasets, make_test_dataset)
from src.prediction import classifier

STATUS_PATH = MODEL_PATH.parent / "retrain_status.json"
CANDIDATE = MODEL_PATH.with_name("candidate.h5")
_running = threading.Lock()


def _set_status(state, message=""):
    STATUS_PATH.write_text(json.dumps({"state": state, "message": message,
                                      "time": datetime.utcnow().isoformat()}))


def get_status():
    return json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {"state": "idle", "message": ""}


def is_running():
    return _running.locked()


def _promote(metrics):
    """Back up the current model, put the candidate in its place and reload the classifier."""
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, MODEL_PATH.with_name("previous_model.h5"))
    shutil.move(str(CANDIDATE), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    classifier.load(force=True)


def train_from_scratch(head_epochs=5, fine_tune_epochs=8):
    """Full training on data/train (same as the notebook). Replaces the current model."""
    if not _running.acquire(blocking=False):
        return {"started": False, "reason": "Another training job is running"}
    try:
        _set_status("running", "Building datasets")
        train_ds, val_ds = make_datasets([TRAIN_DIR])
        _set_status("running", "Training (transfer learning + fine-tuning)")
        model, _, _ = train(train_ds, val_ds, head_epochs, fine_tune_epochs, save_path=CANDIDATE)
        _set_status("running", "Evaluating on the test set")
        metrics = evaluate(tf.keras.models.load_model(CANDIDATE), make_test_dataset(), save=False)
        _promote(metrics)
        db.log_run(0, metrics["accuracy"], metrics["f1"], metrics["recall"], "trained from scratch")
        _set_status("done", "Model trained from scratch and deployed")
        return {"started": True, "metrics": metrics}
    except Exception as e:
        _set_status("error", str(e))
        raise
    finally:
        _running.release()


def retrain(epochs=5, lr=1e-5):
    """Uploaded data (in DB + data/uploads) -> preprocessing -> fine-tune the saved model
    -> evaluate on the test set -> promote only if recall does not get worse."""
    if not _running.acquire(blocking=False):
        return {"started": False, "reason": "A training job is already running"}
    try:
        n_new = db.pending_count()
        if n_new < MIN_RETRAIN_IMAGES:
            _set_status("idle", f"Need at least {MIN_RETRAIN_IMAGES} new images (have {n_new})")
            return {"started": False, "reason": "Not enough new data"}
        _set_status("running", f"Preprocessing {n_new} new images")
        train_ds, val_ds = make_datasets([UPLOAD_DIR, TRAIN_DIR])

        _set_status("running", "Loading saved model as the pre-trained starting point")
        model = tf.keras.models.load_model(MODEL_PATH)      # custom pre-trained model
        compile_model(model, lr)

        _set_status("running", "Fine-tuning")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=compute_class_weights(), callbacks=callbacks(CANDIDATE))

        _set_status("running", "Evaluating on the held-out test set")
        new_metrics = evaluate(tf.keras.models.load_model(CANDIDATE), make_test_dataset(), save=False)
        old = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {"recall": 0}

        promoted = new_metrics["recall"] >= old["recall"] - 0.01  # screening task: protect sensitivity
        if promoted:
            _promote(new_metrics)
        db.mark_used()
        db.log_run(n_new, new_metrics["accuracy"], new_metrics["f1"], new_metrics["recall"],
                   "promoted" if promoted else "rejected")
        _set_status("done", "New model promoted" if promoted else "New model rejected (recall dropped)")
        return {"started": True, "promoted": promoted, "metrics": new_metrics}
    except Exception as e:
        _set_status("error", str(e))
        raise
    finally:
        _running.release()


if __name__ == "__main__":
    print(retrain())
