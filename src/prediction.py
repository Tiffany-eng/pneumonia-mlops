"""Inference: one class that loads the model and predicts single images or batches."""
import os
import threading
from pathlib import Path

import numpy as np

from src.utils.config import CLASS_NAMES, MODEL_PATH, ONNX_PATH
from src.utils.image_io import bytes_to_array


class PneumoniaClassifier:
    """Loads the saved model (.h5, or .onnx when USE_ONNX=1 and the file exists).
    It reloads automatically when the model file changes, so a retrain on one container
    is picked up by the others (they share the models volume)."""

    def __init__(self, model_path=MODEL_PATH, onnx_path=ONNX_PATH, use_onnx=None):
        self.model_path = Path(model_path)
        self.onnx_path = Path(onnx_path)
        self.use_onnx = (os.getenv("USE_ONNX", "0") == "1") if use_onnx is None else use_onnx
        self._lock = threading.Lock()
        self._model = None
        self._session = None
        self._input_name = None
        self._loaded_mtime = None

    def _active_file(self) -> Path:
        return self.onnx_path if (self.use_onnx and self.onnx_path.exists()) else self.model_path

    def load(self, force=False):
        with self._lock:
            path = self._active_file()
            mtime = path.stat().st_mtime
            if not force and self._loaded_mtime == mtime:
                return
            if path.suffix == ".onnx":
                import onnxruntime as ort
                self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self._model = None
            else:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(path)
                self._session = None
            self._loaded_mtime = mtime

    def _probabilities(self, batch: np.ndarray) -> np.ndarray:
        self.load()
        if self._session is not None:
            return self._session.run(None, {self._input_name: batch})[0].ravel()
        return self._model.predict(batch, verbose=0).ravel()

    @staticmethod
    def _format(prob: float) -> dict:
        idx = int(prob >= 0.5)
        return {"label": CLASS_NAMES[idx],
                "confidence": round(prob if idx else 1 - prob, 4),
                "probability_pneumonia": round(prob, 4)}

    def predict(self, data: bytes) -> dict:
        return self._format(float(self._probabilities(bytes_to_array(data))[0]))

    def predict_batch(self, images: list) -> list:
        batch = np.concatenate([bytes_to_array(d) for d in images], axis=0)
        return [self._format(float(p)) for p in self._probabilities(batch)]


classifier = PneumoniaClassifier()
load_model = classifier.load          # kept for notebooks / older imports
predict_bytes = classifier.predict
