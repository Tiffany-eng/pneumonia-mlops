"""Optional: convert the .h5 model to ONNX for faster CPU inference (e.g. serverless / small VMs).
    pip install -r requirements-onnx.txt
    python -m src.onnx_export
Then run the API with USE_ONNX=1. NOTE: tf2onnx works best with TF/Keras 2.x; if the conversion fails on
TF 2.16 (Keras 3), run this script in a separate environment with tensorflow==2.15.*."""
import time

import numpy as np
import tensorflow as tf

from src.utils.config import IMG_SIZE, MODEL_PATH, ONNX_PATH


def export():
    import tf2onnx
    model = tf.keras.models.load_model(MODEL_PATH)
    spec = (tf.TensorSpec((None,) + IMG_SIZE + (3,), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(ONNX_PATH))
    print("Saved", ONNX_PATH)


def benchmark(n=50):
    import onnxruntime as ort
    model = tf.keras.models.load_model(MODEL_PATH)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, *IMG_SIZE, 3).astype("float32") * 255
    for label, fn in [("Keras .h5", lambda: model.predict(x, verbose=0)), ("ONNX Runtime", lambda: sess.run(None, {name: x}))]:
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        print(f"{label}: {1000 * (time.perf_counter() - t0) / n:.1f} ms per image")


if __name__ == "__main__":
    export()
    benchmark()
