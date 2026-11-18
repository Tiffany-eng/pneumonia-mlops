"""Model creation, training, evaluation."""
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.preprocessing import IMG_SIZE, METRICS_PATH, MODEL_PATH, compute_class_weights


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(name="auc")])


def build_model(lr=1e-3):
    """MobileNetV2 transfer learning + augmentation + L2 + dropout (base frozen at first)."""
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False,
                                             weights="imagenet", name="mobilenetv2")
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="scale_to_pm1")(x)  # = MobileNetV2 preprocess_input
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid",
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    model = tf.keras.Model(inputs, outputs)
    compile_model(model, lr)
    return model


def callbacks(path=MODEL_PATH):
    return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True)]


def train(train_ds, val_ds, head_epochs=5, fine_tune_epochs=8, save_path=MODEL_PATH):
    """Phase 1: train the new head with the base frozen. Phase 2: unfreeze the top of the base."""
    weights = compute_class_weights()
    model = build_model()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=weights, callbacks=callbacks(save_path))
    base = model.get_layer("mobilenetv2")
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                   class_weight=weights, callbacks=callbacks(save_path))
    model.save(save_path)  # .h5 file
    return model, h1, h2


def evaluate(model, test_ds, threshold=0.5, save=True) -> dict:
    """Accuracy, loss, precision, recall, F1, AUC + confusion matrix on the test set."""
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
    }
    if save:
        METRICS_PATH.write_text(json.dumps(result, indent=2))
    return result
