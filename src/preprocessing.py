"""Preprocessing pipeline: small single-purpose functions (see src/utils for the helpers).
Names from utils are re-exported here so notebooks and scripts can import everything from one place."""
import tensorflow as tf

from src.utils.config import *                      # noqa: F401,F403  (paths, constants, settings)
from src.utils.data_acquisition import download_dataset  # noqa: F401
from src.utils.image_io import bytes_to_array, thumbnail_b64  # noqa: F401
from src.utils.validation import count_images, has_images, is_valid_image  # noqa: F401
from pathlib import Path


def ensure_class_folders(directory):
    """Keras needs both class folders to exist, even if an upload only had one class."""
    for c in CLASS_NAMES:
        (Path(directory) / c).mkdir(parents=True, exist_ok=True)


def load_split(directory, subset, val_split=0.2, batch_size=BATCH_SIZE):
    """Load one directory as a tf.data dataset: resize to 224x224, binary labels, seeded 80/20 split."""
    return tf.keras.utils.image_dataset_from_directory(
        str(directory), labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
        image_size=IMG_SIZE, batch_size=batch_size, validation_split=val_split,
        subset=subset, seed=SEED)


def make_datasets(data_dirs, val_split=0.2, batch_size=BATCH_SIZE):
    """Combine one or more class-folder directories into train/val pipelines."""
    dirs = [d for d in data_dirs if has_images(d)]
    for d in dirs:
        ensure_class_folders(d)
    train = [load_split(d, "training", val_split, batch_size) for d in dirs]
    val = [load_split(d, "validation", val_split, batch_size) for d in dirs]
    train_ds, val_ds = train[0], val[0]
    for t, v in zip(train[1:], val[1:]):
        train_ds, val_ds = train_ds.concatenate(t), val_ds.concatenate(v)
    return (train_ds.shuffle(1000, seed=SEED).prefetch(tf.data.AUTOTUNE),
            val_ds.prefetch(tf.data.AUTOTUNE))


def make_test_dataset(batch_size=BATCH_SIZE):
    return tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR), labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
        image_size=IMG_SIZE, batch_size=batch_size, shuffle=False).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(directory=TRAIN_DIR) -> dict:
    """Chest X-ray data is imbalanced (~3 pneumonia : 1 normal), so weight the loss."""
    counts = count_images(directory)
    total = sum(counts.values())
    return {i: total / (len(CLASS_NAMES) * max(counts[c], 1)) for i, c in enumerate(CLASS_NAMES)}
