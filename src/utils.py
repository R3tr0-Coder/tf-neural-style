"""
utils.py — Config loading, logging, seeding, and helper utilities.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


def load_config(path: str) -> dict:
    """Load and return a YAML config as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path, level: int = logging.INFO):
    """Configure root logger: console + file handler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def get_gpu_info() -> str:
    """Return a string summarizing available GPU devices."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "No GPU found — using CPU"
    names = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            names.append(details.get("device_name", gpu.name))
        except Exception:
            names.append(gpu.name)
    return f"{len(gpus)} GPU(s): {', '.join(names)}"


def configure_gpu_memory_growth():
    """Enable memory growth to avoid TF grabbing all VRAM at startup."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Must be called before any GPU ops


def count_parameters(model: tf.keras.Model) -> dict:
    """Return total and trainable parameter counts for a Keras model."""
    total = model.count_params()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    return {"total": total, "trainable": int(trainable)}


def print_model_summary(model: tf.keras.Model, name: str = ""):
    """Log a brief model summary."""
    logger = logging.getLogger(__name__)
    counts = count_parameters(model)
    label = name or model.name
    logger.info(f"Model: {label}")
    logger.info(f"  Total parameters    : {counts['total']:,}")
    logger.info(f"  Trainable parameters: {counts['trainable']:,}")
