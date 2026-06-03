"""
preprocessing.py — Image loading, resizing, and VGG19 normalization.

VGG19 expects BGR images with ImageNet mean subtracted (no division by std).
TensorFlow's preprocess_input handles this automatically.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf


# VGG19 ImageNet normalization constants (BGR order, mean subtraction)
VGG_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)


def load_image(
    path: Union[str, Path],
    max_dim: int = 512,
) -> tf.Tensor:
    """
    Load an image from disk, resize (preserving aspect ratio), and
    return a float32 tensor in [0, 1] with shape (1, H, W, 3).

    Args:
        path:    Path to the image file.
        max_dim: Resize so the longest edge equals max_dim.

    Returns:
        tf.Tensor of shape (1, H, W, 3), dtype float32, values in [0, 1].
    """
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0

    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_h = tf.cast(shape[0] * scale, tf.int32)
    new_w = tf.cast(shape[1] * scale, tf.int32)
    img = tf.image.resize(img, [new_h, new_w])

    return tf.expand_dims(img, 0)  # (1, H, W, 3)


def save_image(tensor: tf.Tensor, path: Union[str, Path]):
    """
    Save a float32 tensor (values in [0, 1]) to disk as a JPEG/PNG.

    Args:
        tensor: Shape (1, H, W, 3) or (H, W, 3).
        path:   Output file path (.jpg or .png).
    """
    img = tensor
    if len(img.shape) == 4:
        img = tf.squeeze(img, axis=0)

    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.cast(img * 255.0, tf.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        encoded = tf.image.encode_jpeg(img, quality=95)
    else:
        encoded = tf.image.encode_png(img)

    tf.io.write_file(str(path), encoded)


def preprocess_vgg(image: tf.Tensor) -> tf.Tensor:
    """
    Apply VGG19 preprocessing:
      1. Scale [0,1] → [0,255]
      2. Convert RGB → BGR
      3. Subtract ImageNet channel means

    Args:
        image: float32 tensor (B, H, W, 3) or (H, W, 3) in [0, 1].

    Returns:
        Preprocessed tensor ready for VGG19.
    """
    img = image * 255.0
    img = img[..., ::-1]           # RGB → BGR
    img = img - VGG_MEAN           # subtract means
    return img


def deprocess_vgg(image: tf.Tensor) -> tf.Tensor:
    """
    Reverse VGG19 preprocessing → [0, 1] RGB image.

    Args:
        image: Preprocessed tensor (B, H, W, 3) or (H, W, 3).

    Returns:
        float32 tensor in [0, 1], RGB channel order.
    """
    img = image + VGG_MEAN
    img = img[..., ::-1]           # BGR → RGB
    img = img / 255.0
    return tf.clip_by_value(img, 0.0, 1.0)


def resize_to_match(image: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    """Resize `image` to the spatial dimensions of `reference`."""
    shape = tf.shape(reference)
    h, w = shape[1], shape[2]
    return tf.image.resize(image, [h, w])


def random_noise_image(shape: Tuple[int, int, int, int], seed: int = 42) -> tf.Variable:
    """
    Create a trainable tf.Variable initialized with random noise.
    Used as the starting point for optimization-based style transfer.

    Args:
        shape: (1, H, W, 3)
        seed:  Random seed for reproducibility.

    Returns:
        tf.Variable initialized to uniform noise in [0, 1].
    """
    tf.random.set_seed(seed)
    noise = tf.random.uniform(shape, minval=0.0, maxval=1.0, seed=seed)
    return tf.Variable(noise, trainable=True, dtype=tf.float32)
