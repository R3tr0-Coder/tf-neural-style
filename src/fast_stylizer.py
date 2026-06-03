"""
fast_stylizer.py — Train and apply a Fast Style Transfer feed-forward network.

Based on Johnson et al. (2016): train a lightweight network to transform
any content image into a fixed style in a single forward pass.

Training requires a large image dataset (e.g. MS-COCO train2017).
At inference, stylization takes ~10ms per image on CPU.
"""

import logging
import time
from pathlib import Path
from typing import Iterator

import tensorflow as tf

from .model import StyleFeatureExtractor, build_fast_style_net
from .losses import style_loss, content_loss, total_variation_loss
from .preprocessing import load_image, save_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_image_dataset(
    folder: str,
    image_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline from a folder of images.

    Args:
        folder:     Path to image directory (recursive).
        image_size: Resize images to (image_size, image_size).
        batch_size: Batch size.

    Returns:
        Unbatched, preprocessed tf.data.Dataset.
    """
    folder = Path(folder)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in extensions:
        paths.extend(folder.rglob(ext))
    paths = [str(p) for p in paths]
    logger.info(f"Found {len(paths):,} images in {folder}")

    def _load(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    dataset = (
        tf.data.Dataset.from_tensor_slices(paths)
        .shuffle(min(10_000, len(paths)))
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset, len(paths) // batch_size


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FastStyleTrainer:
    """
    Trains a FastStyleNet to transfer a fixed artistic style to arbitrary images.

    Usage:
        trainer = FastStyleTrainer(cfg)
        trainer.train(style_image_path, dataset_folder)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fs_cfg = cfg["fast_style"]

        self.image_size = fs_cfg["image_size"]
        self.epochs = fs_cfg["epochs"]
        self.batch_size = fs_cfg["batch_size"]
        self.lr = fs_cfg["learning_rate"]
        self.content_weight = fs_cfg["content_weight"]
        self.style_weight = fs_cfg["style_weight"]
        self.tv_weight = fs_cfg["tv_weight"]
        self.checkpoint_dir = Path(fs_cfg["checkpoint_dir"])
        self.save_every = fs_cfg.get("save_every_n_batches", 500)

        # Networks
        self.style_net = build_fast_style_net()
        self.vgg = StyleFeatureExtractor()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

        # TensorBoard
        log_dir = self.checkpoint_dir / "logs"
        self.writer = tf.summary.create_file_writer(str(log_dir))

    def _precompute_style_targets(self, style_img: tf.Tensor):
        """Extract and cache style feature targets from the style image."""
        self._style_targets = self.vgg(style_img)["style"]
        logger.info("Precomputed style feature targets.")

    @tf.function
    def _train_step(self, content_batch: tf.Tensor):
        """Single training step."""
        with tf.GradientTape() as tape:
            generated = self.style_net(content_batch, training=True)

            gen_features = self.vgg(generated)
            content_features = self.vgg(content_batch)["content"]

            c_loss = tf.add_n([
                content_loss(gen_features["content"][layer], content_features[layer])
                for layer in content_features
            ])
            s_loss = style_loss(gen_features["style"], self._style_targets)
            tv_loss = total_variation_loss(generated)

            total = (
                self.content_weight * c_loss
                + self.style_weight * s_loss
                + self.tv_weight * tv_loss
            )

        grads = tape.gradient(total, self.style_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.style_net.trainable_variables))
        return total, c_loss, s_loss, tv_loss

    def train(self, style_image_path: str, dataset_folder: str):
        """
        Train the fast style network.

        Args:
            style_image_path: Path to the target style image.
            dataset_folder:   Path to folder containing training images.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        style_img = load_image(style_image_path, self.image_size)
        self._precompute_style_targets(style_img)

        dataset, steps_per_epoch = make_image_dataset(
            dataset_folder, self.image_size, self.batch_size
        )

        global_step = 0
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.epochs}")
            epoch_start = time.time()

            for batch_idx, content_batch in enumerate(dataset):
                total, c_loss, s_loss, tv_loss = self._train_step(content_batch)
                global_step += 1

                if batch_idx % 50 == 0:
                    logger.info(
                        f"  [{batch_idx:5d}/{steps_per_epoch}]  "
                        f"total={total:.4e}  content={c_loss:.4e}  "
                        f"style={s_loss:.4e}  tv={tv_loss:.4e}"
                    )

                with self.writer.as_default():
                    tf.summary.scalar("loss/total", total, step=global_step)
                    tf.summary.scalar("loss/content", c_loss, step=global_step)
                    tf.summary.scalar("loss/style", s_loss, step=global_step)

                if global_step % self.save_every == 0:
                    self._save_checkpoint(global_step)

            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch {epoch} complete ({epoch_time:.1f}s)")

        self._save_checkpoint("final")
        logger.info(f"\nTraining complete. Checkpoints → {self.checkpoint_dir}")

    def _save_checkpoint(self, tag):
        path = self.checkpoint_dir / f"fast_style_{tag}"
        self.style_net.save(str(path))
        logger.info(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def apply_fast_style(
    checkpoint_path: str,
    image_path: str,
    output_path: str,
    max_dim: int = 1024,
):
    """
    Apply a trained FastStyleNet to a single image.

    Args:
        checkpoint_path: Path to saved Keras model directory.
        image_path:      Input image path.
        output_path:     Output image path.
        max_dim:         Max dimension for resizing input image.
    """
    logger.info(f"Loading model from {checkpoint_path}...")
    model = tf.keras.models.load_model(checkpoint_path)

    img = load_image(image_path, max_dim)
    t0 = time.time()
    stylized = model(img, training=False)
    elapsed = (time.time() - t0) * 1000

    save_image(stylized, output_path)
    logger.info(f"Styled in {elapsed:.1f}ms → saved to {output_path}")
    return stylized
