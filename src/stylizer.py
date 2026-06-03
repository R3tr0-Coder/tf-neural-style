"""
stylizer.py — Optimization-based Neural Style Transfer engine.

Iteratively optimizes pixel values of a generated image to minimize
combined content + style + TV loss using the Adam optimizer.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import tensorflow as tf

from .model import StyleFeatureExtractor
from .losses import combined_loss
from .preprocessing import load_image, save_image, random_noise_image, resize_to_match

logger = logging.getLogger(__name__)


class Stylizer:
    """
    Runs the iterative style transfer optimization loop.

    Usage:
        stylizer = Stylizer(cfg)
        result = stylizer.run(content_path, style_path, output_path)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        st_cfg = cfg["style_transfer"]

        self.image_size = st_cfg["image_size"]
        self.iterations = st_cfg["iterations"]
        self.lr = st_cfg["learning_rate"]
        self.content_weight = st_cfg["content_weight"]
        self.style_weight = st_cfg["style_weight"]
        self.tv_weight = st_cfg["tv_weight"]
        self.save_interval = st_cfg.get("save_interval", 100)

        content_layers = st_cfg.get("content_layers", None)
        style_layers = st_cfg.get("style_layers", None)
        self.extractor = StyleFeatureExtractor(content_layers, style_layers)

        if st_cfg.get("use_mixed_precision", False):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (float16).")

    def _extract_targets(self, content_img: tf.Tensor, style_img: tf.Tensor):
        """Pre-compute fixed content and style targets (only done once)."""
        content_features = self.extractor(content_img)["content"]
        style_features = self.extractor(style_img)["style"]
        return content_features, style_features

    @tf.function
    def _train_step(
        self,
        generated: tf.Variable,
        content_targets,
        style_targets,
        optimizer: tf.optimizers.Optimizer,
    ):
        """Single optimization step with gradient tape."""
        with tf.GradientTape() as tape:
            gen_features = self.extractor(generated)
            total, c_loss, s_loss, tv_loss = combined_loss(
                gen_features,
                content_targets,
                style_targets,
                generated,
                self.content_weight,
                self.style_weight,
                self.tv_weight,
            )

        grads = tape.gradient(total, generated)
        optimizer.apply_gradients([(grads, generated)])

        # Clip pixel values to [0, 1] after each update
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))

        return total, c_loss, s_loss, tv_loss

    def run(
        self,
        content_path: str,
        style_path: str,
        output_path: str,
        init: str = "content",   # "content" | "noise"
    ) -> tf.Tensor:
        """
        Run the style transfer optimization.

        Args:
            content_path: Path to content image.
            style_path:   Path to style image.
            output_path:  Path to save the final stylized image.
            init:         Initialize from "content" image or random "noise".

        Returns:
            Final generated image as a float32 tensor (1, H, W, 3).
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading images...")
        content_img = load_image(content_path, self.image_size)
        style_img = load_image(style_path, self.image_size)
        style_img = resize_to_match(style_img, content_img)

        logger.info(f"Content image shape : {content_img.shape}")
        logger.info(f"Style image shape   : {style_img.shape}")

        # Extract fixed targets
        logger.info("Extracting feature targets from VGG19...")
        content_targets, style_targets = self._extract_targets(content_img, style_img)

        # Initialize generated image
        if init == "content":
            generated = tf.Variable(tf.identity(content_img), trainable=True)
        else:
            generated = random_noise_image(tf.shape(content_img))

        optimizer = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.99, epsilon=1e-1)

        logger.info(f"Starting optimization for {self.iterations} iterations...")
        t0 = time.time()
        best_loss = float("inf")
        best_image = tf.identity(generated)

        for step in range(1, self.iterations + 1):
            total, c_loss, s_loss, tv_loss = self._train_step(
                generated, content_targets, style_targets, optimizer
            )

            loss_val = total.numpy()
            if loss_val < best_loss:
                best_loss = loss_val
                best_image = tf.identity(generated)

            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                logger.info(
                    f"  Step [{step:5d}/{self.iterations}]  "
                    f"total={total:.4e}  "
                    f"content={c_loss:.4e}  "
                    f"style={s_loss:.4e}  "
                    f"tv={tv_loss:.4e}  "
                    f"({elapsed:.1f}s)"
                )

            # Save intermediate results
            if step % self.save_interval == 0:
                interim_path = output_dir / f"step_{step:05d}.jpg"
                save_image(generated, interim_path)
                logger.debug(f"  Saved intermediate → {interim_path}")

        # Save final output
        save_image(best_image, output_path)
        logger.info(f"\nDone! Final image saved → {output_path}")
        logger.info(f"Total time: {time.time() - t0:.1f}s")

        return best_image
