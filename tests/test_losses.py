"""
test_losses.py — Unit tests for loss functions and model shapes.

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import (
    content_loss,
    gram_matrix,
    style_loss_single_layer,
    total_variation_loss,
    combined_loss,
)
from src.model import StyleFeatureExtractor, build_fast_style_net
from src.preprocessing import load_image, random_noise_image


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestContentLoss:
    def test_identical_features_zero_loss(self):
        features = tf.ones((1, 14, 14, 256))
        loss = content_loss(features, features)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_different_features_positive_loss(self):
        a = tf.ones((1, 14, 14, 256))
        b = tf.zeros((1, 14, 14, 256))
        loss = content_loss(a, b)
        assert float(loss) > 0.0

    def test_output_is_scalar(self):
        a = tf.random.normal((1, 8, 8, 64))
        b = tf.random.normal((1, 8, 8, 64))
        loss = content_loss(a, b)
        assert loss.shape == ()


class TestGramMatrix:
    def test_output_shape(self):
        feature_map = tf.random.normal((2, 16, 16, 32))
        gram = gram_matrix(feature_map)
        assert gram.shape == (2, 32, 32)

    def test_symmetry(self):
        feature_map = tf.random.normal((1, 8, 8, 16))
        gram = gram_matrix(feature_map)
        diff = tf.abs(gram - tf.transpose(gram, [0, 2, 1]))
        assert float(tf.reduce_max(diff)) < 1e-5

    def test_identical_features_zero_style_loss(self):
        features = tf.random.normal((1, 8, 8, 32))
        loss = style_loss_single_layer(features, features)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)


class TestTotalVariationLoss:
    def test_constant_image_zero_loss(self):
        img = tf.ones((1, 64, 64, 3))
        loss = total_variation_loss(img)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_noisy_image_higher_loss(self):
        smooth = tf.ones((1, 64, 64, 3))
        noisy = tf.random.uniform((1, 64, 64, 3))
        assert float(total_variation_loss(noisy)) > float(total_variation_loss(smooth))

    def test_output_scalar(self):
        img = tf.random.uniform((1, 32, 32, 3))
        loss = total_variation_loss(img)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestStyleFeatureExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        return StyleFeatureExtractor()

    def test_output_keys(self, extractor):
        dummy = tf.random.uniform((1, 224, 224, 3))
        out = extractor(dummy)
        assert "content" in out
        assert "style" in out

    def test_content_layer_names(self, extractor):
        dummy = tf.random.uniform((1, 224, 224, 3))
        out = extractor(dummy)
        assert "block5_conv2" in out["content"]

    def test_style_layer_count(self, extractor):
        dummy = tf.random.uniform((1, 224, 224, 3))
        out = extractor(dummy)
        assert len(out["style"]) == 5  # 5 default style layers

    def test_not_trainable(self, extractor):
        assert not extractor.extractor.trainable


class TestFastStyleNet:
    @pytest.fixture(scope="class")
    def model(self):
        return build_fast_style_net()

    def test_output_shape(self, model):
        dummy = tf.random.uniform((1, 256, 256, 3))
        out = model(dummy, training=False)
        assert out.shape == (1, 256, 256, 3)

    def test_output_range(self, model):
        dummy = tf.random.uniform((1, 256, 256, 3))
        out = model(dummy, training=False)
        assert float(tf.reduce_min(out)) >= 0.0 - 1e-5
        assert float(tf.reduce_max(out)) <= 1.0 + 1e-5

    def test_variable_input_size(self, model):
        """Model should handle different input sizes."""
        for size in [128, 256, 512]:
            dummy = tf.random.uniform((1, size, size, 3))
            out = model(dummy, training=False)
            assert out.shape == (1, size, size, 3)

    def test_parameter_count(self, model):
        total = model.count_params()
        # FastStyleNet should be < 2M params (lightweight by design)
        assert total < 2_000_000, f"Model too large: {total:,} params"


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_random_noise_image_shape(self):
        noise = random_noise_image((1, 64, 64, 3))
        assert noise.shape == (1, 64, 64, 3)

    def test_random_noise_is_variable(self):
        noise = random_noise_image((1, 32, 32, 3))
        assert isinstance(noise, tf.Variable)
        assert noise.trainable

    def test_random_noise_range(self):
        noise = random_noise_image((1, 32, 32, 3))
        assert float(tf.reduce_min(noise)) >= 0.0
        assert float(tf.reduce_max(noise)) <= 1.0
