"""
model.py — Neural network models for style transfer.

Two architectures:
  1. StyleFeatureExtractor  — VGG19-based feature extractor for optimization NST
  2. FastStyleNet           — Lightweight feed-forward net for real-time stylization
"""

from typing import Dict, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# VGG19 Feature Extractor
# ---------------------------------------------------------------------------

class StyleFeatureExtractor(keras.Model):
    """
    Wraps a pretrained VGG19 to extract intermediate feature maps
    for content and style loss computation.

    Content layers: high-level semantic features (block5_conv2)
    Style layers:   low-to-high level texture features (block1–5_conv1)
    """

    DEFAULT_CONTENT_LAYERS = ["block5_conv2"]
    DEFAULT_STYLE_LAYERS = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    def __init__(
        self,
        content_layers: List[str] = None,
        style_layers: List[str] = None,
    ):
        super().__init__()
        self.content_layers = content_layers or self.DEFAULT_CONTENT_LAYERS
        self.style_layers = style_layers or self.DEFAULT_STYLE_LAYERS
        self._all_layers = self.style_layers + self.content_layers

        # Load VGG19 without the top classification head
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        outputs = {name: vgg.get_layer(name).output for name in self._all_layers}
        self.extractor = keras.Model(inputs=vgg.input, outputs=outputs)
        self.extractor.trainable = False

    def call(self, inputs: tf.Tensor) -> Dict[str, Dict[str, tf.Tensor]]:
        """
        Forward pass. Applies VGG19 preprocessing internally.

        Args:
            inputs: float32 tensor (B, H, W, 3) in [0, 1].

        Returns:
            dict with keys 'content' and 'style', each mapping
            layer names to their feature tensors.
        """
        # VGG19 expects pixel values in [0, 255] with specific preprocessing
        preprocessed = keras.applications.vgg19.preprocess_input(inputs * 255.0)
        features = self.extractor(preprocessed)

        content_features = {
            layer: features[layer] for layer in self.content_layers
        }
        style_features = {
            layer: features[layer] for layer in self.style_layers
        }
        return {"content": content_features, "style": style_features}


# ---------------------------------------------------------------------------
# Fast Style Transfer Network
# ---------------------------------------------------------------------------

class ReflectionPad2D(layers.Layer):
    """Reflection padding layer (better than zero-padding for style transfer)."""

    def __init__(self, padding: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, x: tf.Tensor) -> tf.Tensor:
        p = self.padding
        return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")

    def get_config(self):
        return {**super().get_config(), "padding": self.padding}


class ConvNormRelu(layers.Layer):
    """Conv2D + InstanceNorm + optional ReLU block."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        activation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        pad = kernel_size // 2
        self.pad = ReflectionPad2D(pad)
        self.conv = layers.Conv2D(filters, kernel_size, strides=stride, padding="valid", use_bias=False)
        self.norm = layers.GroupNormalization(groups=filters, axis=-1)  # InstanceNorm approx
        self.relu = layers.ReLU()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.relu(x)
        return x


class ResidualBlock(layers.Layer):
    """Residual block for the Fast Style Transfer network."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvNormRelu(filters, 3)
        self.conv2 = ConvNormRelu(filters, 3, activation=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return x + self.conv2(self.conv1(x, training=training), training=training)


class UpsampleBlock(layers.Layer):
    """Upsample (nearest-neighbor) + Conv instead of transposed conv (avoids checkerboard)."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(size=2, interpolation="nearest")
        self.conv = ConvNormRelu(filters, 3)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.conv(self.upsample(x), training=training)


def build_fast_style_net(input_shape=(None, None, 3)) -> keras.Model:
    """
    Johnson-style feed-forward network for real-time style transfer.

    Architecture:
        Encoder (3 Conv, stride 1/2) → 5 Residual blocks → Decoder (2 Upsample) → Output Conv

    Args:
        input_shape: (H, W, C) — can be None for variable-size inputs.

    Returns:
        Keras model mapping (B, H, W, 3) → (B, H, W, 3) in [0, 1].
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Encoder
    x = ConvNormRelu(32, 9, name="enc_1")(inputs)
    x = ConvNormRelu(64, 3, stride=2, name="enc_2")(x)
    x = ConvNormRelu(128, 3, stride=2, name="enc_3")(x)

    # Residual bottleneck
    for i in range(5):
        x = ResidualBlock(128, name=f"res_{i+1}")(x)

    # Decoder
    x = UpsampleBlock(64, name="dec_1")(x)
    x = UpsampleBlock(32, name="dec_2")(x)

    # Output: tanh → scale to [0, 1]
    x = ReflectionPad2D(4)(x)
    x = layers.Conv2D(3, 9, padding="valid", activation="tanh", name="output_conv")(x)
    outputs = (x + 1.0) / 2.0  # tanh [-1,1] → [0,1]

    return keras.Model(inputs=inputs, outputs=outputs, name="FastStyleNet")
