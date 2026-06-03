"""
losses.py — Loss functions for Neural Style Transfer.

  - content_loss       : MSE between feature maps (preserve structure)
  - gram_matrix        : Compute Gram matrix for style representation
  - style_loss         : MSE between Gram matrices (match textures)
  - total_variation_loss : Spatial smoothness regularizer
  - combined_loss      : Weighted sum of all three losses
"""

from typing import Dict, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Content Loss
# ---------------------------------------------------------------------------

def content_loss(
    generated_features: tf.Tensor,
    content_features: tf.Tensor,
) -> tf.Tensor:
    """
    Mean squared error between feature maps.

    Measures how well the generated image preserves high-level
    semantic content from the content image.

    Args:
        generated_features: Feature tensor from the generated image.
        content_features:   Feature tensor from the content image (target).

    Returns:
        Scalar loss tensor.
    """
    return tf.reduce_mean(tf.square(generated_features - content_features))


# ---------------------------------------------------------------------------
# Style Loss (Gram Matrix)
# ---------------------------------------------------------------------------

def gram_matrix(feature_map: tf.Tensor) -> tf.Tensor:
    """
    Compute the Gram matrix of a feature map.

    The Gram matrix captures pairwise feature correlations across spatial
    positions, encoding texture/style information independent of location.

    Args:
        feature_map: (B, H, W, C) tensor of activations.

    Returns:
        (B, C, C) tensor — normalized Gram matrix.
    """
    b, h, w, c = tf.unstack(tf.shape(feature_map))
    # Reshape to (B, H*W, C)
    features = tf.reshape(feature_map, [b, h * w, c])
    # Gram: (B, C, C)
    gram = tf.matmul(features, features, transpose_a=True)
    # Normalize by number of elements
    return gram / tf.cast(h * w * c, tf.float32)


def style_loss_single_layer(
    generated_features: tf.Tensor,
    style_features: tf.Tensor,
) -> tf.Tensor:
    """
    Style loss for a single layer: MSE between Gram matrices.

    Args:
        generated_features: (B, H, W, C) activations of generated image.
        style_features:     (B, H, W, C) activations of style image.

    Returns:
        Scalar loss tensor.
    """
    gram_gen = gram_matrix(generated_features)
    gram_style = gram_matrix(style_features)
    return tf.reduce_mean(tf.square(gram_gen - gram_style))


def style_loss(
    generated_style_features: Dict[str, tf.Tensor],
    target_style_features: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Total style loss averaged across all style layers.

    Args:
        generated_style_features: {layer_name: feature_tensor} for generated image.
        target_style_features:    {layer_name: feature_tensor} for style image.

    Returns:
        Scalar loss tensor.
    """
    layer_losses = [
        style_loss_single_layer(generated_style_features[layer], target_style_features[layer])
        for layer in generated_style_features
    ]
    return tf.add_n(layer_losses) / len(layer_losses)


# ---------------------------------------------------------------------------
# Total Variation Loss
# ---------------------------------------------------------------------------

def total_variation_loss(image: tf.Tensor) -> tf.Tensor:
    """
    Total Variation (TV) regularization loss.

    Encourages spatial smoothness in the generated image by penalizing
    large differences between adjacent pixels.

    Args:
        image: (B, H, W, 3) float32 tensor in [0, 1].

    Returns:
        Scalar loss tensor.
    """
    # Horizontal and vertical differences
    diff_h = image[:, 1:, :, :] - image[:, :-1, :, :]
    diff_w = image[:, :, 1:, :] - image[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(diff_h)) + tf.reduce_mean(tf.abs(diff_w))


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

def combined_loss(
    generated_features: Dict[str, Dict[str, tf.Tensor]],
    content_targets: Dict[str, tf.Tensor],
    style_targets: Dict[str, tf.Tensor],
    generated_image: tf.Tensor,
    content_weight: float = 1e4,
    style_weight: float = 1e-2,
    tv_weight: float = 30.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the total weighted loss for style transfer.

    Args:
        generated_features: Features extracted from the generated image.
        content_targets:    Content feature targets (from content image).
        style_targets:      Style feature targets (from style image).
        generated_image:    The generated image tensor (for TV loss).
        content_weight:     Weight for content loss term.
        style_weight:       Weight for style loss term.
        tv_weight:          Weight for total variation loss term.

    Returns:
        Tuple of (total_loss, c_loss, s_loss, tv_loss) — all scalar tensors.
    """
    c_loss = tf.add_n([
        content_loss(generated_features["content"][layer], content_targets[layer])
        for layer in content_targets
    ])

    s_loss = style_loss(generated_features["style"], style_targets)
    tv_loss = total_variation_loss(generated_image)

    total = (
        content_weight * c_loss
        + style_weight * s_loss
        + tv_weight * tv_loss
    )
    return total, c_loss, s_loss, tv_loss
