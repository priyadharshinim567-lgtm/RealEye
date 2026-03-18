"""
utils/gradcam.py
----------------
Gradient-weighted Class Activation Mapping (Grad-CAM) for the RealEye CNN.

Grad-CAM works by:
  1. Running a forward pass and recording the activations of the last
     convolutional layer.
  2. Computing the gradient of the predicted class score with respect to
     those activations (via GradientTape).
  3. Globally-average-pooling the gradients to get per-channel weights.
  4. Taking the weighted sum of the activation maps and ReLU-ing the result
     so only positive influences are kept.
  5. Resizing the resulting heatmap to the original image size and overlaying
     it as a colour map.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


# ── Internal helpers ─────────────────────────────────────────────────────────

def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer in the model.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    str
        Layer name, e.g. ``'conv2d_2'``.

    Raises
    ------
    ValueError
        If no Conv2D layer is found.
    """
    last_conv_name: str = ""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
    if not last_conv_name:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv_name


def _compute_gradcam(model: tf.keras.Model,
                     img_batch: np.ndarray,
                     layer_name: str) -> np.ndarray:
    """Core Grad-CAM computation (pure TF/NumPy).

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    img_batch : np.ndarray
        Pre-processed image batch of shape (1, H, W, 3), float32 in [0, 1].
    layer_name : str
        Name of the target Conv2D layer.

    Returns
    -------
    np.ndarray
        Raw heatmap of shape (feat_h, feat_w) with float values ≥ 0.
    """
    # Build a sub-model that outputs (conv_layer_output, final_prediction)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    # Record gradients through a forward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        # For binary classification the single sigmoid output IS the class score
        class_score = predictions[:, 0]

    # Gradient of the class score w.r.t. the conv layer feature maps
    grads = tape.gradient(class_score, conv_outputs)   # shape: (1, fH, fW, C)

    # Global-average-pool the gradients over the spatial axes → (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the pooled gradients
    conv_outputs = conv_outputs[0]                     # (fH, fW, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (fH, fW, 1)
    heatmap = tf.squeeze(heatmap)                      # (fH, fW)

    # ReLU: keep only features that push the score positively
    heatmap = tf.nn.relu(heatmap).numpy()

    return heatmap


def _overlay_heatmap(heatmap: np.ndarray,
                     original_pil: Image.Image,
                     alpha: float = 0.45) -> Image.Image:
    """Resize the heatmap to match the original image and blend them together.

    Parameters
    ----------
    heatmap : np.ndarray
        Raw Grad-CAM heatmap (H×W, float).
    original_pil : PIL.Image.Image
        The original uploaded image (any size).
    alpha : float, optional
        Opacity of the heatmap overlay (0 = original only, 1 = heatmap only).
        Default is 0.45.

    Returns
    -------
    PIL.Image.Image
        RGBA-free, RGB PIL image with the heatmap overlaid.
    """
    # Normalise heatmap to [0, 255]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply JET colour map via OpenCV
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)    # RGB

    # Resize heatmap to the original image dimensions
    orig_w, orig_h = original_pil.size
    heatmap_resized = cv2.resize(heatmap_colored, (orig_w, orig_h))

    # Blend with the original image
    orig_rgb = np.array(original_pil.convert("RGB"), dtype=np.float32)
    overlay = (1 - alpha) * orig_rgb + alpha * heatmap_resized.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_gradcam(model: tf.keras.Model,
                     img_batch: np.ndarray,
                     original_pil: Image.Image,
                     alpha: float = 0.45) -> Image.Image:
    """Generate a Grad-CAM heatmap overlay for the given image.

    This is the single entry-point used by the Streamlit app.

    Parameters
    ----------
    model : tf.keras.Model
        Trained RealEye Keras model.
    img_batch : np.ndarray
        Pre-processed image batch, shape (1, 224, 224, 3), float32 in [0, 1].
    original_pil : PIL.Image.Image
        The raw uploaded image (used for the overlay).
    alpha : float, optional
        Heatmap opacity. Default is 0.45.

    Returns
    -------
    PIL.Image.Image
        Original image with Grad-CAM heatmap blended on top.

    Raises
    ------
    ValueError
        If the model has no Conv2D layer (shouldn't happen for RealEye).
    """
    # ── 1. Find the last Conv2D layer automatically ───────────────────────
    layer_name = _find_last_conv_layer(model)

    # ── 2. Compute the raw heatmap via gradient tape ──────────────────────
    heatmap = _compute_gradcam(model, img_batch, layer_name)

    # ── 3. Resize, colourise, and blend with the original image ──────────
    overlay = _overlay_heatmap(heatmap, original_pil, alpha=alpha)

    return overlay
