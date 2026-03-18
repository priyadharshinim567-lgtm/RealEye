"""
utils/preprocess.py
-------------------
Image preprocessing helpers for the RealEye Streamlit web app.

These functions handle PIL Image objects (from st.file_uploader) rather
than file paths, so they complement — not replace — preprocessing/preprocess.py.
"""

import numpy as np
from PIL import Image

# Must match the input shape the CNN was trained on
IMG_SIZE = (224, 224)


def preprocess_uploaded_image(pil_image: Image.Image,
                               target_size: tuple = IMG_SIZE) -> np.ndarray:
    """Convert a PIL Image from Streamlit's file uploader into a model-ready
    NumPy array.

    Steps
    -----
    1. Ensure the image is in RGB mode (handles RGBA / greyscale uploads).
    2. Resize to the size the CNN expects (224 × 224 by default).
    3. Normalise pixel values from [0, 255] → [0.0, 1.0] (float32).
    4. Add a batch dimension so the shape becomes (1, H, W, 3).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Image object returned by st.file_uploader / Image.open().
    target_size : tuple, optional
        (width, height) to resize to. Defaults to (224, 224).

    Returns
    -------
    np.ndarray
        Float32 array of shape (1, height, width, 3) ready for model.predict().
    """
    # ── Step 1: Convert to RGB (handles PNG with alpha, greyscale, etc.) ──
    img_rgb = pil_image.convert("RGB")

    # ── Step 2: Resize to the model's expected input dimensions ───────────
    img_resized = img_rgb.resize(target_size)

    # ── Step 3: Normalise pixel values to [0, 1] ──────────────────────────
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # ── Step 4: Add batch dimension → (1, 224, 224, 3) ────────────────────
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch
