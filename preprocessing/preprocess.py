"""
preprocessing/preprocess.py
----------------------------
Image preprocessing utilities for the RealEye project.

Functions
---------
load_and_preprocess_image : Load a single image from disk, resize it, and
                            normalise pixel values to [0, 1].
load_dataset              : Walk the dataset directory and return all images
                            as NumPy arrays with binary labels.
"""

import os
import numpy as np
from PIL import Image

# Default target size expected by our CNN
TARGET_SIZE = (224, 224)

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_preprocess_image(image_path: str,
                              target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Load a single image, resize it, and normalise to [0, 1].

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.
    target_size : tuple, optional
        (width, height) to resize to. Default is (224, 224).

    Returns
    -------
    np.ndarray
        A float32 array of shape (height, width, 3) with values in [0, 1].
    """
    # Open the image and ensure it is in RGB mode
    img = Image.open(image_path).convert("RGB")

    # Resize to the target dimensions
    img = img.resize(target_size)

    # Convert to NumPy array and normalise pixel values from [0, 255] → [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array


def load_dataset(dataset_dir: str = "dataset",
                 target_size: tuple = TARGET_SIZE):
    """Load all images from the dataset directory and return arrays + labels.

    Expected directory layout::

        dataset/
            real/            ← label 0
            ai_generated/    ← label 1

    Parameters
    ----------
    dataset_dir : str
        Path to the root dataset folder.
    target_size : tuple, optional
        (width, height) to resize every image to. Default is (224, 224).

    Returns
    -------
    X : np.ndarray
        Array of shape (N, height, width, 3) containing all images.
    y : np.ndarray
        Array of shape (N,) containing labels (0 = Real, 1 = AI-Generated).
    """
    images = []
    labels = []

    # Map sub-folder names to numeric labels
    class_map = {
        "real": 0,
        "ai_generated": 1,
    }

    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARNING] Directory not found: {class_dir} — skipping.")
            continue

        file_count = 0
        for filename in os.listdir(class_dir):
            # Only process supported image formats
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(class_dir, filename)
            try:
                img_array = load_and_preprocess_image(filepath, target_size)
                images.append(img_array)
                labels.append(label)
                file_count += 1
            except Exception as e:
                print(f"[WARNING] Could not process {filepath}: {e}")

        print(f"Loaded {file_count} images from '{class_name}' (label={label})")

    if len(images) == 0:
        raise ValueError(
            "No images found! Please add images to "
            f"'{dataset_dir}/real/' and '{dataset_dir}/ai_generated/'."
        )

    X = np.array(images)
    y = np.array(labels)

    print(f"\nTotal images loaded : {len(y)}")
    print(f"  Real (label 0)    : {np.sum(y == 0)}")
    print(f"  AI-Gen (label 1)  : {np.sum(y == 1)}")

    return X, y
