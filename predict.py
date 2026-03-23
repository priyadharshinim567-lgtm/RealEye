"""
predict.py
----------
Classify a single image as **Real** or **AI Generated** using the trained
RealEye model.

Usage::

    python predict.py path/to/image.jpg
"""

import sys
import os
import numpy as np
from keras.models import load_model

# ── Local project import ─────────────────────────────────────────────
from preprocessing.preprocess import load_and_preprocess_image

# ── Configuration ────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "realeye_model.keras")
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.5          # decision boundary


def predict_image(image_path: str, model_path: str = MODEL_PATH) -> None:
    """Load the model, preprocess the image, print the prediction.

    Parameters
    ----------
    image_path : str
        Path to the image file to classify.
    model_path : str, optional
        Path to the saved Keras model.
    """
    # ── Validate inputs ──────────────────────────────────────────
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Trained model not found at '{model_path}'.")
        print("        Run 'python train_model.py' first to train the model.")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────
    print(f"Loading model from '{model_path}' ...")
    model = load_model(model_path)

    # ── Preprocess image ─────────────────────────────────────────
    img_array = load_and_preprocess_image(image_path, target_size=IMG_SIZE)
    # The model expects a batch dimension → (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # ── Make prediction ──────────────────────────────────────────
    prediction = model.predict(img_batch, verbose=0)[0][0]

    # prediction is the probability of being AI-Generated (label 1)
    if prediction >= THRESHOLD:
        label = "AI Generated"
        confidence = prediction * 100
    else:
        label = "Real"
        confidence = (1 - prediction) * 100

    # ── Display result ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2f} %")
    print("=" * 50 + "\n")


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python predict.py <image_path>")
        print("Example: python predict.py dataset/real/photo1.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])
