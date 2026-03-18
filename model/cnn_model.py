"""
model/cnn_model.py
-------------------
Defines a simple Convolutional Neural Network (CNN) for binary image
classification: Real vs AI-Generated.

Architecture
~~~~~~~~~~~~
Input (224×224×3)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)   ← probability of being AI-Generated
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)


def build_model(input_shape: tuple = (224, 224, 3)) -> Sequential:
    """Build and compile a binary-classification CNN.

    Parameters
    ----------
    input_shape : tuple, optional
        Shape of a single input image. Default is (224, 224, 3).

    Returns
    -------
    tensorflow.keras.models.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────────────────────────
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # ── Block 2 ──────────────────────────────────────────────
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # ── Block 3 ──────────────────────────────────────────────
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # ── Classifier head ─────────────────────────────────────
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),          # helps prevent over-fitting
        Dense(1, activation="sigmoid"),   # single output → binary
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Quick sanity check when running this file directly ──────────────
if __name__ == "__main__":
    model = build_model()
    model.summary()
