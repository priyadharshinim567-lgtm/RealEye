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

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)


def build_model(input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """Build and compile a binary-classification CNN using the Functional API.

    The Functional API is used instead of Sequential to ensure the model
    graph is properly defined for Grad-CAM and internal layer access.

    Parameters
    ----------
    input_shape : tuple, optional
        Shape of a single input image. Default is (224, 224, 3).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Block 1 ──────────────────────────────────────────────
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 2 ──────────────────────────────────────────────
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ── Block 3 ──────────────────────────────────────────────
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ── Classifier head ─────────────────────────────────────
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)          # helps prevent over-fitting
    outputs = Dense(1, activation="sigmoid")(x)   # single output → binary

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
