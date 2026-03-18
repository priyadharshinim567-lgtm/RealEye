"""
train_model.py
--------------
Train the RealEye CNN on images stored in the ``dataset/`` folder.

Usage::

    python train_model.py

The trained model is saved to ``model/realeye_model.keras`` and a plot of the
training history (accuracy & loss) is saved to ``model/training_history.png``.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── Local project imports ────────────────────────────────────────────
from preprocessing.preprocess import load_dataset
from model.cnn_model import build_model

# ── Configuration ────────────────────────────────────────────────────
DATASET_DIR   = "dataset"
MODEL_SAVE    = os.path.join("model", "realeye_model.keras")
HISTORY_PLOT  = os.path.join("model", "training_history.png")
IMG_SIZE      = (224, 224)
EPOCHS        = 10
BATCH_SIZE    = 32
VAL_SPLIT     = 0.2      # 80 % train, 20 % validation
RANDOM_STATE  = 42


def plot_history(history, save_path: str) -> None:
    """Save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ─────────────────────────────────────────────────
    ax1.plot(history.history["accuracy"],     label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── Loss ─────────────────────────────────────────────────────
    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining history plot saved to: {save_path}")


def main():
    # ── 1. Load dataset ──────────────────────────────────────────
    print("=" * 60)
    print("  RealEye — Model Training  ")
    print("=" * 60)
    print(f"\nLoading images from '{DATASET_DIR}/' ...\n")

    X, y = load_dataset(DATASET_DIR, target_size=IMG_SIZE)

    # ── 2. Train / validation split ──────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,       # keep class balance in both splits
    )
    print(f"\nTraining samples   : {len(y_train)}")
    print(f"Validation samples : {len(y_val)}")

    # ── 3. Build model ───────────────────────────────────────────
    model = build_model(input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────
    print("\nStarting training ...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # ── 5. Save model ────────────────────────────────────────────
    model.save(MODEL_SAVE)
    print(f"\nModel saved to: {MODEL_SAVE}")

    # ── 6. Plot training curves ──────────────────────────────────
    plot_history(history, HISTORY_PLOT)

    # ── 7. Final summary ─────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  Final Validation Accuracy : {val_acc * 100:.2f} %")
    print(f"  Final Validation Loss     : {val_loss:.4f}")
    print(f"{'=' * 60}")

    print("\nDone! You can now run:  python predict.py <image_path>")


if __name__ == "__main__":
    main()
