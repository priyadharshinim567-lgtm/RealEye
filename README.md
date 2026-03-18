# 🔍 RealEye — AI vs Real Image Classifier

**Week 1 Prototype** — A CNN-based binary classifier that detects whether an
image is *Real* or *AI-Generated*.

---

## 📁 Project Structure

```
RealEye/
├── dataset/
│   ├── real/                  ← Place real photos here
│   └── ai_generated/          ← Place AI-generated images here
├── model/
│   ├── cnn_model.py           ← CNN architecture definition
│   ├── realeye_model.keras    ← (generated after training)
│   └── training_history.png   ← (generated after training)
├── preprocessing/
│   └── preprocess.py          ← Image loading & preprocessing
├── train_model.py             ← Train the model
├── predict.py                 ← Predict on a single image
├── requirements.txt           ← Python dependencies
└── README.md                  ← You are here
```

---

## 🗂️ Setting Up the Dataset

1. Collect images — aim for **at least 50–100 images per class** for
   meaningful results. More is better.
2. Place **real photographs** (camera shots, phone photos, etc.) inside:
   ```
   dataset/real/
   ```
3. Place **AI-generated images** (from DALL·E, Midjourney, Stable Diffusion,
   etc.) inside:
   ```
   dataset/ai_generated/
   ```
4. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

> **Tip:** You can download datasets from Kaggle. Search for
> *"AI-generated vs real images"* to find ready-made datasets.

---

## ⚙️ Installation

```bash
# 1. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess all images from `dataset/`
- Split into 80 % training / 20 % validation
- Train the CNN for 10 epochs
- Save the model to `model/realeye_model.keras`
- Save accuracy/loss plots to `model/training_history.png`

### Predict on a Single Image

```bash
python predict.py path/to/image.jpg
```

Example output:

```
==================================================
  Image      : photo1.jpg
  Prediction : Real
  Confidence : 94.32 %
==================================================
```

---

## 🧠 Model Architecture

| Layer             | Output Shape       | Params  |
|-------------------|--------------------|---------|
| Conv2D (32, 3×3)  | 222 × 222 × 32    | 896     |
| MaxPooling2D      | 111 × 111 × 32    | 0       |
| Conv2D (64, 3×3)  | 109 × 109 × 64    | 18,496  |
| MaxPooling2D      | 54 × 54 × 64      | 0       |
| Conv2D (128, 3×3) | 52 × 52 × 128     | 73,856  |
| MaxPooling2D      | 26 × 26 × 128     | 0       |
| Flatten           | 86,528             | 0       |
| Dense (128, ReLU) | 128                | 11,075,712 |
| Dropout (0.5)     | 128                | 0       |
| Dense (1, Sigmoid)| 1                  | 129     |

- **Optimiser:** Adam
- **Loss:** Binary Cross-Entropy
- **Metric:** Accuracy

---

## 📌 Week 1 Scope

- ✅ Dataset folder structure
- ✅ Image preprocessing (resize, normalise)
- ✅ CNN model definition
- ✅ Model training with train/val split
- ✅ Single-image prediction script
- ❌ Web interface (planned for later weeks)
