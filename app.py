"""
app.py  –  RealEye Week 3
-------------------------
Enhanced Streamlit application with:
  • Improved branding & layout
  • Grad-CAM heatmap explanation
  • Side-by-side image / heatmap display
  • Confidence bar + percentage
  • Robust error handling

Run:
    streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from keras.models import load_model

# ── Local project imports ────────────────────────────────────────────────────
from utils.preprocess import preprocess_uploaded_image
from utils.gradcam    import generate_gradcam

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
MODEL_PATH    = os.path.join("model", "realeye_model.keras")
THRESHOLD     = 0.5          # sigmoid decision boundary
IMG_SIZE      = (224, 224)   # must match training
ALLOWED_TYPES = ["jpg", "jpeg", "png"]

# ════════════════════════════════════════════════════════════════════════════
# Page setup
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RealEye – AI Image Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Injected CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header band ─────────────────────────────────────────────────── */
.header-band {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.header-band h1 {
    color: #ffffff;
    font-size: 2.4rem;
    letter-spacing: 1px;
    margin: 0 0 0.3rem;
}
.header-band p {
    color: #b0b8d8;
    font-size: 1rem;
    margin: 0;
}

/* ── Section card ────────────────────────────────────────────────── */
.card {
    background: #1a1a2e;
    border: 1px solid #2d2d50;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6c7aff;
    margin-bottom: 0.6rem;
}

/* ── Result chip ─────────────────────────────────────────────────── */
.chip {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 30px;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.chip-ai   { background:#ff4b4b22; border:2px solid #ff4b4b; color:#ff4b4b; }
.chip-real { background:#00d26a22; border:2px solid #00d26a; color:#00d26a; }

/* ── Confidence text ─────────────────────────────────────────────── */
.conf-text {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.4rem 0 0;
}
.conf-sub {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0;
}

/* ── Heatmap legend ──────────────────────────────────────────────── */
.legend {
    font-size: 0.8rem;
    color: #999;
    text-align: center;
    margin-top: 0.3rem;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #555;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #2d2d50;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Cached resource helpers
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading RealEye model…")
def load_realeye_model(path: str):
    """Load and cache the trained Keras model.

    Returns None if the model file is not found, so the caller can show
    a helpful error instead of crashing.
    """
    if not os.path.isfile(path):
        return None
    model = load_model(path)
    return load_model(path)

# ════════════════════════════════════════════════════════════════════════════
# Prediction logic
# ════════════════════════════════════════════════════════════════════════════

def run_prediction(model, pil_image: Image.Image) -> tuple[str, float, Image.Image]:
    """Preprocess → predict → generate Grad-CAM.

    Parameters
    ----------
    model : tf.keras.Model
    pil_image : PIL.Image.Image – the raw uploaded image

    Returns
    -------
    label : str        – "AI Generated" or "Original"
    confidence : float – percentage 0-100
    heatmap_img : PIL.Image.Image – Grad-CAM overlay
    """
    # ── 1. Preprocess ─────────────────────────────────────────────────────
    img_batch = preprocess_uploaded_image(pil_image, target_size=IMG_SIZE)

    # ── 2. Forward pass ───────────────────────────────────────────────────
    raw_score = float(model.predict(img_batch, verbose=0)[0][0])

    # ── 3. Decide label and confidence ─────────────────────────────────── 
    if raw_score >= THRESHOLD:
        label      = "AI Generated"
        confidence = raw_score * 100
    else:
        label      = "Original"
        confidence = (1.0 - raw_score) * 100

    # ── 4. Grad-CAM ───────────────────────────────────────────────────────
    heatmap_img = generate_gradcam(model, img_batch, pil_image)

    return label, confidence, heatmap_img


# ════════════════════════════════════════════════════════════════════════════
# UI – Header
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-band">
    <h1>🔍 RealEye</h1>
    <p>AI Image Detection &nbsp;·&nbsp; Upload an image to find out if it's real or AI-generated</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# Load model – show clear error if missing
# ════════════════════════════════════════════════════════════════════════════
model = load_realeye_model(MODEL_PATH)

if model is None:
    st.error(
        f"⚠️ **Model not found** at `{MODEL_PATH}`.\n\n"
        "Run `python train_model.py` to train and save the model first, "
        "then restart this app."
    )
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# UI – Upload section
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📂 Upload Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Choose a JPG or PNG image",
    type=ALLOWED_TYPES,
    help="Supported formats: JPG, JPEG, PNG",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("👆 Upload a JPG or PNG image above to get started.")
    st.stop()

# ── Open the image safely ────────────────────────────────────────────────────
try:
    pil_image = Image.open(uploaded_file)
except Exception as exc:
    st.error(
        f"❌ **Could not read the uploaded file.**\n\n"
        f"Make sure you upload a valid JPG or PNG image.\n\n"
        f"_Technical detail: `{exc}`_"
    )
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# Run prediction
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("🧠 Analysing image…"):
    try:
        label, confidence, heatmap_img = run_prediction(model, pil_image)
    except Exception as exc:
        st.error(
            f"❌ **Prediction failed.**\n\n"
            f"_Technical detail: `{exc}`_"
        )
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# UI – Prediction result + confidence
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📊 Prediction Result</div>', unsafe_allow_html=True)

# Choose colour scheme based on label
is_ai      = label == "AI Generated"
chip_class = "chip-ai" if is_ai else "chip-real"
icon       = "🤖" if is_ai else "✅"
conf_color = "#ff4b4b" if is_ai else "#00d26a"

# Three-column layout: label | spacer | confidence
col_label, col_gap, col_conf = st.columns([2, 0.4, 1.6])

with col_label:
    st.markdown(
        f'<div class="card">'
        f'  <div class="section-label">Classification</div>'
        f'  <span class="chip {chip_class}">{icon} &nbsp; {label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_conf:
    st.markdown(
        f'<div class="card">'
        f'  <div class="section-label">Confidence</div>'
        f'  <p class="conf-text" style="color:{conf_color}">{confidence:.1f}%</p>'
        f'  <p class="conf-sub">model certainty</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Progress bar (confidence as 0-100 int)
st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# UI – Side-by-side image + Grad-CAM heatmap
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">🌡️ Explainability – Grad-CAM Heatmap</div>', unsafe_allow_html=True)
st.caption(
    "The heatmap highlights **which regions of the image influenced the model's decision**. "
    "Warmer colours (red/yellow) indicate areas of higher importance."
)

col_orig, col_heat = st.columns(2)

with col_orig:
    st.image(
        pil_image,
        use_container_width=True,
        caption=f"📷 Original – {uploaded_file.name}",
    )

with col_heat:
    st.image(
        heatmap_img,
        use_container_width=True,
        caption="🌡️ Grad-CAM Overlay (red = high influence)",
    )

st.markdown(
    '<p class="legend">🔴 High influence &nbsp;·&nbsp; 🟡 Medium &nbsp;·&nbsp; 🔵 Low influence</p>',
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">'
    'RealEye &nbsp;·&nbsp; Week 3 Prototype &nbsp;·&nbsp; '
    'CNN + Grad-CAM &nbsp;·&nbsp; Real vs AI-Generated Image Detection'
    '</div>',
    unsafe_allow_html=True,
)
