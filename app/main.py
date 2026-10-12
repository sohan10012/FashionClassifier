import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fashion AI Classifier // Neural Vision Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# HTML Rendering Helper
# Strips leading indentation from each line so Markdown parser NEVER interprets
# HTML lines as indented code blocks.
# ---------------------------------------------------------------------------
def render_html(html_str: str):
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Custom CSS: Modern Dark Mode UI
# ---------------------------------------------------------------------------
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');

    /* Global Dark Canvas */
    .stApp {
        background-color: #0a0a0f !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.14) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 55%) !important;
        color: #f8f7fa !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #f8f7fa !important;
    }

    p, span, label {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Header */
    .top-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .brand-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #cbc3d7 70%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }

    .brand-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #a19fad;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    .telemetry-badges {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
    }

    .telemetry-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        padding: 0.35rem 0.75rem;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbc3d7;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Cohesive Showcase Cards */
    .showcase-card {
        background: rgba(18, 17, 26, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.5rem;
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.75);
        margin-bottom: 1.5rem;
    }

    .card-title-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.75rem;
    }

    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.18rem;
        font-weight: 500;
        color: #f8f7fa;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8b5cf6;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Uniform Preset Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #e4e1e9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        line-height: 42px !important;
        padding: 0 0.5rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: rgba(139, 92, 246, 0.25) !important;
        border-color: rgba(192, 132, 252, 0.5) !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* File Uploader Restyling */
    [data-testid="stFileUploader"] {
        background: rgba(10, 10, 15, 0.45) !important;
        border: 1px dashed rgba(139, 92, 246, 0.35) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.75) !important;
    }

    /* Image Display Frame */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.8) !important;
        max-width: 250px !important;
        max-height: 250px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        display: block;
        margin: 0 auto;
        image-rendering: -webkit-optimize-contrast;
    }

    /* Prediction Showcase Card */
    .prediction-hero {
        background: linear-gradient(145deg, rgba(26, 24, 38, 0.88) 0%, rgba(18, 17, 26, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.38);
        border-radius: 10px;
        padding: 1.75rem;
        box-shadow: 0 0 32px -4px rgba(139, 92, 246, 0.22);
        margin-bottom: 1.5rem;
    }

    .pred-category-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.35rem;
    }

    .pred-class-name {
        font-family: 'Playfair Display', serif;
        font-size: 2.25rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    }

    .pred-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #a19fad;
        margin-bottom: 1.25rem;
    }

    .confidence-meter-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin: 1.25rem 0;
        background: rgba(10, 10, 15, 0.55);
        padding: 1rem 1.25rem;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .confidence-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #10b981;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    .confidence-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #a19fad;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    .telemetry-cell {
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        padding-left: 1.25rem;
    }

    .telemetry-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #c084fc;
        line-height: 1.1;
    }

    .telemetry-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        color: #f8f7fa;
        line-height: 1.1;
    }

    /* Styling Grid */
    .styling-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-top: 1.25rem;
    }

    .styling-cell {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
    }

    .styling-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #8b5cf6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .styling-val {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #f8f7fa;
        font-weight: 400;
        line-height: 1.35;
    }

    /* Probability Distribution Bars */
    .bar-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
        font-size: 0.83rem;
    }

    .bar-name {
        width: 120px;
        color: #cbc3d7;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .bar-name-active {
        color: #ffffff;
        font-weight: 600;
    }

    .bar-track {
        flex: 1;
        height: 7px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        margin: 0 1rem;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #8b5cf6, #c084fc);
    }

    .bar-fill-secondary {
        height: 100%;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.22);
    }

    .bar-percent {
        width: 60px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #a19fad;
    }

    .bar-percent-active {
        color: #10b981;
        font-weight: 600;
    }
    </style>
    """
)

# ---------------------------------------------------------------------------
# Fashion-MNIST Taxonomy & All-English Metadata
# ---------------------------------------------------------------------------
CLASS_METADATA = {
    0: {
        "name": "T-shirt / Top",
        "description": "Casual Everyday Wardrobe Essential",
        "icon": "👕",
        "fit": "Minimalist Regular Fit",
        "pairing": "Tailored trousers or raw dark denim",
        "season": "Spring / Summer",
        "fabric": "Supima Cotton / Breathable Jersey",
    },
    1: {
        "name": "Trouser",
        "description": "Structured Formal & Casual Pants",
        "icon": "👖",
        "fit": "Structured Straight-Leg Cut",
        "pairing": "Oversized blazer, button-up shirt, or knitwear",
        "season": "All-Season Transition",
        "fabric": "Virgin Wool Blend / Cotton Twill",
    },
    2: {
        "name": "Pullover",
        "description": "Knitwear & Layering Sweater",
        "icon": "🧶",
        "fit": "Relaxed Cozy Fit",
        "pairing": "Pleated midi skirt or selvedge chinos",
        "season": "Autumn / Winter",
        "fabric": "Merino Wool / Cashmere Blend",
    },
    3: {
        "name": "Dress",
        "description": "Formal & Occasion Wear",
        "icon": "👗",
        "fit": "Fluid Column / A-Line Cut",
        "pairing": "Structured trench coat & block heels",
        "season": "Cocktail / Summer Evening",
        "fabric": "Mulberry Silk / Chiffon Fabric",
    },
    4: {
        "name": "Coat",
        "description": "Outerwear & Overcoat",
        "icon": "🧥",
        "fit": "Double-Breasted Statement Cut",
        "pairing": "Monochrome knit sweater & leather boots",
        "season": "Winter Wardrobe",
        "fabric": "Warm Wool / Heavyweight Twill",
    },
    5: {
        "name": "Sandal",
        "description": "Open-Toe Summer Footwear",
        "icon": "👡",
        "fit": "Sculpted Strap Design",
        "pairing": "Linen trousers, shorts, or sundress",
        "season": "High Summer",
        "fabric": "Smooth Leather / Breathable Canvas",
    },
    6: {
        "name": "Shirt",
        "description": "Button-Up Collared Apparel",
        "icon": "👔",
        "fit": "Crisp Tailored Collar & Cuffs",
        "pairing": "High-waist trousers or pencil skirt",
        "season": "Year-Round Wardrobe",
        "fabric": "Egyptian Cotton Poplin / Oxford Weave",
    },
    7: {
        "name": "Sneaker",
        "description": "Athletic & Street Footwear",
        "icon": "👟",
        "fit": "Low-Profile Athletic Cut",
        "pairing": "Casual suit, denim, or relaxed joggers",
        "season": "All-Year Streetwear",
        "fabric": "Full-Grain Leather / Technical Mesh",
    },
    8: {
        "name": "Bag",
        "description": "Handbag & Carry Accessory",
        "icon": "👜",
        "fit": "Structured Geometric Design",
        "pairing": "Complements any casual or tailored outfit",
        "season": "Perennial Essential",
        "fabric": "Textured Leather / Coated Canvas",
    },
    9: {
        "name": "Ankle Boot",
        "description": "All-Weather Ankle Footwear",
        "icon": "🥾",
        "fit": "Block-Heel Ankle Cut",
        "pairing": "Cropped denim, chinos, or long trench coat",
        "season": "Autumn / Winter",
        "fabric": "Polished Calfskin / Suede",
    },
}

CLASS_NAMES = [CLASS_METADATA[i]["name"] for i in range(10)]

# ---------------------------------------------------------------------------
# Model Loader with Cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classification_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(base_dir, "trained_model", "trained_fashion_mnist_model.h5"),
        os.path.join(base_dir, "..", "app", "trained_model", "trained_fashion_mnist_model.h5"),
    ]
    for path in model_paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("Trained Fashion-MNIST model file (.h5) not found.")

try:
    model = load_classification_model()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)

# ---------------------------------------------------------------------------
# Preprocessing Helper (Auto-detects light vs dark backgrounds)
# ---------------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    gray = pil_img.convert("L")
    arr_raw = np.array(gray)
    # If corners are bright (white background product shot), invert so garment is bright on dark
    corners = [arr_raw[0, 0], arr_raw[0, -1], arr_raw[-1, 0], arr_raw[-1, -1]]
    if np.mean(corners) > 128:
        arr_raw = 255 - arr_raw
        gray = Image.fromarray(arr_raw)

    gray_resized = gray.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(gray_resized, dtype=np.float32) / 255.0
    return arr.reshape((1, 28, 28, 1))

def predict_fashion_item(img_tensor: np.ndarray):
    start_time = time.perf_counter()
    logits = model.predict(img_tensor, verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    latency_ms = (time.perf_counter() - start_time) * 1000
    top_idx = int(np.argmax(probabilities))
    return top_idx, probabilities, latency_ms

# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------
render_html(
    """
    <div class="top-header">
        <div>
            <h1 class="brand-title">FASHION AI CLASSIFIER</h1>
            <div class="brand-tagline">Deep Learning Image Classification // Fashion-MNIST Neural Network</div>
        </div>
        <div class="telemetry-badges">
            <div class="telemetry-badge">
                <span class="pulse-dot"></span>
                <span>NEURAL ENGINE ACTIVE</span>
            </div>
            <div class="telemetry-badge">
                <span>TENSORFLOW 2.15</span>
            </div>
            <div class="telemetry-badge">
                <span>DEEP CONVNET</span>
            </div>
            <div class="telemetry-badge">
                <span>10 CATEGORIES</span>
            </div>
        </div>
    </div>
    """
)

if not model_ready:
    st.error(f"Model initialization error: {model_error}")
    st.stop()

# ---------------------------------------------------------------------------
# High-Resolution Preset Wardrobe Setup
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
preset_dir = os.path.join(base_dir, "sample_wardrobe")

# Concise 1-word labels to prevent button text wrapping
PRESET_ITEMS = [
    {"label": "👕 T-Shirt", "file": "00_t-shirt.png", "title": "T-Shirt"},
    {"label": "👖 Trouser", "file": "01_trouser.png", "title": "Trouser"},
    {"label": "🧶 Sweater", "file": "02_pullover.png", "title": "Pullover Sweater"},
    {"label": "👗 Dress", "file": "03_dress.png", "title": "Dress"},
    {"label": "🧥 Coat", "file": "04_coat.png", "title": "Overcoat"},
    {"label": "👡 Sandal", "file": "05_sandal.png", "title": "Sandal"},
    {"label": "👔 Shirt", "file": "06_shirt.png", "title": "Button Shirt"},
    {"label": "👟 Sneaker", "file": "07_sneaker.png", "title": "Sneaker"},
    {"label": "👜 Bag", "file": "08_bag.png", "title": "Handbag"},
    {"label": "🥾 Boot", "file": "09_ankle_boot.png", "title": "Ankle Boot"},
]

# Initialize Session State
if "current_image" not in st.session_state:
    default_path = os.path.join(preset_dir, "00_t-shirt.png")
    if os.path.exists(default_path):
        st.session_state["current_image"] = Image.open(default_path)
        st.session_state["current_label"] = "Preset: T-Shirt"
    else:
        st.session_state["current_image"] = None
        st.session_state["current_label"] = ""

# ---------------------------------------------------------------------------
# Main Layout: Left Column (Inputs & Preview) | Right Column (Predictions)
# ---------------------------------------------------------------------------
col_input, col_results = st.columns([1.05, 1.35], gap="large")

with col_input:
    # Card 1: Curated Selection & Upload
    render_html(
        """
        <div class="showcase-card">
            <div class="card-title-bar">
                <h3 class="card-title">✦ Garment Selection</h3>
                <span class="card-meta">Presets & Upload</span>
            </div>
            <p style="font-size: 0.83rem; color: #a19fad; margin-bottom: 0.75rem;">
                Click any of the 10 preset garments below to test immediately, or upload your own photo:
            </p>
        </div>
        """
    )

    # 10 Preset Buttons in 2 even rows of 5 columns (short labels that never wrap)
    row1 = PRESET_ITEMS[:5]
    row2 = PRESET_ITEMS[5:]

    cols1 = st.columns(5)
    for i, item in enumerate(row1):
        with cols1[i]:
            if st.button(item["label"], key=f"btn_{item['file']}"):
                p_path = os.path.join(preset_dir, item["file"])
                if os.path.exists(p_path):
                    st.session_state["current_image"] = Image.open(p_path)
                    st.session_state["current_label"] = f"Preset: {item['title']}"

    cols2 = st.columns(5)
    for i, item in enumerate(row2):
        with cols2[i]:
            if st.button(item["label"], key=f"btn_{item['file']}"):
                p_path = os.path.join(preset_dir, item["file"])
                if os.path.exists(p_path):
                    st.session_state["current_image"] = Image.open(p_path)
                    st.session_state["current_label"] = f"Preset: {item['title']}"

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Custom Apparel Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
        help="Upload a centered apparel photo (JPG, PNG, WEBP)",
    )

    if uploaded_file is not None:
        try:
            st.session_state["current_image"] = Image.open(uploaded_file)
            st.session_state["current_label"] = f"Uploaded: {uploaded_file.name}"
        except Exception as err:
            st.error(f"Image load error: {err}")

    # Card 2: Unified Specimen Preview Card (Clean, Framed, Crisp)
    active_img = st.session_state.get("current_image")
    active_lbl = st.session_state.get("current_label", "")

    if active_img is not None:
        w, h = active_img.size
        render_html(
            f"""
            <div class="showcase-card" style="margin-top: 1.25rem;">
                <div class="card-title-bar">
                    <h3 class="card-title">✦ Selected Garment Preview</h3>
                    <span class="card-meta">{w}×{h} PX • {active_img.format or 'PNG'}</span>
                </div>
                <div style="font-size: 0.8rem; color: #cbc3d7; margin-bottom: 0.75rem;">
                    Active Source: <strong style="color: #ffffff;">{active_lbl}</strong>
                </div>
            </div>
            """
        )

        # Centered, crisp image preview (Fixed width=240 so it is never stretched or blurry!)
        col_pad1, col_center, col_pad2 = st.columns([0.2, 0.6, 0.2])
        with col_center:
            try:
                st.image(active_img, width=240)
            except Exception:
                st.image(active_img)

# ---------------------------------------------------------------------------
# Right Column: Neural Inference & Results Showcase
# ---------------------------------------------------------------------------
with col_results:
    if active_img is not None:
        img_tensor = preprocess_image(active_img)
        top_idx, probabilities, latency_ms = predict_fashion_item(img_tensor)

        top_meta = CLASS_METADATA[top_idx]
        confidence_pct = probabilities[top_idx] * 100.0

        # Primary Classification Showcase Card
        render_html(
            f"""
            <div class="prediction-hero">
                <div class="pred-category-badge">Identified Fashion Category // Class #{top_idx}</div>
                <div class="pred-class-name">{top_meta['icon']} {top_meta['name']}</div>
                <div class="pred-subtitle">{top_meta['description']}</div>

                <div class="confidence-meter-container">
                    <div>
                        <div class="confidence-score">{confidence_pct:.1f}%</div>
                        <div class="confidence-label">Confidence Score</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-num">{latency_ms:.1f} <span style="font-size: 0.75rem; color: #a19fad;">ms</span></div>
                        <div class="confidence-label">Inference Latency</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-text">Deep CNN</div>
                        <div class="confidence-label">Model Architecture</div>
                    </div>
                </div>

                <div class="styling-grid">
                    <div class="styling-cell">
                        <div class="styling-label">Style & Fit</div>
                        <div class="styling-val">{top_meta['fit']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Fabric Profile</div>
                        <div class="styling-val">{top_meta['fabric']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Suggested Pairing</div>
                        <div class="styling-val">{top_meta['pairing']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Best Season</div>
                        <div class="styling-val">{top_meta['season']}</div>
                    </div>
                </div>
            </div>
            """
        )

        # Full 10-Class Confidence Distribution Spectrum
        bars_html = [
            '<div class="showcase-card">',
            '  <div class="card-title-bar">',
            '    <h3 class="card-title">✦ Class Confidence Distribution</h3>',
            '    <span class="card-meta">Softmax Spectrum</span>',
            '  </div>',
        ]

        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            prob = probabilities[idx] * 100.0
            is_winner = (idx == top_idx)
            name_class = "bar-name bar-name-active" if is_winner else "bar-name"
            fill_class = "bar-fill" if is_winner else "bar-fill-secondary"
            pct_class = "bar-percent bar-percent-active" if is_winner else "bar-percent"
            icon = CLASS_METADATA[idx]["icon"]
            label_text = f"{icon} {CLASS_METADATA[idx]['name']}"

            bars_html.append(f"""
            <div class="bar-row">
                <span class="{name_class}">{label_text}</span>
                <div class="bar-track">
                    <div class="{fill_class}" style="width: {prob:.2f}%;"></div>
                </div>
                <span class="{pct_class}">{prob:.1f}%</span>
            </div>
            """)

        bars_html.append('</div>')
        render_html("\n".join(bars_html))

    else:
        render_html(
            """
            <div class="showcase-card" style="text-align: center; padding: 3.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.6;">✧</div>
                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #f8f7fa; margin-bottom: 0.5rem;">
                    Awaiting Garment Selection
                </h3>
                <p style="color: #a19fad; font-size: 0.88rem; max-width: 360px; margin: 0 auto; line-height: 1.6;">
                    Click any of the 10 preset fashion garments on the left or upload an apparel photograph to see instant deep learning predictions.
                </p>
                <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap;">
                    <span class="telemetry-badge">10 Apparel Classes</span>
                    <span class="telemetry-badge">Conv2D + MaxPool</span>
                    <span class="telemetry-badge">Softmax Calibrated</span>
                </div>
            </div>
            """
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render_html(
    """
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); text-align: center; color: #5a576b; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;">
        FASHION AI CLASSIFIER // CONVOLUTIONAL NEURAL NETWORK // DATASET: ZALANDO RESEARCH FASHION-MNIST
    </div>
    """
)
