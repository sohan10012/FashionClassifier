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
# Strips all leading indentation to prevent Markdown from interpreting
# indented HTML lines as code blocks.
# ---------------------------------------------------------------------------
def render_html(html_str: str):
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Custom CSS: Premium Dark Mode UI
# Obsidian Void • Electric Violet • Emerald Accents • Sleek Glassmorphism
# ---------------------------------------------------------------------------
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');

    /* Global Dark Canvas */
    .stApp {
        background-color: #0a0a0f !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 55%) !important;
        color: #f8f7fa !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Hide standard Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography Hierarchy */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #f8f7fa !important;
        letter-spacing: -0.01em !important;
    }

    p, span, label {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Header Banner */
    .top-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
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
        line-height: 1.2;
    }

    .brand-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
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

    /* Vitrine Cards */
    .vitrine-card {
        background: rgba(18, 17, 26, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
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
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.75rem;
    }

    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 500;
        color: #f8f7fa;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #8b5cf6;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Prediction Showcase Card */
    .prediction-hero {
        background: linear-gradient(145deg, rgba(26, 24, 38, 0.88) 0%, rgba(18, 17, 26, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.38);
        border-radius: 8px;
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

    /* Preset Selection Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #cbc3d7 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.6rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: rgba(139, 92, 246, 0.25) !important;
        border-color: rgba(192, 132, 252, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Uploader Restyling */
    [data-testid="stFileUploader"] {
        background: rgba(18, 17, 26, 0.45) !important;
        border: 1px dashed rgba(139, 92, 246, 0.35) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.75) !important;
    }

    /* Single Image Framing */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important;
        display: block;
        margin: 0 auto;
    }
    </style>
    """
)

# ---------------------------------------------------------------------------
# Fashion-MNIST Taxonomy & All-English Fashion Metadata
# ---------------------------------------------------------------------------
CLASS_METADATA = {
    0: {
        "name": "T-shirt / Top",
        "description": "Casual Everyday Wardrobe Essential",
        "icon": "👕",
        "silhouette": "Minimalist Regular Fit",
        "pairing": "Tailored trousers or raw dark denim",
        "season": "Spring / Summer",
        "fabric": "Supima Cotton / Breathable Jersey",
    },
    1: {
        "name": "Trouser",
        "description": "Structured Formal & Casual Pants",
        "icon": "👖",
        "silhouette": "Structured Straight-Leg Cut",
        "pairing": "Oversized blazer, button-up shirt, or knitwear",
        "season": "All-Season Transition",
        "fabric": "Virgin Wool Blend / Gabardine Twill",
    },
    2: {
        "name": "Pullover",
        "description": "Knitwear & Layering Sweater",
        "icon": "🧶",
        "silhouette": "Relaxed Cozy Silhouette",
        "pairing": "Pleated midi skirt or selvedge chinos",
        "season": "Autumn / Winter",
        "fabric": "Merino Wool / Cashmere Blend",
    },
    3: {
        "name": "Dress",
        "description": "Formal & Occasion Wear",
        "icon": "👗",
        "silhouette": "Fluid Column / A-Line Drape",
        "pairing": "Structured trench coat & block heels",
        "season": "Cocktail / Summer Evening",
        "fabric": "Mulberry Silk / Crêpe de Chine",
    },
    4: {
        "name": "Coat",
        "description": "Outerwear & Overcoat",
        "icon": "🧥",
        "silhouette": "Double-Breasted Statement Coat",
        "pairing": "Monochrome knit sweater & leather boots",
        "season": "Winter Wardrobe",
        "fabric": "Melton Wool / Heavyweight Twill",
    },
    5: {
        "name": "Sandal",
        "description": "Open-Toe Summer Footwear",
        "icon": "👡",
        "silhouette": "Sculpted Strap Design",
        "pairing": "Linen trousers, shorts, or sundress",
        "season": "High Summer",
        "fabric": "Smooth Nappa Leather / Canvas",
    },
    6: {
        "name": "Shirt",
        "description": "Button-Up Collared Apparel",
        "icon": "👔",
        "silhouette": "Crisp Tailored Collar & Cuffs",
        "pairing": "High-waist trousers or pencil skirt",
        "season": "Year-Round Wardrobe",
        "fabric": "Egyptian Cotton Poplin / Oxford Weave",
    },
    7: {
        "name": "Sneaker",
        "description": "Athletic & Street Footwear",
        "icon": "👟",
        "silhouette": "Low-Profile Athletic Silhouette",
        "pairing": "Casual suit, denim, or relaxed joggers",
        "season": "All-Year Streetwear",
        "fabric": "Full-Grain Leather / Technical Mesh",
    },
    8: {
        "name": "Bag",
        "description": "Handbag & Carry Accessory",
        "icon": "👜",
        "silhouette": "Architectural Geometric Tote",
        "pairing": "Complements any casual or tailored outfit",
        "season": "Perennial Essential",
        "fabric": "Saffiano Leather / Coated Canvas",
    },
    9: {
        "name": "Ankle Boot",
        "description": "All-Weather Ankle Footwear",
        "icon": "🥾",
        "silhouette": "Block-Heel Ankle Profile",
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
# Image Preprocessing & Model Prediction
# ---------------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocess image to normalized (1, 28, 28, 1) float array."""
    gray_img = pil_img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(gray_img, dtype=np.float32) / 255.0
    return arr.reshape((1, 28, 28, 1))

def predict_fashion_item(img_tensor: np.ndarray):
    """Run neural inference and return calibrated softmax probabilities."""
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
                <span>NEURAL ENGINE READY</span>
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
# Preset Sample Wardrobe Locator (10 High Confidence Sample Garments)
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
preset_candidates = [
    os.path.join(base_dir, "sample_wardrobe"),
    os.path.join(base_dir, "..", "test_images", "sample_wardrobe"),
    os.path.join(base_dir, "test_images", "sample_wardrobe"),
    os.path.join(base_dir, "test_images"),
    os.path.join(base_dir, "..", "test_images"),
]

preset_dir = None
for p in preset_candidates:
    if os.path.exists(p) and len([f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) >= 3:
        preset_dir = p
        break

# Map preset files to nice English labels with icons
PRESET_ITEMS = [
    {"label": "👕 T-shirt", "filename": "00_t-shirt.png", "class_name": "T-shirt / Top"},
    {"label": "👖 Trouser", "filename": "01_trouser.png", "class_name": "Trouser"},
    {"label": "🧶 Pullover", "filename": "02_pullover.png", "class_name": "Pullover"},
    {"label": "👗 Dress", "filename": "03_dress.png", "class_name": "Dress"},
    {"label": "🧥 Coat", "filename": "04_coat.png", "class_name": "Coat"},
    {"label": "👡 Sandal", "filename": "05_sandal.png", "class_name": "Sandal"},
    {"label": "👔 Shirt", "filename": "06_shirt.png", "class_name": "Shirt"},
    {"label": "👟 Sneaker", "filename": "07_sneaker.png", "class_name": "Sneaker"},
    {"label": "👜 Bag", "filename": "08_bag.png", "class_name": "Bag"},
    {"label": "🥾 Ankle Boot", "filename": "09_ankle_boot.png", "class_name": "Ankle Boot"},
]

# Initialize Session State for Selection
if "current_image" not in st.session_state:
    # Default to first preset if available
    default_img_path = None
    if preset_dir:
        candidate = os.path.join(preset_dir, "00_t-shirt.png")
        if os.path.exists(candidate):
            default_img_path = candidate
        else:
            files = [f for f in sorted(os.listdir(preset_dir)) if f.lower().endswith(('.png', '.jpg'))]
            if files:
                default_img_path = os.path.join(preset_dir, files[0])

    if default_img_path:
        st.session_state["current_image"] = Image.open(default_img_path)
        st.session_state["current_label"] = "Preset: T-shirt"
    else:
        st.session_state["current_image"] = None
        st.session_state["current_label"] = ""

# ---------------------------------------------------------------------------
# Main Layout: Left Column (Inputs & Preview) & Right Column (Results)
# ---------------------------------------------------------------------------
col_input, col_results = st.columns([1.05, 1.35], gap="large")

with col_input:
    # 1. Preset Images Selection alongside Upload
    render_html(
        """
        <div class="vitrine-card">
            <div class="card-title-bar">
                <h3 class="card-title">✦ Choose an Image</h3>
                <span class="card-meta">Instant Presets & Upload</span>
            </div>
            <p style="font-size: 0.84rem; color: #a19fad; margin-bottom: 0.85rem;">
                Select any preloaded garment below to test instantly, or upload your own image.
            </p>
        </div>
        """
    )

    st.markdown("**✦ Instant Preset Garments:**")
    # Render preset buttons in 2 compact rows of 5 columns
    p_rows = [PRESET_ITEMS[:5], PRESET_ITEMS[5:]]
    for row in p_rows:
        cols = st.columns(len(row))
        for i, item in enumerate(row):
            with cols[i]:
                if st.button(item["label"], key=f"btn_{item['filename']}"):
                    if preset_dir:
                        path = os.path.join(preset_dir, item["filename"])
                        if os.path.exists(path):
                            st.session_state["current_image"] = Image.open(path)
                            st.session_state["current_label"] = f"Preset: {item['class_name']}"

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("**✦ Or Upload Custom Image:**")
    uploaded_file = st.file_uploader(
        "Upload Fashion Item",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Upload a centered fashion item photograph (JPG, PNG, WEBP)",
    )

    if uploaded_file is not None:
        try:
            st.session_state["current_image"] = Image.open(uploaded_file)
            st.session_state["current_label"] = f"Uploaded: {uploaded_file.name}"
        except Exception as err:
            st.error(f"Unable to read uploaded image: {err}")

    # 2. Selected Image Preview (Clean Single Image Only - No 2nd image!)
    active_img = st.session_state.get("current_image")
    active_lbl = st.session_state.get("current_label", "")

    if active_img is not None:
        render_html(
            f"""
            <div class="vitrine-card" style="margin-top: 1.5rem;">
                <div class="card-title-bar">
                    <h3 class="card-title">✦ Selected Image</h3>
                    <span class="card-meta">{active_img.size[0]}×{active_img.size[1]} PX</span>
                </div>
            </div>
            """
        )

        try:
            st.image(active_img, caption=active_lbl, use_column_width=True)
        except TypeError:
            st.image(active_img, caption=active_lbl, use_container_width=True)

    else:
        st.info("Select a preset garment above or upload a photograph to view classification.")

# ---------------------------------------------------------------------------
# Right Column: Neural Inference & Classification Results
# ---------------------------------------------------------------------------
with col_results:
    if active_img is not None:
        # Preprocess and execute inference
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
                        <div class="confidence-label">Inference Time</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-text">Deep CNN</div>
                        <div class="confidence-label">Model Architecture</div>
                    </div>
                </div>

                <div class="styling-grid">
                    <div class="styling-cell">
                        <div class="styling-label">Silhouette</div>
                        <div class="styling-val">{top_meta['silhouette']}</div>
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
            '<div class="vitrine-card">',
            '  <div class="card-title-bar">',
            '    <h3 class="card-title">✦ Class Confidence Distribution</h3>',
            '    <span class="card-meta">Softmax Probabilities</span>',
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
            <div class="vitrine-card" style="text-align: center; padding: 3.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.6;">✧</div>
                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #f8f7fa; margin-bottom: 0.5rem;">
                    Awaiting Garment Selection
                </h3>
                <p style="color: #a19fad; font-size: 0.88rem; max-width: 360px; margin: 0 auto; line-height: 1.6;">
                    Click any of the 10 preset fashion garments on the left or upload your own apparel photograph to see instant deep learning predictions.
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
