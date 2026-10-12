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
    page_title="ATELIER SYNTHÉTIQUE // AI Fashion Classifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS: "Atelier Synthétique" Design System
# Glassmorphism • Obsidian Void • Electric Violet • Editorial Serifs
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');

    /* Global Viewport Foundation */
    .stApp {
        background-color: #0a0a0f !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.07) 0px, transparent 50%) !important;
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

    /* Editorial Header Banner */
    .editorial-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .brand-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.3rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 30%, #cbc3d7 70%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }

    .brand-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #a19fad;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    .telemetry-badges {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
    }

    .telemetry-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #cbc3d7;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Atelier Vitrine Cards */
    .vitrine-card {
        background: rgba(18, 17, 26, 0.70);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.5rem;
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.75);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .vitrine-card:hover {
        border-color: rgba(139, 92, 246, 0.28);
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

    /* Prediction Hero Showcase */
    .prediction-hero {
        background: linear-gradient(145deg, rgba(26, 24, 38, 0.85) 0%, rgba(18, 17, 26, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 8px;
        padding: 1.75rem;
        box-shadow: 0 0 32px -4px rgba(139, 92, 246, 0.22);
        margin-bottom: 1.5rem;
        position: relative;
    }

    .pred-category-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.3rem;
    }

    .pred-class-name {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }

    .pred-french-subtitle {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 1rem;
        color: #a19fad;
        margin-bottom: 1.25rem;
    }

    .confidence-meter-container {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin: 1.25rem 0;
        background: rgba(10, 10, 15, 0.5);
        padding: 1rem;
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
        font-size: 0.75rem;
        color: #a19fad;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Confidence Progress Bars */
    .bar-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.7rem;
        font-size: 0.83rem;
    }

    .bar-name {
        width: 110px;
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
        position: relative;
    }

    .bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #8b5cf6, #c084fc);
        transition: width 0.6s ease-out;
    }

    .bar-fill-secondary {
        height: 100%;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.25);
    }

    .bar-percent {
        width: 58px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #a19fad;
    }

    .bar-percent-active {
        color: #10b981;
        font-weight: 600;
    }

    /* Style Advice Card */
    .styling-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .styling-cell {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 0.75rem;
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
    }

    /* Custom Streamlit File Uploader Restyling */
    [data-testid="stFileUploader"] {
        background: rgba(18, 17, 26, 0.4) !important;
        border: 1px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.7) !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-size: 0.82rem !important;
        padding: 0.65rem 1.6rem !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(139, 92, 246, 0.6) !important;
        border-color: rgba(192, 132, 252, 0.5) !important;
    }

    /* Radio / Selection tabs styling */
    .stRadio > div {
        background: rgba(18, 17, 26, 0.6) !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Image Display Framing */
    [data-testid="stImage"] img {
        border-radius: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fashion-MNIST Taxonomy & Haute Couture Knowledge Base
# ---------------------------------------------------------------------------
CLASS_METADATA = {
    0: {
        "name": "T-shirt / Top",
        "french": "Haut & T-shirt Essentiel",
        "icon": "👕",
        "silhouette": "Minimalist Casual",
        "pairing": "Tailored trousers or raw denim",
        "season": "Spring / Summer",
        "fabric": "Supima Cotton / Silk Jersey",
    },
    1: {
        "name": "Trouser",
        "french": "Pantalon de Costume",
        "icon": "👖",
        "silhouette": "Structured Straight-Leg",
        "pairing": "Oversized blazer or knitwear",
        "season": "All-Season Transition",
        "fabric": "Virgin Wool / Gabardine",
    },
    2: {
        "name": "Pullover",
        "french": "Pull en Tricot Fin",
        "icon": "🧶",
        "silhouette": "Relaxed Cocoon",
        "pairing": "Pleated midi skirt or selvedge chinos",
        "season": "Autumn / Winter",
        "fabric": "Cashmere / Merino Wool",
    },
    3: {
        "name": "Dress",
        "french": "Robe de Créateur",
        "icon": "👗",
        "silhouette": "Fluid Column / A-Line",
        "pairing": "Structured trench & kitten heels",
        "season": "Cocktail / Resort",
        "fabric": "Mulberry Silk / Crêpe de Chine",
    },
    4: {
        "name": "Coat",
        "french": "Manteau Architectural",
        "icon": "🧥",
        "silhouette": "Double-Breasted Statement",
        "pairing": "Monochrome knit & leather boots",
        "season": "Winter Editorial",
        "fabric": "Melton Wool / Double-Faced Cashmere",
    },
    5: {
        "name": "Sandal",
        "french": "Sandale Minimaliste",
        "icon": "👡",
        "silhouette": "Sculptural Strap",
        "pairing": "Linen attire & woven accessories",
        "season": "High Summer",
        "fabric": "Calfskin Nappa / Box Leather",
    },
    6: {
        "name": "Shirt",
        "french": "Chemise Structurée",
        "icon": "👔",
        "silhouette": "Crisp Tailored Collar",
        "pairing": "High-waist trousers or leather skirt",
        "season": "Year-Round Wardrobe",
        "fabric": "Egyptian Cotton Poplin",
    },
    7: {
        "name": "Sneaker",
        "french": "Baskets Urbaines",
        "icon": "👟",
        "silhouette": "Low-Profile Athletic",
        "pairing": "Tailored suit or relaxed loungewear",
        "season": "Contemporary Urban",
        "fabric": "Full-Grain Leather / Mesh",
    },
    8: {
        "name": "Bag",
        "french": "Sac Iconique",
        "icon": "👜",
        "silhouette": "Architectural Geometry",
        "pairing": "Elevates any tailored silhouette",
        "season": "Perennial Investment",
        "fabric": "Saffiano / Smooth Box Leather",
    },
    9: {
        "name": "Ankle Boot",
        "french": "Bottine Contemporaine",
        "icon": "🥾",
        "silhouette": "Pointed / Block-Heel Precision",
        "pairing": "Cropped denim or long trench",
        "season": "Autumn / Winter",
        "fabric": "Polished Burnished Calfskin",
    },
}

CLASS_NAMES = [CLASS_METADATA[i]["name"] for i in range(10)]

# ---------------------------------------------------------------------------
# Model Loading with Streamlit Resource Caching
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
    model_status = True
except Exception as e:
    model_status = False
    model_error = str(e)

# ---------------------------------------------------------------------------
# Preprocessing & Inference Helpers
# ---------------------------------------------------------------------------
def preprocess_for_inference(pil_img: Image.Image) -> np.ndarray:
    """Preprocess image to normalized (1, 28, 28, 1) float array."""
    gray_img = pil_img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(gray_img, dtype=np.float32) / 255.0
    return arr.reshape((1, 28, 28, 1)), gray_img

def run_inference(img_tensor: np.ndarray):
    """Compute softmax probabilities and return prediction metrics."""
    start_time = time.perf_counter()
    logits = model.predict(img_tensor, verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    latency_ms = (time.perf_counter() - start_time) * 1000
    top_idx = int(np.argmax(probabilities))
    return top_idx, probabilities, latency_ms

# ---------------------------------------------------------------------------
# Header: Atelier Synthétique Editorial Brand
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="editorial-header">
        <div>
            <h1 class="brand-title">ATELIER SYNTHÉTIQUE</h1>
            <div class="brand-tagline">Haute Couture Neural Classification Engine // Fashion-MNIST</div>
        </div>
        <div class="telemetry-badges">
            <div class="telemetry-badge">
                <span class="pulse-dot"></span>
                <span>NEURAL ENGINE ACTIVE</span>
            </div>
            <div class="telemetry-badge">
                <span>KERAS / TF 2.15</span>
            </div>
            <div class="telemetry-badge">
                <span>28×28 CONVNET</span>
            </div>
            <div class="telemetry-badge">
                <span>10 COUTURE TAXA</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_status:
    st.error(f"Model initialization failure: {model_error}")
    st.stop()

# ---------------------------------------------------------------------------
# Sample Wardrobe Locator
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
sample_search_paths = [
    os.path.join(base_dir, "test_images"),
    os.path.join(base_dir, "..", "test_images"),
    os.path.join(base_dir, "sample_images"),
]
sample_dir = None
for p in sample_search_paths:
    if os.path.exists(p) and len(os.listdir(p)) > 0:
        sample_dir = p
        break

sample_files = {}
if sample_dir:
    for fname in sorted(os.listdir(sample_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            label = fname
            if "1" in fname:
                label = "🥾 Curated Sample: Ankle Boot"
            elif "2" in fname:
                label = "🧶 Curated Sample: Pullover Knit"
            elif "3" in fname:
                label = "🥾 Curated Sample: Suede Boot"
            sample_files[label] = os.path.join(sample_dir, fname)

# ---------------------------------------------------------------------------
# Two-Column Layout: Input / Curation (Left) & Neural Telemetry (Right)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.05, 1.35], gap="large")

with col_left:
    st.markdown(
        """
        <div class="vitrine-card">
            <div class="card-title-bar">
                <h3 class="card-title">✦ Specimen Selection</h3>
                <span class="card-meta">Curate Input</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Source Method",
        options=["Select from Curated Atelier Wardrobe", "Upload Custom Garment Image"],
        label_visibility="collapsed",
    )

    selected_image = None
    image_source_label = ""

    if input_mode == "Select from Curated Atelier Wardrobe" and sample_files:
        chosen_sample_label = st.selectbox(
            "Select Wardrobe Specimen",
            options=list(sample_files.keys()),
            help="Pre-loaded Fashion-MNIST evaluation specimens",
        )
        sample_path = sample_files[chosen_sample_label]
        try:
            selected_image = Image.open(sample_path)
            image_source_label = chosen_sample_label
        except Exception as err:
            st.warning(f"Could not load specimen: {err}")
    else:
        uploaded_file = st.file_uploader(
            "Upload Garment Photograph",
            type=["jpg", "jpeg", "png", "webp"],
            help="High contrast or centered fashion apparel image",
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                image_source_label = uploaded_file.name
            except Exception as err:
                st.error(f"Image read error: {err}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Image Vitrine Preview Card
    if selected_image is not None:
        st.markdown(
            f"""
            <div class="vitrine-card">
                <div class="card-title-bar">
                    <h3 class="card-title">✦ Specimen Vitrine</h3>
                    <span class="card-meta">{selected_image.size[0]}×{selected_image.size[1]} PX • {selected_image.format or 'RGB'}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        sub_col1, sub_col2 = st.columns([1.2, 0.8])
        with sub_col1:
            try:
                st.image(selected_image, caption=f"Source: {image_source_label}", use_column_width=True)
            except TypeError:
                st.image(selected_image, caption=f"Source: {image_source_label}", use_container_width=True)
        with sub_col2:
            img_tensor, gray_preview = preprocess_for_inference(selected_image)
            try:
                st.image(gray_preview, caption="Normalized 28×28 Tensor", use_column_width=True)
            except TypeError:
                st.image(gray_preview, caption="Normalized 28×28 Tensor", use_container_width=True)
            st.caption("Preprocessing: Lanczos Resample → Grayscale → [0.0, 1.0] Tensor")

        classify_clicked = st.button("✦ Diagnose Specimen with Neural Engine", key="classify_btn")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Select a wardrobe specimen above or upload a photograph to begin neural classification.")
        classify_clicked = False

# ---------------------------------------------------------------------------
# Right Column: Neural Inference & Haute Couture Diagnosis
# ---------------------------------------------------------------------------
with col_right:
    # Auto-run if sample is chosen, or when button clicked
    should_run = (selected_image is not None) and (
        classify_clicked or ("auto_inferred" not in st.session_state)
    )

    if selected_image is not None:
        img_tensor, _ = preprocess_for_inference(selected_image)
        top_idx, probabilities, latency_ms = run_inference(img_tensor)
        st.session_state["auto_inferred"] = True

        top_meta = CLASS_METADATA[top_idx]
        confidence_pct = probabilities[top_idx] * 100.0

        # Primary Haute Couture Classification Showcase
        st.markdown(
            f"""
            <div class="prediction-hero">
                <div class="pred-category-badge">Identified Couture Taxon // Class #{top_idx}</div>
                <div class="pred-class-name">{top_meta['icon']} {top_meta['name']}</div>
                <div class="pred-french-subtitle">{top_meta['french']}</div>

                <div class="confidence-meter-container">
                    <div>
                        <div class="confidence-score">{confidence_pct:.1f}%</div>
                        <div class="confidence-label">Confidence Index</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255, 255, 255, 0.08); padding-left: 1.25rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; color: #c084fc;">
                            {latency_ms:.1f} <span style="font-size: 0.8rem; color: #a19fad;">ms</span>
                        </div>
                        <div class="confidence-label">Inference Latency</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255, 255, 255, 0.08); padding-left: 1.25rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; color: #f8f7fa;">
                            Deep CNN
                        </div>
                        <div class="confidence-label">Network Backbone</div>
                    </div>
                </div>

                <div class="styling-grid">
                    <div class="styling-cell">
                        <div class="styling-label">Silhouette</div>
                        <div class="styling-val">{top_meta['silhouette']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Fabric Archetype</div>
                        <div class="styling-val">{top_meta['fabric']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Seasonal Pairing</div>
                        <div class="styling-val">{top_meta['pairing']}</div>
                    </div>
                    <div class="styling-cell">
                        <div class="styling-label">Seasonality</div>
                        <div class="styling-val">{top_meta['season']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Full 10-Class Confidence Spectrum
        st.markdown(
            """
            <div class="vitrine-card">
                <div class="card-title-bar">
                    <h3 class="card-title">✦ Probability Distribution Spectrum</h3>
                    <span class="card-meta">Softmax Telemetry</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Sort classes by probability descending
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            prob = probabilities[idx] * 100.0
            is_winner = (idx == top_idx)
            name_class = "bar-name bar-name-active" if is_winner else "bar-name"
            fill_class = "bar-fill" if is_winner else "bar-fill-secondary"
            pct_class = "bar-percent bar-percent-active" if is_winner else "bar-percent"
            icon = CLASS_METADATA[idx]["icon"]
            label_text = f"{icon} {CLASS_METADATA[idx]['name']}"

            st.markdown(
                f"""
                <div class="bar-row">
                    <span class="{name_class}">{label_text}</span>
                    <div class="bar-track">
                        <div class="{fill_class}" style="width: {prob:.2f}%;"></div>
                    </div>
                    <span class="{pct_class}">{prob:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder vitrine before selection
        st.markdown(
            """
            <div class="vitrine-card" style="text-align: center; padding: 3.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.6;">✧</div>
                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #f8f7fa; margin-bottom: 0.5rem;">
                    Awaiting Garment Specimen
                </h3>
                <p style="color: #a19fad; font-size: 0.88rem; max-width: 360px; margin: 0 auto; line-height: 1.6;">
                    Select one of the curated Fashion-MNIST test specimens or upload an apparel photograph on the left to initiate real-time deep neural inference.
                </p>
                <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 0.8rem;">
                    <span class="telemetry-badge">10 Apparel Classes</span>
                    <span class="telemetry-badge">93.3k Parameters</span>
                    <span class="telemetry-badge">Softmax Calibrated</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); text-align: center; color: #5a576b; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;">
        ATELIER SYNTHÉTIQUE // ARCHITECTURE: 3× CONV2D + MAXPOOLING + DENSE // DATASET: ZALANDO RESEARCH FASHION-MNIST
    </div>
    """,
    unsafe_allow_html=True,
)
