# FashionClassifier ✦ Atelier Synthétique

An editorial, haute couture AI Fashion Item Classifier built with **Streamlit** and **TensorFlow/Keras** trained on the Fashion-MNIST dataset. Powered by the *Atelier Synthétique* luxury dark-mode design system.

![Fashion Classifier](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Design](https://img.shields.io/badge/Design-Atelier%20Synth%C3%A9tique-8B5CF6)

---

## Features

- **Haute Couture Aesthetic**: Designed using Google Stitch (*Atelier Synthétique* system) featuring Obsidian void backgrounds, electric violet highlights, and atmospheric glassmorphism.
- **Deep Convolutional Neural Network**: High-precision 3-layer Conv2D + MaxPooling architecture trained on Zalando Research Fashion-MNIST.
- **Calibrated Softmax Probability Spectrum**: Real-time confidence scores and full 10-class probability distribution bars.
- **Editorial Fashion Intelligence**: Instant categorization with French couture taxonomy, silhouette classifications, fabric archetypes, and seasonal styling recommendations.
- **Interactive Specimen Testing**:
  - One-click instant curation from pre-loaded test specimens.
  - Drag-and-drop custom photograph uploader with live 28×28 grayscale tensor normalization preview.
- **Low-Latency Neural Telemetry**: Microsecond execution timers and live model telemetry metrics.

---

## 10 Fashion Taxa Classified

| Class ID | Garment Category | French Editorial Taxon |
| :---: | :--- | :--- |
| `0` | **T-shirt / Top** | *Haut & T-shirt Essentiel* |
| `1` | **Trouser** | *Pantalon de Costume* |
| `2` | **Pullover** | *Pull en Tricot Fin* |
| `3` | **Dress** | *Robe de Créateur* |
| `4` | **Coat** | *Manteau Architectural* |
| `5` | **Sandal** | *Sandale Minimaliste* |
| `6` | **Shirt** | *Chemise Structurée* |
| `7` | **Sneaker** | *Baskets Urbaines* |
| `8` | **Bag** | *Sac Iconique* |
| `9` | **Ankle Boot** | *Bottine Contemporaine* |

---

## Setup & How to Run

### 1. Run Locally (Recommended)

```bash
# 1. Activate the Python 3.11 virtual environment
source .venv/bin/activate

# 2. Run the Streamlit application
streamlit run app/main.py
```

Open your browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

### 2. Run with Docker

```bash
cd app
docker build -t fashion-classifier .
docker run -p 80:80 fashion-classifier
```

Then visit **[http://localhost:80](http://localhost:80)**.
