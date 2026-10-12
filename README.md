# FashionClassifier ✦ AI Fashion Item Classifier

A modern, high-performance AI Fashion Item Classifier built with **Streamlit** and **TensorFlow/Keras** trained on the Fashion-MNIST dataset, featuring a sleek, professional dark-mode user interface.

![Fashion Classifier](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Design](https://img.shields.io/badge/Theme-Modern%20Dark%20UI-8B5CF6)

---

## Features

- **Professional Dark Theme**: Clean obsidian dark background with electric violet accents and modern glassmorphic cards.
- **Deep Convolutional Neural Network**: High-precision 3-layer Conv2D + MaxPooling architecture trained on Zalando Research Fashion-MNIST.
- **Calibrated Softmax Probability Spectrum**: Real-time confidence scores and full 10-category probability distribution bars.
- **Fashion Details & Styling Guide**: Category classification, garment description, fit details, fabric composition, and seasonal styling recommendations.
- **Instant Preset Selection & Upload**:
  - One-click instant selection from 10 high-resolution pre-loaded garment samples.
  - Drag-and-drop custom photograph uploader with automatic background contrast normalization.
- **Low-Latency Neural Telemetry**: Real-time execution timer (in milliseconds) and model status indicators.

---

## 10 Fashion Categories Classified

| Class ID | Category Name | Description |
| :---: | :--- | :--- |
| `0` | **T-shirt / Top** | Casual Everyday Wardrobe Essential |
| `1` | **Trouser** | Structured Formal & Casual Pants |
| `2` | **Pullover** | Knitwear & Layering Sweater |
| `3` | **Dress** | Formal & Occasion Wear |
| `4` | **Coat** | Outerwear & Overcoat |
| `5` | **Sandal** | Open-Toe Summer Footwear |
| `6` | **Shirt** | Button-Up Collared Apparel |
| `7` | **Sneaker** | Athletic & Street Footwear |
| `8` | **Bag** | Handbag & Carry Accessory |
| `9` | **Ankle Boot** | All-Weather Ankle Footwear |

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
