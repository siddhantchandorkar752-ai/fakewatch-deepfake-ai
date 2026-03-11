
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:FF0000,100:FF6B00&height=250&section=header&text=FAKEWATCH&fontSize=90&fontColor=ffffff&fontAlignY=38&desc=Deepfake%20Forensic%20Intelligence%20System%20v2.0&descAlignY=60&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=22&duration=3000&pause=800&color=FF4500&center=true&vCenter=true&multiline=true&width=800&height=120&lines=🕵️+Detecting+Deepfakes+with+Surgical+Precision;EfficientNet+%2B+Temporal+Transformer+%2B+GradCAM;Spatial+%2B+Temporal+Feature+Fusion+Pipeline;Production-Grade+%7C+ONNX+%7C+FastAPI+%7C+Docker)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet_B4-Spatial-FF6B00?style=for-the-badge)
![Transformer](https://img.shields.io/badge/Transformer-Temporal-FF0000?style=for-the-badge)
![GradCAM](https://img.shields.io/badge/GradCAM-XAI-FF4500?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FF4500?style=for-the-badge)

<br/>

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/siddhantchandorkar/fakewatch-deepfake-ai)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai/fakewatch-deepfake-ai)

<br/>

> **🕵️ Not just a classifier — a forensic intelligence system that sees what the human eye cannot.**

</div>

---

## 🔥 WHAT IS FAKEWATCH?

```
╔══════════════════════════════════════════════════════════════════════════╗
║     FAKEWATCH — Deepfake Forensic Intelligence System v2.0             ║
║                                                                          ║
║     "Most detectors look at frames. FAKEWATCH looks at time."           ║
║                                                                          ║
║     SPATIAL  → EfficientNet-B4   → Per-frame spatial artifacts          ║
║     TEMPORAL → Transformer       → Cross-frame inconsistencies          ║
║     FUSION   → Feature Fusion    → Combined representation              ║
║     XAI      → GradCAM           → Forgery localization heatmap         ║
║     DEPLOY   → FastAPI + ONNX    → Production-ready inference           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

FAKEWATCH is a **research-grade deepfake detection system** that combines **spatial** and **temporal** deep learning to detect manipulated videos with surgical precision. Unlike simple frame-level classifiers, FAKEWATCH models **temporal inconsistencies across frames** using a Transformer architecture — the same technique used in cutting-edge research labs.

Built to production engineering standards with modular architecture, ONNX export, FastAPI deployment, Docker support, and full explainability via GradCAM heatmaps.

---

## 😔 PROBLEM STATEMENT

```
2025 Reality:
─────────────────────────────────────────────────────────────
96% of deepfake videos target real people without consent
Deepfake fraud cases up 3000% since 2022
Election interference via synthetic media on the rise
Identity theft using face-swap technology — untraceable
─────────────────────────────────────────────────────────────

Why existing detectors fail:
• Frame-level only        — miss temporal artifacts
• Single dataset training — poor generalization
• Black box               — no explainability
• Not deployable          — Jupyter notebooks only
─────────────────────────────────────────────────────────────

FAKEWATCH solves all four.
```

---

## 🌍 REAL WORLD IMPACT

| Use Case | How FAKEWATCH Helps |
|----------|-------------------|
| 📰 **News Verification** | Journalists verify viral videos before publishing |
| ⚖️ **Court Evidence** | Verify if submitted video evidence is tampered |
| 🗳️ **Election Integrity** | Detect synthetic political speeches |
| 🏢 **HR Screening** | Verify video interviews are genuine |
| 📱 **Social Media** | Flag deepfake content before it goes viral |
| 🔐 **Identity Verification** | Prevent deepfake-based KYC fraud |

---

## ⚡ CORE FEATURES

| Feature | Description | Tech |
|---------|-------------|------|
| 🧠 **Spatial Analysis** | Per-frame feature extraction from face regions | EfficientNet-B4 |
| ⏱️ **Temporal Modeling** | Cross-frame inconsistency detection | Vision Transformer |
| 🔀 **Feature Fusion** | Spatial + temporal combined representation | Custom Fusion Module |
| 👁️ **Forgery Localization** | Heatmap showing exactly where fake is | GradCAM |
| 📊 **Cross-Dataset Eval** | Test generalization across 3 benchmarks | FF++ + Celeb-DF + DFDC |
| 🚀 **Production API** | REST endpoint for video analysis | FastAPI |
| 📦 **ONNX Export** | Optimized inference for deployment | ONNX Runtime |
| 🐳 **Docker Ready** | One command deployment | Docker |
| 📋 **Experiment Tracking** | Structured logging + checkpointing | Custom Logger |
| ⏹️ **Early Stopping** | Prevent overfitting automatically | Patience-based |

---

## 🏗️ ARCHITECTURE

```
VIDEO INPUT
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   FAKEWATCH PIPELINE v2.0                   │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │  PREPROCESSING                       │                  │
│  │  Face Detection → Frame Sampling     │                  │
│  │  Normalization → Tensor (B,T,C,H,W)  │                  │
│  └─────────────────┬────────────────────┘                  │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │   SPATIAL BRANCH    │                            │
│         │   EfficientNet-B4   │                            │
│         │   Per-frame feats   │                            │
│         │   → (B, T, 1792)    │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │   TEMPORAL BRANCH   │                            │
│         │   Transformer       │                            │
│         │   Positional Enc    │                            │
│         │   8 heads, 4 layers │                            │
│         │   → (B, 512)        │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │   FUSION MODULE     │                            │
│         │   Spatial + Temporal│                            │
│         │   LayerNorm + ReLU  │                            │
│         │   → (B, 256)        │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│      ┌─────────────┴──────────────┐                        │
│      ▼                            ▼                        │
│  CLASSIFIER                   GradCAM                      │
│  REAL / FAKE + prob           Forgery Heatmap              │
│  → Confidence Score           → Localization Map           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 PROJECT STRUCTURE

```
fakewatch/
├── src/
│   ├── models/
│   │   ├── spatial.py          ← EfficientNet-B4 spatial extractor
│   │   ├── temporal.py         ← Transformer + positional encoding
│   │   ├── fusion.py           ← Feature fusion module
│   │   ├── classifier.py       ← Classification head
│   │   ├── fakewatch.py        ← Master model (all modules combined)
│   │   └── __init__.py
│   ├── data/
│   │   ├── preprocess.py       ← Frame extraction + face crop + normalization
│   │   ├── dataset.py          ← DeepfakeDataset + DataLoader
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py          ← FakeWatchTrainer (train + eval + checkpoint)
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluator.py        ← Cross-dataset evaluation + metrics
│   │   └── __init__.py
│   ├── explainability/
│   │   ├── gradcam.py          ← GradCAM heatmap generation
│   │   └── __init__.py
│   ├── deployment/
│   │   ├── inference.py        ← FakeWatchInference + ONNX export
│   │   ├── api.py              ← FastAPI server
│   │   └── __init__.py
│   └── utils/
│       ├── logger.py           ← Structured logging
│       ├── config.py           ← YAML config loader
│       └── __init__.py
├── configs/
│   └── config.yaml             ← All hyperparameters + settings
├── scripts/
│   ├── download_datasets.py    ← Dataset folder structure creator
│   └── evaluate.py             ← Cross-dataset evaluation script
├── datasets/                   ← NOT committed to git
├── checkpoints/                ← NOT committed to git
├── logs/                       ← NOT committed to git
├── train.py                    ← Training entry point
├── inference.py                ← CLI inference script
├── app.py                      ← Gradio web UI
├── setup.py
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## 🗄️ DATASETS

| Dataset | Size | Type | Access |
|---------|------|------|--------|
| **FaceForensics++** | 1000 real + 4000 fake | Face swap + reenactment | [Request Form](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF** | 590 real + 5639 fake | Celebrity deepfakes | [Request Form](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** | 100k+ videos | Competition dataset | [Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge) |

```bash
# Create dataset folder structure
python scripts/download_datasets.py
```

---

## 🛠️ INSTALLATION

```bash
# 1. Clone
git clone https://github.com/siddhantchandorkar752-ai/fakewatch-deepfake-ai.git
cd fakewatch-deepfake-ai

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Install
pip install -r requirements.txt

# 4. Dataset structure
python scripts/download_datasets.py
```

---

## 🚀 USAGE

```bash
# Train
python train.py

# Inference
python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pt

# Evaluate across datasets
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Gradio UI
python app.py

# FastAPI
uvicorn src.deployment.api:app --reload --port 8000

# Docker
docker build -t fakewatch .
docker run -p 7860:7860 fakewatch
```

---

## 🧪 EXAMPLE OUTPUT

```
==================================================
       FAKEWATCH ANALYSIS RESULT
==================================================
  Prediction : FAKE
  Fake Prob  : 0.9732
  Real Prob  : 0.0268
  Confidence : 0.9732
==================================================

GradCAM → eye region + jaw boundary highlighted
         Classic face-swap artifact localized
```

---

## 📊 METRICS

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve — primary metric |
| **F1 Score** | Harmonic mean of precision + recall |
| **Precision** | Fake predictions that were actually fake |
| **Recall** | Actual fakes correctly detected |
| **Cross-Dataset AUC** | Generalization across FF++, Celeb-DF, DFDC |

---

## 🔑 API ENDPOINTS

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System info |
| GET | `/health` | Health check + model status |
| POST | `/analyze` | Upload video → get prediction |

---

## 🛠️ TECH STACK

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet_B4-FF6B00?style=for-the-badge)
![Transformer](https://img.shields.io/badge/Transformer-FF0000?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🔮 FUTURE IMPROVEMENTS

- [ ] Multi-face tracking — detect multiple subjects simultaneously
- [ ] Audio deepfake detection — voice synthesis detection
- [ ] Real-time video stream analysis
- [ ] Fine-tuning on custom enterprise datasets
- [ ] Mobile deployment — TensorFlow Lite export
- [ ] Browser extension for social media verification

---

## 📄 LICENSE

Distributed under the MIT License.

---

## 👨‍💻 AUTHOR

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0D1117,100:FF4500&height=60&text=Siddhant%20Chandorkar&fontSize=28&fontColor=ffffff&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-siddhantchandorkar-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just build AI. I build AI that understands humans."*

</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6B00,50:FF0000,100:0D1117&height=120&section=footer&text=FAKEWATCH%20v2.0&fontSize=28&fontColor=ffffff&fontAlignY=65&animation=fadeIn" width="100%"/>
</div>
