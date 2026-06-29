# AI Detection of Thoracic Diseases

A hybrid CNN-RNN deep learning system for multi-class chest X-ray classification targeting **COVID-19**, **Pneumonia**, and **Normal** cases. Built entirely in TensorFlow/Keras, this project explores four distinct modeling approaches — from standalone CNN and RNN baselines to a true architectural CNN–Transformer hybrid and a system-level ensemble — to compare how different architectures handle thoracic disease detection from radiographic images.

---

## Datasets

This project uses two publicly available Kaggle datasets:

| Dataset | Source |
|---------|--------|
| **Chest X-Ray Pneumonia** | [kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **COVID-19 Radiography Database** | [kaggle.com/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |

**Preprocessing & Splitting:**
- Images are resized to **224×224** pixels and preprocessed with model-specific functions (ResNet50/DenseNet121 preprocessing or simple normalization for the RNN).
- Data augmentation is applied to training data: rotation (±15°), width/height shifts (10%), horizontal flips, zoom (15%), and shear (10%).
- The normal class is capped to match the combined count of COVID-19 + Pneumonia images to mitigate class imbalance.
- Stratified train/validation/test splits are used (approximately 70%/15%/15%) with a fixed random seed (42) for reproducibility.
- Class weights are computed via `sklearn.utils.class_weight.compute_class_weight("balanced")` to further address imbalance during training.

---

## Architecture Overview

This project implements and compares **four** approaches to thoracic disease classification:

### 1. CNN Baseline (`cnn/`)
A **ResNet50** backbone trained from scratch (no ImageNet weights) with global average pooling, BatchNormalization, Dense(256, ReLU), Dropout(0.5), and a 3-class Softmax head. Trained with Adam (lr=3e-4) and categorical crossentropy.

### 2. Pure RNN (`rnn/`)
A **Bidirectional LSTM** model that treats each 224×224 X-ray image as a sequence of 224 timesteps, where each timestep is a flattened row of 672 features (224 pixels × 3 RGB channels). Two stacked Bidirectional LSTM layers (256 units, dropout=0.3) are followed by BatchNorm, Dense(256, ReLU), Dropout(0.5), and Softmax. This provides a purely sequential perspective on medical images.

### 3. True Architectural Hybrid (`hybrid/`)
A **CNN–Transformer** model that combines ResNet50 feature extraction with Transformer-based reasoning in a single end-to-end architecture:
- ResNet50 (ImageNet-pretrained, last 20 layers fine-tuned) extracts spatial feature maps.
- A 1×1 convolution projects features to 256-dimensional embeddings.
- Feature maps are flattened into a token sequence with learnable positional embeddings.
- Two Transformer encoder blocks (4-head self-attention, key_dim=64, FFN dim=256) process the token sequence.
- Global average pooling → LayerNorm → Dense → Softmax produces the final prediction.

### 4. System-Level Ensemble (`ensemble/`)
A **probability-averaging ensemble** of two independently trained CNNs:
- **ResNet50** (ImageNet-pretrained, last 20 layers fine-tuned)
- **DenseNet121** (ImageNet-pretrained, last 20 layers fine-tuned)

Each model has its own preprocessing pipeline. At inference, their softmax probabilities are averaged to produce the final class prediction. This leverages architectural diversity (residual vs. dense connections) for improved generalization.

---

## Repository Structure

```
AI-Thoracic-Detection-Model/
├── README.md                    ← You are here
├── requirements.txt             ← Consolidated Python dependencies
├── .gitignore                   ← Standard ignores for Python, data, models
│
├── cnn/                         ← CNN baseline (ResNet50, trained from scratch)
│   ├── cnn_thoracic_disease.py
│   ├── cnn_thoracic_disease.ipynb
│   └── README.md
│
├── rnn/                         ← Pure RNN (Bidirectional LSTM)
│   ├── rnn_thoracic_diseases.py
│   ├── rnn_thoracic_diseases.ipynb
│   └── README.md
│
├── hybrid/                      ← True architectural hybrid (CNN + Transformer)
│   ├── true_architectural_hybrid_thoracic_diseases.py
│   ├── true_architectural_hybrid_thoracic_diseases.ipynb
│   └── README.md
│
└── ensemble/                    ← System-level ensemble (ResNet50 + DenseNet121)
    ├── system_level_hybrid_ensemble_thoracic_diseases.py
    ├── system_level_hybrid_ensemble_thoracic_diseases.ipynb
    └── README.md
```

Each folder contains:
- A `.py` script — the clean, documented Python code extracted from the Kaggle notebook.
- The original `.ipynb` notebook — preserved for reference (view on Kaggle to see outputs/plots).
- A `README.md` — a short description of that specific component.

---

## ⚠️ Hardware Note

> **This project was developed and trained entirely on Kaggle's free GPU runtime (NVIDIA Tesla T4).** The training code is **NOT** intended to be re-run on a local CPU or low-VRAM machine without modification. Training involves large CNN backbones (ResNet50, DenseNet121) with mixed-precision (float16) and memory growth settings tuned for Kaggle's environment.

**To re-run the training**, use the original Kaggle notebooks directly — they are configured to run with one click on Kaggle's GPU:

| Component | Kaggle Notebook |
|-----------|----------------|
| CNN (ResNet50) | [Open on Kaggle](https://www.kaggle.com/code/siddharthsenguttuvan/cnn-thoracic-disease) |
| RNN (Bidirectional LSTM) | [Open on Kaggle](https://www.kaggle.com/code/siddharthsenguttuvan/rnn-thoracic-diseases) |
| Hybrid (CNN + Transformer) | [Open on Kaggle](https://www.kaggle.com/code/siddharthsenguttuvan/true-architectural-hybrid-thoracic-diseases) |
| Ensemble (ResNet50 + DenseNet121) | [Open on Kaggle](https://www.kaggle.com/code/siddharthsenguttuvan/system-level-hybrid-ensemble-thoracic-diseases) |

---

## How to Use

If you want to **adapt the code** for your own experiments:

```bash
# Clone the repository
git clone https://github.com/SiddharthSeng/AI-Thoracic-Detection-Model.git
cd AI-Thoracic-Detection-Model

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** You will need a CUDA-capable GPU with sufficient VRAM (≥16 GB recommended) and the appropriate CUDA/cuDNN versions for TensorFlow 2.x. Alternatively, upload the notebooks to Kaggle or Google Colab for free GPU access.

You will also need to download the datasets from Kaggle and update the `BASE_ROOT` / `COVID_PATH` / `PNEUMONIA_PATH` variables in each script to point to your local data directory.

---

## Results & Metrics

> **Note:** The notebook outputs (accuracy numbers, confusion matrices, classification reports) are visible in the original Kaggle notebooks linked above. The `.ipynb` files in this repository were downloaded without cell outputs. Please refer to the Kaggle links to view the full training logs, plots, and evaluation metrics.

All four models produce:
- **Classification reports** (precision, recall, F1-score per class)
- **Confusion matrices** (heatmaps)
- **Per-class accuracy** breakdowns
- **Training/validation accuracy and loss curves**

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **CNN Backbones** | ResNet50, DenseNet121 |
| **Sequence Models** | Bidirectional LSTM |
| **Attention** | Multi-Head Self-Attention (Transformer Encoder) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **ML Utilities** | scikit-learn (train/test split, class weights, metrics) |
| **Training Hardware** | Kaggle GPU (NVIDIA Tesla T4) |
| **Precision** | Mixed precision (float16) for GPU acceleration |

---

## Author

**Siddharth Senguttuvan**
B.Tech Computer Science (AI & ML)
Hindustan Institute of Technology and Science (HITS)

---

*This repository is a portfolio project demonstrating the design, training, and evaluation of multiple deep learning architectures for medical image classification.*
