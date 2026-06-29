# AI Detection of Thoracic Diseases

A hybrid CNN-RNN deep learning system for multi-class chest X-ray classification targeting **COVID-19**, **Pneumonia**, and **Normal** cases. Built entirely in TensorFlow/Keras, this project explores four distinct modeling approaches — from standalone CNN and RNN baselines to a true architectural CNN–Transformer hybrid and a system-level ensemble — to compare how different architectures handle thoracic disease detection from radiographic images.

---

## Experimental Results & Performance Evaluation

All four architectures were evaluated under identical data preprocessing, augmentation, and splitting protocols on a test set extracted from 2,768 frontal chest radiographs.

### 🏆 Overall Model Comparison

| Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Approx. Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 CNN Baseline** (`cnn/`) | 96.6% | 0.9741 | 0.9730 | 0.9735 | ~15 min |
| **Pure RNN BiLSTM** (`rnn/`) | 91.7% | 0.9212 | 0.9227 | 0.9092 | ~25 min |
| **True CNN–Transformer Hybrid** (`hybrid/`) | 97.5% | 0.9742 | 0.9739 | 0.9733 | ~18 min |
| **ResNet50 + DenseNet121 Ensemble** (`ensemble/`) | **97.7%** | **0.9789** | **0.9783** | **0.9800** | ~30 min |

---

### 🔍 Detailed Per-Class Accuracy Breakdown

| Architecture | COVID-19 Accuracy | Pneumonia Accuracy | Normal Accuracy |
| :--- | :---: | :---: | :---: |
| **ResNet50 CNN Baseline** | 97.85% | 97.86% | 95.66% |
| **Pure RNN BiLSTM** | 91.14% | 96.91% | 88.88% |
| **True CNN–Transformer Hybrid** | 97.79% | 96.53% | 97.58% |
| **ResNet50 + DenseNet121 Ensemble** | **97.95%** | 95.54% | **98.79%** |

---

### 💡 Key Findings & Error Analysis

1. **Ensemble Methods Maximize Accuracy & Reliability:** The system-level ensemble achieved the highest overall test accuracy (**97.7%**) and macro F1-score (**0.9800**), alongside the highest per-class accuracy for Normal cases (**98.79%**). Averaging predictions across diverse convolutional backbones effectively smoothed out individual model variance and misclassifications.
2. **Attention Refines Global Relationships:** The True CNN–Transformer hybrid demonstrated a notable improvement over the baseline on Normal cases (**97.58%** vs. 95.66%), proving that self-attention mechanisms successfully re-weight spatial features based on global context.
3. **Pure Sequence Models Struggle with Static Radiographs:** The BiLSTM RNN lagged significantly behind (**91.7%** accuracy), particularly on Normal cases (**88.88%**). Linearizing 2D radiographs into 1D temporal sequences discards crucial spatial relationships, causing the network to occasionally confuse subtle normal anatomical variations with early disease opacities.
4. **CNN Baselines Remain Highly Competitive:** The ResNet50 baseline converged rapidly (~15 minutes) and achieved strong balanced performance (**96.6%** accuracy), reaffirming that well-tuned transfer learning with moderate data augmentation is highly effective for medical image screening.

---

## Datasets

This project uses two publicly available Kaggle datasets:

| Dataset | Source |
|---|---|
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
├── README.md                    ← Main project documentation & experimental results
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

---

## ⚠️ Hardware Note

> **This project was developed and trained entirely on Kaggle's free GPU runtime (NVIDIA Tesla T4).** The training code is **NOT** intended to be re-run on a local CPU or low-VRAM machine without modification. Training involves large CNN backbones (ResNet50, DenseNet121) with mixed-precision (float16) and memory growth settings tuned for Kaggle's environment.

**To re-run the training**, use the original Kaggle notebooks directly — they are configured to run with one click on Kaggle's GPU:

| Component | Kaggle Notebook |
|---|---|
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

## Tech Stack

| Category | Technologies |
|---|---|
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

## Project Team & Authors

**Siddharth Senguttuvan** (22143049) — *System Design, Hybrid/Ensemble Modeling, GPU Training & Analysis*  
B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
**Hindustan Institute of Technology and Science (HITS)**, Chennai  

---

*This repository is a portfolio project demonstrating the design, training, and evaluation of multiple deep learning architectures for medical image classification.*
