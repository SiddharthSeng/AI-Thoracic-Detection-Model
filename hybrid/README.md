# Hybrid Component — CNN + Transformer Thoracic Disease Classifier

This folder contains the true architectural hybrid model that fuses **CNN** and **Transformer** components within a single end-to-end network. A fine-tuned ResNet50 backbone extracts spatial feature maps, which are projected to 256-dimensional embeddings via a 1×1 convolution, flattened into a token sequence, and augmented with learnable positional embeddings. Two Transformer encoder blocks (4-head self-attention, key_dim=64, FFN dim=256) then process these tokens. Global average pooling, LayerNorm, and a Dense classifier produce the final 3-class prediction.

Unlike the ensemble approach (which combines separate models at inference), this hybrid model integrates CNN feature extraction and Transformer-based reasoning in a single trainable architecture.

### 📊 Experimental Results

| Metric | Score | Per-Class Accuracy Breakdown |
| :--- | :---: | :--- |
| **Test Accuracy** | **97.5%** | • **COVID-19:** 97.79% |
| **Macro Precision** | **0.9742** | • **Pneumonia:** 96.53% |
| **Macro Recall** | **0.9739** | • **Normal:** 97.58% |
| **Macro F1-Score** | **0.9733** | *Approx. Training Time: ~18 min (Tesla T4)* |

> **Note:** Incorporating self-attention improved Normal-class detection accuracy from 95.66% (in the baseline CNN) to 97.58%, demonstrating the value of global context modeling.

| File | Description |
|---|---|
| `true_architectural_hybrid_thoracic_diseases.py` | Clean Python script extracted from the notebook |
| `true_architectural_hybrid_thoracic_diseases.ipynb` | Original Kaggle notebook (with outputs) |
