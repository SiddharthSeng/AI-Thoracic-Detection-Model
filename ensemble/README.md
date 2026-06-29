# Ensemble Component — System-Level Hybrid (ResNet50 + DenseNet121)

This folder contains the system-level ensemble approach to thoracic disease classification. Two independent CNN models — **ResNet50** and **DenseNet121** — are each fine-tuned (last 20 layers trainable, ImageNet-pretrained) on the same dataset with their respective preprocessing pipelines. At inference, prediction probabilities from both models are averaged to produce the final ensemble classification.

This approach leverages architectural diversity (residual connections vs. dense connections) to reduce prediction variance and improve generalization compared to any single model.

### 📊 Experimental Results

| Metric | Score | Per-Class Accuracy Breakdown |
| :--- | :---: | :--- |
| **Test Accuracy** | **97.7%** | • **COVID-19:** 97.95% |
| **Macro Precision** | **0.9789** | • **Pneumonia:** 95.54% |
| **Macro Recall** | **0.9783** | • **Normal:** 98.79% |
| **Macro F1-Score** | **0.9800** | *Approx. Training Time: ~30 min (Tesla T4)* |

> **Note:** This system-level ensemble achieved the highest overall macro F1-score (0.9800) and the highest sensitivity for Normal radiographs (98.79%) across all four evaluated architectures.

| File | Description |
|---|---|
| `system_level_hybrid_ensemble_thoracic_diseases.py` | Clean Python script extracted from the notebook |
| `system_level_hybrid_ensemble_thoracic_diseases.ipynb` | Original Kaggle notebook (with outputs) |
