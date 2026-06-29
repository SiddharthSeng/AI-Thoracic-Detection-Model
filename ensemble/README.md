# Ensemble Component — System-Level Hybrid (ResNet50 + DenseNet121)

This folder contains the system-level ensemble approach to thoracic disease classification. Two independent CNN models — **ResNet50** and **DenseNet121** — are each fine-tuned (last 20 layers trainable, ImageNet-pretrained) on the same dataset with their respective preprocessing pipelines. At inference, prediction probabilities from both models are averaged to produce the final ensemble classification.

This approach leverages architectural diversity (residual connections vs. dense connections) to reduce prediction variance and improve generalization compared to any single model.

| File | Description |
|------|-------------|
| `system_level_hybrid_ensemble_thoracic_diseases.py` | Clean Python script extracted from the notebook |
| `system_level_hybrid_ensemble_thoracic_diseases.ipynb` | Original Kaggle notebook (with outputs) |
