# CNN Component — ResNet50 Thoracic Disease Classifier

This folder contains the standalone CNN model for thoracic disease classification. It uses a **ResNet50** backbone (trained from scratch, without ImageNet weights) with a custom classification head (BatchNorm → Dense 256 → Dropout → Softmax) for 3-class prediction: COVID-19, Pneumonia, and Normal. The model is trained on 224×224 chest X-ray images with data augmentation (rotation, shifts, flips, zoom, shear) and class-weight balancing.

This component serves as the baseline CNN approach. Its predictions are also used as one input to the system-level ensemble in `../ensemble/`.

| File | Description |
|------|-------------|
| `cnn_thoracic_disease.py` | Clean Python script extracted from the notebook |
| `cnn_thoracic_disease.ipynb` | Original Kaggle notebook (with outputs) |
