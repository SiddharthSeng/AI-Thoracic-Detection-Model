# RNN Component — Bidirectional LSTM Thoracic Disease Classifier

This folder contains the pure RNN model for thoracic disease classification. It uses a **Bidirectional LSTM** architecture that treats each 224×224 chest X-ray as a sequence: the image is reshaped into 224 timesteps of 672 features (224 pixels × 3 channels per row), allowing the LSTM to scan the image row-by-row. Two stacked Bidirectional LSTM layers (256 units each) with dropout are followed by BatchNorm, a Dense layer, and a Softmax classifier.

This component demonstrates a pure sequence-based approach to image classification without any convolutional layers, serving as a complementary architecture to the CNN baseline.

| File | Description |
|------|-------------|
| `rnn_thoracic_diseases.py` | Clean Python script extracted from the notebook |
| `rnn_thoracic_diseases.ipynb` | Original Kaggle notebook (with outputs) |
