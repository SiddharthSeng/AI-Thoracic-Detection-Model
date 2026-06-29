"""
True Architectural Hybrid: CNN + Transformer for Thoracic Disease Detection
============================================================================

Originally developed as a Kaggle notebook. Converted to a standalone script.

This module implements a hybrid deep-learning architecture that combines:
  - **CNN backbone (ResNet50)**: Extracts spatial feature maps from chest
    X-ray images. The last 20 layers are fine-tuned while earlier layers
    remain frozen, leveraging ImageNet pre-trained weights.
  - **Transformer encoder**: The ResNet50 feature maps are projected to a
    lower-dimensional embedding space via a 1×1 convolution, reshaped into
    a sequence of patch tokens, augmented with learnable positional
    embeddings, and then processed by two stacked Transformer encoder
    blocks (multi-head self-attention + feed-forward) to capture global
    contextual relationships across spatial regions of the radiograph.

The model classifies chest X-rays into three categories:
  COVID-19 | Viral Pneumonia | Normal

Key design choices preserved from the original notebook:
  - Mixed-precision (float16) training when a GPU is available.
  - Balanced class weights computed from the training split.
  - Data augmentation (rotation, shift, flip, zoom, shear) on training set.
  - Early stopping, learning-rate reduction, and model checkpointing.
"""

# --- Imports ---

import os
import warnings
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

warnings.filterwarnings("ignore")

# --- Environment & GPU Configuration ---

print("TensorFlow Version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPU.")
    except Exception as e:
        print("Could not set memory growth for GPU:", e)

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
else:
    print("No GPU detected, running on CPU.")

# --- Dataset Paths (Kaggle layout) ---

BASE_ROOT    = "/kaggle/input/modeltestcnn/Throacic Detection using AI"
CONTENT_ROOT = os.path.join(BASE_ROOT, "content")
COVID_PATH   = os.path.join(CONTENT_ROOT, "COVID-19_Radiography_Dataset")

print("COVID_PATH:", COVID_PATH)

# --- Hyperparameters & Constants ---

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20
NUM_CLASSES = 3
CLASS_NAMES = ["COVID-19", "Pneumonia", "Normal"]

np.random.seed(42)
tf.random.set_seed(42)


# --- Data Loading ---

def build_file_dataframe():
    """Scan the dataset directories and return a DataFrame of filepaths + labels."""
    filepaths = []
    labels = []

    covid_images = glob(os.path.join(COVID_PATH, "COVID", "images", "*.png"))
    filepaths.extend(covid_images)
    labels.extend(["COVID-19"] * len(covid_images))

    pneumonia_images = glob(os.path.join(COVID_PATH, "Viral Pneumonia", "images", "*.png"))
    filepaths.extend(pneumonia_images)
    labels.extend(["Pneumonia"] * len(pneumonia_images))

    normal_images = glob(os.path.join(COVID_PATH, "Normal", "images", "*.png"))
    max_normals = len(covid_images) + len(pneumonia_images)
    normal_images = normal_images[: max_normals]
    filepaths.extend(normal_images)
    labels.extend(["Normal"] * len(normal_images))

    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print("Total images:", len(df))
    print(df["label"].value_counts())
    return df


# --- Train / Val / Test Splitting ---

def split_dataframe(df):
    """Stratified split into ~70% train, ~15% val, ~15% test."""
    label_map = {"COVID-19": 0, "Pneumonia": 1, "Normal": 2}
    df["label_id"] = df["label"].map(label_map)

    from sklearn.model_selection import train_test_split

    df_temp, df_test = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label_id"],
        random_state=42,
    )

    df_train, df_val = train_test_split(
        df_temp,
        test_size=0.176,
        stratify=df_temp["label_id"],
        random_state=42,
    )

    print("\nSplit sizes:")
    print("Train:", len(df_train), "Val:", len(df_val), "Test:", len(df_test))
    return df_train, df_val, df_test


# --- Data Generators (with augmentation) ---

def create_generators(df_train, df_val, df_test):
    """Create Keras ImageDataGenerators for train (augmented), val, and test."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess
    )

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_gen = test_datagen.flow_from_dataframe(
        df_val,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_dataframe(
        df_test,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# --- Transformer Encoder Block ---

def transformer_encoder(x, num_heads, key_dim, ff_dim, dropout=0.1):
    """Single Transformer encoder block: LayerNorm -> MHSA -> Add -> LayerNorm -> FFN -> Add."""
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(x_norm, x_norm)
    x = layers.Add()([x, attn_output])

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    ff_output = layers.Dense(ff_dim, activation="relu")(x_norm)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    x = layers.Add()([x, ff_output])
    return x


# --- Hybrid CNN-Transformer Model ---

def build_cnn_transformer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=3):
    """
    Build the hybrid model:
      ResNet50 (fine-tuned) -> 1x1 Conv projection -> Positional Embedding
      -> 2× Transformer Encoder -> GlobalAvgPool -> Dense head.
    """
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
    )

    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)      

    embed_dim = 256
    x = layers.Conv2D(embed_dim, kernel_size=1, padding="same")(x)  

    H = x.shape[1]
    W = x.shape[2]
    x = layers.Reshape((H * W, embed_dim))(x) 

    num_tokens = H * W
    pos_indices = tf.range(start=0, limit=num_tokens, delta=1)
    pos_embed_layer = layers.Embedding(
        input_dim=num_tokens, output_dim=embed_dim, name="pos_embedding"
    )
    pos_embeds = pos_embed_layer(pos_indices)           
    pos_embeds = tf.expand_dims(pos_embeds, axis=0)     
    x = x + pos_embeds                                  

    x = transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=256, dropout=0.1)
    x = transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=256, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="CNN_Transformer_CXR")
    return model


# --- Plotting Utilities ---

def plot_history(history, tag):
    """Plot training / validation accuracy and loss curves, save to PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{tag} Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{tag} Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"training_history_{tag}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()


# --- Training & Evaluation Pipeline ---

def train_and_evaluate(model, train_gen, val_gen, test_gen, df_train, df_test, tag):
    """Compile, train, evaluate, and save the model. Returns model, history, accuracy, confusion matrix."""
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_train_ids = df_train["label"].map(label_map).values
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_ids),
        y=y_train_ids,
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"\n[{tag}] Class weights:", class_weights_dict)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    checkpoint_path = f"/kaggle/working/best_{tag}.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, tag)

    final_path = f"/kaggle/working/thoracic_{tag}_final.keras"
    model.save(final_path)
    print(f"[{tag}] Final model saved to {final_path}")
    print(f"[{tag}] Best checkpoint saved to {checkpoint_path}")

    # Test evaluation
    y_true = df_test["label"].map(label_map).values
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = y_true[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{tag}] Test Accuracy: {acc*100:.2f}%")

    print(f"\n[{tag}] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{tag}] Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        cls_acc = cm[i, i] / cm[i, :].sum()
        print(f"{cls}: {cls_acc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title(f"Confusion Matrix ({tag})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{tag}.png", dpi=200, bbox_inches="tight")
    plt.show()

    return model, history, acc, cm


# --- Main Orchestrator ---

def main(train_from_scratch=True):
    """End-to-end pipeline: data loading -> splitting -> model build -> training -> evaluation."""
    print("=" * 70)
    print(" THORACIC DISEASE ANALYSIS - CNN + Transformer HYBRID ")
    print("=" * 70)

    print("\nSTEP 1: BUILDING FILE LISTS AND SPLITS...")
    df = build_file_dataframe()
    df_train, df_val, df_test = split_dataframe(df)

    print("\nSTEP 2: CREATING GENERATORS...")
    train_gen, val_gen, test_gen = create_generators(df_train, df_val, df_test)

    print("\nSTEP 3: BUILDING CNN-TRANSFORMER MODEL...")
    model = build_cnn_transformer_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
    )
    model.summary()

    print("\nSTEP 4: TRAINING AND EVALUATION...")
    model, history, test_acc, cm = train_and_evaluate(
        model, train_gen, val_gen, test_gen, df_train, df_test, tag="cnn_transformer"
    )

    return model, history, test_acc, cm


# --- Script Entry Point ---

if __name__ == "__main__":
    model, history, test_acc, cm = main(train_from_scratch=True)
