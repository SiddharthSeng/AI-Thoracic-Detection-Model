"""System-Level Hybrid Ensemble for Thoracic Disease Detection.

This module implements a system-level hybrid ensemble approach for classifying
chest X-ray images into three categories: COVID-19, Pneumonia, and Normal.

Two CNN backbones — ResNet50 and DenseNet121 — are trained independently on the
same data splits (with backbone-specific preprocessing). At inference time, the
predicted probability vectors from both models are averaged element-wise, and the
class with the highest average probability is selected as the final ensemble
prediction. This probability-averaging strategy provides a simple yet effective
way to combine complementary feature representations learned by each architecture.

Note: This script was originally developed and executed as a Kaggle notebook.
Kaggle-specific paths (e.g. /kaggle/input/*, /kaggle/working/*) are retained
as-is from the original notebook.
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
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

warnings.filterwarnings("ignore")

# --- Environment & GPU Setup ---

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

# --- Data Paths ---

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
    """Scan the dataset directories and return a DataFrame of filepaths and labels."""
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


# --- Train / Validation / Test Split ---

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


# --- Data Generators ---

def create_generators(df_train, df_val, df_test, preprocess_func):
    """Create Keras ImageDataGenerators with augmentation for training."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
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


# --- Model Architectures ---

def build_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=3):
    """Build a ResNet50-based classifier with partial fine-tuning (last 20 layers)."""
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="ResNet50_CXR")
    return model


def build_densenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=3):
    """Build a DenseNet121-based classifier with partial fine-tuning (last 20 layers)."""
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="DenseNet121_CXR")
    return model


# --- Visualization Helpers ---

def plot_history(history, tag):
    """Plot and save training/validation accuracy and loss curves."""
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


# --- Training Loop ---

def train_cnn(model, train_gen, val_gen, df_train, tag):
    """Compile and train a single CNN with class weighting and callbacks."""
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y = df_train["label"].map(label_map).values
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
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

    return model, history, checkpoint_path


# --- Ensemble Evaluation ---

def evaluate_ensemble(models, test_gens, df_test):
    """Average predicted probabilities from all models and compute ensemble metrics."""
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_true = df_test["label"].map(label_map).values

    probs_list = []
    for model, test_gen in zip(models, test_gens):
        p = model.predict(test_gen, verbose=1)
        probs_list.append(p)

    avg_probs = np.mean(probs_list, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    y_true = y_true[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    print("\nEnsemble Test Accuracy: {:.2f}%".format(acc * 100))

    print("\nEnsemble Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nEnsemble per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        cls_acc = cm[i, i] / cm[i, :].sum()
        print(f"{cls}: {cls_acc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix (ResNet50 + DenseNet121 Ensemble)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_ensemble_resnet_densenet.png", dpi=200, bbox_inches="tight")
    plt.show()

    return acc, cm


# --- Main Pipeline ---

def main(train_from_scratch=True):
    """End-to-end pipeline: data prep → train both CNNs → ensemble evaluation."""
    print("=" * 70)
    print(" THORACIC DISEASE ANALYSIS - CNN ENSEMBLE (ResNet50 + DenseNet121) ")
    print("=" * 70)

    print("\nSTEP 1: BUILDING FILE LISTS AND SPLITS...")
    df = build_file_dataframe()
    df_train, df_val, df_test = split_dataframe(df)

    print("\nSTEP 2: CREATING GENERATORS...")
    res_train_gen, res_val_gen, res_test_gen = create_generators(
        df_train, df_val, df_test, preprocess_func=resnet_preprocess
    )
    den_train_gen, den_val_gen, den_test_gen = create_generators(
        df_train, df_val, df_test, preprocess_func=densenet_preprocess
    )

    if train_from_scratch:
        print("\nSTEP 3: BUILDING AND TRAINING ResNet50...")
        resnet_model = build_resnet_model(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=NUM_CLASSES,
        )
        resnet_model.summary()
        resnet_model, res_hist, res_ckpt = train_cnn(
            resnet_model, res_train_gen, res_val_gen, df_train, tag="resnet50"
        )

        print("\nSTEP 4: BUILDING AND TRAINING DenseNet121...")
        densenet_model = build_densenet_model(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=NUM_CLASSES,
        )
        densenet_model.summary()
        densenet_model, den_hist, den_ckpt = train_cnn(
            densenet_model, den_train_gen, den_val_gen, df_train, tag="densenet121"
        )
    else:
        print("\nLOADING SAVED MODELS...")
        resnet_model = keras.models.load_model("/kaggle/working/thoracic_resnet50_final.keras")
        densenet_model = keras.models.load_model("/kaggle/working/thoracic_densenet121_final.keras")

    print("\nSTEP 5: ENSEMBLE EVALUATION...")
    ensemble_acc, cm = evaluate_ensemble(
        [resnet_model, densenet_model],
        [res_test_gen, den_test_gen],
        df_test,
    )

    return resnet_model, densenet_model, ensemble_acc, cm


# --- Entry Point ---

if __name__ == "__main__":
    resnet_model, densenet_model, ensemble_acc, cm = main(train_from_scratch=True)
