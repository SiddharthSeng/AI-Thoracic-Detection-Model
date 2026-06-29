"""
CNN (ResNet50) component for thoracic disease classification.

This script trains a ResNet50-based convolutional neural network to classify
chest X-ray images into three categories: COVID-19, Pneumonia, and Normal.
The model is trained from scratch (no ImageNet weights) using mixed-precision
training when a GPU is available.

Originally developed as a Kaggle notebook and converted to a standalone script.
Kaggle-specific paths (e.g. /kaggle/input/*, /kaggle/working/*) are preserved
from the original notebook environment.
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

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
        print("Could not set memory growth:", e)

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
else:
    print("No GPU detected, running on CPU.")

# --- Dataset Paths ---

BASE_ROOT    = "/kaggle/input/modeltestcnn/Throacic Detection using AI"
CONTENT_ROOT = os.path.join(BASE_ROOT, "content")
COVID_PATH   = os.path.join(CONTENT_ROOT, "COVID-19_Radiography_Dataset")
PNEUMONIA_PATH = os.path.join(CONTENT_ROOT, "chest_xray")

print("COVID_PATH:", COVID_PATH)
print("PNEUMONIA_PATH:", PNEUMONIA_PATH)

# --- Hyperparameters & Constants ---

IMG_SIZE   = 224
BATCH_SIZE = 16          
EPOCHS     = 20          
NUM_CLASSES = 3
CLASS_NAMES = ["COVID-19", "Pneumonia", "Normal"]

# --- Reproducibility Seeds ---

np.random.seed(42)
tf.random.set_seed(42)


# --- Data Preparation ---

def build_file_dataframe():
    """Scan dataset directories and return a DataFrame of filepaths and labels."""
    filepaths = []
    labels = []

    covid_images = glob(os.path.join(COVID_PATH, "COVID", "images", "*.png"))
    filepaths.extend(covid_images)
    labels.extend(["COVID-19"] * len(covid_images))

    pneumonia_images_1 = glob(os.path.join(COVID_PATH, "Viral Pneumonia", "images", "*.png"))
    pneumonia_train = glob(os.path.join(PNEUMONIA_PATH, "train", "PNEUMONIA", "*.jpeg"))
    pneumonia_test  = glob(os.path.join(PNEUMONIA_PATH, "test",  "PNEUMONIA", "*.jpeg"))
    pneumonia_images = pneumonia_images_1 + pneumonia_train + pneumonia_test
    filepaths.extend(pneumonia_images)
    labels.extend(["Pneumonia"] * len(pneumonia_images))

    normal_images_1 = glob(os.path.join(COVID_PATH, "Normal", "images", "*.png"))
    normal_train    = glob(os.path.join(PNEUMONIA_PATH, "train", "NORMAL", "*.jpeg"))
    normal_test     = glob(os.path.join(PNEUMONIA_PATH, "test",  "NORMAL", "*.jpeg"))
    normal_images = normal_images_1 + normal_train + normal_test

    max_normals = len(covid_images) + len(pneumonia_images)
    normal_images = normal_images[: max_normals]

    filepaths.extend(normal_images)
    labels.extend(["Normal"] * len(normal_images))

    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print("Total images:", len(df))
    print(df["label"].value_counts())
    return df


def split_dataframe(df):
    """Split the DataFrame into train / validation / test sets (stratified)."""
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

def create_generators(df_train, df_val, df_test):
    """Create Keras ImageDataGenerators with augmentation for training."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
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
        class_mode="categorical",
        classes=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# --- Model Architecture ---

def build_resnet_model(input_shape=(224, 224, 3), num_classes=3):
    """Build a ResNet50 classifier (trained from scratch, no ImageNet weights)."""
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg",
    )

    for layer in base_model.layers:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    return model


# --- Training ---

def train_model(model, train_gen, val_gen, epochs=20):
    """Compile and train the model with class-balanced weights and callbacks."""
    labels = train_gen.classes
    from sklearn.utils.class_weight import compute_class_weight

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print("\nClass weights:", class_weights_dict)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    checkpoint_path = "/kaggle/working/best_model.keras"

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
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    return history, checkpoint_path


# --- Visualization ---

def plot_training_history(history):
    """Plot accuracy and loss curves for training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=CLASS_NAMES):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.show()
    return cm


# --- Evaluation ---

def evaluate_model(model, test_gen):
    """Run inference on the test set and print classification metrics."""
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    print("\nTest Accuracy: {:.2f}%".format(acc * 100))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        cls_acc = cm[i, i] / cm[i, :].sum()
        print(f"{cls}: {cls_acc * 100:.2f}%")

    return acc


# --- Main Pipeline ---

def main():
    """End-to-end pipeline: data loading, training, evaluation, and saving."""
    print("=" * 70)
    print(" THORACIC DISEASE ANALYSIS - ResNet50 (OFFLINE, NO IMAGENET WEIGHTS) ")
    print("=" * 70)

    print("\nSTEP 1: BUILDING FILE LISTS AND SPLITS...")
    df = build_file_dataframe()
    df_train, df_val, df_test = split_dataframe(df)

    print("\nSTEP 2: CREATING GENERATORS...")
    train_gen, val_gen, test_gen = create_generators(df_train, df_val, df_test)

    print("\nSTEP 3: BUILDING MODEL...")
    model = build_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.summary()

    print("\nSTEP 4: TRAINING MODEL...")
    history, ckpt_path = train_model(model, train_gen, val_gen, epochs=EPOCHS)

    print("\nSTEP 5: PLOTTING TRAINING HISTORY...")
    plot_training_history(history)

    print("\nSTEP 6: EVALUATING MODEL...")
    test_acc = evaluate_model(model, test_gen)

    print("\nSTEP 7: SAVING FINAL MODEL...")
    final_path = "/kaggle/working/thoracic_disease_resnet50_final_offline.keras"
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    print(f"Best checkpoint saved to {ckpt_path}")

    return model, history, test_acc


# --- Entry Point ---

if __name__ == "__main__":
    model, history, test_acc = main()
