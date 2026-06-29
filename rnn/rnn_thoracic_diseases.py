# Originally developed as a Kaggle notebook and converted to a standalone script.
"""
RNN (Bidirectional LSTM) model for thoracic disease classification.

This module classifies chest X-ray images into three categories —
COVID-19, Pneumonia, and Normal — using a pure recurrent neural network
built with bidirectional LSTM layers.

**Image-to-sequence conversion:**  Each image is resized to 224×224 RGB and
then reshaped into a 2-D sequence of shape (224, 672), where each of the
224 pixel rows becomes a timestep and the 672-element feature vector holds
the flattened RGB values of that row (224 pixels × 3 channels).  This lets
a standard RNN process spatial information row-by-row without convolutions.
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

warnings.filterwarnings("ignore")

# --- Environment / GPU Setup ---

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

# --- Dataset Paths ---
# NOTE: These paths point to the original Kaggle input directories.
# Update them to match your local or cloud storage layout.

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

# --- Reproducibility ---

np.random.seed(42)
tf.random.set_seed(42)

# --- Data Loading ---


def build_file_dataframe():
    """Scan dataset directories and return a DataFrame of file paths and labels."""
    filepaths = []
    labels = []

    # COVID-19
    covid_images = glob(os.path.join(COVID_PATH, "COVID", "images", "*.png"))
    filepaths.extend(covid_images)
    labels.extend(["COVID-19"] * len(covid_images))

    # Pneumonia
    pneumonia_images_1 = glob(os.path.join(COVID_PATH, "Viral Pneumonia", "images", "*.png"))
    pneumonia_train = glob(os.path.join(PNEUMONIA_PATH, "train", "PNEUMONIA", "*.jpeg"))
    pneumonia_test  = glob(os.path.join(PNEUMONIA_PATH, "test",  "PNEUMONIA", "*.jpeg"))
    pneumonia_images = pneumonia_images_1 + pneumonia_train + pneumonia_test
    filepaths.extend(pneumonia_images)
    labels.extend(["Pneumonia"] * len(pneumonia_images))

    # Normal
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


# --- Train / Val / Test Split ---


def split_dataframe(df):
    """Stratified split into ~70 % train, ~15 % val, ~15 % test."""
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


# --- RNN Sequence Generator ---


class RNNImageSequence(keras.utils.Sequence):
    """Keras Sequence that reshapes each image into (timesteps, features) for RNN input."""

    def __init__(self, df, batch_size, img_size, class_names, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.df.loc[batch_indexes, "filepath"].values
        batch_labels = self.df.loc[batch_indexes, "label"].values

        timesteps = self.img_size
        feature_dim = self.img_size * 3

        batch_x = np.zeros((len(batch_indexes), timesteps, feature_dim), dtype="float32")
        batch_y = np.zeros((len(batch_indexes), self.num_classes), dtype="float32")

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            img_raw = tf.keras.utils.load_img(
                path,
                target_size=(self.img_size, self.img_size),
                color_mode="rgb",
            )
            img = tf.keras.utils.img_to_array(img_raw)
            img = img / 255.0
            img_seq = np.reshape(img, (timesteps, feature_dim))
            batch_x[i] = img_seq

            label_idx = self.class_names.index(label)
            batch_y[i, label_idx] = 1.0

        return batch_x, batch_y


def create_rnn_generators(df_train, df_val, df_test):
    """Instantiate train / val / test RNNImageSequence generators."""
    train_gen = RNNImageSequence(df_train, BATCH_SIZE, IMG_SIZE, CLASS_NAMES, shuffle=True)
    val_gen   = RNNImageSequence(df_val,   BATCH_SIZE, IMG_SIZE, CLASS_NAMES, shuffle=False)
    test_gen  = RNNImageSequence(df_test,  BATCH_SIZE, IMG_SIZE, CLASS_NAMES, shuffle=False)
    return train_gen, val_gen, test_gen


# --- Model Architecture ---


def build_pure_rnn_model(timesteps=IMG_SIZE, feature_dim=IMG_SIZE * 3, num_classes=3):
    """Build a Bidirectional LSTM model for chest X-ray classification."""
    inputs = layers.Input(shape=(timesteps, feature_dim))

    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3)
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=False, dropout=0.3)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="Pure_RNN_CXR_224")
    return model


# --- Training ---


def train_model(model, train_gen, val_gen, df_train, epochs=EPOCHS):
    """Compile and train the model with class-balanced weights and callbacks."""
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y = df_train["label"].map(label_map).values
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print("\nClass weights:", class_weights_dict)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )

    # NOTE: Checkpoint path points to the original Kaggle working directory.
    checkpoint_path = "/kaggle/working/best_pure_rnn_model_224.keras"

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


# --- Evaluation ---


def evaluate_model_fast(model, test_gen):
    """Run predictions on the test set and print metrics / confusion matrix."""
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_true = test_gen.df["label"].map(label_map).values

    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = y_true[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    print("\nTest Accuracy: {:.2f}%".format(acc * 100))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        cls_acc = cm[i, i] / cm[i, :].sum()
        print(f"{cls}: {cls_acc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix (Pure RNN 224)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_pure_rnn_224.png", dpi=200, bbox_inches="tight")
    plt.show()

    return acc, cm


# --- Visualisation ---


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
    plt.savefig("training_history_pure_rnn_224.png", dpi=200, bbox_inches="tight")
    plt.show()


# --- Main Orchestration ---


def main(train_from_scratch=True):
    """End-to-end pipeline: data → model → train → evaluate."""
    print("=" * 70)
    print(" THORACIC DISEASE ANALYSIS - PURE RNN (LSTM, 224x224) ")
    print("=" * 70)

    print("\nSTEP 1: BUILDING FILE LISTS AND SPLITS...")
    df = build_file_dataframe()
    df_train, df_val, df_test = split_dataframe(df)

    print("\nSTEP 2: CREATING RNN GENERATORS...")
    train_gen, val_gen, test_gen = create_rnn_generators(df_train, df_val, df_test)

    if train_from_scratch:
        print("\nSTEP 3: BUILDING PURE RNN MODEL...")
        model = build_pure_rnn_model(
            timesteps=IMG_SIZE,
            feature_dim=IMG_SIZE * 3,
            num_classes=NUM_CLASSES
        )
        model.summary()

        print("\nSTEP 4: TRAINING MODEL...")
        history, ckpt_path = train_model(
            model, train_gen, val_gen, df_train=df_train, epochs=EPOCHS
        )

        print("\nSTEP 5: PLOTTING TRAINING HISTORY...")
        plot_training_history(history)

        print("\nSTEP 6: EVALUATING MODEL (FAST)...")
        test_acc, cm = evaluate_model_fast(model, test_gen)

        print("\nSTEP 7: SAVING FINAL MODEL...")
        # NOTE: Save path points to the original Kaggle working directory.
        final_path = "/kaggle/working/thoracic_disease_pure_rnn_224_final.keras"
        model.save(final_path)
        print(f"Final model saved to {final_path}")
        print(f"Best checkpoint saved to {ckpt_path}")
    else:
        print("\nLOADING SAVED MODEL AND ONLY EVALUATING...")
        model = keras.models.load_model(
            "/kaggle/working/thoracic_disease_pure_rnn_224_final.keras"
        )
        test_acc, cm = evaluate_model_fast(model, test_gen)
        history = None
        ckpt_path = None

    return model, history, test_acc, cm


# --- Entry Point (originally Cell 2 of the Kaggle notebook) ---

if __name__ == "__main__":
    model, history, test_acc, cm = main(train_from_scratch=True)
