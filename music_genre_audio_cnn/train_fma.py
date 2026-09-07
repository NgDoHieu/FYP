"""Train a CNN on FMA Small audio and mel spectrograms."""

from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = int(DURATION_SECONDS / CHUNK_SECONDS)
N_MELS = 128
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / 512) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
# Using the 8 true FMA Small genres
GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_mel(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
    
    chunks = []
    for i in range(CHUNKS_PER_TRACK):
        start = i * TARGET_FRAMES
        end = start + TARGET_FRAMES
        chunk = mel[:, start:end]
        if chunk.shape[1] < TARGET_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TARGET_FRAMES - chunk.shape[1])))
        chunks.append(chunk[:, :, np.newaxis])
    return np.array(chunks)


def is_readable_audio(path: str) -> bool:
    try:
        load_mel(path)
    except Exception as error:
        print(f"Skipping unreadable audio: {path} ({error})")
        return False
    return True


def collect_audio_files(audio_root: Path) -> list[tuple[str, int]]:
    collected: list[tuple[str, int]] = []
    for label, genre in enumerate(GENRES):
        genre_dir = audio_root / genre
        if not genre_dir.exists():
            continue
        for path in sorted(genre_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            path_string = str(path)
            if is_readable_audio(path_string):
                collected.append((path_string, label))
    return collected


def make_dataset(paths: list[str], labels: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels.astype(np.int32)))

    def extract(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        mels = tf.numpy_function(load_mel, [path], tf.float32)
        mels.set_shape((CHUNKS_PER_TRACK, N_MELS, TARGET_FRAMES, 1))
        chunk_labels = tf.repeat(label, CHUNKS_PER_TRACK)
        return mels, chunk_labels

    dataset = dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.cache()
    
    if training:
        dataset = dataset.shuffle(len(paths) * CHUNKS_PER_TRACK, seed=SEED, reshuffle_each_iteration=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(mel: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mel = tf.image.random_brightness(mel, max_delta=0.08)
    mel = tf.clip_by_value(mel + tf.random.normal(tf.shape(mel), stddev=0.015), 0.0, 1.0)
    def mask_time(value: tf.Tensor) -> tf.Tensor:
        time_start = tf.random.uniform((), 0, TARGET_FRAMES - 20, dtype=tf.int32)
        time_width = tf.random.uniform((), 5, 20, dtype=tf.int32)
        mask = tf.concat(
            [
                tf.ones((N_MELS, time_start, 1)),
                tf.zeros((N_MELS, time_width, 1)),
                tf.ones((N_MELS, TARGET_FRAMES - time_start - time_width, 1)),
            ],
            axis=1,
        )
        return value * mask

    mel = tf.cond(tf.random.uniform(()) < 0.5, lambda: mask_time(mel), lambda: mel)
    return mel, label


def make_model(class_count: int) -> tf.keras.Model:
    l2_reg = tf.keras.regularizers.l2(0.001)
    inputs = tf.keras.Input((N_MELS, TARGET_FRAMES, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2_reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    epochs = 40
    batch_size = 32
    set_seed()
    project_root = Path(__file__).resolve().parents[1]
    audio_root = project_root / "raw" / "fma_small_organized" / "genres"
    output_dir = Path(__file__).resolve().parent / "results_fma"
    output_dir.mkdir(exist_ok=True)

    collected = collect_audio_files(audio_root)
    if not collected:
        raise FileNotFoundError(f"No labeled audio files found in {audio_root}. Please run the organize script first.")

    paths = [path for path, _ in collected]
    labels = [label for _, label in collected]

    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.15 / 0.85, random_state=SEED, stratify=train_labels)
    train = make_dataset(train_paths, np.asarray(train_labels), batch_size, True)
    validation = make_dataset(validation_paths, np.asarray(validation_labels), batch_size, False)
    test = make_dataset(test_paths, np.asarray(test_labels), batch_size, False)

    model = make_model(len(GENRES))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(output_dir / "fma_audio_cnn.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    history = model.fit(train, validation_data=validation, epochs=epochs, callbacks=callbacks)
    test_loss, test_accuracy = model.evaluate(test, verbose=0)

    true_labels, predicted_labels = [], []
    for batch_audio, batch_labels in test:
        predictions = model.predict(batch_audio, verbose=0).argmax(axis=1)
        true_labels.extend(batch_labels.numpy())
        predicted_labels.extend(predictions)
    
    report = classification_report(true_labels, predicted_labels, target_names=GENRES, output_dict=True, zero_division=0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    
    ConfusionMatrixDisplay(confusion_matrix(true_labels, predicted_labels), display_labels=GENRES).plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    results = {
        "dataset": "FMA Small audio",
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": report,
        "track_split": {"train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)},
        "genres": GENRES,
        "augmentation": ["brightness", "Gaussian noise", "time masking", "L2 Regularization"],
        "model_file": str(output_dir / "fma_audio_cnn.keras"),
        "training_graph": str(output_dir / "training_curves.png"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"FMA Small audio test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {output_dir / 'results.json'}")

if __name__ == "__main__":
    main()
