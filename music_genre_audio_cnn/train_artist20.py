"""Train a simple mel-spectrogram CNN on an Artist20-style directory dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

SEED = 42
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
CHUNK_SECONDS = 3
CHUNKS_PER_TRACK = DURATION_SECONDS // CHUNK_SECONDS
N_MELS = 128
TARGET_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / 512) + 1
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


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
    for chunk_index in range(CHUNKS_PER_TRACK):
        start = chunk_index * TARGET_FRAMES
        chunk = mel[:, start : start + TARGET_FRAMES]
        if chunk.shape[1] < TARGET_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, TARGET_FRAMES - chunk.shape[1])))
        chunks.append(chunk[:, :, np.newaxis])
    return np.asarray(chunks, dtype=np.float32)


def collect_audio_files(audio_root: Path) -> tuple[list[str], list[int], list[str]]:
    artist_dirs = sorted(path for path in audio_root.iterdir() if path.is_dir())
    artists = [path.name for path in artist_dirs]
    paths: list[str] = []
    labels: list[int] = []
    for label, artist_dir in enumerate(artist_dirs):
        for path in sorted(artist_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                paths.append(str(path))
                labels.append(label)
    return paths, labels, artists


def make_dataset(paths: list[str], labels: list[int], batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, np.asarray(labels, dtype=np.int32)))

    def extract(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        mel = tf.numpy_function(load_mel, [path], tf.float32)
        mel.set_shape((CHUNKS_PER_TRACK, N_MELS, TARGET_FRAMES, 1))
        return mel, tf.repeat(label, CHUNKS_PER_TRACK)

    dataset = dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    if training:
        dataset = dataset.shuffle(len(paths) * CHUNKS_PER_TRACK, seed=SEED, reshuffle_each_iteration=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(mel: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mel = tf.image.random_brightness(mel, max_delta=0.08)
    mel = tf.clip_by_value(mel + tf.random.normal(tf.shape(mel), stddev=0.015), 0.0, 1.0)
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
    return tf.cond(tf.random.uniform(()) < 0.5, lambda: mel * mask, lambda: mel), label


def make_model(class_count: int) -> tf.keras.Model:
    l2_regularizer = tf.keras.regularizers.l2(0.001)
    inputs = tf.keras.Input((N_MELS, TARGET_FRAMES, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2_regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_regularizer)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_tracks(model: tf.keras.Model, paths: list[str], labels: list[int]) -> tuple[float, float, list[int]]:
    probabilities = []
    for path in paths:
        chunks = load_mel(path)
        probabilities.append(model.predict(chunks, verbose=0).mean(axis=0))
    probability_array = np.asarray(probabilities, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    predictions = np.argmax(probability_array, axis=1)
    loss = tf.keras.losses.sparse_categorical_crossentropy(label_array, probability_array).numpy().mean()
    accuracy = float(np.mean(predictions == label_array))
    return float(loss), accuracy, predictions.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Artist20 audio classifier.") 
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "artist20" / "artist20" / "mp3s-32k",
        help="Directory containing one subdirectory per artist.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Artist20 directory not found: {args.data_dir}. "
            "Expected one folder per artist containing audio files."
        )

    paths, labels, artists = collect_audio_files(args.data_dir)
    if len(artists) < 2 or not paths:
        raise ValueError("Expected audio files in at least two artist subdirectories.")
    if any(labels.count(label) < 7 for label in range(len(artists))):
        raise ValueError("Each artist needs at least seven audio files for train/validation/test splits.")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
    )
    train = make_dataset(train_paths, train_labels, args.batch_size, training=True)
    validation = make_dataset(validation_paths, validation_labels, args.batch_size, training=False)
    test = make_dataset(test_paths, test_labels, args.batch_size, training=False)

    output_dir = Path(__file__).resolve().parent / "results_artist20"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "artist20_audio_cnn.keras"
    model = make_model(len(artists))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    test_loss, test_accuracy, predicted_labels = evaluate_tracks(model, test_paths, test_labels)
    results = {
        "dataset": "Artist20",
        "artists": artists,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "evaluation": "Track-level accuracy after averaging predictions from ten three-second excerpts.",
        "augmentation": ["brightness", "Gaussian noise", "time masking"],
        "classification_report": classification_report(
            test_labels, predicted_labels, target_names=artists, output_dict=True, zero_division=0
        ),
        "track_split": {
            "train": len(train_paths), "validation": len(validation_paths), "test": len(test_paths)
        },
        "model_file": str(model_path),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with (output_dir / "artist_labels.json").open("w", encoding="utf-8") as file:
        json.dump(artists, file, indent=2)
    print(f"Artist20 test accuracy: {test_accuracy:.4f}")
    print(f"Saved model and results to: {output_dir}")


if __name__ == "__main__":
    main()