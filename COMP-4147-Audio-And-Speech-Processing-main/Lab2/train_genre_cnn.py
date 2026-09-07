"""Train a basic CNN genre classifier on the local GTZAN dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def make_model(input_shape: tuple[int, int, int], class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_data(
    paths: list[str], labels: list[int], validation_size: float, test_size: float
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=relative_validation_size,
        random_state=SEED,
        stratify=train_labels,
    )
    return (
        train_paths,
        validation_paths,
        test_paths,
        np.asarray(train_labels),
        np.asarray(validation_labels),
        np.asarray(test_labels),
    )


def image_dataset(
    paths: list[str], labels: np.ndarray, image_size: tuple[int, int], batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    def load_image(path: str) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(image_size)
            return np.asarray(image, dtype=np.float32)

    images = np.stack([load_image(path) for path in paths])
    dataset = tf.data.Dataset.from_tensor_slices((images, labels.astype(np.int32)))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def train_model(
    train: tf.data.Dataset,
    validation: tf.data.Dataset,
    test: tf.data.Dataset,
    input_shape: tuple[int, int, int],
    class_names: list[str],
    output_path: Path,
    epochs: int,
    train_steps: int,
) -> tuple[float, dict[str, list[float]]]:
    model = make_model(input_shape, len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
    ]
    training_history = model.fit(
        train.repeat(),
        validation_data=validation,
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
    )
    _, accuracy = model.evaluate(test, verbose=0)
    with output_path.with_suffix(".labels.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    return float(accuracy), training_history.history


def train_gtzan(root: Path, args: argparse.Namespace) -> float:
    image_root = root / "raw" / "gtzan" / "Data" / "images_original"
    paths: list[str] = []
    labels: list[int] = []
    class_names = [genre for genre in GENRES if (image_root / genre).is_dir()]
    for label, genre in enumerate(class_names):
        for path in sorted((image_root / genre).glob("*.png")):
            paths.append(str(path))
            labels.append(label)
    if not paths:
        raise FileNotFoundError(f"No GTZAN spectrogram images found in {image_root}")
    splits = split_data(paths, labels, args.validation_size, args.test_size)
    train_paths, validation_paths, test_paths, train_labels, validation_labels, test_labels = splits
    image_size = (args.image_width, args.image_height)
    train = image_dataset(train_paths, train_labels, image_size, args.batch_size, True)
    validation = image_dataset(validation_paths, validation_labels, image_size, args.batch_size, False)
    test = image_dataset(test_paths, test_labels, image_size, args.batch_size, False)
    accuracy, history = train_model(
        train,
        validation,
        test,
        (args.image_height, args.image_width, 3),
        class_names,
        args.output_dir / "gtzan_cnn.keras",
        args.epochs,
        int(np.ceil(len(train_paths) / args.batch_size)),
    )
    results = {
        "dataset": "GTZAN",
        "test_accuracy": accuracy,
        "class_names": class_names,
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "test_samples": len(test_paths),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "image_size": {"height": args.image_height, "width": args.image_width},
        "model_file": str(args.output_dir / "gtzan_cnn.keras"),
        "training_graph": str(args.output_dir / "gtzan_cnn_training.png"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "gtzan_cnn_training.png", dpi=150)
    plt.close(figure)
    with (args.output_dir / "gtzan_cnn_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    set_seed()
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = train_gtzan(root, args)
    print(f"GTZAN test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()