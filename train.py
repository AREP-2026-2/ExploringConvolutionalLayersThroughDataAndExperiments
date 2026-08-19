"""
SageMaker training entry point for the CIFAR-10 CNN designed in the notebook.

This script mirrors the `build_cnn` architecture from Section 4 of
cnn_cifar10_workshop.ipynb (2 conv+pool blocks, 3x3 kernels, dense head).
It is kept separate because SageMaker's TensorFlow "script mode" runs this
file inside a training container, not the notebook itself.
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def build_cnn(filters=(32, 64), kernel_size=3, pool_size=2, dense_units=64):
    return Sequential([
        Conv2D(filters[0], kernel_size, strides=1, padding="same",
               activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size),
        Conv2D(filters[1], kernel_size, strides=1, padding="same", activation="relu"),
        MaxPooling2D(pool_size),
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dense(10, activation="linear"),
    ], name="cnn_cifar10")


def load_cifar10():
    # Prefer data staged in S3 and mounted by SageMaker into the "training"
    # channel (see export_cifar10.py + the notebook's estimator.fit(inputs=...)).
    # This is needed when the training environment has no general internet
    # access, so it can't reach the Keras/Toronto download URL directly.
    channel_dir = os.environ.get("SM_CHANNEL_TRAINING")
    if channel_dir:
        npz_candidates = glob.glob(os.path.join(channel_dir, "**", "*.npz"), recursive=True)
        if npz_candidates:
            print("Loading CIFAR-10 from staged file:", npz_candidates[0])
            data = np.load(npz_candidates[0])
            return (data["X_train"], data["y_train"]), (data["X_test"], data["y_test"])

    print("No staged data found in SM_CHANNEL_TRAINING; downloading CIFAR-10 directly.")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (X_train, y_train.reshape(-1)), (X_test, y_test.reshape(-1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kernel-size", type=int, default=3)

    # SageMaker sets this env var to the directory where the trained model
    # must be saved so it can be packaged and later deployed to an endpoint.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    return parser.parse_args()


def main():
    args = parse_args()

    tf.random.set_seed(7)

    (X_train_raw, y_train), (X_test_raw, y_test) = load_cifar10()

    X_train = X_train_raw.astype("float32") / 255.0
    X_test = X_test_raw.astype("float32") / 255.0

    model = build_cnn(kernel_size=args.kernel_size)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Final test loss:", test_loss)
    print("Final test accuracy:", test_accuracy)

    # Save in TensorFlow SavedModel format under a numbered version directory
    # ("1"), which is the layout the TensorFlow Serving container (used by
    # the SageMaker TensorFlow endpoint) expects.
    export_path = os.path.join(args.model_dir, "1")
    model.export(export_path)
    print("Model exported to:", export_path)


if __name__ == "__main__":
    main()
