"""
Run this LOCALLY (where cnn_cifar10_workshop.ipynb already downloaded CIFAR-10
successfully) to export the dataset to a single compact .npz file.

Why: SageMaker Studio's network in this account has no general internet access
(confirmed via curl timeouts to google.com, storage.googleapis.com, and the
Toronto host Keras downloads from), so CIFAR-10 can't be downloaded directly
from inside Studio. Uploading this .npz to S3 and reading it from there
sidesteps that restriction entirely.

Usage:
    python export_cifar10.py

Produces:
    cifar10_data.npz  (~170 MB, same folder as this script)
"""

import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

np.savez_compressed(
    "cifar10_data.npz",
    X_train=X_train,
    y_train=y_train.reshape(-1),
    X_test=X_test,
    y_test=y_test.reshape(-1),
)

print("Wrote cifar10_data.npz")
print("X_train:", X_train.shape, " X_test:", X_test.shape)
