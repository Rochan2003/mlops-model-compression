"""Loads Fashion-MNIST and the small sample used for int8 calibration."""

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]
SEED = 42


def load():
    """Return (x_train, y_train), (x_test, y_test), scaled to 0-1."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = (x_train / 255.0).astype("float32")
    x_test = (x_test / 255.0).astype("float32")

    return (x_train, y_train), (x_test, y_test)


def representative_dataset(x_train, samples=200):
    """Yields examples so full int8 conversion can pick activation ranges.

    Converting weights to int8 only needs the weights, but converting the
    activations as well means the converter has to see real inputs to learn
    what range each layer actually produces.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x_train), size=min(samples, len(x_train)), replace=False)

    def generator():
        for i in idx:
            yield [x_train[i].reshape(1, 28, 28).astype("float32")]

    return generator
