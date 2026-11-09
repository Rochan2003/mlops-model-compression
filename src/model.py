"""The baseline network and the training loop every variant starts from."""

import tensorflow as tf

from src.data import SEED

INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10


def build():
    """A small convolutional net, big enough that compression has an effect."""
    tf.keras.utils.set_random_seed(SEED)

    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
        tf.keras.layers.Reshape((*INPUT_SHAPE, 1)),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
