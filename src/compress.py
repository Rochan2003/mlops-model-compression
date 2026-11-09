"""Step 2: build every compressed variant of the baseline model.

Seven variants, so the size and accuracy tradeoffs can be compared against each
other rather than against the baseline alone:

    tflite_float32    straight conversion, no compression
    tflite_dynamic    weights to int8, activations stay float
    tflite_float16    weights to 16 bit float
    tflite_int8       weights and activations to int8, needs calibration data
    qat_int8          trained while simulating int8, then converted
    pruned            75 percent of weights forced to zero
    pruned_dynamic    pruned, then quantized as well

Usage: python -m src.compress --epochs 2
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from src.data import load, representative_dataset
from src.model import compile_model
from src.train_baseline import ARTIFACTS, BASELINE_PATH

FINAL_SPARSITY = 0.75
BATCH_SIZE = 128


def _write(name, tflite_bytes):
    path = os.path.join(ARTIFACTS, f"{name}.tflite")
    with open(path, "wb") as fh:
        fh.write(tflite_bytes)
    print(f"[compress] {name:16} {len(tflite_bytes) / 1024:8.1f} KB -> {path}")
    return path


def convert_plain(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_dynamic(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_float16(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(model, x_train):
    """Full integer quantization.

    Input and output stay float32 so every variant can be benchmarked the same
    way. The weights and activations inside the graph are still int8.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()


def train_control(baseline, x_train, y_train, epochs):
    """The baseline given the same extra epochs as QAT and pruning.

    Without this the comparison is unfair: QAT and pruning each fine tune for
    `epochs` more, so any accuracy they gain is partly just more training. This
    is what they should actually be measured against.
    """
    control = compile_model(baseline)
    control.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs,
                validation_split=0.1, verbose=2)
    return control


def train_qat(baseline, x_train, y_train, epochs):
    """Quantization aware training.

    Inserts fake quantize operations so the model learns weights that survive
    being rounded to int8, instead of being rounded after the fact.
    """
    qat_model = compile_model(tfmot.quantization.keras.quantize_model(baseline))
    qat_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs,
                  validation_split=0.1, verbose=2)
    return qat_model


def train_pruned(baseline, x_train, y_train, epochs):
    """Magnitude pruning, ramped up over training.

    Sparsity climbs from 0 to FINAL_SPARSITY so the model can redistribute
    weight as connections are removed, rather than losing them all at once.
    """
    steps_per_epoch = int(np.ceil(len(x_train) * 0.9 / BATCH_SIZE))
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=FINAL_SPARSITY,
        begin_step=0,
        end_step=steps_per_epoch * epochs,
    )

    pruned = compile_model(
        tfmot.sparsity.keras.prune_low_magnitude(baseline, pruning_schedule=schedule)
    )
    pruned.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs,
               validation_split=0.1, verbose=2,
               callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

    # Removes the pruning wrappers and leaves a plain model full of zeros.
    return tfmot.sparsity.keras.strip_pruning(pruned)


def sparsity_of(model):
    """Share of weights that are exactly zero, which is what compresses."""
    zeros = total = 0
    for weights in model.get_weights():
        zeros += int(np.sum(weights == 0))
        total += int(weights.size)
    return zeros / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2,
                        help="epochs for the QAT and pruning fine tunes")
    args = parser.parse_args()

    (x_train, y_train), _ = load()
    baseline = tf.keras.models.load_model(BASELINE_PATH)

    _write("tflite_float32", convert_plain(baseline))
    _write("tflite_dynamic", convert_dynamic(baseline))
    _write("tflite_float16", convert_float16(baseline))
    _write("tflite_int8", convert_int8(baseline, x_train))

    print("\n[compress] control: baseline with the same extra epochs")
    control = train_control(tf.keras.models.load_model(BASELINE_PATH),
                            x_train, y_train, args.epochs)
    _write("control_float32", convert_plain(control))

    print("\n[compress] quantization aware training")
    qat_model = train_qat(tf.keras.models.load_model(BASELINE_PATH),
                          x_train, y_train, args.epochs)
    _write("qat_int8", convert_dynamic(qat_model))

    print("\n[compress] pruning")
    pruned = train_pruned(tf.keras.models.load_model(BASELINE_PATH),
                          x_train, y_train, args.epochs)
    print(f"[compress] weights that are zero after pruning: {sparsity_of(pruned):.1%}")
    pruned.save(os.path.join(ARTIFACTS, "pruned.h5"))

    _write("pruned", convert_plain(pruned))
    _write("pruned_dynamic", convert_dynamic(pruned))

    print("\n[compress] done")


if __name__ == "__main__":
    main()
