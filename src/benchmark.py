"""Step 3: measure every variant on size, accuracy and latency.

Size is reported twice. Raw file size is what you store; gzipped size is what
you ship over a wire. They matter separately because pruning barely changes the
raw file, since a zero still occupies its slot, but makes it compress far
better.

Usage: python -m src.benchmark --latency-runs 300
"""

import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
import tensorflow as tf

from src.data import load
from src.train_baseline import ARTIFACTS

RESULTS_PATH = "evidence/results.json"

# Printed in this order so the table reads from least to most compressed.
ORDER = [
    "tflite_float32", "control_float32", "tflite_float16", "tflite_dynamic",
    "tflite_int8", "qat_int8", "pruned", "pruned_dynamic",
]


def gzipped_size(path):
    with open(path, "rb") as fh:
        return len(gzip.compress(fh.read(), compresslevel=9))


def evaluate(interpreter, x_test, y_test):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    correct = 0
    for image, label in zip(x_test, y_test):
        interpreter.set_tensor(input_index, image[np.newaxis, ...].astype("float32"))
        interpreter.invoke()
        if int(np.argmax(interpreter.get_tensor(output_index)[0])) == int(label):
            correct += 1
    return correct / len(x_test)


def measure_latency(interpreter, sample, runs):
    """Median single image inference time in milliseconds.

    Median rather than mean, because the first few calls and the occasional
    scheduler hiccup produce outliers that a mean would absorb.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    sample = sample[np.newaxis, ...].astype("float32")

    for _ in range(20):            # warm up
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        timings.append((time.perf_counter() - start) * 1000)

    return float(np.median(timings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-runs", type=int, default=300)
    parser.add_argument("--eval-samples", type=int, default=2000,
                        help="test images used for accuracy; the full set is slow "
                             "because TFLite runs one image at a time")
    args = parser.parse_args()

    _, (x_test, y_test) = load()
    x_eval, y_eval = x_test[:args.eval_samples], y_test[:args.eval_samples]

    found = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(ARTIFACTS, "*.tflite"))}
    names = [n for n in ORDER if n in found] + sorted(set(found) - set(ORDER))

    results = {}
    print(f"{'variant':<17}{'KB':>9}{'gzip KB':>10}{'accuracy':>10}{'ms':>8}")
    for name in names:
        path = found[name]
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        entry = {
            "size_kb": round(os.path.getsize(path) / 1024, 1),
            "gzip_kb": round(gzipped_size(path) / 1024, 1),
            "accuracy": round(evaluate(interpreter, x_eval, y_eval), 4),
            "latency_ms": round(measure_latency(interpreter, x_test[0], args.latency_runs), 4),
        }
        results[name] = entry
        print(f"{name:<17}{entry['size_kb']:>9}{entry['gzip_kb']:>10}"
              f"{entry['accuracy']:>10}{entry['latency_ms']:>8.3f}")

    baseline = results.get("tflite_float32")
    if baseline:
        for name, entry in results.items():
            entry["size_vs_baseline"] = round(entry["size_kb"] / baseline["size_kb"], 3)
            entry["gzip_vs_baseline"] = round(entry["gzip_kb"] / baseline["gzip_kb"], 3)
            entry["accuracy_drop"] = round(baseline["accuracy"] - entry["accuracy"], 4)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump({"eval_samples": int(args.eval_samples),
                   "latency_runs": int(args.latency_runs),
                   "variants": results}, fh, indent=2)
    print(f"\n[benchmark] wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
