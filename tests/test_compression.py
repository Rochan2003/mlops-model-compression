"""Tests for the data, the model and the compression results."""

import gzip
import json
import os

import numpy as np
import pytest

from src import data, model
from src.compress import sparsity_of

ARTIFACTS = "artifacts"
RESULTS_PATH = "evidence/results.json"

VARIANTS = [
    "tflite_float32", "control_float32", "tflite_float16", "tflite_dynamic",
    "tflite_int8", "qat_int8", "pruned", "pruned_dynamic",
]


@pytest.fixture(scope="module")
def results():
    return json.load(open(RESULTS_PATH))["variants"]


# ------------------------------------------------------------------- data --

def test_dataset_shape():
    (x_train, y_train), (x_test, y_test) = data.load()

    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert len(data.CLASS_NAMES) == 10


def test_pixels_are_scaled():
    (x_train, _), _ = data.load()

    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0


def test_representative_dataset_yields_single_images():
    (x_train, _), _ = data.load()

    batches = list(data.representative_dataset(x_train, samples=5)())

    assert len(batches) == 5
    assert batches[0][0].shape == (1, 28, 28)


# ------------------------------------------------------------------ model --

def test_model_output_shape():
    built = model.build()

    assert built.output_shape == (None, model.NUM_CLASSES)


def test_sparsity_counts_zeros():
    class Fake:
        def get_weights(self):
            return [np.array([0.0, 0.0, 1.0, 2.0])]

    assert sparsity_of(Fake()) == 0.5


# ------------------------------------------------------------- artifacts --

@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_was_produced(name):
    assert os.path.exists(os.path.join(ARTIFACTS, f"{name}.tflite"))


@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_was_benchmarked(name, results):
    entry = results[name]

    for key in ("size_kb", "gzip_kb", "accuracy", "latency_ms"):
        assert key in entry
        assert entry[key] > 0


# ------------------------------------------------------- the actual claims --

def test_quantization_shrinks_the_model(results):
    baseline = results["tflite_float32"]["size_kb"]

    for name in ("tflite_dynamic", "tflite_int8", "tflite_float16", "qat_int8"):
        assert results[name]["size_kb"] < baseline


def test_float16_is_about_half_the_size(results):
    ratio = results["tflite_float16"]["size_kb"] / results["tflite_float32"]["size_kb"]

    assert 0.45 < ratio < 0.55


def test_pruning_does_not_shrink_the_raw_file(results):
    """Zeros still occupy their slot, so the file is the same size."""
    assert results["pruned"]["size_kb"] == results["tflite_float32"]["size_kb"]


def test_pruning_does_shrink_the_compressed_file(results):
    """Which is the whole point of pruning: runs of zeros compress."""
    assert results["pruned"]["gzip_kb"] < 0.5 * results["tflite_float32"]["gzip_kb"]


def test_pruning_and_quantization_stack(results):
    """Combined should beat either one alone, on gzipped size."""
    combined = results["pruned_dynamic"]["gzip_kb"]

    assert combined < results["pruned"]["gzip_kb"]
    assert combined < results["tflite_dynamic"]["gzip_kb"]


def test_int8_is_faster_than_float32(results):
    assert results["tflite_int8"]["latency_ms"] < results["tflite_float32"]["latency_ms"]


def test_no_variant_collapses(results):
    """Compression should cost accuracy, not destroy the model."""
    for name in VARIANTS:
        assert results[name]["accuracy"] > 0.85


def test_compression_is_measured_against_the_fair_baseline(results):
    """The control exists and was trained longer, so it should score higher
    than the short baseline. Without it the comparison flatters compression."""
    assert results["control_float32"]["accuracy"] > results["tflite_float32"]["accuracy"]


def test_gzip_helper_matches_the_recorded_size():
    path = os.path.join(ARTIFACTS, "pruned.tflite")
    with open(path, "rb") as fh:
        recomputed = len(gzip.compress(fh.read(), compresslevel=9)) / 1024

    recorded = json.load(open(RESULTS_PATH))["variants"]["pruned"]["gzip_kb"]
    assert abs(recomputed - recorded) < 1.0
