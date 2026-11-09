"""Step 1: train the uncompressed model everything else is measured against.

Usage: python -m src.train_baseline --epochs 4
"""

import argparse
import os

from src.data import load
from src.model import build, compile_model

ARTIFACTS = "artifacts"
BASELINE_PATH = os.path.join(ARTIFACTS, "baseline.h5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load()

    model = compile_model(build())
    model.fit(x_train, y_train, epochs=args.epochs, validation_split=0.1, verbose=2)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    os.makedirs(ARTIFACTS, exist_ok=True)
    model.save(BASELINE_PATH)

    print(f"[baseline] test accuracy {accuracy:.4f}")
    print(f"[baseline] saved -> {BASELINE_PATH}")


if __name__ == "__main__":
    main()
