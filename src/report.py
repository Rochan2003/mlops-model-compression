"""Step 4: draw the size against accuracy tradeoff from the benchmark results.

Usage: python -m src.report
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_PATH = "evidence/results.json"
OUT_PATH = "evidence/size-vs-accuracy.png"

INK, MUTED = "#2C2C2A", "#5F5E5A"

# Grouped by what was done to the model, not by rank.
FAMILY = {
    "tflite_float32": ("uncompressed", "#7F77DD"),
    "control_float32": ("uncompressed", "#7F77DD"),
    "tflite_float16": ("quantized", "#1D9E75"),
    "tflite_dynamic": ("quantized", "#1D9E75"),
    "tflite_int8": ("quantized", "#1D9E75"),
    "qat_int8": ("quantized", "#1D9E75"),
    "pruned": ("pruned", "#D85A30"),
    "pruned_dynamic": ("pruned", "#D85A30"),
}
LABEL_OFFSET = {
    "tflite_float32": (-10, 6), "control_float32": (-10, -16),
    "tflite_float16": (-10, 10), "tflite_dynamic": (8, -4),
    "tflite_int8": (8, -14), "qat_int8": (10, -4),
    "pruned": (10, 2), "pruned_dynamic": (8, 8),
}
# labels placed to the left of their point need right alignment
LABEL_ALIGN = {"tflite_float32": "right", "control_float32": "right",
               "tflite_float16": "right"}


def main():
    results = json.load(open(RESULTS_PATH))["variants"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    seen = set()

    for name, entry in results.items():
        family, colour = FAMILY.get(name, ("other", MUTED))
        label = family if family not in seen else None
        seen.add(family)

        ax.scatter(entry["gzip_kb"], entry["accuracy"], s=90, color=colour,
                   label=label, zorder=3, edgecolors="white", linewidths=1.5)
        dx, dy = LABEL_OFFSET.get(name, (8, 4))
        ax.annotate(name, (entry["gzip_kb"], entry["accuracy"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=LABEL_ALIGN.get(name, "left"), fontsize=8, color=MUTED)

    control = results.get("control_float32")
    if control:
        ax.axhline(control["accuracy"], color=MUTED, linewidth=0.8,
                   linestyle="--", zorder=1)
        ax.annotate("fairly trained baseline", (ax.get_xlim()[0], control["accuracy"]),
                    textcoords="offset points", xytext=(8, 6), ha="left",
                    fontsize=8, color=MUTED)

    ax.set_xlabel("gzipped size (KB), smaller is better", color=MUTED, fontsize=9)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    ax.set_title("What each compression method costs", color=INK, fontsize=12, pad=14)

    ax.grid(color="#D3D1C7", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B4B2A9")
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=8, loc="center left")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[report] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
