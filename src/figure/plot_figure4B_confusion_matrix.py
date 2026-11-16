#!/usr/bin/env python3
"""
Figure 4B - Confusion matrix of the multimodal model (out-of-fold predictions)
=============================================================================

Input
-----
multimodal_oof_predictions.csv
    Written by the nested 5-fold CV notebook (PLOS_Multimodal_nestedCV.ipynb).
    Required columns: y_true_code, y_pred_code, y_true, y_pred

    Each of the 622 diplotype-drug records appears exactly once, predicted by
    the model of the outer fold in which it was held out.  No model weights are
    loaded here; the figure is a pure re-plot of those stored predictions.

Output
------
figure4B_confusion_matrix.tiff   600 dpi, LZW compressed (PLOS submission)
figure4B_confusion_matrix.pdf    vector
figure4B_confusion_matrix.png    preview

Usage
-----
    python plot_figure4B_confusion_matrix.py
    python plot_figure4B_confusion_matrix.py --oof path/to/oof.csv --outdir figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

NUM_CLASSES = 3
DPI = 600

# Internal class code -> label shown in the figure.
DISPLAY_NAMES = {
    "de": "Decreased",
    "f": "Non-functional",
    "nor": "Normal",
    "Decrease": "Decreased",
    "No function": "Non-functional",
    "Normal": "Normal",
}

# Journal-friendly fonts; Arial if available, otherwise a sans-serif fallback.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable fonts
matplotlib.rcParams["ps.fonttype"] = 42


def load_oof(path):
    """Load out-of-fold predictions and recover the class order from the codes."""
    df = pd.read_csv(path)
    for col in ("y_true_code", "y_pred_code", "y_true", "y_pred"):
        if col not in df.columns:
            raise KeyError(f"'{col}' column missing in {path}")

    df["y_true"] = df["y_true"].astype(str).str.strip()
    df["y_pred"] = df["y_pred"].astype(str).str.strip()

    # code -> class string, ordered by the integer code used during training
    code_to_name = (
        df[["y_true_code", "y_true"]]
        .drop_duplicates()
        .set_index("y_true_code")["y_true"]
        .to_dict()
    )
    class_names = [code_to_name[c] for c in sorted(code_to_name)]
    display_labels = [DISPLAY_NAMES.get(c, c) for c in class_names]

    y_true = df["y_true_code"].to_numpy()
    y_pred = df["y_pred_code"].to_numpy()
    return y_true, y_pred, display_labels


def plot_confusion_matrix(y_true, y_pred, display_labels, out_basename, dpi=DPI):
    """Row-normalized confusion matrix: counts in bold, recall % underneath."""
    n = len(display_labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
    )

    # Single-hue sequential colormap (white -> dark blue).
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#ffffff", "#9ecae1", "#3182bd", "#08306b"]
    )

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_yticklabels(display_labels, fontsize=8, rotation=90, va="center")
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Text colour flips on dark cells so the annotation stays legible.
    for i in range(n):
        for j in range(n):
            frac = cm_norm[i, j]
            txt_color = "white" if frac > 0.5 else "#222222"
            ax.text(
                j, i - 0.12, f"{cm[i, j]:d}",
                ha="center", va="center",
                fontsize=11, color=txt_color, fontweight="bold",
            )
            if row_sums[i, 0] > 0:
                ax.text(
                    j, i + 0.22, f"{frac * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=7, color=txt_color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within true class", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(
        f"{out_basename}.tiff", dpi=dpi,
        pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight",
    )
    fig.savefig(f"{out_basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_basename}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return cm


def report(cm, y_true, y_pred, display_labels):
    """Print the numbers that must match the caption and the main text."""
    acc = accuracy_score(y_true, y_pred)
    recalls = recall_score(
        y_true, y_pred, labels=list(range(len(display_labels))),
        average=None, zero_division=0,
    )

    print("=" * 64)
    print("[Figure 4B] out-of-fold predictions from nested 5-fold CV")
    print("            (each record predicted once, by the fold that held it out)")
    print(f"N (records in matrix) = {int(cm.sum())}")
    print(f"Overall accuracy      = {acc:.4f}")
    print(f"Total errors          = {int(cm.sum() - np.trace(cm))}")
    print("-" * 64)
    for name, r in zip(display_labels, recalls):
        print(f"  recall[{name:<16}] = {r:.4f}")
    print("-" * 64)
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "".join(f"{d:>16}" for d in display_labels))
    for i, d in enumerate(display_labels):
        print(f"{d:>12}" + "".join(f"{cm[i, j]:>16d}" for j in range(len(display_labels))))
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="multimodal_oof_predictions.csv",
                        help="out-of-fold prediction CSV from the nested CV notebook")
    parser.add_argument("--outdir", default=".", help="directory for the figure files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_basename = str(outdir / "figure4B_confusion_matrix")

    y_true, y_pred, display_labels = load_oof(args.oof)
    cm = plot_confusion_matrix(y_true, y_pred, display_labels, out_basename)
    report(cm, y_true, y_pred, display_labels)
    print(f"saved {out_basename}.{{tiff,pdf,png}}")


if __name__ == "__main__":
    main()
