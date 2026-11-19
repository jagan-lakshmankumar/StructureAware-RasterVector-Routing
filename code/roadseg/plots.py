"""Plotting utilities: training curves and confusion matrices.

Saves PNG images (no GUI needed, uses the non-interactive Agg backend):
    training_curves.png          train/val loss, accuracy, IoU, F1 per epoch
    confusion_matrix_*.png       2x2 confusion-matrix heatmaps
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["background", "road"]


def save_training_curves(history, out_path):
    """Plot train/val loss, accuracy, IoU and F1 curves from a history list."""
    if not history:
        print("[plots] Empty history, skipping training curves.")
        return

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(history)]

    panels = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_accuracy", "val_accuracy"),
        ("IoU", "train_iou", "val_iou"),
        ("F1 / Dice", None, None),  # handled via nested metrics below
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (title, tr_key, va_key) in zip(axes.flat, panels):
        if tr_key is None:
            tr = [(r.get("train_metrics") or {}).get("f1") for r in history]
            va = [(r.get("val_metrics") or {}).get("f1") for r in history]
        else:
            tr = [r.get(tr_key) for r in history]
            va = [r.get(va_key) for r in history]

        # Skip panel if no data is present for it.
        if all(v is None for v in tr) and all(v is None for v in va):
            ax.axis("off")
            continue

        ax.plot(epochs, tr, "o-", label="train", linewidth=2, markersize=4)
        ax.plot(epochs, va, "s-", label="val", linewidth=2, markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved training curves -> {Path(out_path).resolve()}")


def save_confusion_matrix(tp, fp, fn, tn, out_path, title="Confusion Matrix"):
    """Save a 2x2 confusion-matrix heatmap from raw TP/FP/FN/TN counts.

    Layout (rows = actual, columns = predicted):
        [[TN, FP],
         [FN, TP]]
    """
    cm = np.array([[int(tn), int(fp)],
                   [int(fn), int(tp)]], dtype=float)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = 100.0 * v / total if total else 0.0
            ax.text(j, i, f"{int(v)}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if v > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved confusion matrix -> {Path(out_path).resolve()}")


def _counts_from_metrics(metrics, prefix=""):
    """Extract tp/fp/fn/tn from a metrics dict, with a fallback for flat keys."""
    metrics = metrics or {}
    def get(key):
        v = metrics.get(key)
        if v is None and prefix:
            v = metrics.get(f"{prefix}_{key}")
        return int(v or 0)
    return get("tp"), get("fp"), get("fn"), get("tn")
