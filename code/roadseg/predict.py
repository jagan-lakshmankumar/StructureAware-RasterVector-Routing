"""Prediction and evaluation on full-resolution test images."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import RoadDataset, pair_files
from .metrics import MetricTracker
from .plots import save_confusion_matrix
from .tiling import predict_tiled


def predict(args, model, device):
    data_dir = Path(args.data_dir)
    test_pairs = pair_files(data_dir / "tiff" / "test", data_dir / "tiff" / "test_labels")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "binary_masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Tiled Prediction on {len(test_pairs)} test images (1500x1500px)...")
    print(f"Output Directory: {out_dir.resolve()}\n")

    meter = MetricTracker()

    for img_path, mask_path in _tqdm(test_pairs):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        prob_map = predict_tiled(model, rgb, device=device, tile=args.tile,
                                overlap=args.overlap)
        pred_mask = (prob_map >= args.threshold).astype(np.uint8)

        meter.update(pred_mask, gt_mask)

        stem = img_path.stem
        cv2.imwrite(str(masks_dir / f"{stem}_pred.png"), pred_mask * 255)

        overlay = bgr.copy()
        overlay[pred_mask == 1] = [0, 255, 0]
        blended = cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0)
        cv2.imwrite(str(overlays_dir / f"{stem}_overlay.jpg"), blended)

    final_metrics = meter.compute()
    print("\n=======================================================")
    print(f"Test Set Evaluation Results ({getattr(args, 'model', 'model')}):")
    for k in ["iou", "miou", "dice", "f1", "precision", "recall", "accuracy", "kappa"]:
        print(f"  {k.capitalize():12s}: {final_metrics[k]:.4f}")
    print("=======================================================\n")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save the test confusion-matrix image.
    try:
        save_confusion_matrix(
            final_metrics["tp"], final_metrics["fp"],
            final_metrics["fn"], final_metrics["tn"],
            out_dir / "confusion_matrix_test.png",
            title=f"Test Confusion Matrix ({getattr(args, 'model', 'model')})",
        )
    except Exception as e:  # pragma: no cover - plotting is non-critical
        print(f"[plots] Warning: could not save test confusion matrix: {e}")

    return final_metrics


def _tqdm(iterable):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc="Testing")
    except Exception:
        return iterable
