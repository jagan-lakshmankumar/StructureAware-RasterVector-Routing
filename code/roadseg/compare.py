"""Aggregate metrics across all model runs into a comparison table + CSV."""

import csv
import json
from pathlib import Path

KEY_METRICS = [
    "iou", "miou", "dice", "f1", "precision", "recall",
    "sensitivity", "specificity", "accuracy", "kappa",
]


def compare_runs(runs_dir: Path, preds_dir: Path, out_csv: str = "results_comparison.csv"):
    rows = []
    for model_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        row = {"model": model}

        hist_path = model_dir / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            if hist:
                best = max(hist, key=lambda r: r.get("val_iou", 0))
                row["best_epoch"] = best.get("epoch")
                row["train_loss"] = round(best.get("train_loss", 0), 4)
                row["val_loss"] = round(best.get("val_loss", 0), 4)
                tm = best.get("train_metrics", {})
                vm = best.get("val_metrics", {})
                for k in KEY_METRICS:
                    row[f"train_{k}"] = round(tm.get(k, 0), 4) if k in tm else None
                    row[f"val_{k}"] = round(vm.get(k, 0), 4) if k in vm else None
                row["train_accuracy"] = round(best.get("train_accuracy", 0), 4)
                row["val_accuracy"] = round(best.get("val_accuracy", 0), 4)

        test_path = preds_dir / model / "test_metrics.json"
        if test_path.exists():
            tm = json.loads(test_path.read_text())
            for k in KEY_METRICS:
                row[f"test_{k}"] = round(tm.get(k, 0), 4) if k in tm else None

        rows.append(row)

    if not rows:
        print("No runs found to compare.")
        return

    # write CSV
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # pretty print
    print("\n================= MODEL COMPARISON =================")
    header = f"{'model':<16}{'test_iou':>9}{'test_miou':>11}{'test_f1':>9}{'test_prec':>10}{'test_rec':>9}{'test_acc':>10}{'test_kappa':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "test_iou" in r:
            print(f"{r['model']:<16}{r.get('test_iou',0):>9.4f}{r.get('test_miou',0):>11.4f}"
                  f"{r.get('test_f1',0):>9.4f}{r.get('test_precision',0):>10.4f}"
                  f"{r.get('test_recall',0):>9.4f}{r.get('test_accuracy',0):>10.4f}"
                  f"{r.get('test_kappa',0):>11.4f}")
        else:
            print(f"{r['model']:<16} (no test metrics)")
    print(f"\nFull comparison written to: {Path(out_csv).resolve()}")
    print("====================================================\n")
    return rows
