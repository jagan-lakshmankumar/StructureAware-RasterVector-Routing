"""Metric tracking for binary semantic segmentation.

Accumulates confusion-matrix counts (TP/FP/FN/TN) over a whole epoch / test set
and derives a rich set of metrics: IoU, mIoU, Dice/F1, Precision, Recall,
Specificity, Accuracy and Cohen's Kappa.
"""

import torch


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, preds, targets):
        """Accepts torch tensors or numpy arrays of arbitrary shape."""
        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)
        p = preds.bool().reshape(-1)
        t = targets.bool().reshape(-1)
        self.tp += int((p & t).sum().item())
        self.fp += int((p & ~t).sum().item())
        self.fn += int((~p & t).sum().item())
        self.tn += int((~p & ~t).sum().item())

    def compute(self):
        eps = 1e-8
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + tn + fp + fn

        iou_fg = tp / (tp + fp + fn + eps)            # road class IoU
        iou_bg = tn / (tn + fp + fn + eps)            # background class IoU
        miou = (iou_fg + iou_bg) / 2.0

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)                  # sensitivity
        specificity = tn / (tn + fp + eps)
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)     # = dice
        accuracy = (tp + tn) / (total + eps)

        # Cohen's kappa
        if total > 0:
            po = accuracy
            pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total + eps)
            kappa = (po - pe) / (1.0 - pe + eps)
        else:
            kappa = 0.0

        return {
            "iou": float(iou_fg),
            "miou": float(miou),
            "dice": float(f1),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "kappa": float(kappa),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    def __repr__(self):
        m = self.compute()
        return (f"MetricTracker(iou={m['iou']:.4f}, miou={m['miou']:.4f}, "
                f"f1={m['f1']:.4f}, precision={m['precision']:.4f}, "
                f"recall={m['recall']:.4f}, accuracy={m['accuracy']:.4f}, "
                f"kappa={m['kappa']:.4f})")
