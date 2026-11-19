"""Train a CPU-friendly weakly supervised GMU + GAT road-exposure model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent


class GATLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels, bias=False)
        self.attn_src = nn.Parameter(torch.empty(channels))
        self.attn_dst = nn.Parameter(torch.empty(channels))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.attn_src, -.1, .1)
        nn.init.uniform_(self.attn_dst, -.1, .1)

    def forward(self, features, edge_index):
        transformed = self.linear(features)
        source, destination = edge_index
        logits = F.leaky_relu((transformed[source] * self.attn_src).sum(-1) + (transformed[destination] * self.attn_dst).sum(-1), .2)
        weights = torch.exp(torch.clamp(logits, -10, 10))
        normalizer = torch.zeros(len(features), device=features.device).index_add_(0, destination, weights)
        alpha = weights / normalizer[destination].clamp_min(1e-9)
        output = torch.zeros_like(transformed).index_add_(0, destination, transformed[source] * alpha.unsqueeze(-1))
        return F.elu(output)


class WeakGMUGAT(nn.Module):
    def __init__(self, raster_dim, vector_dim, hidden=64):
        super().__init__()
        self.raster = nn.Sequential(nn.Linear(raster_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.vector = nn.Sequential(nn.Linear(vector_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gate = nn.Linear(hidden * 2, hidden)
        self.gat1, self.gat2 = GATLayer(hidden), GATLayer(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(.15), nn.Linear(hidden // 2, 1))

    def forward(self, raster, vector, edge_index):
        raster_embedding, vector_embedding = self.raster(raster), self.vector(vector)
        gate = torch.sigmoid(self.gate(torch.cat([raster_embedding, vector_embedding], dim=-1)))
        fused = gate * raster_embedding + (1 - gate) * vector_embedding
        topology = self.gat2(self.gat1(fused, edge_index), edge_index)
        return self.head(topology).squeeze(-1), gate


def load(path: Path):
    data = np.load(path, allow_pickle=False)
    return tuple(torch.from_numpy(data[name]).float() if name != "edge_index" else torch.from_numpy(data[name]).long()
                 for name in ("raster", "vector", "labels", "edge_index"))


def metrics(logits, labels, threshold):
    prediction = torch.sigmoid(logits) >= threshold
    truth = labels >= .5
    tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9), "iou": tp / max(tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def best_metrics(logits, labels):
    return max((metrics(logits, labels, float(threshold)) for threshold in np.arange(.10, .91, .02)), key=lambda item: item["iou"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "dataset/weak_road_edge_chimanimani")
    parser.add_argument("--out", type=Path, default=PROJECT / "results/weak_gmu_gat_chimanimani")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train, validation, test = (load(args.data / f"{split}.npz") for split in ("train", "internal_val", "test"))
    if not len(train[2]) or not len(validation[2]):
        raise ValueError("Weak-label dataset has no train or validation samples.")
    model = WeakGMUGAT(train[0].shape[1], train[1].shape[1])
    weight = torch.tensor([(len(train[2]) - train[2].sum()) / train[2].sum().clamp_min(1)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.
    for epoch in range(1, args.epochs + 1):
        model.train(); logits, _ = model(train[0], train[1], train[3]); loss = F.binary_cross_entropy_with_logits(logits, train[2], pos_weight=weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(validation[0], validation[1], validation[3]); report = best_metrics(validation_logits, validation[2])
        history.append({"epoch": epoch, "loss": float(loss.detach()), "validation": report})
        if report["iou"] > best:
            best = report["iou"]
            torch.save({"model_state_dict": model.state_dict(), "raster_dim": train[0].shape[1], "vector_dim": train[1].shape[1],
                        "epoch": epoch, "weak_label_definition": "landslide-exposed road segment within 500 m of ground-truth landslide", "validation": report}, args.out / "best.pt")
        if epoch == 1 or epoch % 20 == 0: print(f"Epoch {epoch}/{args.epochs}: val weak-label IoU={report['iou']:.4f} F1={report['f1']:.4f}", flush=True)
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad():
        test_logits, gates = model(test[0], test[1], test[3])
        test_report = metrics(test_logits, test[2], checkpoint["validation"]["threshold"]) if int(test[2].sum()) else {
            "not_meaningful": True,
            "reason": "Held-out split contains zero ground-truth-landslide-exposed road pieces; road-exposure recall/IoU cannot be measured.",
            "samples": int(len(test[2])), "positive_weak_exposure": 0,
        }
    report = {"weak_supervision": True, "label_definition": checkpoint["weak_label_definition"], "best_validation": checkpoint["validation"],
              "held_out_weak_label_test": test_report, "mean_raster_gate": float(gates.mean()), "epochs_selected": checkpoint["epoch"]}
    (args.out / "history.json").write_text(json.dumps(history, indent=2)); (args.out / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
