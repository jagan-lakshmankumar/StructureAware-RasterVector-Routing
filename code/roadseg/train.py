"""Training loop shared by all models."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RoadDataset, pair_files, seed_everything
from .losses import build_loss
from .metrics import MetricTracker
from .plots import save_confusion_matrix, save_training_curves


def train(args, model, device):
    seed_everything(getattr(args, "seed", 42))

    data_dir = Path(args.data_dir)
    train_pairs = pair_files(data_dir / "tiff" / "train", data_dir / "tiff" / "train_labels")
    val_pairs = pair_files(data_dir / "tiff" / "val", data_dir / "tiff" / "val_labels")
    print(f"Dataset-2 Loaded: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    train_ds = RoadDataset(train_pairs, patch_size=args.patch_size, is_train=True)
    val_ds = RoadDataset(val_pairs, patch_size=args.patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = build_loss(getattr(args, "loss", "bce_dice"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_iou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_meter = MetricTracker()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_meter.update((torch.sigmoid(logits) >= 0.5), (masks >= 0.5))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= max(1, len(train_loader))
        train_metrics = train_meter.compute()

        # Validation
        model.eval()
        val_loss = 0.0
        val_meter = MetricTracker()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                val_loss += loss.item()
                val_meter.update((torch.sigmoid(logits) >= 0.5), (masks >= 0.5))

        val_loss /= max(1, len(val_loader))
        val_metrics = val_meter.compute()

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}, "
              f"P: {train_metrics['precision']:.4f}, R: {train_metrics['recall']:.4f}, "
              f"Kappa: {train_metrics['kappa']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, "
              f"Kappa: {val_metrics['kappa']:.4f}")

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            # convenience flat keys
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
        }
        history.append(record)

        ckpt = {
            "model_name": getattr(args, "model", "unknown"),
            "model_kwargs": getattr(args, "_model_kwargs", {}),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_iou": val_metrics["iou"],
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  >>> Best checkpoint updated! (Val IoU: {best_val_iou:.4f})")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ---- save plots: training curves + confusion matrices ----
    try:
        save_training_curves(history, out_dir / "training_curves.png")
        best_epoch = max(history, key=lambda r: r.get("val_iou", 0))
        ep = best_epoch.get("epoch")
        train_cm = best_epoch.get("train_metrics", {})
        val_cm = best_epoch.get("val_metrics", {})
        save_confusion_matrix(
            train_cm.get("tp", 0), train_cm.get("fp", 0),
            train_cm.get("fn", 0), train_cm.get("tn", 0),
            out_dir / "confusion_matrix_train.png",
            title=f"Training Confusion Matrix (epoch {ep})",
        )
        save_confusion_matrix(
            val_cm.get("tp", 0), val_cm.get("fp", 0),
            val_cm.get("fn", 0), val_cm.get("tn", 0),
            out_dir / "confusion_matrix_val.png",
            title=f"Validation Confusion Matrix (epoch {ep})",
        )
    except Exception as e:  # pragma: no cover - plotting is non-critical
        print(f"[plots] Warning: could not save plots: {e}")

    print(f"\nTraining Complete! Best Val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {out_dir.resolve()}")
    return best_val_iou
