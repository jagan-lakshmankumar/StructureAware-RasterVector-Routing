"""Sliding-window tiled prediction with Hann blending for full-resolution images."""

import numpy as np
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def predict_tiled(model, image_rgb, device="cuda", tile=512, overlap=64, threshold=None):
    """Returns probability map (H, W) in [0,1]."""
    device = torch.device(device)  # accept str or torch.device
    h, w = image_rgb.shape[:2]
    stride = tile - overlap

    # padding
    if h <= tile:
        pad_h = tile - h
    else:
        pad_h = (stride - (h - tile) % stride) % stride
    if w <= tile:
        pad_w = tile - w
    else:
        pad_w = (stride - (w - tile) % stride) % stride

    padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    total_h, total_w = padded_img.shape[:2]

    norm = padded_img.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()

    accum = torch.zeros((total_h, total_w), device=device)
    counts = torch.zeros((total_h, total_w), device=device)

    w1d = np.hanning(tile).astype(np.float32)
    w1d = np.maximum(w1d, 1e-3)
    window = torch.from_numpy(np.outer(w1d, w1d)).to(device)

    model.eval()
    for y in range(0, total_h - tile + 1, stride):
        for x in range(0, total_w - tile + 1, stride):
            patch = tensor[:, :, y:y + tile, x:x + tile].to(device)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(patch)
            prob = torch.sigmoid(logits)[0, 0]
            accum[y:y + tile, x:x + tile] += prob * window
            counts[y:y + tile, x:x + tile] += window

    prob_map = (accum / counts.clamp_min(1e-8))[:h, :w].cpu().numpy()
    if threshold is not None:
        return (prob_map >= threshold).astype(np.uint8)
    return prob_map
