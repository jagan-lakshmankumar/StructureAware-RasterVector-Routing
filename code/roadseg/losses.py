"""Loss functions for binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0, eps=1e-7):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return self.bce_w * bce + self.dice_w * dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return dice.mean()


def build_loss(name="bce_dice"):
    name = name.lower()
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {name}")
