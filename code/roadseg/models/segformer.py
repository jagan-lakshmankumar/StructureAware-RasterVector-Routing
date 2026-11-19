"""SegFormer (MiT encoder + lightweight MLP decoder)."""

import torch.nn as nn
import torch.nn.functional as F

# Official MiT configs: encoder depths, hidden sizes, decoder hidden size
VARIANTS = {
    "mit_b0": dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
                   decoder_hidden_size=256),
    "mit_b1": dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                   decoder_hidden_size=256),
    "mit_b2": dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                   decoder_hidden_size=768),
    "mit_b3": dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                   decoder_hidden_size=768),
    "mit_b4": dict(depths=[3, 8, 40, 3], hidden_sizes=[64, 128, 320, 512],
                   decoder_hidden_size=768),
    "mit_b5": dict(depths=[3, 16, 56, 3], hidden_sizes=[64, 128, 320, 512],
                   decoder_hidden_size=768),
}


class SegFormer(nn.Module):
    """Wraps HuggingFace SegFormer so it returns raw logits like the other models."""

    def __init__(self, variant="mit_b0", pretrained=False, classes=1):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except Exception as e:
            raise ImportError("transformers is required for SegFormer") from e

        variant = variant.lower().replace("-", "_")
        cfg = VARIANTS.get(variant)
        if cfg is None:
            raise ValueError(f"Unknown SegFormer variant: {variant}. "
                             f"Choose from {list(VARIANTS)}")

        if pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    f"nvidia/{variant}-ade",
                    num_labels=classes,
                    ignore_mismatched_sizes=True,
                )
                print(f"[SegFormer] Loaded pretrained weights from nvidia/{variant}")
            except Exception as e:
                print(f"[SegFormer] Pretrained load failed ({e}); using random init")
                self.model = SegformerForSemanticSegmentation(
                    SegformerConfig(num_labels=classes, **cfg))
        else:
            self.model = SegformerForSemanticSegmentation(
                SegformerConfig(num_labels=classes, **cfg))

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits
