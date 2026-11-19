"""Dataset utilities for Dataset-2 (Massachusetts Roads, 1500x1500 GeoTIFF)."""

import os
from pathlib import Path

# Silence OpenCV TIFF metadata warnings for GeoTIFF tags (33550, 33922,
# 34735, 34737, 42112). Must be set BEFORE cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_EXTS = {".tiff", ".tif", ".png", ".jpg", ".jpeg"}


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_files(image_dir, mask_dir):
    """Pairs images and masks by matching identical file stems."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing directories: {image_dir} or {mask_dir}")
    images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise RuntimeError(f"No matching pairs found between {image_dir} and {mask_dir}")
    return [(images[k], masks[k]) for k in common]


class RoadDataset(Dataset):
    """1500x1500px aerial imagery + binary road mask with patch cropping."""

    def __init__(self, pairs, patch_size=512, is_train=True, augment=True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.is_train = is_train

        if is_train and augment:
            self.transform = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        mask = (mask_raw > 127).astype(np.float32)

        if self.is_train and self.transform is not None:
            augmented = self.transform(image=rgb, mask=mask)
            rgb = augmented["image"]
            mask = augmented["mask"]
        elif (not self.is_train) and self.patch_size is not None:
            h, w = rgb.shape[:2]
            if h != self.patch_size or w != self.patch_size:
                crop = A.CenterCrop(height=min(self.patch_size, h),
                                    width=min(self.patch_size, w))
                res = crop(image=rgb, mask=mask)
                rgb = res["image"]
                mask = res["mask"]

        img_norm = rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        return img_tensor, mask_tensor
