"""Fuse landslide probabilities into vector-road SAFE/DANGER decisions.

The active landslide input is the georeferenced result of
``roboflow_landslide_inference.py`` using Roboflow's hosted SegFormer model.
This script emits a strict, safety-first road status: ``SAFE`` or
``DANGER_DO_NOT_USE``.  A road is dangerous if any sampled point in its
10-metre corridor reaches the landslide probability threshold.

Usage
-----
    python code/fuse_landslide_road_confidence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MAXSCENES = PROJECT / "dataset" / "MaxScenes" / "Dataset_MaxScenes"
DEFAULT_RGB = MAXSCENES / "Shirur_2024_16-07-2024" / "POST" / "2024-09-30" / "S2_RGB.tif"
DEFAULT_ROADS = PROJECT / "dataset" / "Vector" / "gis_osm_roads_free_1.shp"
DEFAULT_LANDSLIDE_PROBABILITY = PROJECT / "results" / "roboflow_landslide_shirur" / "landslide_probability.tif"
DEFAULT_OUT = PROJECT / "results" / "maxscenes_shirur_confidence"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DRIVABLE_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                    "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
                    "unclassified", "living_street"}


def load_model(checkpoint_path: Path, device):
    """Load either the local U-Net or the equivalent Colab notebook U-Net."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from roadseg.models.unet import UNet
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = int(checkpoint.get("base_channels", 16))
    keys = checkpoint["model_state_dict"].keys()
    if any(key.startswith("i.net.") for key in keys):
        # Architecture embedded in colab_train_sentinel2_road.ipynb.
        class DoubleConv(nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
            def forward(self, x): return self.net(x)
        class Down(nn.Module):
            def __init__(self, i, o): super().__init__(); self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(i, o))
            def forward(self, x): return self.net(x)
        class Up(nn.Module):
            def __init__(self, i, o): super().__init__(); self.up = nn.ConvTranspose2d(i, i // 2, 2, 2); self.conv = DoubleConv(i, o)
            def forward(self, x, skip):
                x = self.up(x); dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
                x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
                return self.conv(torch.cat([skip, x], 1))
        class ColabUNet(nn.Module):
            def __init__(self, base):
                super().__init__(); c = [base, base * 2, base * 4, base * 8, base * 16]
                self.i = DoubleConv(3, c[0]); self.d1 = Down(c[0], c[1]); self.d2 = Down(c[1], c[2]); self.d3 = Down(c[2], c[3]); self.d4 = Down(c[3], c[4])
                self.u1 = Up(c[4], c[3]); self.u2 = Up(c[3], c[2]); self.u3 = Up(c[2], c[1]); self.u4 = Up(c[1], c[0]); self.out = nn.Conv2d(c[0], 1, 1)
            def forward(self, x):
                a = self.i(x); b = self.d1(a); c = self.d2(b); d = self.d3(c); e = self.d4(d)
                return self.out(self.u4(self.u3(self.u2(self.u1(e, d), c), b), a))
        model = ColabUNet(base)
    else:
        model = UNet(in_channels=3, out_channels=1, base=base)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def tiled_probability(model, image: np.ndarray, device, tile: int, overlap: int) -> np.ndarray:
    """Hann-blended tiled prediction for a CxHxW normalized image."""
    import torch
    _, height, width = image.shape
    stride = tile - overlap
    pad_h = (tile - height) if height <= tile else (stride - (height - tile) % stride) % stride
    pad_w = (tile - width) if width <= tile else (stride - (width - tile) % stride) % stride
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape[1:]
    tensor = torch.from_numpy(padded).unsqueeze(0).float()
    total = torch.zeros((ph, pw), device=device)
    weights = torch.zeros((ph, pw), device=device)
    w1 = np.maximum(np.hanning(tile).astype(np.float32), 1e-3)
    window = torch.from_numpy(np.outer(w1, w1)).to(device)
    with torch.no_grad():
        for y in range(0, ph - tile + 1, stride):
            for x in range(0, pw - tile + 1, stride):
                logits = model(tensor[:, :, y:y + tile, x:x + tile].to(device))
                probability = torch.sigmoid(logits)[0, 0]
                total[y:y + tile, x:x + tile] += probability * window
                weights[y:y + tile, x:x + tile] += window
    return (total / weights.clamp_min(1e-8))[:height, :width].cpu().numpy()


def percentile_rgb(cwh: np.ndarray) -> np.ndarray:
    rgb = np.empty_like(cwh, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(cwh[c], (2, 98))
        rgb[c] = np.clip((cwh[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def filter_roads(roads):
    if "fclass" in roads.columns:
        roads = roads[roads["fclass"].isin(DRIVABLE_CLASSES)].copy()
    elif "highway" in roads.columns:
        roads = roads[roads["highway"].isin(DRIVABLE_CLASSES)].copy()
    return roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()


def segment_landslide_risk(geometry, landslide_prob, transform, shape, buffer_m=10.0):
    from rasterio.features import rasterize
    corridor = geometry.buffer(buffer_m)
    mask = rasterize([(corridor, 1)], out_shape=shape, transform=transform,
                     all_touched=True, dtype="uint8").astype(bool)
    valid = mask & np.isfinite(landslide_prob) & (landslide_prob >= 0)
    if not valid.any():
        # A segment with no observed hazard pixels must never be called safe.
        return 1.0
    # Strict safety rule: any high-confidence landslide exposure blocks the
    # whole road feature.  A percentile could hide a short hazardous crossing.
    landslide_risk = float(landslide_prob[valid].max())
    return landslide_risk


def write_probability(path: Path, array: np.ndarray, profile):
    import rasterio
    meta = {**profile, "driver": "GTiff", "count": 1, "dtype": "float32", "compress": "lzw"}
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype(np.float32), 1)
    Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB, help="MaxScenes S2_RGB.tif post-event scene")
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS, help="OSM/official pre-event roads vector")
    parser.add_argument("--landslide-probability", type=Path, default=DEFAULT_LANDSLIDE_PROBABILITY,
                        help="GeoTIFF from roboflow_landslide_inference.py")
    parser.add_argument("--landslide-model", default="segformer-landslide-detection/2 (Roboflow)",
                        help="Provenance label for the supplied landslide-probability raster.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--danger-threshold", type=float, default=.77,
                        help="Landslide probability at which a road becomes DANGER_DO_NOT_USE.")
    parser.add_argument("--road-corridor-buffer-m", type=float, default=10.0,
                        help="Half-width of the road corridor sampled for landslide exposure.")
    parser.add_argument("--landslide-safety-buffer-m", type=float, default=0.0,
                        help="Extra clearance around detected landslide regions. Added to the road corridor buffer.")
    args = parser.parse_args()
    if not 0 <= args.danger_threshold <= 1:
        raise ValueError("--danger-threshold must be between 0 and 1")
    if args.road_corridor_buffer_m < 0 or args.landslide_safety_buffer_m < 0:
        raise ValueError("Road and landslide buffers must be non-negative")

    import geopandas as gpd
    import pyogrio
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    args.out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.rgb) as src:
        profile, crs, transform = src.profile.copy(), src.crs, src.transform
        shape = (src.height, src.width)
    if crs is None:
        raise ValueError("The post-event raster must be georeferenced.")

    print(f"Vector-only road fusion; image={shape[1]}x{shape[0]}; CRS={crs}")
    if not args.landslide_probability.exists():
        raise FileNotFoundError(
            f"Missing {args.landslide_probability}. Run code/roboflow_landslide_inference.py first."
        )
    with rasterio.open(args.landslide_probability) as landslide_source:
        if landslide_source.crs != crs or landslide_source.transform != transform or landslide_source.shape != shape:
            raise ValueError("Roboflow landslide raster must match the post-event image grid and CRS.")
        landslide_prob = landslide_source.read(1).astype(np.float32)
    write_probability(args.out / "landslide_probability.tif", landslide_prob, profile)

    source_crs = pyogrio.read_info(args.roads)["crs"]
    bounds = array_bounds(*shape, transform)
    source_bounds = transform_bounds(crs, source_crs, *bounds)
    roads = gpd.read_file(args.roads, bbox=source_bounds).to_crs(crs)
    roads = filter_roads(roads)
    roads = roads[roads.geometry.intersects(box(*bounds))].copy()
    if roads.empty:
        raise RuntimeError("No drivable OSM roads found in the post-event raster extent.")

    total_buffer_m = args.road_corridor_buffer_m + args.landslide_safety_buffer_m
    risk = [segment_landslide_risk(g, landslide_prob, transform, shape, total_buffer_m) for g in roads.geometry]
    roads["landslide_risk"] = risk
    roads["road_status"] = np.where(roads["landslide_risk"] >= args.danger_threshold,
                                    "DANGER_DO_NOT_USE", "SAFE")
    roads["is_blocked"] = roads["road_status"].eq("DANGER_DO_NOT_USE")
    roads["status_rule"] = (
        f"DANGER_DO_NOT_USE if maximum probability within {total_buffer_m:.1f} m road/hazard clearance >= {args.danger_threshold:.2f}"
    )
    keep = [c for c in ("osm_id", "fclass", "name", "landslide_risk",
                        "road_status", "is_blocked", "status_rule", "geometry") if c in roads.columns]
    roads[keep].to_file(args.out / "road_confidence.geojson", driver="GeoJSON")

    report = {"rgb": str(args.rgb), "roads": str(args.roads),
              "landslide_source": str(args.landslide_probability), "landslide_model": args.landslide_model, "road_count": len(roads),
              "safe_roads": int((~roads["is_blocked"]).sum()), "danger_roads": int(roads["is_blocked"].sum()),
              "danger_threshold": args.danger_threshold,
              "road_corridor_buffer_m": args.road_corridor_buffer_m,
              "landslide_safety_buffer_m": args.landslide_safety_buffer_m,
              "total_exposure_buffer_m": total_buffer_m,
              "rule": "A road is DANGER_DO_NOT_USE when any pixel inside the buffered road/hazard clearance reaches the landslide threshold. Road geometry comes directly from the vector layer; no road-extraction model is used.",
              "warning": "Model-derived risk still requires event-level road-condition validation before operational use."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
