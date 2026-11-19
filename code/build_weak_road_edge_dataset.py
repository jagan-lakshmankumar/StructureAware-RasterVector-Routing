"""Build Chimanimani weak road-exposure labels and GMU/GAT input features.

Labels are deliberately named ``weak_exposed``: a 50 m pre-event road piece
is positive when it intersects a 500 m buffer around a *ground-truth*
landslide mask.  This is exposure supervision, not independently observed
road damage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union

PROJECT = Path(__file__).resolve().parent.parent


def probability_features(geometry, probability: np.ndarray, transform) -> np.ndarray:
    step = max(abs(transform.a), abs(transform.e), 10.0)
    distances = np.arange(0, max(geometry.length, step), step)
    inverse = ~transform
    values = []
    for distance in distances:
        point = geometry.interpolate(float(distance))
        col, row = inverse * (point.x, point.y)
        col, row = int(col), int(row)
        if 0 <= row < probability.shape[0] and 0 <= col < probability.shape[1]:
            value = probability[row, col]
            if np.isfinite(value) and value >= 0:
                values.append(float(value))
    if not values:
        return np.zeros(4, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    return np.asarray([values.mean(), values.max(), np.quantile(values, .95), (values >= .81).mean()], dtype=np.float32)


def graph_edges(rows: list[dict]) -> np.ndarray:
    by_node: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_node.setdefault(row["u"], []).append(index)
        by_node.setdefault(row["v"], []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in by_node.values():
        for left in members:
            pairs.add((left, left))
            for right in members:
                pairs.add((left, right))
    return np.asarray(sorted(pairs), dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, default=PROJECT / "results/chimanimani_study_graph_v2_local_buffer_update/road_edges.geojson")
    parser.add_argument("--s2", type=Path, default=PROJECT / "dataset/Sen12Landslides_S2_14GB/s2")
    parser.add_argument("--manifests", type=Path, default=PROJECT / "dataset/Sen12Landslides_S2_14GB/temporal_cache_chimanimani")
    parser.add_argument("--probability-root", type=Path, default=PROJECT / "results/chimanimani_full_post_event_coverage")
    parser.add_argument("--buffer-m", type=float, default=500.)
    parser.add_argument("--max-train-scenes", type=int, default=120,
                        help="Use a reproducible annotated training-scene subset for the CPU prototype; 0 uses all.")
    parser.add_argument("--out", type=Path, default=PROJECT / "dataset/weak_road_edge_chimanimani")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_edges = gpd.read_file(args.edges)
    # Keep one physical direction only; the GAT line graph is undirected.
    roads = all_edges[all_edges.edge_id.str.contains(r"_0_s")].copy()
    classes = sorted(roads.fclass.fillna("unclassified").astype(str).unique())
    class_index = {name: index for index, name in enumerate(classes)}
    report = {"label_definition": "weak_exposed: 50 m road piece intersects a 500 m buffer around ground-truth landslide mask",
              "buffer_m": args.buffer_m, "raster_features": ["mean_probability", "max_probability", "p95_probability", "fraction_probability_ge_0.81"],
              "vector_features": ["log_length", "speed_normalized", "u_degree", "v_degree", *[f"fclass_{name}" for name in classes]],
              "splits": {}}

    for split in ("train", "internal_val", "test"):
        manifest = json.loads((args.manifests / f"{split}_manifest.json").read_text())
        manifest = [item for item in manifest if item.get("annotated", False)]
        if split == "train" and args.max_train_scenes:
            manifest = manifest[:args.max_train_scenes]
        rows: list[dict] = []
        for scene_number, item in enumerate(manifest, start=1):
            nc_path = args.s2 / item["file"]
            probability_path = args.probability_root / f"{nc_path.stem}_landslide_probability.tif"
            if not nc_path.exists() or not probability_path.exists():
                continue
            with xr.open_dataset(nc_path) as dataset:
                truth = dataset["MASK"].isel(time=int(item["post"])).values.astype(np.uint8)
                x, y = dataset["x"].values.astype(float), dataset["y"].values.astype(float)
            dx, dy = abs(float(np.median(np.diff(x)))), abs(float(np.median(np.diff(y))))
            bounds = box(float(x.min() - dx / 2), float(y.min() - dy / 2), float(x.max() + dx / 2), float(y.max() + dy / 2))
            truth_geometries = [shape(geometry) for geometry, value in shapes(truth, mask=truth > 0,
                               transform=rasterio.transform.from_origin(float(x.min() - dx / 2), float(y.max() + dy / 2), dx, dy)) if value == 1]
            if not truth_geometries:
                continue
            hazard_buffer = unary_union(truth_geometries).buffer(args.buffer_m)
            candidate_indices = roads.sindex.query(bounds, predicate="intersects")
            with rasterio.open(probability_path) as source:
                probability, transform = source.read(1), source.transform
            for index in candidate_indices:
                edge = roads.iloc[int(index)]
                if edge.geometry.intersects(bounds):
                    fclass = str(edge.fclass) if edge.fclass is not None else "unclassified"
                    one_hot = np.zeros(len(classes), dtype=np.float32)
                    one_hot[class_index[fclass]] = 1
                    vector = np.concatenate([np.asarray([
                        np.log1p(float(edge.length_m)), float(edge.speed_kph) / 80.,
                        min(float(edge.get("u_degree", 1)), 8.) / 8., min(float(edge.get("v_degree", 1)), 8.) / 8.,
                    ], dtype=np.float32), one_hot])
                    rows.append({"edge_id": edge.edge_id, "u": edge.u, "v": edge.v,
                                 "raster": probability_features(edge.geometry, probability, transform), "vector": vector,
                                 "label": int(edge.geometry.intersects(hazard_buffer))})
            if scene_number % 20 == 0 or scene_number == len(manifest):
                print(f"{split}: processed {scene_number}/{len(manifest)} annotated scenes", flush=True)
        # An edge can touch a tile boundary; keep just one sample deterministically.
        unique = {row["edge_id"]: row for row in rows}
        rows = list(unique.values())
        raster_features = np.stack([row["raster"] for row in rows]) if rows else np.empty((0, 4), dtype=np.float32)
        vector_features = np.stack([row["vector"] for row in rows]) if rows else np.empty((0, 4 + len(classes)), dtype=np.float32)
        labels = np.asarray([row["label"] for row in rows], dtype=np.float32)
        np.savez_compressed(args.out / f"{split}.npz", raster=raster_features, vector=vector_features, labels=labels,
                            edge_index=graph_edges(rows), edge_ids=np.asarray([row["edge_id"] for row in rows]))
        report["splits"][split] = {"samples": len(rows), "positive_weak_exposure": int(labels.sum()),
                                    "negative": int(len(labels) - labels.sum())}
        print(split, report["splits"][split], flush=True)
    (args.out / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
