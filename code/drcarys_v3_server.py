"""Local DR-Carys V3 server: weak GMU+GAT fusion and safety-only routing.

Run from the project root:
    .\\.venv\\Scripts\\python.exe code\\drcarys_v3_server.py

Uploads may be georeferenced landslide-probability GeoTIFFs (values 0..1) or
the temporal Sen12 NetCDF scenes used by the project's trained model.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

import geopandas as gpd
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

PROJECT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT / "app" / "drcarys-v3"
DIST = APP_DIR / "dist"
RUNTIME = APP_DIR / "runtime"
MASTER_EDGES_PATH = PROJECT / "results" / "chimanimani_study_graph_v2_local_buffer_update" / "road_edges.geojson"
CHECKPOINT = PROJECT / "results" / "sen12_temporal_chimanimani_finetuned" / "best.pt"
WEAK_GMU_GAT = PROJECT / "results" / "weak_gmu_gat_chimanimani" / "best.pt"
DEMO_PROBABILITY_DIR = PROJECT / "results" / "chimanimani_full_post_event_coverage"
THRESHOLD, BUFFER_M = .81, 500.0

app = Flask(__name__, static_folder=None)
MASTER_EDGES = gpd.read_file(MASTER_EDGES_PATH)
PHYSICAL_EDGES = MASTER_EDGES[MASTER_EDGES.edge_id.str.contains(r"_0_s")].copy()


def probability_from_netcdf(path: Path, event_date: str) -> tuple[np.ndarray, object, object]:
    """Run the frozen temporal U-Net on a supported Sen12 NetCDF scene."""
    import rasterio.crs
    import torch
    import xarray as xr
    from rasterio.transform import from_origin
    from roadseg.models.unet import UNet
    from fuse_landslide_road_confidence import tiled_probability

    with xr.open_dataset(path) as data:
        times = data["time"].values.astype("datetime64[ns]")
        event = np.datetime64(event_date)
        before, after = np.flatnonzero(times < event), np.flatnonzero(times > event)
        if not len(before) or not len(after):
            raise ValueError("No pre-event and post-event acquisition bracket the selected event date.")
        pre_index, post_index = int(before[-1]), int(after[0])
        pre = np.stack([data[band].isel(time=pre_index).values for band in ("B02", "B03", "B04", "B08")]).astype(np.float32)
        post = np.stack([data[band].isel(time=post_index).values for band in ("B02", "B03", "B04", "B08")]).astype(np.float32)
        x, y = data["x"].values.astype(float), data["y"].values.astype(float)
        crs = rasterio.crs.CRS.from_string(str(data.attrs["crs"]))
    dx, dy = abs(float(np.median(np.diff(x)))), abs(float(np.median(np.diff(y))))
    raster_transform = from_origin(float(x.min() - dx / 2), float(y.max() + dy / 2), dx, dy)
    image = np.concatenate([(pre / 10000.0 - .5) / .5, (post / 10000.0 - .5) / .5, (post - pre) / 10000.0])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = UNet(int(checkpoint["in_channels"]), 1, int(checkpoint["base_channels"])).float().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return tiled_probability(model, image, torch.device("cpu"), tile=128, overlap=16), raster_transform, crs


def probability_from_tiff(path: Path) -> tuple[np.ndarray, object, object]:
    import rasterio
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError("GeoTIFF uploads must have a projected CRS.")
        values = source.read(1).astype(np.float32)
        if not np.isfinite(values).any() or values.min() < -0.01 or values.max() > 1.01:
            raise ValueError("GeoTIFF must be a landslide-probability raster with values from 0 to 1.")
        return values, source.transform, source.crs


def scene_update(probability: np.ndarray, raster_transform, crs) -> tuple[list[str], dict]:
    """Return local blocked graph-edge IDs and predicted-hazard GeoJSON."""
    from rasterio.features import shapes
    if str(crs) != str(MASTER_EDGES.crs):
        raise ValueError(f"Raster CRS {crs} does not match study graph CRS {MASTER_EDGES.crs}.")
    mask = np.isfinite(probability) & (probability >= THRESHOLD)
    polygons = [shape(geometry).simplify(20, preserve_topology=True)
                for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=raster_transform) if value == 1]
    if not polygons:
        return [], {"type": "FeatureCollection", "features": []}
    hazard_buffer = unary_union(polygons).buffer(BUFFER_M)
    height, width = probability.shape
    left, top = raster_transform * (0, 0)
    right, bottom = raster_transform * (width, height)
    coverage = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
    selected = MASTER_EDGES.geometry.intersects(coverage) & MASTER_EDGES.geometry.intersects(hazard_buffer)
    edge_ids = MASTER_EDGES.loc[selected, "edge_id"].tolist()
    transformer = Transformer.from_crs(crs, 4326, always_xy=True).transform
    hazard_wgs = gpd.GeoDataFrame({"status": ["PREDICTED_LANDSLIDE"] * len(polygons)},
                                  geometry=[transform(transformer, polygon) for polygon in polygons], crs=4326)
    return edge_ids, json.loads(hazard_wgs.to_json())


def _road_feature(edge) -> np.ndarray:
    """Vector feature schema used when the weak GMU+GAT was trained."""
    return np.asarray([np.log1p(float(edge.length_m)), float(edge.speed_kph) / 80.,
                       min(float(getattr(edge, "u_degree", 1)), 8.) / 8.,
                       min(float(getattr(edge, "v_degree", 1)), 8.) / 8., 1.], dtype=np.float32)


def _edge_index(rows) -> torch.Tensor:
    by_endpoint: dict[str, list[int]] = {}
    for index, edge in enumerate(rows):
        by_endpoint.setdefault(edge.u, []).append(index); by_endpoint.setdefault(edge.v, []).append(index)
    pairs = set()
    for members in by_endpoint.values():
        for left in members:
            for right in members: pairs.add((left, right))
    return torch.tensor(sorted(pairs), dtype=torch.long).T if pairs else torch.empty((2, 0), dtype=torch.long)


def gmu_gat_update(probability: np.ndarray, raster_transform, crs) -> tuple[list[str], dict, dict]:
    """Run GMU+GAT only where the raster provides landslide support.

    The learned model refines road risk; it must not create a danger decision
    from vector topology alone.  The support region deliberately matches the
    V2 safety policy: a 500 m buffer around pixels above the segmentation
    threshold.
    """
    from build_weak_road_edge_dataset import probability_features
    from train_weak_gmu_gat import WeakGMUGAT
    from rasterio.features import shapes
    if str(crs) != str(MASTER_EDGES.crs):
        raise ValueError(f"Raster CRS {crs} does not match study graph CRS {MASTER_EDGES.crs}.")
    height, width = probability.shape
    left, top = raster_transform * (0, 0); right, bottom = raster_transform * (width, height)
    coverage = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
    selected = PHYSICAL_EDGES[PHYSICAL_EDGES.geometry.intersects(coverage)].copy()
    if selected.empty:
        return [], {"type": "FeatureCollection", "features": []}, {"risk_threshold": None, "mean_risk": None}
    hazard_mask = np.isfinite(probability) & (probability >= THRESHOLD)
    polygons = [shape(geometry) for geometry, value in shapes(hazard_mask.astype("uint8"), mask=hazard_mask,
                transform=raster_transform) if value == 1]
    if not polygons:
        return [], {"type": "FeatureCollection", "features": []}, {
            "risk_threshold": None, "mean_risk": None, "physical_edges_assessed": len(selected),
            "physical_edges_raster_supported": 0, "physical_edges_danger": 0,
            "decision_policy": "no raster landslide support; all roads remain safe",
        }
    hazard_support = unary_union(polygons).buffer(BUFFER_M)
    selected["raster_supported"] = selected.geometry.intersects(hazard_support)
    raster = np.stack([probability_features(edge.geometry, probability, raster_transform) for edge in selected.itertuples(index=False)])
    vector = np.stack([_road_feature(edge) for edge in selected.itertuples(index=False)])
    checkpoint = torch.load(WEAK_GMU_GAT, map_location="cpu", weights_only=False)
    model = WeakGMUGAT(checkpoint["raster_dim"], checkpoint["vector_dim"]); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad():
        logits, gates = model(torch.from_numpy(raster).float(), torch.from_numpy(vector).float(), _edge_index(list(selected.itertuples(index=False))))
        risk = torch.sigmoid(logits).numpy()
    threshold = float(checkpoint["validation"]["threshold"])
    selected["weak_gmu_gat_risk"] = risk
    # Hard evidence gate: topology and vector attributes can refine only a
    # road piece already supported by a predicted raster hazard.
    selected["is_danger"] = (risk >= threshold) & selected["raster_supported"]
    danger_pieces = selected[selected.is_danger]
    # Preserve the exact 50 m physical piece in both travel directions.  The
    # reverse direction has reversed segment numbering, so identify it from
    # swapped endpoint nodes rather than the edge-id suffix.
    directed_ids = set(danger_pieces.edge_id)
    for edge in danger_pieces.itertuples(index=False):
        reverse = MASTER_EDGES[(MASTER_EDGES.source_edge_index == edge.source_edge_index)
                               & (MASTER_EDGES.u == edge.v) & (MASTER_EDGES.v == edge.u)]
        directed_ids.update(reverse.edge_id)
    directed_ids = MASTER_EDGES.loc[MASTER_EDGES.edge_id.isin(directed_ids), "edge_id"].tolist()
    danger = selected[selected.is_danger].to_crs(4326)
    return directed_ids, json.loads(danger[["edge_id", "weak_gmu_gat_risk", "geometry"]].to_json()), {
        "risk_threshold": threshold, "mean_risk": float(risk.mean()), "mean_raster_gate": float(gates.mean()),
        "physical_edges_assessed": len(selected), "physical_edges_raster_supported": int(selected.raster_supported.sum()),
        "physical_edges_danger": int(selected.is_danger.sum()),
        "decision_policy": "GMU+GAT risk within 500 m raster-supported hazard buffer",
    }


def update_from_probability(probability: np.ndarray, raster_transform, crs, name: str, source: str) -> tuple[list[str], dict, dict]:
    """Produce one transparent raster overlay and one learned road update."""
    _, hazard = scene_update(probability, raster_transform, crs)
    edge_ids, _, fusion = gmu_gat_update(probability, raster_transform, crs)
    scene = {"name": name, "source": source, "blocked_directed_edges": len(edge_ids),
             "probability_max": float(np.nanmax(probability)), "fusion": fusion}
    return edge_ids, hazard, scene


@app.get("/")
def index():
    return send_from_directory(DIST, "index.html")


@app.get("/data/<path:name>")
def data_file(name: str):
    return send_from_directory(DIST / "data", name)


@app.get("/runtime/<path:name>")
def runtime_file(name: str):
    return send_from_directory(RUNTIME, name)


@app.post("/api/analyze")
def analyze():
    uploads = request.files.getlist("rasters")
    if not uploads:
        return jsonify(error="Upload at least one .nc or probability .tif file."), 400
    event_date = request.form.get("event_date", "2019-03-15")
    job_id = uuid.uuid4().hex
    job_dir = RUNTIME / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    all_edges: set[str] = set()
    all_features, scenes = [], []
    for uploaded in uploads:
        filename = secure_filename(uploaded.filename or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".nc", ".tif", ".tiff"}:
            return jsonify(error=f"{filename}: supported formats are .nc, .tif, and .tiff."), 400
        target = job_dir / filename
        uploaded.save(target)
        try:
            if suffix == ".nc":
                probability, raster_transform, crs = probability_from_netcdf(target, event_date)
                source_kind = "temporal_model_inference"
            else:
                probability, raster_transform, crs = probability_from_tiff(target)
                source_kind = "uploaded_probability_raster"
            # V3 uses the learned edge-risk model for the graph update while
            # retaining the raster hazard polygons for transparent display.
            edge_ids, hazard, scene = update_from_probability(probability, raster_transform, crs, filename, source_kind)
            all_edges.update(edge_ids)
            all_features.extend(hazard["features"])
            scenes.append(scene)
        except Exception as error:
            return jsonify(error=f"{filename}: {error}"), 400
    return jsonify(job_id=job_id, threshold=THRESHOLD, buffer_m=BUFFER_M, fusion_method="weak_gmu_gat",
                   risk_threshold=float(torch.load(WEAK_GMU_GAT, map_location="cpu", weights_only=False)["validation"]["threshold"]),
                   scenes=scenes, blocked_edge_ids=sorted(all_edges), hazards={"type": "FeatureCollection", "features": all_features})


@app.post("/api/analyze-demo")
def analyze_demo():
    """Run selected packaged Chimanimani demo scenes through V3 GMU+GAT."""
    scene_ids = (request.get_json(silent=True) or {}).get("scene_ids", [])
    if not isinstance(scene_ids, list) or not scene_ids:
        return jsonify(error="Select at least one demonstration scene."), 400
    all_edges: set[str] = set()
    all_features, scenes = [], []
    for scene_id in scene_ids:
        scene_id = str(scene_id)
        if not re.fullmatch(r"\d+", scene_id):
            return jsonify(error="Invalid demonstration scene ID."), 400
        path = DEMO_PROBABILITY_DIR / f"chimanimani_s2_{scene_id}_landslide_probability.tif"
        if not path.exists():
            return jsonify(error=f"Demonstration scene {scene_id} is unavailable."), 404
        probability, raster_transform, crs = probability_from_tiff(path)
        edge_ids, hazard, scene = update_from_probability(probability, raster_transform, crs, f"Scene {scene_id}", "packaged_demo_probability")
        all_edges.update(edge_ids)
        all_features.extend(hazard["features"])
        scenes.append(scene)
    checkpoint = torch.load(WEAK_GMU_GAT, map_location="cpu", weights_only=False)
    return jsonify(fusion_method="weak_gmu_gat", risk_threshold=float(checkpoint["validation"]["threshold"]),
                   scenes=scenes, blocked_edge_ids=sorted(all_edges),
                   hazards={"type": "FeatureCollection", "features": all_features})


if __name__ == "__main__":
    RUNTIME.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=8766, debug=False)
