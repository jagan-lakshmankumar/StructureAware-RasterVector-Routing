"""Smoke-test the live DR-Carys probability-raster fusion endpoint."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from drcarys_v3_server import PROJECT, app, gmu_gat_update, probability_from_tiff


def main() -> None:
    fixture = PROJECT / "results/chimanimani_full_post_event_coverage/chimanimani_s2_1072_landslide_probability.tif"
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/data/dashboard_data.js").status_code == 200
    with fixture.open("rb") as first:
        response = client.post("/api/analyze", data={"rasters": [(io.BytesIO(first.read()), fixture.name)]},
                               content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    one = response.get_json()
    assert one["fusion_method"] == "weak_gmu_gat"
    assert one["scenes"][0]["source"] == "uploaded_probability_raster"
    assert one["blocked_edge_ids"], one
    probability, transform, crs = probability_from_tiff(fixture)
    blocked, _, fusion = gmu_gat_update(np.zeros_like(probability), transform, crs)
    assert blocked == []
    assert fusion["physical_edges_danger"] == 0
    response = client.post("/api/analyze-demo", json={"scene_ids": ["1072"]})
    assert response.status_code == 200, response.get_json()
    demo = response.get_json()
    assert demo["fusion_method"] == "weak_gmu_gat"
    assert demo["scenes"][0]["source"] == "packaged_demo_probability"
    assert set(demo["blocked_edge_ids"]) == set(one["blocked_edge_ids"])
    # Every danger piece must have only its opposite-direction twin, never
    # every piece of the same parent OSM road.
    assert len(one["blocked_edge_ids"]) == 2 * one["scenes"][0]["fusion"]["physical_edges_danger"]
    with fixture.open("rb") as first, fixture.open("rb") as second:
        response = client.post("/api/analyze", data={"rasters": [(io.BytesIO(first.read()), fixture.name),
                                                                   (io.BytesIO(second.read()), "second_" + fixture.name)]},
                               content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    two = response.get_json()
    assert len(two["scenes"]) == 2 and set(two["blocked_edge_ids"]) == set(one["blocked_edge_ids"])
    temporal = PROJECT / "dataset/Sen12Landslides_S2_14GB/s2/chimanimani_s2_1072.nc"
    with temporal.open("rb") as source:
        response = client.post("/api/analyze", data={"rasters": [(io.BytesIO(source.read()), temporal.name)]},
                               content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    temporal_result = response.get_json()
    assert temporal_result["scenes"][0]["source"] == "temporal_model_inference"
    print({"single_scene_blocked_edges": len(one["blocked_edge_ids"]), "demo_scene_count": len(demo["scenes"]), "multi_scene_count": len(two["scenes"]),
           "temporal_model_blocked_edges": len(temporal_result["blocked_edge_ids"])})


if __name__ == "__main__":
    main()
