"""Illustrate one fixed saved Euclidean PointNet++ group; no recomputed KNN.

Uses a private case-render configuration and its already completed provenance.
The original renderer/source and its24 illustrations remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv
from render_real_cases import bound, camera, read_bound, sha


def main(config_path, previous_provenance):
    start = time.monotonic()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    previous = json.loads(Path(previous_provenance).read_text(encoding="utf-8"))
    assert sha(config_path) == previous["private_configuration_sha256"]
    assert sha(Path(__file__).with_name("render_real_cases.py")) == previous["renderer_source_sha256"]
    manifest = read_bound(config["manifest"])
    case = manifest["cases"][0]
    assert case["role"] == "externalDEV"
    assert previous["inputs"][0]["source_id"] == case["source_id"]
    with np.load(bound(case["mapping"]), allow_pickle=False) as data:
        xyz = data["point_world"].astype(float)
    with np.load(bound(case["common"]), allow_pickle=False) as data:
        s1 = data["s1"].copy()
        g1 = data["g1"].copy()
    assert xyz.shape == (2048, 3) and s1.shape == (128,) and g1.shape == (128, 32)
    query = int(s1[0])
    neighbor_indices = g1[0]
    assert np.all((neighbor_indices >= 0) & (neighbor_indices < len(xyz)))
    center = (xyz.max(0) + xyz.min(0)) * .5
    span = previous["camera"]["shared_span_mm"]
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1600))
    plotter.set_background("white")
    plotter.add_points(xyz, color="#bfc3c9", opacity=.28, point_size=5,
                       render_points_as_spheres=True)
    plotter.add_points(xyz[s1], color="#86b5d7", point_size=11,
                       render_points_as_spheres=True)
    # Lines are only the saved query-to-neighbor grouping relations, not vessels.
    line_points = np.vstack((xyz[query], xyz[neighbor_indices]))
    line_indices = np.column_stack((np.full(32, 2), np.zeros(32, dtype=int), np.arange(1, 33)))
    line_data = pv.PolyData(line_points)
    line_data.lines = line_indices.ravel()
    plotter.add_mesh(line_data, color="#cd9717", line_width=2.5, opacity=.85)
    plotter.add_points(xyz[neighbor_indices], color="#e4ad25", point_size=14,
                       render_points_as_spheres=True)
    plotter.add_points(xyz[query:query+1], color="#b2182b", point_size=24,
                       render_points_as_spheres=True)
    plotter.camera_position = camera(center, span)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = span * .60
    plotter.camera.clipping_range = (span * .01, span * 12)
    output = Path(config["output_dir"]) / "CaseA_grouping.png"
    plotter.screenshot(str(output), transparent_background=False)
    renderer = next((line.strip() for line in plotter.render_window.ReportCapabilities().splitlines()
                     if "OpenGL renderer string" in line), "UNAVAILABLE")
    plotter.close()
    assert any(k in renderer.lower() for k in ("llvmpipe", "softpipe", "software"))
    private_output = Path(previous_provenance).with_name("grouping_render_provenance.json")
    assert "private_build" in private_output.parts
    result = dict(version="FIXED_QUERY_S1ROW0_EUCLIDEAN_GROUP_RENDER_V1", display_label="CaseA",
        source_id=case["source_id"], common=case["common"], mapping=case["mapping"],
        previous_provenance_sha256=sha(previous_provenance), source_sha256=sha(__file__),
        parent_renderer_sha256=previous["renderer_source_sha256"], full_points=2048, anchors=128,
        query_anchor_row=0, query_full_point_index=query, neighbor_slots=32,
        unique_neighbor_points=int(np.unique(neighbor_indices).size),
        self_query_slots=int((neighbor_indices == query).sum()),
        neighbor_full_point_indices=neighbor_indices.tolist(), camera=previous["camera"],
        grouping="Saved Euclidean g1[0], not vessel connectivity or new nearest-neighbor computation",
        output_file=output.name, output_sha256=sha(output), OpenGL_renderer=renderer,
        new_fits=0, model_forwards=0, new_neighbor_searches=0, raw_image_reads=0,
        FINAL_reads=0, original65_reads=0, elapsed_seconds=time.monotonic()-start)
    private_output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("output_file", "output_sha256", "neighbor_slots", "unique_neighbor_points", "self_query_slots", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--previous-provenance", required=True, type=Path)
    arguments = parser.parse_args()
    main(arguments.config, arguments.previous_provenance)
