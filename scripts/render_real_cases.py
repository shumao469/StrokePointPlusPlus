"""Render a prespecified external-DEV case series, never fit or infer a model.

Run with a private, hash-bound JSON configuration. No patient path or source ID
is embedded in this distributable script. The private configuration/provenance
must not be included in a public package. Requires numpy, nibabel, matplotlib,
scikit-image and optionally pyvista/VTK; CPU offscreen rendering only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import nibabel as nib
from skimage.measure import marching_cubes

ARMS = ("BASE3D", "CONCAT", "YZ_UNGATED", "YZ_GATE")
RENDERING_EVIDENCE = {}
PALETTE = {
    1: ("BA", "#7b3294"), 2: ("R-PCA", "#c51b7d"),
    3: ("L-PCA", "#de77ae"), 4: ("R-ICA", "#2166ac"),
    5: ("R-MCA", "#4393c3"), 6: ("L-ICA", "#b2182b"),
    7: ("L-MCA", "#d6604d"), 8: ("R-Pcom", "#1b9e77"),
    9: ("L-Pcom", "#66a61e"), 10: ("Acom", "#e6ab02"),
    11: ("R-ACA", "#7570b3"), 12: ("L-ACA", "#a6761d"),
    15: ("3rd-A2", "#e7298a")
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound(record):
    path = Path(record["path"]).resolve()
    if sha(path) != record["sha256"]:
        raise ValueError("Bound source hash mismatch")
    return path


def read_bound(record):
    return json.loads(bound(record).read_text(encoding="utf-8-sig"))


def colors(labels):
    if not set(np.unique(labels)).issubset(PALETTE):
        raise ValueError("Unexpected native anatomical label")
    return np.asarray([to_rgb(PALETTE[int(k)][1]) for k in labels])


def camera(center, span):
    direction = np.array([0.15, -1., 1.25])
    direction /= np.linalg.norm(direction)
    return (tuple(center + 3 * span * direction), tuple(center), (0., 0., 1.))


def vtk_render(path, xyz, rgb, center, span, faces=None):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1600))
    plotter.set_background("white")
    if faces is None:
        plotter.add_points(xyz, scalars=(rgb * 255).astype(np.uint8), rgb=True,
                           point_size=9, render_points_as_spheres=True,
                           ambient=.35, diffuse=.65)
    else:
        mesh = pv.PolyData(xyz, faces)
        plotter.add_mesh(mesh, color="#9d273b", smooth_shading=True,
                         ambient=.25, diffuse=.75, specular=.15)
    plotter.camera_position = camera(center, span)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = span * .60
    plotter.camera.clipping_range = (span * .01, span * 12)
    plotter.screenshot(str(path), transparent_background=False)
    capabilities = plotter.render_window.ReportCapabilities()
    renderer = next((line.strip() for line in capabilities.splitlines()
                     if "OpenGL renderer string" in line), "UNAVAILABLE")
    RENDERING_EVIDENCE["OpenGL_renderer"] = renderer
    RENDERING_EVIDENCE["software_renderer_verified"] = any(
        name in renderer.lower() for name in ("llvmpipe", "softpipe", "software"))
    plotter.close()


def mpl_render(path, xyz, rgb, center, span, faces=None):
    fig = plt.figure(figsize=(9, 8), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    if faces is None:
        ax.scatter(*xyz.T, c=rgb, s=6, alpha=1, linewidths=0, depthshade=False)
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        tri = faces.reshape(-1, 4)[:, 1:]
        collection = Poly3DCollection(xyz[tri], facecolors="#9d273b", linewidths=0)
        ax.add_collection3d(collection)
    for axis, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(value - span * .53, value + span * .53)
    ax.view_init(elev=51, azim=-81.5)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def render(path, xyz, rgb, center, span, faces=None):
    try:
        vtk_render(path, xyz, rgb, center, span, faces)
        return "pyvista_VTK_CPU_offscreen"
    except Exception:
        mpl_render(path, xyz, rgb, center, span, faces)
        return "matplotlib_Agg_CPU_fallback"


def main(config_path):
    started = time.monotonic()
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest = read_bound(configuration["manifest"])
    protocol = read_bound(configuration["protocol"])
    review_inputs = read_bound(configuration["review_inputs"])
    numerical_review = read_bound(configuration["independent_review"])
    prediction_lock = read_bound(configuration["prediction_lock"])
    if not numerical_review["qualified"] or numerical_review["mismatches"]:
        raise ValueError("Numerical review must already be complete")
    if not prediction_lock["locked"]:
        raise ValueError("All predictions must already be locked")
    selected = manifest["cases"][:4]
    if len(selected) != 4 or any(c["role"] != "externalDEV" for c in selected):
        raise ValueError("Exactly first four fixed-order DEV cases required")
    if any(c["source_id"] != protocol["cases"][i]["source_id"] for i, c in enumerate(selected)):
        raise ValueError("Protocol and fixed manifest order differ")
    output = Path(configuration["output_dir"]).resolve()
    provenance_path = Path(configuration["provenance_output"]).resolve()
    if "private_build" not in provenance_path.parts:
        raise ValueError("Identifiable provenance must remain private")
    output.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = bound(review_inputs["review_arrays"])
    cases, inputs, products = [], [], []
    with np.load(archive_path, allow_pickle=False) as arrays:
        for i, case in enumerate(selected):
            label = "Case" + "ABCD"[i]
            entry = review_inputs["cases"][i]
            if entry["patient"] != case["patient"]:
                raise ValueError("Reviewed point-label case correspondence")
            key = entry["key_prefix"]
            mapping_path, common_path = bound(case["mapping"]), bound(case["common"])
            with np.load(mapping_path, allow_pickle=False) as mapping:
                xyz = mapping["point_world"].astype(float)
                native_ijk = mapping["point_voxel_IJK"].copy()
                affine = mapping["affine"].copy()
                np.testing.assert_array_equal(native_ijk, arrays[key + "__point_voxel_IJK"])
                np.testing.assert_allclose(mapping["point_xyz"], arrays[key + "__point_xyz"], atol=1e-6)
                np.testing.assert_allclose(nib.affines.apply_affine(affine, native_ijk), xyz, atol=1e-4)
            with np.load(common_path, allow_pickle=False) as common:
                s1, s2 = common["s1"].copy(), common["s2"].copy()
            gt = arrays[key + "__native_y"].copy()
            predictions = {arm: arrays[key + "__prediction__" + arm].copy() for arm in ARMS}
            if xyz.shape != (2048, 3) or gt.shape != (2048,):
                raise ValueError("Full shared2048-point sample required")
            cases.append(dict(label=label, xyz=xyz, truth=gt, predictions=predictions, s1=s1, s2=s2))
            inputs.append(dict(display_label=label, source_id=case["source_id"], role=case["role"],
                               mapping=case["mapping"], common=case["common"], review_key=key,
                               point_count=2048, point_world_sha256=hashlib.sha256(xyz.tobytes()).hexdigest()))
    span = max(float(np.ptp(c["xyz"], axis=0).max()) for c in cases) * 1.04
    for case in cases:
        center = (case["xyz"].max(0) + case["xyz"].min(0)) * .5
        case["center"] = center
        for variant, labels in [("truth", case["truth"]), *case["predictions"].items()]:
            path = output / (case["label"] + "_" + variant + ".png")
            backend = render(path, case["xyz"], colors(labels), center, span)
            products.append(dict(file=path.name, sha256=sha(path), backend=backend,
                                 display_label=case["label"], point_count=2048))
            print("rendered", path.name, flush=True)
    first = cases[0]
    for level, index, color in (("s1", first["s1"], "#377eb8"),
                                 ("s2", first["s1"][first["s2"]], "#9d273b")):
        path = output / ("CaseA_" + level + ".png")
        selected_points = first["xyz"][index]
        backend = render(path, selected_points, np.tile(to_rgb(color), (len(index), 1)), first["center"], span)
        products.append(dict(file=path.name, sha256=sha(path), backend=backend, point_count=len(index)))
    source = protocol["cases"][0]
    image_path, mask_path = bound(source["image"]), bound(source["native_mask"])
    image, mask = nib.load(image_path), nib.load(mask_path)
    np.testing.assert_allclose(image.affine, mask.affine, atol=1e-6)
    raw = np.asarray(image.dataobj, dtype=np.float32)
    foreground = np.asarray(mask.dataobj) > 0
    if raw.shape != foreground.shape:
        raise ValueError("Raw MRA/given foreground correspondence")
    coordinates = np.argwhere(foreground)
    lo, hi = coordinates.min(0), coordinates.max(0) + 1
    padded_lo, padded_hi = np.maximum(lo - 2, 0), np.minimum(hi + 2, foreground.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(padded_lo, padded_hi))
    vertices, faces, _, _ = marching_cubes(foreground[sl].astype(np.float32), .5)
    vertices = nib.affines.apply_affine(image.affine, vertices + padded_lo)
    faces = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    path = output / "CaseA_surface.png"
    backend = render(path, vertices, None, first["center"], span, faces)
    products.append(dict(file=path.name, sha256=sha(path), backend=backend,
                         source="Given foreground union, unsmoothed marching-cubes geometry", vertices=len(vertices)))
    # Explicit source-image MIPs, cropped only by given foreground bounding box.
    margin = np.ceil(12 / nib.affines.voxel_sizes(image.affine)).astype(int)
    mlo, mhi = np.maximum(lo - margin, 0), np.minimum(hi + margin, raw.shape)
    crop = raw[tuple(slice(int(a), int(b)) for a, b in zip(mlo, mhi))]
    low, high = np.percentile(crop[np.isfinite(crop)], (1, 99.7))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), dpi=200, facecolor="white")
    voxel_mm = nib.affines.voxel_sizes(image.affine)
    for axis, dim, title in zip(axes, (0, 1, 2), ("Sagittal MIP", "Coronal MIP", "Axial MIP")):
        remaining_axes = [k for k in range(3) if k != dim]
        physical_extent = (0, crop.shape[remaining_axes[0]] * voxel_mm[remaining_axes[0]],
                           0, crop.shape[remaining_axes[1]] * voxel_mm[remaining_axes[1]])
        axis.imshow(np.max(crop, axis=dim).T, origin="lower", cmap="gray", vmin=low, vmax=high,
                    extent=physical_extent, aspect="equal")
        axis.set_title(title, fontsize=12, color="#252525")
        axis.set_axis_off()
    fig.tight_layout(pad=.7)
    path = output / "CaseA_mra.png"
    fig.savefig(path, facecolor="white", dpi=200)
    plt.close(fig)
    products.append(dict(file=path.name, sha256=sha(path), backend="matplotlib_Agg_CPU",
                         crop="Given foreground bounding box plus12mm", intensity_window_percentiles=[1, 99.7]))
    provenance = dict(version="FIXED_FIRST4_REVIEWED_EXTERNAL_DEV_RENDER_V1", inputs=inputs,
        renderer_source_sha256=sha(__file__), reviewed_arrays=review_inputs["review_arrays"],
        rendering_evidence=RENDERING_EVIDENCE,
        private_configuration_sha256=sha(config_path), original_mra=source["image"],
        given_foreground_source_mask=source["native_mask"], metadata_bindings={k: configuration[k] for k in
            ("manifest", "protocol", "review_inputs", "independent_review", "prediction_lock")},
        products=products, class_palette={str(k): dict(name=v[0], hex=v[1]) for k, v in PALETTE.items()},
        camera=dict(direction=[.15, -1, 1.25], parallel_projection=True, shared_span_mm=span,
                    centering="Per-case geometry bounding box only; same camera and scale for all methods/cases"),
        selection="First four frozen manifest cases, no outcome-based selection", new_fits=0,
        model_forwards=0, FINAL_raw_reads=0, original65_reads=0, raw_cases_decoded=1,
        elapsed_seconds=time.monotonic()-started)
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(products=len(products), elapsed_seconds=provenance["elapsed_seconds"], new_fits=0, model_forwards=0)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    main(parser.parse_args().config)
