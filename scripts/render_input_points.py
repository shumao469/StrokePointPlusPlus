"""Render the same fixed DEV point input without any anatomical identity color."""
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib.colors import to_rgb
from render_real_cases import bound, read_bound, render, sha


def main(config_path, previous_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    assert sha(config_path) == previous["private_configuration_sha256"]
    assert sha(Path(__file__).with_name("render_real_cases.py")) == previous["renderer_source_sha256"]
    case = read_bound(config["manifest"])["cases"][0]
    assert case["role"] == "externalDEV" and case["source_id"] == previous["inputs"][0]["source_id"]
    with np.load(bound(case["mapping"]), allow_pickle=False) as mapping:
        xyz = mapping["point_world"].astype(float)
    assert xyz.shape == (2048, 3)
    center = (xyz.max(0) + xyz.min(0)) * .5
    output = Path(config["output_dir"]) / "CaseA_input.png"
    backend = render(output, xyz, np.tile(to_rgb("#959ca5"), (2048, 1)),
                     center, previous["camera"]["shared_span_mm"])
    target = Path(previous_path).with_name("input_render_provenance.json")
    assert "private_build" in target.parts
    provenance = dict(display_label="CaseA", source_id=case["source_id"], mapping=case["mapping"],
        previous_provenance_sha256=sha(previous_path), source_sha256=sha(__file__),
        parent_renderer_sha256=previous["renderer_source_sha256"], camera=previous["camera"],
        output_file=output.name, output_sha256=sha(output), point_count=2048,
        identity_labels_or_predictions_read=False, raw_images_read=0, new_fits=0,
        model_forwards=0, FINAL_reads=0, original65_reads=0, backend=backend)
    target.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("output_file", "output_sha256", "point_count")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--previous-provenance", required=True, type=Path)
    args = parser.parse_args()
    main(args.config, args.previous_provenance)
