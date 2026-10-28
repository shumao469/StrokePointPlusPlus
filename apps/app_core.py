"""Small, torch-free research-demo data and upload validation helpers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import math
import stat
import zipfile
import zlib

import numpy as np

ARMS = ("BASE3D", "CONCAT", "YZ_UNGATED", "YZ_GATE")
DISPLAY_NAMES = {
    "BASE3D": "StrokePoint++-3D",
    "CONCAT": "StrokePoint++-Fuse",
    "YZ_UNGATED": "Frozen PointNet++ · ungated",
    "YZ_GATE": "StrokePoint++-Gate",
}
RARE_BRANCHES = ("R-Pcom", "L-Pcom", "Acom", "3rd-A2")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_POINTS = 100_000
MAX_HEADER_BYTES = 16_384
MAX_RENDER_POINTS = 25_000


def contained_path(root: Path, candidate: Path) -> Path:
    root, candidate = root.resolve(), candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Asset must remain inside the release directory")
    return candidate


def find_summary(release: Path) -> Path | None:
    for folder in ("results/aggregate", "data/aggregate"):
        candidate = release / folder / "DEV_FINAL_four_arm_summary_v1.json"
        if candidate.is_file():
            return contained_path(release, candidate)
        dev, final = release/folder/"externalDEV_Lausanne.json", release/folder/"externalFINAL_IXI_HH.json"
        if dev.is_file() and final.is_file():
            contained_path(release, final)
            return contained_path(release, dev)
    return None


def _count_from_patient_mean(statistic: dict, n: int):
    value = statistic.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Not available"
    total = float(value)*n
    if abs(total-round(total)) > 1e-6:
        raise ValueError("Reviewed count mean is inconsistent with its patient denominator")
    return int(round(total))


def _cohort_from_packaged_wrapper(wrapper: dict, role: str) -> dict:
    review, statistics = wrapper.get("review", {}), wrapper.get("statistics", {})
    if review.get("qualified") is not True or review.get("mismatches") not in ([], 0):
        raise ValueError("A qualified independent review is required")
    n = int(wrapper["patient_count"])
    if n != statistics.get("n_patients") or statistics.get("external_role") != role:
        raise ValueError("Cohort role/patient denominator mismatch")
    arms = {}
    for arm in ARMS:
        source = statistics["summary"][arm]
        rare = {}
        for name, label in zip(RARE_BRANCHES, (8, 9, 10, 15)):
            r = next(row for row in statistics["per_class"][arm] if row["label"] == label)
            rare[name] = dict(full_native_positive_patients=r.get("full_native_positive_patients"),
                sampled_positive_patients=r["positive_patients"],
                native_positive_but_sampled_zero_patients=r.get("native_positive_but_sampled_zero_patients"),
                absent_FPR_patient_denominator=r.get("full_native_negative_patients"),
                point_recall=r["recall"],native_absent_FPR=r.get("absent_FPR"))
        refs = {}
        for ref in ("BASE3D", "YZ_UNGATED"):
            event = statistics["patient_event"][arm][ref]
            refs[ref] = dict(corrected_points=_count_from_patient_mean(source.get("corrected_vs"+ref,{}),n),
                new_error_points=_count_from_patient_mean(source.get("new_errors_vs"+ref,{}),n),
                rare_true_positive_points_lost=_count_from_patient_mean(source.get("rare_TP_lost_vs"+ref,{}),n),
                changed_patients="Not available in aggregate",
                population_one_sided95_CP_risk_upper=event["one_sided95_CP_upper"])
        arms[arm] = dict(point_macroDice=source["Dice"],patientmean_native_absent_FPR=source["native_absent_FPR"],
                         rare_branches=rare,correction_and_incremental_error_counts=refs)
    return dict(independently_reviewed=True,n_patients=n,original_arm_statistics=arms,
                original_primary_contrasts=statistics["contrasts"])


def load_public_summary(file: Path) -> dict:
    if file.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Packaged aggregate JSON exceeds the display limit")
    data = json.loads(file.read_text(encoding="utf-8"))
    if "statistics" in data and file.name == "externalDEV_Lausanne.json":
        final_file=contained_path(file.parent,file.parent/"externalFINAL_IXI_HH.json")
        if final_file.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("Packaged FINAL aggregate exceeds the display limit")
        final_wrapper=json.loads(final_file.read_text(encoding="utf-8"))
        data={"cohorts":{"Lausanne_DEV":_cohort_from_packaged_wrapper(data,"DEV"),
                         "IXI_HH_FINAL":_cohort_from_packaged_wrapper(final_wrapper,"FINAL")}}
    cohorts = data.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("Expected reviewed cohort aggregates")
    for cohort in cohorts.values():
        if cohort.get("independently_reviewed") is not True:
            raise ValueError("Unreviewed aggregates must not be displayed")
        if set(cohort.get("original_arm_statistics", {})) != set(ARMS):
            raise ValueError("All four fixed comparison arms must be retained")
    return data


def statistic_text(value, *, percent: bool = True) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("value"), (int, float)):
        return "Not available / 未判定"
    point = float(value["value"])
    if not math.isfinite(point):
        return "Not available / 未判定"
    scale, suffix = (100., "%") if percent else (1., "")
    result = f"{point * scale:.2f}{suffix}"
    interval = value.get("CI")
    if interval is None:
        return result + " · CI undefined"
    if len(interval) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in interval):
        return result + " · CI undefined"
    return result + f" [{interval[0] * scale:.2f}, {interval[1] * scale:.2f}]"


def support_text(value) -> str:
    return str(value) if isinstance(value, (int, float)) and math.isfinite(value) else "Not available"


def _npy_header(member) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=MAX_HEADER_BYTES)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=MAX_HEADER_BYTES)
    else:
        raise ValueError("Only ordinary NPY v1/v2 arrays are accepted")
    if dtype.hasobject or dtype.kind not in "fiu" or dtype.fields is not None:
        raise ValueError("Only plain numeric arrays are accepted; object/pickle data are forbidden")
    if any(not isinstance(n, int) or n < 0 for n in shape):
        raise ValueError("Invalid array dimensions")
    if math.prod(shape) * dtype.itemsize > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Declared array allocation exceeds the limit")
    return shape, dtype


def validate_pointcloud_npz(payload: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate ZIP and NPY headers before NumPy allocation; never extract/write."""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_UPLOAD_BYTES:
        raise ValueError("Select an NPZ file no larger than 8 MiB")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            items = archive.infolist()
            names = [item.filename for item in items]
            if not items or len(items) > 2 or len(set(names)) != len(names):
                raise ValueError("Expected one point array and at most one labels array")
            if not set(names) <= {"xyz.npy", "points.npy", "labels.npy"}:
                raise ValueError("Only xyz/points and optional labels arrays are permitted")
            point_names = set(names) & {"xyz.npy", "points.npy"}
            if len(point_names) != 1:
                raise ValueError("Provide exactly one of xyz or points")
            if sum(item.file_size for item in items) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Uncompressed archive exceeds 16 MiB")
            headers = {}
            for item in items:
                if item.flag_bits & 1 or item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Encrypted files, directories and links are forbidden")
                with archive.open(item) as member:
                    headers[item.filename] = _npy_header(member)
            point_name = next(iter(point_names))
            shape, _ = headers[point_name]
            if len(shape) != 2 or shape[1] != 3 or not 1 <= shape[0] <= MAX_POINTS:
                raise ValueError("Coordinates must have shape N×3, with 1–100,000 points")
            if "labels.npy" in headers:
                label_shape, label_dtype = headers["labels.npy"]
                if label_shape != (shape[0],) or label_dtype.kind not in "iu":
                    raise ValueError("Optional labels must be a one-dimensional integer array of length N")
        with np.load(BytesIO(payload), allow_pickle=False, max_header_size=MAX_HEADER_BYTES) as arrays:
            points = np.asarray(arrays[point_name[:-4]], dtype=np.float32)
            labels = np.array(arrays["labels"], copy=True) if "labels" in arrays else None
        if not np.isfinite(points).all() or np.abs(points).max() > 1e8:
            raise ValueError("Coordinates must be finite and within the supported numerical range")
        if labels is not None and ((labels < 0).any() or (labels > 255).any()):
            raise ValueError("Display labels must be integers between 0 and 255")
        return points, labels
    except (zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error, OSError, EOFError, OverflowError, TypeError, RuntimeError) as exc:
        raise ValueError("Invalid or unsupported numeric NPZ archive") from exc


def display_subset(points: np.ndarray, labels: np.ndarray | None):
    indices = np.linspace(0, len(points)-1, min(len(points), MAX_RENDER_POINTS), dtype=int)
    return points[indices], (labels[indices] if labels is not None else None)


def packaged_images(release: Path, folder: str) -> list[Path]:
    directory = contained_path(release, release/folder)
    return sorted(contained_path(release, p) for p in directory.glob("*.png") if p.is_file())
