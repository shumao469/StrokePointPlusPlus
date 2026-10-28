"""NC01-R0 v2 Y/Z reconstruction: privacy-adapted public source.

Publication adaptation replaces the ordered source-ID allowlist with its SHA256
commitment. Exact historical binding, row counts, estimators and numerical
operations are unchanged; this file is not byte-identical to the local original.

Pure interfaces only: no import-time IO, patient access, checkpoint loading,
cross-validation, optimizer steps, or model fitting. `fit_two_heads` fits
ONLY when explicitly called with allow_fitting=True and a caller ledger
callback. It accepts passed arrays, never file paths. The root protocol owns
all raw/geometry/checkpoint access, scope gates, budgets, and serialization.

Historical source detail must not be silently repaired: Y training used each
of48 patients' OWN outerfold two-seed mean F/L/cal; unseen external inference
used allfour fold/seed models. Mean F/L averages independently trained hidden
coordinates. This module preserves that archived recipe without declaring
those coordinates aligned or changing the method to a new fusion recipe.

Segment embeddings are [meanF,maxF,meanL,maxL], each64 dimensions. Patient
features are coordinatewise max over DBSCAN segments, over Pcom sides, and
over CT/MR. Comments in oldY said average modalities, but executed code was
max; max is reproduced. Existence targets use complete native annotations,
not sampled-point absence. Targets never enter feature construction.

V2 adds one required root-owned raw_state_save callback immediately after each
successful clf.fit return, before any post-fit assertion or replay. The original
v1 helper remains unchanged. This is an execution-preservation repair, not a
changed estimator, extra optimization, threshold change, or numerical certificate.

There are exactly TWO logistic fits, no CV fits and no new threshold search.
Frozen thresholds .05/.05 come from the sealed FP0Y training_info. This is
a reconstruction of the deployed final estimators, not a rerun of historical
CV. Fixed thresholds are not a newly validated external calibration.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit


LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)
PCOM_INDICES = (7, 8)
A2_INDEX = 12
FEATURE_DIM = 256
SEEDS = (20261020, 20261021)
FIXED_THRESHOLDS = {"pcom": .05, "a2": .05}
HEAD_ORDER = ("pcom", "a2")
EXPECTED_TRAINING_COUNTS = {"patients": 48, "pcom_positive": 30, "a2_positive": 7}
ORIGINAL48_PATIENT_BINDING_SHA256 = "59e8ca830a33562b52d896139b35db9abd0c5bb26450451659aee3e842e7378a"
# Commitment encoding: ordered JSON string array, ensure_ascii=True,
# separators=(",", ":"), ASCII bytes, with no trailing newline.
def _matches_original48(patient_ids: Sequence[str]) -> bool:
    patients = tuple(patient_ids)
    if len(patients) != 48 or any(not isinstance(p, str) for p in patients):
        return False
    payload = json.dumps(list(patients), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest() == ORIGINAL48_PATIENT_BINDING_SHA256


DBSCAN_EPS = .015
DBSCAN_MIN_SAMPLES = 3


def _finite_array(value: Any, shape: tuple[int, ...] | None = None,
                  name: str = "array") -> np.ndarray:
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite floating data")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} shape {a.shape}, expected {shape}")
    return a


def aggregate_scan_sources(packets: Sequence[Mapping[str, Any]],
                           expected_sources: Sequence[tuple[int, int]],
                           expected_n_points: int = 2048) -> dict[str, np.ndarray]:
    """Mean the exact ordered source list, preserving historical float32 math.

    Root must pass [(ownfold,seed20),(ownfold,seed21)] for a training scan,
    and [(0,seed20),(0,seed21),(1,seed20),(1,seed21)] externally. This helper
    validates the order but does not infer a patient's training ownership.
    """
    expected = [(int(f), int(s)) for f, s in expected_sources]
    actual = [(int(p["fold"]), int(p["seed"])) for p in packets]
    if actual != expected or len(actual) not in (2, 4) or len(set(actual)) != len(actual):
        raise ValueError("Exact ordered two-seed training or four-source external binding required")
    if len(actual) == 2 and (actual[0][0] != actual[1][0] or tuple(s for _, s in actual) != SEEDS):
        raise ValueError("Training requires same-fold seed20,seed21")
    if len(actual) == 4 and actual != [(f, s) for f in (0, 1) for s in SEEDS]:
        raise ValueError("External source order must exactly match archived fold/seed iteration")
    if expected_n_points < 1:
        raise ValueError("Positive point count required")
    fields = {"cal": 13, "F": 64, "L": 64}
    result = {}
    for key, columns in fields.items():
        arrays = [_finite_array(p[key], (expected_n_points, columns), key) for p in packets]
        if any(a.dtype != np.float32 for a in arrays):
            raise ValueError("Historical F/L/cal ensemble inputs must be float32; no silent fp64 repair")
        result[key] = np.mean(arrays, axis=0)
        if result[key].dtype != np.float32:
            raise RuntimeError("Archived mean dtype changed")
    return result


def detect_segments(xyz: np.ndarray, pred: np.ndarray, class_index: int) -> list[np.ndarray]:
    """The original sklearn DBSCAN rule, without labels or GT candidate IDs."""
    from sklearn.cluster import DBSCAN
    coordinates = _finite_array(xyz, name="xyz")
    p = np.asarray(pred)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or p.shape != (len(coordinates),):
        raise ValueError("Expected xyz[N,3] and predicted class-index[N]")
    if not np.issubdtype(p.dtype, np.integer) or np.any((p < 0) | (p >= 13)):
        raise ValueError("Predictions must be13-class indices")
    if class_index not in range(13):
        raise ValueError("Unknown class index")
    mask = p == class_index
    if int(mask.sum()) < DBSCAN_MIN_SAMPLES:
        return []
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                    metric="euclidean").fit(coordinates[mask]).labels_
    point_indices = np.where(mask)[0]
    return [point_indices[labels == sid] for sid in range(int(labels.max()) + 1)
            if int((labels == sid).sum()) >= DBSCAN_MIN_SAMPLES]


def segment_embedding(F: np.ndarray, L: np.ndarray, segment: np.ndarray) -> np.ndarray:
    ff = _finite_array(F, name="F")
    ll = _finite_array(L, ff.shape, "L")
    if ff.ndim != 2 or ff.shape[1] != 64:
        raise ValueError("F,L must be N,64")
    indices = np.asarray(segment)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Segment indices must be one-dimensional integers")
    if len(indices) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float64)
    if np.any((indices < 0) | (indices >= len(ff))):
        raise ValueError("Segment index outside scan")
    f, l = ff[indices], ll[indices]
    # No pre-cast: mean/max precede float64 exactly as archivedY.
    return np.concatenate((f.mean(0), f.max(0), l.mean(0), l.max(0))).astype(np.float64)


def patient_embedding(F: np.ndarray, L: np.ndarray, xyz: np.ndarray,
                      pred: np.ndarray, class_index: int) -> np.ndarray:
    segments = detect_segments(xyz, pred, class_index)
    if not segments:
        return np.zeros(FEATURE_DIM, dtype=np.float64)
    return np.max([segment_embedding(F, L, s) for s in segments], axis=0)


def scan_features(xyz: np.ndarray, cal: np.ndarray, F: np.ndarray,
                  L: np.ndarray) -> dict[str, np.ndarray]:
    coordinates = _finite_array(xyz, name="xyz")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("xyz must be N,3")
    logits = _finite_array(cal, (len(coordinates), 13), "cal")
    _finite_array(F, (len(coordinates), 64), "F")
    _finite_array(L, (len(coordinates), 64), "L")
    pred = logits.argmax(-1)
    return dict(pcom=np.maximum(patient_embedding(F, L, coordinates, pred, 7),
                                patient_embedding(F, L, coordinates, pred, 8)),
                a2=patient_embedding(F, L, coordinates, pred, A2_INDEX))


def pool_patient_scan_features(features_by_modality: Mapping[str, Mapping[str, np.ndarray]],
                               require_both_modalities: bool = True) -> dict[str, np.ndarray]:
    mods = tuple(m for m in ("ct", "mr") if m in features_by_modality)
    if set(features_by_modality) - {"ct", "mr"} or not mods:
        raise ValueError("Only explicit ct/mr scan features accepted")
    if require_both_modalities and mods != ("ct", "mr"):
        raise ValueError("Archived training recipe requires both CT and MR")
    return {head: np.max([_finite_array(features_by_modality[m][head], (FEATURE_DIM,), head)
                         for m in mods], axis=0) for head in HEAD_ORDER}


def native_presence_targets(native_present_by_modality: Mapping[str, Sequence[int]],
                             require_both_modalities: bool = True) -> dict[str, int]:
    """Target-only function; never called by scan_features or inference pooling."""
    mods = tuple(m for m in ("ct", "mr") if m in native_present_by_modality)
    if set(native_present_by_modality) - {"ct", "mr"} or not mods:
        raise ValueError("Native presence needs explicit ct/mr provenance")
    if require_both_modalities and mods != ("ct", "mr"):
        raise ValueError("Complete paired native-presence labels required")
    present: set[int] = set()
    for mod in mods:
        current = set(int(v) for v in native_present_by_modality[mod])
        if not current.issubset(set(LABELS)):
            raise ValueError("Native presence uses canonical13 vessel labels, not indices")
        present.update(current)
    return dict(pcom=int(8 in present or 9 in present), a2=int(15 in present))


def build_training_arrays(patient_ids: Sequence[str],
                           scan_feature_map: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
                           native_present_map: Mapping[tuple[str, str], Sequence[int]],
                           enforce_historical_counts: bool = True) -> dict[str, np.ndarray]:
    patients = [str(p) for p in patient_ids]
    if patients != sorted(set(patients)) or any(len(p) != 3 or not p.isdigit() for p in patients):
        raise ValueError("Patient IDs must be unique sorted canonical3-digit strings")
    if enforce_historical_counts and not _matches_original48(patients):
        raise ValueError("R0 reconstruction must use the frozen original48 patients")
    expected_keys = {(p, m) for p in patients for m in ("ct", "mr")}
    if set(scan_feature_map) != expected_keys or set(native_present_map) != expected_keys:
        raise ValueError("Exactly the bound48x2 feature and native-label records required")
    x = {h: [] for h in HEAD_ORDER}
    y = {h: [] for h in HEAD_ORDER}
    for pid in patients:
        visible = pool_patient_scan_features({m: scan_feature_map[pid, m] for m in ("ct", "mr")})
        targets = native_presence_targets({m: native_present_map[pid, m] for m in ("ct", "mr")})
        for h in HEAD_ORDER:
            x[h].append(visible[h]); y[h].append(targets[h])
    out = {f"X_{h}": np.asarray(x[h], np.float64) for h in HEAD_ORDER}
    out.update({f"y_{h}": np.asarray(y[h], np.int64) for h in HEAD_ORDER})
    out["patient_ids"] = np.asarray(patients)
    out["groups"] = np.asarray([f"tc_{p}" for p in patients])
    if enforce_historical_counts:
        for h in HEAD_ORDER:
            if int(out[f"y_{h}"].sum()) != EXPECTED_TRAINING_COUNTS[f"{h}_positive"]:
                raise ValueError(f"{h} label counts do not reproduce historical native existence")
    return out


def _digest_arrays(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str(a.dtype).encode("ascii")); digest.update(str(a.shape).encode("ascii")); digest.update(a.tobytes())
    return digest.hexdigest()


def predict_serialized(head: Mapping[str, Any], X: np.ndarray) -> dict[str, np.ndarray]:
    """Stable executable sklearn-equivalent binary head without pickle or fit."""
    xx = _finite_array(X, name="X").astype(np.float64, copy=False)
    if xx.ndim != 2 or xx.shape[1] != FEATURE_DIM or list(head["classes"]) != [0, 1]:
        raise ValueError("Binary256-dimensional classifier contract required")
    mean = _finite_array(head["scaler"]["mean"], (FEATURE_DIM,), "scalermean")
    scale = _finite_array(head["scaler"]["scale"], (FEATURE_DIM,), "scalerscale")
    coef = _finite_array(head["coef"], (1, FEATURE_DIM), "coef")
    intercept = _finite_array(head["intercept"], (1,), "intercept")
    if np.any(scale <= 0):
        raise ValueError("StandardScaler scale must be strictly positive")
    decision = ((xx - mean) / scale) @ coef[0] + intercept[0]
    positive = expit(decision)
    return dict(decision=decision, probability=positive,
                prediction=(positive >= float(head["threshold"])).astype(np.int64))


def fit_two_heads(arrays: Mapping[str, np.ndarray], *, allow_fitting: bool = False,
                   before_fit: Callable[[str], None] | None = None,
                   raw_state_save: Callable[[str, Any, Any, Sequence[Any]], None] | None = None,
                   after_fit: Callable[[str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Exactly two final LR fits; caller must separately freeze/authorize scope.

    This function performs no CV and no threshold selection. A failed fit still
    consumes a caller-ledger attempt. There are no retries or constant-model
    substitutions. StandardScaler estimation is part of each disclosed head
    fitting session; the caller must not count it as a hidden third estimator.
    """
    if not allow_fitting or before_fit is None or raw_state_save is None or after_fit is None:
        raise PermissionError("Frozen R0 protocol, ledger, raw-state save and final per-head save callbacks required")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    import sklearn
    import scipy
    patients = np.asarray(arrays["patient_ids"])
    if patients.shape != (48,) or not _matches_original48(patients.tolist()):
        raise ValueError("Exactly48 sorted original patient rows required")
    prepared = {}
    for name in HEAD_ORDER:
        xx = _finite_array(arrays[f"X_{name}"], (48, FEATURE_DIM), name).astype(np.float64, copy=False)
        yy = np.asarray(arrays[f"y_{name}"])
        if yy.shape != (48,) or not np.issubdtype(yy.dtype, np.integer) or set(yy.tolist()) != {0, 1}:
            raise ValueError("Both binary classes must be present; no fallback recipe")
        if int(yy.sum()) != EXPECTED_TRAINING_COUNTS[f"{name}_positive"]:
            raise ValueError("Native existence positives differ from sealedY recipe")
        prepared[name] = (xx, yy)
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    serialized = {}; replay = {}
    for name in HEAD_ORDER:
        xx, yy = prepared[name]
        before_fit(name)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            scaler = StandardScaler().fit(xx)
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=20261019)
            clf.fit(scaler.transform(xx), yy)
            # First action after fit returns: persist raw fitted state BEFORE
            # any class assertion, conversion, serialization or numerical replay.
            # The root-owned callback performs IO and records the consumed fit.
            raw_state_save(name, scaler, clf, tuple(recorded))
        if clf.classes_.tolist() != [0, 1] or clf.coef_.shape != (1, FEATURE_DIM):
            raise RuntimeError("Unexpected sklearn binary classifier structure")
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in recorded)
        payload = dict(head=name, classes=clf.classes_.tolist(), labels=list(LABELS),
                       coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(),
                       threshold=FIXED_THRESHOLDS[name], sklearn_parameters=clf.get_params(),
                       n_iter=clf.n_iter_.tolist(), library_versions=versions,
                       scaler=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                                   variance=scaler.var_.tolist(), n_samples_seen=int(scaler.n_samples_seen_),
                                   with_mean=bool(scaler.with_mean), with_std=bool(scaler.with_std)),
                       n_training=48, positive_patients=int(yy.sum()),
                       target_definition=("native Pcom at least one side in CT or MR" if name == "pcom" else "native3rd-A2 in CT or MR"),
                       warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in recorded],
                       converged_without_ConvergenceWarning=converged,
                       cv_fits=0, threshold_searches=0,
                       historical_hidden_averaging_preserved=True,
                       parameter_sha256=_digest_arrays((scaler.mean_, scaler.scale_, scaler.var_, clf.coef_, clf.intercept_, clf.classes_)))
        manual = predict_serialized(payload, xx)
        sklearn_decision = clf.decision_function(scaler.transform(xx))
        sklearn_probability = clf.predict_proba(scaler.transform(xx))[:, 1]
        decision_difference = float(np.max(np.abs(manual["decision"] - sklearn_decision)))
        probability_difference = float(np.max(np.abs(manual["probability"] - sklearn_probability)))
        if not np.allclose(manual["decision"], sklearn_decision, atol=1e-10, rtol=1e-12) or not np.allclose(manual["probability"], sklearn_probability, atol=2e-12, rtol=1e-12):
            raise RuntimeError("Serialized coefficient/scaler replay mismatch")
        payload["training_replay"] = dict(decision_max_abs_difference=decision_difference,
                                          probability_max_abs_difference=probability_difference,
                                          threshold_prediction_exact=bool(np.array_equal(manual["prediction"], sklearn_probability >= FIXED_THRESHOLDS[name])),
                                          input_sha256=_digest_arrays((xx, yy)))
        if after_fit is not None:
            after_fit(name, payload)
        serialized[name] = payload
        replay.update({f"{name}_{k}": v for k, v in manual.items()})
        replay[f"{name}_target"] = yy.copy()
    replay["patient_ids"] = patients.copy()
    return dict(heads=serialized, training_replay_arrays=replay, library_versions=versions,
                estimator_fit_calls=2, cv_fits=0, threshold_searches=0,
                qualified=all(h["converged_without_ConvergenceWarning"] for h in serialized.values()),
                scope="Original48 final-estimator reconstruction only, not external validation")


def apply_patient_gate(cal: np.ndarray, pcom_probability: float,
                        a2_probability: float) -> np.ndarray:
    logits = _finite_array(cal, name="cal")
    if logits.ndim != 2 or logits.shape[1] != 13:
        raise ValueError("cal must be N,13")
    if not np.isfinite([pcom_probability, a2_probability]).all() or not (0 <= pcom_probability <= 1 and 0 <= a2_probability <= 1):
        raise ValueError("Binary probabilities required")
    blocked = ([] if pcom_probability >= FIXED_THRESHOLDS["pcom"] else list(PCOM_INDICES))
    blocked += [] if a2_probability >= FIXED_THRESHOLDS["a2"] else [A2_INDEX]
    pred = logits.argmax(-1)
    if blocked:
        changed = np.isin(pred, blocked)
        if changed.any():
            alternative = logits.copy(); alternative[:, blocked] = -np.inf
            pred[changed] = alternative[changed].argmax(-1)
    return pred


def synthetic_checks() -> dict[str, Any]:
    """Array/DBSCAN/manual-estimator tests only. Does not call fit_two_heads."""
    checks = {}
    rng = np.random.default_rng(20261027)
    packets = [dict(fold=0, seed=seed, cal=rng.normal(size=(9, 13)).astype(np.float32),
                    F=rng.normal(size=(9, 64)).astype(np.float32), L=rng.normal(size=(9, 64)).astype(np.float32)) for seed in SEEDS]
    agg = aggregate_scan_sources(packets, [(0, s) for s in SEEDS], expected_n_points=9)
    checks["float32_source_mean_exact"] = all(np.array_equal(agg[k], np.mean([p[k] for p in packets], axis=0)) and agg[k].dtype == np.float32 for k in ("cal", "F", "L"))
    try:
        aggregate_scan_sources(packets, [(0, SEEDS[1]), (0, SEEDS[0])], 9)
        checks["source_order_mismatch_rejected"] = False
    except ValueError:
        checks["source_order_mismatch_rejected"] = True
    xyz = np.array([[0, 0, 0], [.001, 0, 0], [.002, 0, 0], [.1, 0, 0], [.101, 0, 0], [.102, 0, 0], [.5, 0, 0], [.501, 0, 0], [.502, 0, 0]], np.float32)
    pred = np.array([7] * 3 + [8] * 3 + [12] * 3)
    cal = np.full((9, 13), -10., np.float32); cal[np.arange(9), pred] = 10.
    ff = np.arange(9 * 64, dtype=np.float32).reshape(9, 64)
    ll = -ff
    features = scan_features(xyz, cal, ff, ll)
    checks["segment_embedding_order_and_dtype"] = np.array_equal(segment_embedding(ff, ll, np.arange(3)), np.concatenate((ff[:3].mean(0), ff[:3].max(0), ll[:3].mean(0), ll[:3].max(0))).astype(np.float64))
    checks["Pcom_side_max_not_mean"] = np.array_equal(features["pcom"], np.maximum(segment_embedding(ff, ll, np.arange(3)), segment_embedding(ff, ll, np.arange(3, 6))))
    checks["A2_index12_maps_canonical15"] = np.array_equal(features["a2"], segment_embedding(ff, ll, np.arange(6, 9)))
    checks["missing_predicted_class_zero_embedding"] = bool((patient_embedding(ff, ll, xyz, pred, 0) == 0).all())
    mod_features = {"ct": features, "mr": {h: features[h] + 2 for h in HEAD_ORDER}}
    pooled = pool_patient_scan_features(mod_features)
    checks["CT_MR_pool_is_max_not_mean"] = all(np.array_equal(pooled[h], features[h] + 2) for h in HEAD_ORDER)
    checks["native_complete_presence_OR"] = native_presence_targets({"ct": [1, 8], "mr": [1, 15]}) == {"pcom": 1, "a2": 1}
    checks["native_absent_zero"] = native_presence_targets({"ct": [1], "mr": [2]}) == {"pcom": 0, "a2": 0}
    fake = dict(classes=[0, 1], coef=np.zeros((1, 256)).tolist(), intercept=[0.], threshold=.05,
                scaler=dict(mean=np.zeros(256).tolist(), scale=np.ones(256).tolist()))
    fake["coef"][0][0] = 1.
    xx = np.zeros((3, 256)); xx[:, 0] = [-1000., 0., 1000.]
    replay = predict_serialized(fake, xx)
    checks["stable_serialized_logistic_extremes"] = np.array_equal(replay["probability"], np.array([0., .5, 1.]))
    checks["fixed_threshold_greater_equal"] = np.array_equal(replay["prediction"], np.array([0, 1, 1]))
    ungated = apply_patient_gate(cal, .05, .05)
    checks["threshold_equality_keeps_predictions"] = np.array_equal(ungated, pred)
    blocked = apply_patient_gate(cal, .049, .049)
    checks["blocked_classes_reassigned_not_unknown"] = not np.isin(blocked, [7, 8, 12]).any() and np.all((blocked >= 0) & (blocked < 13))
    try:
        fit_two_heads({}, allow_fitting=False)
        checks["fits_require_explicit_root_authorization"] = False
    except PermissionError:
        checks["fits_require_explicit_root_authorization"] = True
    try:
        fit_two_heads({}, allow_fitting=True, before_fit=lambda name: None,
                      after_fit=lambda name, payload: None)
        checks["raw_state_callback_required_before_any_fitting"] = False
    except PermissionError:
        checks["raw_state_callback_required_before_any_fitting"] = True
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(qualified=all(checks.values()), checks=checks, actual_fit_calls=0,
                cross_validation_calls=0, patient_data_reads=0,
                note="Synthetic checks intentionally do not estimate scaler or fit a logistic model")
