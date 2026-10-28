"""Synthetic privacy/binding tests; no patient arrays, weights, or fitting."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from strokepointpp._frozen import gate_arrays as gate

ROOT = Path(__file__).resolve().parents[1]


def commitment(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=True,
        separators=(",", ":")).encode("ascii")).hexdigest()


def synthetic_rows():
    ids = tuple(f"{100+i:03d}" for i in range(48))
    features = {(p, m): {h: np.full(256, i, dtype=np.float64) for h in gate.HEAD_ORDER}
                for i, p in enumerate(ids) for m in ("ct", "mr")}
    targets = {(p, m): [1] + ([8] if i < 30 else []) + ([15] if i < 7 else [])
               for i, p in enumerate(ids) for m in ("ct", "mr")}
    return ids, features, targets


class PrivacyContractTests(unittest.TestCase):
    def test_public_commitment_encoding_is_explicit_and_no_plain_allowlist(self):
        self.assertEqual(gate.ORIGINAL48_PATIENT_BINDING_SHA256,
            "59e8ca830a33562b52d896139b35db9abd0c5bb26450451659aee3e842e7378a")
        self.assertFalse(hasattr(gate, "ORIGINAL48_PATIENT_IDS"))
        for relative in ("strokepointpp/_frozen/gate_arrays.py",
                "archive/algorithms/work/nc01_yz_reconstruction_v2_20261027.py"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("privacy-adapted public source", text)
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    literals = [n.value for n in node.elts if isinstance(n, ast.Constant)]
                    self.assertFalse(len(literals) >= 3 and all(isinstance(v, str)
                        and len(v) == 3 and v.isascii() and v.isdigit() for v in literals))

    def test_order_count_and_type_binding_matches_old_tuple_equality(self):
        ids, _, _ = synthetic_rows()
        variations = [ids, ids[::-1], ids[:-1], ids + ids[:1],
            tuple(int(p) for p in ids), tuple(p + " " for p in ids),
            tuple("0" + p for p in ids), ids[:24] + ids[25:] + ids[24:25]]
        with patch.object(gate, "ORIGINAL48_PATIENT_BINDING_SHA256", commitment(ids)):
            for candidate in variations:
                self.assertEqual(gate._matches_original48(candidate), tuple(candidate) == ids)
        self.assertFalse(gate._matches_original48(ids))

    def test_training_array_order_counts_and_native_targets_preserved(self):
        ids, features, targets = synthetic_rows()
        with patch.object(gate, "ORIGINAL48_PATIENT_BINDING_SHA256", commitment(ids)):
            arrays = gate.build_training_arrays(ids, features, targets)
            self.assertEqual(tuple(arrays["patient_ids"]), ids)
            self.assertEqual(arrays["X_pcom"].shape, (48, 256))
            np.testing.assert_array_equal(arrays["X_pcom"][:, 0], np.arange(48))
            self.assertEqual(int(arrays["y_pcom"].sum()), 30)
            self.assertEqual(int(arrays["y_a2"].sum()), 7)
            with self.assertRaises(ValueError):
                gate.build_training_arrays(ids[::-1], features, targets)
            changed = dict(targets)
            changed[ids[0], "ct"] = [1]
            changed[ids[0], "mr"] = [1]
            with self.assertRaises(ValueError):
                gate.build_training_arrays(ids, features, changed)

    def test_fit_preflight_uses_same_commitment_and_never_reaches_fit(self):
        ids, features, targets = synthetic_rows()
        class StopBeforeFit(RuntimeError):
            pass
        callbacks = []
        def stop(name):
            callbacks.append(name)
            raise StopBeforeFit()
        kwargs = dict(allow_fitting=True, before_fit=stop,
            raw_state_save=lambda *args: None, after_fit=lambda *args: None)
        with patch.object(gate, "ORIGINAL48_PATIENT_BINDING_SHA256", commitment(ids)):
            arrays = gate.build_training_arrays(ids, features, targets)
            with patch("sklearn.linear_model.LogisticRegression.fit",
                    side_effect=AssertionError("Fitting forbidden in this test")) as fit:
                with self.assertRaises(StopBeforeFit):
                    gate.fit_two_heads(arrays, **kwargs)
                self.assertEqual(callbacks, ["pcom"])
                callbacks.clear()
                wrong = dict(arrays, patient_ids=arrays["patient_ids"][::-1])
                with self.assertRaises(ValueError):
                    gate.fit_two_heads(wrong, **kwargs)
                self.assertEqual(callbacks, [])
                fit.assert_not_called()

    def test_builder_rejects_plain_source_lists_and_accepts_digest(self):
        spec = importlib.util.spec_from_file_location("release_builder", ROOT / "scripts/build_release.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        synthetic = tuple(f"{i+100:03d}" for i in range(4))
        with self.assertRaises(RuntimeError):
            builder._validate_public_source_privacy("BOUND_IDS = " + repr(synthetic), "example.py")
        builder._validate_public_source_privacy(
            "BOUND_SHA256 = " + repr(commitment(synthetic)), "example.py")
        for relative in ("strokepointpp/_frozen/gate_arrays.py",
                "archive/algorithms/work/nc01_yz_reconstruction_v2_20261027.py"):
            builder._validate_public_source_privacy((ROOT / relative).read_text(), relative)


if __name__ == "__main__":
    unittest.main()
