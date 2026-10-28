from io import BytesIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core import (ARMS, MAX_POINTS, MAX_UPLOAD_BYTES, contained_path,
    display_subset, load_public_summary, statistic_text, validate_pointcloud_npz)


def npz_bytes(**arrays):
    buffer=BytesIO();np.savez_compressed(buffer,**arrays);return buffer.getvalue()


class AppCoreTests(unittest.TestCase):
    def test_numeric_points_and_labels(self):
        p,y=validate_pointcloud_npz(npz_bytes(xyz=np.zeros((6,3),np.float32),labels=np.arange(6)))
        self.assertEqual(p.shape,(6,3));self.assertEqual(y.tolist(),list(range(6)))

    def test_points_alias(self):
        p,y=validate_pointcloud_npz(npz_bytes(points=np.ones((2,3))))
        self.assertEqual(p.dtype,np.float32);self.assertIsNone(y)

    def test_objects_and_pickle_rejected(self):
        with self.assertRaises(ValueError):validate_pointcloud_npz(npz_bytes(xyz=np.array([[object(),1,2]],dtype=object)))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):validate_pointcloud_npz(npz_bytes(xyz=np.full((2,3),np.nan)))

    def test_shape_and_limit_rejected(self):
        for value in (np.zeros((3,2)),np.zeros((0,3)),np.zeros((MAX_POINTS+1,3))):
            with self.assertRaises(ValueError):validate_pointcloud_npz(npz_bytes(xyz=value))

    def test_label_schema(self):
        for value in (np.zeros((2,1),int),np.zeros(2,float),np.array([-1,2]),np.array([1,300])):
            with self.assertRaises(ValueError):validate_pointcloud_npz(npz_bytes(xyz=np.zeros((2,3)),labels=value))

    def test_extra_arrays_and_ambiguous_points(self):
        for payload in (npz_bytes(xyz=np.zeros((2,3)),points=np.ones((2,3))),npz_bytes(xyz=np.zeros((2,3)),secret=np.zeros(2))):
            with self.assertRaises(ValueError):validate_pointcloud_npz(payload)

    def test_non_zip_and_bytes_cap(self):
        for payload in (b"not a zip",b"x"*(MAX_UPLOAD_BYTES+1)):
            with self.assertRaises(ValueError):validate_pointcloud_npz(payload)

    def test_path_traversal_members(self):
        buffer=BytesIO()
        with zipfile.ZipFile(buffer,"w") as archive:archive.writestr("../xyz.npy",b"invalid")
        with self.assertRaises(ValueError):validate_pointcloud_npz(buffer.getvalue())

    def test_duplicate_zip_member(self):
        buffer=BytesIO()
        with zipfile.ZipFile(buffer,"w") as archive:
            archive.writestr("xyz.npy",b"a");archive.writestr("xyz.npy",b"b")
        with self.assertRaises(ValueError):validate_pointcloud_npz(buffer.getvalue())

    def test_render_cap(self):
        p,c=display_subset(np.zeros((50000,3)),np.zeros(50000,int))
        self.assertEqual(len(p),25000);self.assertEqual(len(c),25000)

    def test_asset_cannot_escape_release(self):
        with self.assertRaises(ValueError):contained_path(Path("/tmp/release"),Path("/tmp/private.json"))

    def test_undefined_ci_not_promoted(self):
        self.assertIn("undefined",statistic_text({"value":.5,"CI":None,"conditional_finite_only_CI":[.4,.6]}))

    def test_review_gate_all_four_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"fixture.json"
            data={"cohorts":{"DEV":{"independently_reviewed":True,"original_arm_statistics":{a:{} for a in ARMS}}}}
            file.write_text(json.dumps(data));self.assertTrue(load_public_summary(file))
            data["cohorts"]["DEV"]["independently_reviewed"]=False
            file.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_public_summary(file)


if __name__ == "__main__":unittest.main()
