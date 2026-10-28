"""Data-independent tests: no checkpoint/patient reads, no fitting."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
 def test_metadata_is_candid(self):
  from strokepointpp.cli import resource
  p=resource('release_summary_v1.json')
  self.assertFalse(p['NC_ready']); self.assertFalse(p['NMI_ready'])
  self.assertTrue(p['original_STOP_preserved'])

 def test_no_weights_distributed(self):
  from strokepointpp.cli import resource
  assets=resource('model_assets_v1.json')['assets']
  self.assertGreater(len(assets),16)
  self.assertTrue(all(not a['distributed'] and len(a['sha256'])==64 for a in assets))

 def test_missing_asset_fails_without_fallback(self):
  from strokepointpp.adapters import _assets
  # Existing placeholder config; no fitted state or patient arrays are opened.
  with self.assertRaises((FileNotFoundError,ValueError)):
   _assets(ROOT/'configs/model_assets.local.example.json')('gate_projection')

 def test_bundle_hashes(self):
  from strokepointpp.cli import verify
  r=verify()
  self.assertTrue(r['bundle']['qualified'])


class ArrayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import numpy as np
  cls.np=np

 def test_serialized_head_and_gating(self):
  from strokepointpp._frozen.gate_arrays import synthetic_checks
  r=synthetic_checks()
  self.assertTrue(r['qualified']); self.assertEqual(r['actual_fit_calls'],0)

 def test_four_source_probabilities(self):
  from strokepointpp.gate_engine import SOURCE_ORDER,predict_from_packets
  np=self.np; rng=np.random.default_rng(20261028)
  xyz=rng.normal(size=(2048,3)).astype(np.float32)
  packets=[dict(fold=f,seed=s,cal=rng.normal(size=(2048,13)).astype(np.float32),
   F=rng.normal(size=(2048,64)).astype(np.float32),L=rng.normal(size=(2048,64)).astype(np.float32)) for f,s in SOURCE_ORDER]
  h=dict(classes=[0,1],coef=np.zeros((1,256)).tolist(),intercept=[-1000.],threshold=.05,
   scaler=dict(mean=np.zeros(256).tolist(),scale=np.ones(256).tolist()))
  out=predict_from_packets(packets,xyz,dict(pcom=h,a2=h))
  self.assertEqual(out['gated_probs'].shape,(2048,13))
  np.testing.assert_allclose(out['gated_probs'].sum(-1),1.,atol=2e-6)
  self.assertFalse(np.isin(out['gated_labels'],[0,8,9,15]).any())

 def test_original_pointnet_dimensions(self):
  import torch
  from strokepointpp._frozen.pointnet import PointNetPlusPlus,geometry
  torch.set_num_threads(2); np=self.np
  xyz=np.random.default_rng(20).normal(size=(48,3)).astype(np.float32)
  geo={k:torch.from_numpy(v[None]) for k,v in geometry(xyz).items()}
  model=PointNetPlusPlus().eval()
  with torch.inference_mode():
   logits=model(torch.from_numpy(xyz[None]),geo)
  self.assertEqual(tuple(logits.shape),(1,48,13)); self.assertTrue(bool(torch.isfinite(logits).all()))

 def test_spatial_architecture_contract_without_forward(self):
  from strokepointpp._frozen.spatial_models import create_model
  for arm in ('BASE3D','CONCAT','JOINT','NO_FEEDBACK'):
   m=create_model(arm)
   self.assertEqual(m.point_head.out_features,13)
   self.assertEqual(m.metadata()['dense_label_order'],[0,1,2,3,4,5,6,7,8,9,10,11,12,15])
   self.assertFalse(m.metadata()['fully_trained_nnunet_certification'])
   del m


if __name__=='__main__': unittest.main()
