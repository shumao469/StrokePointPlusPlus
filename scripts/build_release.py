"""Reproducible, allowlisted source package builder; never imports study code.

Only reads source/JSON metadata, copies exact local private source archives,
generates import/publication-privacy adaptations and de-identified aggregate JSON.
No patient payload/model weight is copied, loaded or inferred.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
NC02='PostCTX3_NC02_mra_external_joint_pilot_20261027_v1'
NC03='PostCTX3_NC03_Lausanne_external_DEV_20261028_v1'
NC04='PostCTX3_NC04_IXI_HH_FINAL_20261028_v1'
R0='PostCTX3_NC01_YZ_fixed_recipe_reconstruction_20261027_v2'
PURE={
 'src/vascutime_pointnet.py':'pointnet.py',
 'work/c0_local_global_core_v1.py':'local_global.py',
 'work/c1_sameinfo_core_v1.py':'full_information.py',
 'work/nc01_yz_reconstruction_v2_20261027.py':'gate_arrays.py',
 'work/nc02_spatial_models_v2_20261027.py':'spatial_models.py',
 'work/avcr1_original_evaluator_v2.py':'evaluator.py'}
SEEDS=list(PURE)+[
 'work/nc01_yz_reconstruction_runner_v2_20261027.py','work/nc01_spatial_mapping_20261027.py',
 'work/nc02_data_20261027.py','work/nc02_train_20261027.py','work/nc02_evaluate_20261027.py',
 'work/nc02_independent_review_20261027.py','work/nc02_diag_interventions_20261027.py',
 'work/nc02_diag_trajectory_20261027.py','work/nc02_diag_patient_statistics_20261027.py',
 'work/nc02_diag_counterfactual_statistics_20261027.py',
 'work/nc03_yz_interface_20261028.py','work/nc03_yz_interface_v2_20261028.py',
 'work/nc03_external_dev_20261028.py','work/nc03_external_dev_resume_v3_20261028.py',
 'work/nc03_external_independent_review_20261028.py','work/nc04_yz_final_interface_20261028.py',
 'work/nc04_final_evaluation_20261028.py','work/nc04_final_independent_review_20261028.py',
 'work/nc04_transport_resume_driver_20261028.py','work/nc04_transport_extension_driver_20261028.py',
 'work/nc04_ixi_transport_resume_20261028.py','work/nc04_ixi_transport_extension_20261028.py',
 'work/nc04_final_source_audit_20261028.py','work/nc04_dev_final_summary_20261028.py',
 'work/nc04_final_decision_receipt_20261028.py']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''): h.update(b)
 return h.hexdigest()


def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


PATIENT_BINDING_SHA256='59e8ca830a33562b52d896139b35db9abd0c5bb26450451659aee3e842e7378a'
PRIVACY_SOURCE='work/nc01_yz_reconstruction_v2_20261027.py'


def _privacy_adapt_gate_source(text):
 """Remove the local historical ID list without weakening its ordered binding.

 This consumes source metadata only, never patient arrays or model assets.
 The original source is preserved separately in the local-only archive.
 """
 tree=ast.parse(text); matches=[n for n in tree.body if isinstance(n,ast.Assign)
  and any(isinstance(t,ast.Name) and t.id=='ORIGINAL48_PATIENT_IDS' for t in n.targets)]
 if len(matches)!=1: raise RuntimeError('Expected one original ordered patient binding')
 node=matches[0]; ids=ast.literal_eval(node.value)
 encoded=json.dumps(list(ids),ensure_ascii=True,separators=(',',':')).encode('ascii')
 if len(ids)!=48 or hashlib.sha256(encoded).hexdigest()!=PATIENT_BINDING_SHA256:
  raise RuntimeError('Historical source binding differs; no automatic approval of a new cohort')
 replacement='''ORIGINAL48_PATIENT_BINDING_SHA256 = "59e8ca830a33562b52d896139b35db9abd0c5bb26450451659aee3e842e7378a"
# Commitment encoding: ordered JSON string array, ensure_ascii=True,
# separators=(",", ":"), ASCII bytes, with no trailing newline.
def _matches_original48(patient_ids: Sequence[str]) -> bool:
    patients = tuple(patient_ids)
    if len(patients) != 48 or any(not isinstance(p, str) for p in patients):
        return False
    payload = json.dumps(list(patients), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest() == ORIGINAL48_PATIENT_BINDING_SHA256


'''
 lines=text.splitlines(True); lines[node.lineno-1:node.end_lineno]=[replacement]; adapted=''.join(lines)
 substitutions={
  '"""NC01-R0 v2 faithful Y/Z reconstruction with pre-certification raw-state save.':
   '"""NC01-R0 v2 Y/Z reconstruction: privacy-adapted public source.\n\n'
   'Publication adaptation replaces the ordered source-ID allowlist with its SHA256\n'
   'commitment. Exact historical binding, row counts, estimators and numerical\n'
   'operations are unchanged; this file is not byte-identical to the local original.',
  'import hashlib\n':'import hashlib\nimport json\n',
  'enforce_historical_counts and tuple(patients) != ORIGINAL48_PATIENT_IDS':
   'enforce_historical_counts and not _matches_original48(patients)',
  'patients.shape != (48,) or tuple(patients.tolist()) != ORIGINAL48_PATIENT_IDS':
   'patients.shape != (48,) or not _matches_original48(patients.tolist())'}
 for before,after in substitutions.items():
  if adapted.count(before)!=1: raise RuntimeError('Historical privacy transform anchor differs')
  adapted=adapted.replace(before,after)
 ast.parse(adapted)
 return adapted


def _validate_public_source_privacy(text,rel):
 """Reject source-level ID allowlists even when their bare IDs have no prefix."""
 if not rel.endswith('.py'): return
 for node in ast.walk(ast.parse(text)):
  if not isinstance(node,(ast.List,ast.Tuple,ast.Set)): continue
  values=[n.value for n in node.elts if isinstance(n,ast.Constant)]
  canonical=[v for v in values if isinstance(v,str) and re.fullmatch(r'[0-9]{3}',v)]
  if len(canonical)>=3:
   raise RuntimeError('Plain source-ID collection in public Python: '+rel)


def safe_aggregate(obj):
 if isinstance(obj,dict):
  return {k:safe_aggregate(v) for k,v in obj.items()
    if k not in ('input_sha256','source_sha256','patient_ids','patients','per_patient','per_case','patient_rows')
    and not re.match(r'^(?:[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/[A-Za-z0-9_.-]+/)',k)}
 if isinstance(obj,list): return [safe_aggregate(v) for v in obj]
 if isinstance(obj,str):
  obj=re.sub(r'(?:[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/[A-Za-z0-9_.-]+/)[^\s\"<>]*','<LOCAL_PATH_OMITTED>',obj)
  obj=re.sub(r'IXI\d{3}-HH-\d{4}|sub-\d+_ses-\d+','<SOURCE_ID_OMITTED>',obj)
  return obj
 return obj


def function_source(path,names):
 text=path.read_text(encoding='utf-8'); tree=ast.parse(text); lines=text.splitlines(True)
 return '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body
    if isinstance(n,ast.FunctionDef) and n.name in names)


EXPLICIT_PUBLIC = {
 '.gitignore','.gitattributes','.streamlit/config.toml',
 'README.md','DEPLOYMENT.md','LICENSE_NOTICE.md','pyproject.toml',
 'requirements-app.txt','requirements-figures.txt','requirements-inference-reference.txt',
 'deployment/start_local.ps1',
 'apps/app_core.py','apps/streamlit_app.py','apps/requirements.txt',
 'apps/tests/test_app_core.py','apps/tests/test_streamlit_smoke.py',
 'scripts/build_release.py','scripts/make_publication_figures.py',
 'scripts/render_real_cases.py','scripts/render_grouping.py','scripts/render_input_points.py',
 'docs/FIGURE_PLAN.md','docs/MANUSCRIPT_FIGURE_OUTLINE.md','docs/RELEASE_QA.md',
 'source_data/figure_data.json','source_data/figure_estimates.csv',
 'assets/README_hero.png','assets/README_hero.svg','assets/cases/ATTRIBUTION.md',
 'figures/artwork_manifest.json','figures/FIGURE_LEGENDS.md',
 'figures/StrokePointPlusPlus_FigureBook.pdf'}
TEXT_SUFFIXES={'.py','.json','.md','.toml','.txt','.ps1','.csv','.svg','.html','.css','.js'}
PRIVATE_RE=re.compile(r'(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/[A-Za-z0-9_.-]+/)')
SOURCE_ID_RE=re.compile(r'IXI\d{3}-HH-\d{4}|sub-\d+_ses-\d+')
SECRET_RE=re.compile(r'sk-[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?i:api[_-]?key|password|authorization)\s*[:=]\s*[\"\'][A-Za-z0-9_+/=.-]{20,}[\"\']')


def _public_candidates():
 # Exact top-level files plus narrow, named artifact families. Never traverse
 # private_build/private_archive/env/cache/log/runtime/test-run/dist directories.
 paths={ROOT/p for p in EXPLICIT_PUBLIC if (ROOT/p).is_file()}
 for folder in ('strokepointpp','archive/algorithms','configs','results/aggregate','manifests','tests'):
  base=ROOT/folder
  for path in base.rglob('*') if base.is_dir() else []:
   rel=path.relative_to(ROOT)
   if (path.is_file() and '__pycache__' not in rel.parts and path.suffix in TEXT_SUFFIXES
       and rel.as_posix()!='manifests/public_files_v1.json'):
    paths.add(path)
 for suffix in ('*.png','*.svg'):
  paths.update(path for path in (ROOT/'assets').glob(suffix) if path.is_file())
 for path in (ROOT/'assets/cases').glob('Case*.png'):
  if re.fullmatch(r'Case[A-D]_[A-Za-z0-9_]+\.png',path.name):
   paths.add(path)
 for folder,pattern in [('figures/main',r'fig[1-5]_[a-z0-9_]+\.(?:svg|pdf|png)'),
                        ('figures/extended_data',r'edfig[1-4]_[a-z0-9_]+\.(?:svg|pdf|png)'),
                        ('figures/supplementary',r'suppfig[1-3]_[a-z0-9_]+\.(?:svg|pdf|png)')]:
  for path in (ROOT/folder).glob('*'):
   if path.is_file() and re.fullmatch(pattern,path.name): paths.add(path)
 return sorted(paths)


def _metadata_text(path):
 """Inspect text and raster/PDF metadata, never mis-scan random compressed pixels."""
 if path.suffix in TEXT_SUFFIXES or path.name in ('.gitignore','.gitattributes'):
  return path.read_text(encoding='utf-8-sig')
 blob=path.read_bytes()
 if path.suffix=='.png':
  if not blob.startswith(b'\x89PNG\r\n\x1a\n'): raise RuntimeError('Not a valid PNG signature')
  offset=8; metadata=[]
  while offset+12<=len(blob):
   n=int.from_bytes(blob[offset:offset+4],'big'); kind=blob[offset+4:offset+8]
   end=offset+12+n
   if end>len(blob): raise RuntimeError('Truncated PNG chunk')
   data=blob[offset+8:offset+8+n]
   if zlib.crc32(kind+data)&0xffffffff!=int.from_bytes(blob[offset+8+n:end],'big'):
    raise RuntimeError('PNG chunk CRC mismatch')
   if kind in (b'tEXt',b'iTXt'): metadata.append(data.decode('utf-8',errors='replace'))
   if kind==b'zTXt':
    at=data.find(b'\x00')
    if at<0 or data[at+1:at+2]!=b'\x00': raise RuntimeError('Invalid compressed PNG metadata')
    dec=zlib.decompressobj(); text=dec.decompress(data[at+2:],1024**2+1)
    if len(text)>1024**2: raise RuntimeError('Oversized PNG text metadata')
    metadata.append(text.decode('utf-8',errors='replace'))
   offset=end
   if kind==b'IEND': break
  return '\n'.join(metadata)
 if path.suffix=='.pdf':
  if not blob.startswith(b'%PDF-') or b'%%EOF' not in blob[-1024:]: raise RuntimeError('Invalid PDF signature/EOF')
  fields=re.findall(rb'/(?:Title|Author|Creator|Producer|Subject|Keywords)\s*\((.*?)\)',blob,re.S)
  return b'\n'.join(fields).decode('utf-8',errors='replace')
 raise RuntimeError('Unexpected public binary type: '+path.suffix)


def _collect_public():
 files=[]
 for path in _public_candidates():
  rel=path.relative_to(ROOT).as_posix()
  if path.is_symlink(): raise RuntimeError('No public symlink entries')
  if path.stat().st_size>50*1024**2: raise RuntimeError('Unexpected large public artifact: '+rel)
  try: text=_metadata_text(path)
  except Exception as exc: raise RuntimeError('Public artifact validation failed: '+rel+' : '+str(exc)) from exc
  _validate_public_source_privacy(text,rel)
  if PRIVATE_RE.search(text): raise RuntimeError('Private absolute path in public artifact: '+rel)
  if SOURCE_ID_RE.search(text): raise RuntimeError('Subject source ID in public artifact: '+rel)
  if SECRET_RE.search(text): raise RuntimeError('Credential-like string in public artifact: '+rel)
  if path.suffix=='.svg' and 'file://' in text: raise RuntimeError('SVG links private local file: '+rel)
  files.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path)))
 return files


def _validate_artwork_manifest(public):
 path=ROOT/'figures/artwork_manifest.json'
 if not path.is_file(): return dict(available=False)
 doc=read(path); allowed={r['path'] for r in public}; checked=0
 for rel,digest in doc['outputs'].items():
  if rel=='figures/artwork_manifest.json':
   raise RuntimeError('Artwork manifest cannot certify its own current file hash')
  p=ROOT/rel
  if not p.resolve().is_relative_to(ROOT.resolve()) or rel not in allowed or not p.is_file() or sha(p)!=digest:
   raise RuntimeError('Artwork manifest SHA/output membership mismatch: '+rel)
  checked+=1
 return dict(available=True,qualified=True,checked_outputs=checked,
  main_figures=doc['main_figures'],extended_data_figures=doc['extended_data_figures'],supplementary_figures=doc['supplementary_figures'])


def _local_code_archive(audit):
 out=ROOT/'dist'/('StrokePointPlusPlus-local-code-archive-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
 out.parent.mkdir(exist_ok=True)
 provenance=dict(local_only=True,not_for_public_upload=True,patient_data=False,fitted_assets=False,
  scope='Exact historical source code contains investigator-local paths; not the public distribution',files=audit)
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for row in audit:
   rel=row['relative_source']; p=ROOT/'private_archive/source'/rel
   if p.suffix!='.py' or sha(p)!=row['sha256']: raise RuntimeError('Private source archive mismatch')
   z.writestr('local_code/source/'+rel,p.read_bytes())
  z.writestr('local_code/provenance_manifest.json',json.dumps(provenance,indent=2))
 with zipfile.ZipFile(out) as z:
  if z.testzip() is not None: raise RuntimeError('Local code ZIP CRC failure')
 return dict(filename=out.name,sha256=sha(out),bytes=out.stat().st_size,source_count=len(audit),
  local_only=True,patient_data=False,weights=False)


def build(ws,stage_only=False,local_code_archive=False,local_bindings=False):
 ws=Path(ws).resolve(); resources=ROOT/'strokepointpp/resources'; resources.mkdir(parents=True,exist_ok=True)
 audit=[]; queue=list(SEEDS); seen=set(); external=set()
 while queue:
  rel=queue.pop(0)
  if rel in seen: continue
  seen.add(rel); source=ws/rel
  if not source.is_file(): raise FileNotFoundError('Required study source absent: '+rel)
  blob=source.read_bytes(); tree=ast.parse(blob.decode('utf-8-sig')); imports=[]
  for node in ast.walk(tree):
   mods=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
   for mod in mods:
    imports.append(mod)
    candidates=[mod.replace('.','/')+'.py','work/'+mod.replace('.','/')+'.py']
    local=next((p for p in candidates if (ws/p).is_file()),None)
    if local: queue.append(local)
    else: external.add(mod.split('.')[0])
  dest=ROOT/'private_archive/source'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and sha(dest)!=sha(source): raise RuntimeError('Do not overwrite an older private source archive')
  if not dest.exists(): shutil.copyfile(source,dest)
  audit.append(dict(relative_source=rel,sha256=sha(source),bytes=len(blob),imports=sorted(set(imports))))
 frozen=ROOT/'strokepointpp/_frozen'; frozen.mkdir(parents=True,exist_ok=True)
 (frozen/'__init__.py').write_text('"""Import/privacy adaptations of archived algorithms; not fitted assets."""\n',encoding='utf-8')
 transformations=[]
 for rel,name in PURE.items():
  original=ws/rel; exact=ROOT/'archive/algorithms'/rel; exact.parent.mkdir(parents=True,exist_ok=True)
  text=original.read_text(encoding='utf-8')
  privacy_adapted=rel==PRIVACY_SOURCE
  if privacy_adapted:
   text=_privacy_adapt_gate_source(text)
   exact.write_text(text,encoding='utf-8')
  else:
   shutil.copyfile(original,exact)
  text=text.replace('from src.vascutime_pointnet import','from .pointnet import')
  text=text.replace('from work.c0_local_global_core_v1 import','from .local_global import')
  target=frozen/name; target.write_text(text,encoding='utf-8')
  transformations.append(dict(original=rel,original_sha256=sha(original),public_module='strokepointpp/_frozen/'+name,
    public_sha256=sha(target),public_archive_sha256=sha(exact),
    privacy_adapted=privacy_adapted,public_archive_byte_identical_to_original=not privacy_adapted,
    change=('Publication privacy: replace ordered source-ID list with exact SHA256 binding; '
     'numerical estimators, order/count constraints and deployment operations unchanged'
     if privacy_adapted else 'Only import namespace substitutions; numerical code unchanged')))
 # Engineering adapter uses the exact six numerical geometry/extraction/gating
 # functions; only their module imports are redirected to bundled dependencies.
 interface=ws/'work/nc04_yz_final_interface_20261028.py'
 names={'validate_visible','build_batch','_extract','calibrated_logits','predict_from_packets','predict_yz'}
 body=function_source(interface,names).replace('from src.vascutime_pointnet import','from ._frozen.pointnet import')
 body=body.replace('from nc01_yz_reconstruction_v2_20261027 import','from ._frozen.gate_arrays import')
 header='"""Release engineering extraction of six fixed numerical functions; import adaptation only."""\nfrom typing import Mapping,Any\nimport numpy as np\n'
 header+='LABELS=(1,2,3,4,5,6,7,8,9,10,11,12,15)\nSOURCE_ORDER=((0,20261020),(0,20261021),(1,20261020),(1,20261021))\n'
 header+="GEOMETRY_KEYS=('xyz','s1','s2','g1','g2','i21','w21','i10','w10','path_features','native_gd','euclidean_ids','euclidean_mask','euclidean_active')\n"
 (ROOT/'strokepointpp/gate_engine.py').write_text(header+body+'\n',encoding='utf-8')
 spec=read(ws/'work/nc01_yz_reconstruction_spec_20261027.json'); assets=[]; local_asset_paths={}
 def add(asset_id,kind,path,digest):
  assets.append(dict(asset_id=asset_id,kind=kind,filename=Path(path).name,sha256=digest,distributed=False))
  local_asset_paths[asset_id]=str(path)
 for m in spec['models']:
  for k in ('checkpoint','normalization','readout'):
   r=m[k]; add(f"gate_f{m['fold']}_s{m['seed']}_{k}",k,r['path'],r['sha256'])
 add('gate_projection','projection',spec['projection']['path'],spec['projection']['sha256'])
 for aid,rel in [('gate_priors','fold_priors_v1.json'),('gate_pcom_head','heads/pcom_pipeline_v1.json'),('gate_a2_head','heads/a2_pipeline_v1.json')]:
  p=ws/'artifacts'/R0/rel; add(aid,'fixed_fitted_asset',p,sha(p))
 for m in read(ws/'artifacts'/NC02/'prediction_lock_v1.json')['models']:
  add(f"spatial_f{m['fold']}_{m['arm']}",'short_pilot_spatial_checkpoint',m['checkpoint_path'],m['checkpoint_sha256'])
 model_manifest=dict(assets=assets,labels=spec['labels']['canonical_order'],
  gate_recipe_protocol_sha256=sha(ws/'artifacts'/R0/'protocol_freeze_v1.json'),
  deployment_variant='Gate-MRA-R0v2: single MRA, original4-source latent mean, not original pairedCTMR input',
  no_weights_or_scalers_distributed=True)
 save(resources/'model_assets_v1.json',model_manifest)
 local_binding_report=None
 if local_bindings:
  checked=[]
  for asset in assets:
   actual=Path(local_asset_paths[asset['asset_id']])
   if not actual.is_file() or sha(actual)!=asset['sha256']:
    raise RuntimeError('Local asset binding SHA mismatch: '+asset['asset_id'])
   checked.append(dict(asset_id=asset['asset_id'],sha256=asset['sha256'],bytes=actual.stat().st_size))
  save(ROOT/'private_build/model_assets.local.json',dict(local_only=True,runtime='WSL Ubuntu-22.04',
   assets=local_asset_paths,no_patient_data=True,no_fits_or_model_forwards=True))
  local_binding_report=dict(qualified=True,checked_assets=len(checked),assets=checked,
   mapping_relative_path='private_build/model_assets.local.json',local_only=True,model_forwards=0,fits=0,
   verification='Exact asset-byte SHA only, no deserialization or patient access')
  save(ROOT/'logs/local_model_assets_verification_v1.json',local_binding_report)
 save(ROOT/'configs/model_assets.local.example.json',{'assets':{a['asset_id']:'${ASSET_ROOT}/'+a['asset_id']+'/'+a['filename'] for a in assets}})
 aggregate={}
 for role,folder in [('externalDEV_Lausanne',NC03),('externalFINAL_IXI_HH',NC04)]:
  base=ws/'artifacts'/folder; review=read(base/'independent_review_v1.json')
  if review.get('qualified') is not True or review.get('mismatches')!=[]: raise RuntimeError('Unreviewed aggregate cannot be published')
  if review['input_sha256']['main_statistics']!=sha(base/'statistics_v1.json'): raise RuntimeError('Statistics differ from independently reviewed hash')
  result=safe_aggregate(read(base/'statistics_v1.json'))
  aggregate[role]=dict(patient_count=review['patient_count'],statistics=result,
    review=safe_aggregate(review),statistics_source_sha256=sha(base/'statistics_v1.json'),
    note='Given-manual-binary-foreground, sampled2048point evaluation; not whole-volume end-to-end segmentation')
  save(ROOT/'results/aggregate'/(role+'.json'),aggregate[role])
 summary=dict(name='StrokePoint++',chinese_name='卒中脑血管点云识别工具箱',version='0.1.0.dev0',
  purpose='Research cerebral vessel anatomical labeling, not validated stroke diagnosis/prognosis',
  components={'Gate-AppYZ-v1':'Sealed historical pairedCTMR internal65 application; not modified',
   'Gate-MRA-R0v2':'Fixed-recipe reconstructed heads; single-MRA external transfer, not original fitted state recovery',
   'StrokePoint-3D':'Enhanced3D U-Net short fixed pilot, not fully trained complete nnU-Net',
   'StrokePoint-Fuse':'Same-input simple concatenation comparator, not a proven new mechanism',
   'research/archive':'Joint/feedback and previous negative studies; original STOP retained'},
  clinical_ready=False,NC_ready=False,NMI_ready=False,original_STOP_preserved=True,
  patient_data_in_bundle=False,fitted_assets_in_bundle=False,automatic_training=False,
  reproducibility_scope='Executable parameterized inference/array algorithms with caller-owned exact assets and prepared visible inputs; no turnkey raw-image preprocessing validation claimed')
 save(resources/'release_summary_v1.json',summary); save(resources/'aggregate_results_v1.json',aggregate)
 train=read(ws/'artifacts'/NC02/'training_protocol_freeze_v1.json')
 config=dict(arms=train['arms'],updates_per_fit=train['updates_per_fit'],seed=train['seed'],
  optimizer=train['optimizer'],loss=train['loss'],augmentation=train['augmentation'],
  versions=train['dependency_versions'],model_architectures={k:{kk:vv for kk,vv in v.items() if kk not in ('pointnet_source',)} for k,v in train['models'].items()},
  no_patient_split_or_target_payload_distributed=True,short_development_pilot=True)
 save(ROOT/'configs/spatial_fixed_pilot_v1.json',safe_aggregate(config))
 save(ROOT/'configs/gate_fixed_recipe_v1.json',safe_aggregate(dict(labels=spec['labels'],
  thresholds=spec['thresholds'],feature_recipe=spec['feature_recipe'],historical_limitations=spec['historical_limitations'])))
 save(ROOT/'manifests/source_dependency_audit_v1.json',dict(files=audit,external_import_roots=sorted(external),
  private_archive_excluded_from_public_zip=True,original_sources_byte_preserved=True,
  byte_preservation_scope='Local-only private source archive; public gate source is privacy-adapted',
  privacy_adapted_public_sources=[PRIVACY_SOURCE]))
 save(ROOT/'manifests/import_adaptations_v1.json',dict(files=transformations,gate_engine_original_source_sha256=sha(interface),
  gate_engine_functions=sorted(names),gate_engine_only_import_rewrites=True))
 public=_collect_public()
 artwork_check=_validate_artwork_manifest(public)
 save(ROOT/'manifests/public_files_v1.json',dict(files=public,no_patient_data=True,no_fitted_assets=True,
  no_private_absolute_paths=True,archive_sources_separate_private=True,
  case_images='Approved de-identified publication renderings only, not raw images or patient prediction arrays',
  exact_script_allowlist=True))
 if stage_only:
  report=dict(qualified=True,state='FULL_SCOPE_READY_AWAIT_ROOT_FIGURE_AND_APP_QA',public_files=len(public)+1,
   public_bytes=sum(r['bytes'] for r in public),root_QA_still_required=True,no_final_public_zip_created=True,
   patient_arrays_copied=0,weights_copied=0,real_patient_forwards=0,fits=0,artwork_manifest=artwork_check)
  if local_code_archive:
   report['local_code_archive']=_local_code_archive(audit)
   save(ROOT/'logs/local_code_archive_v1.json',report['local_code_archive'])
  if local_binding_report: report['local_asset_binding']=local_binding_report
  save(ROOT/'logs/full_scope_stage_v1.json',report); print(json.dumps(report,indent=2)); return report
 dist=ROOT/'dist'; dist.mkdir(exist_ok=True)
 out=dist/'StrokePointPlusPlus-0.1.0dev0-full-research.zip'
 if out.exists():
  out=dist/('StrokePointPlusPlus-0.1.0dev0-full-research-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
 members=[r['path'] for r in public]+['manifests/public_files_v1.json']
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for rel in sorted(members):
   info=zipfile.ZipInfo('StrokePointPlusPlus/'+rel,date_time=(2026,10,28,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
   info.create_system=3
   info.external_attr=(0o100644<<16); z.writestr(info,(ROOT/rel).read_bytes())
 with zipfile.ZipFile(out) as z:
  if z.testzip() is not None: raise RuntimeError('Built ZIP CRC mismatch')
 report=dict(qualified=True,zip_filename=out.name,zip_sha256=sha(out),zip_bytes=out.stat().st_size,
  public_file_count=len(members),private_archived_sources=len(audit),asset_manifest_count=len(assets),
  patient_payload_reads=0,patient_data_copied=0,fitted_assets_copied=0,model_forwards=0,fits=0,
  artwork_manifest=artwork_check,
  scope='Build/metadata/hash checks, not a real-patient inference validation')
 if local_code_archive:
  report['local_code_archive']=_local_code_archive(audit)
  save(ROOT/'logs/local_code_archive_v1.json',report['local_code_archive'])
 if local_binding_report: report['local_asset_binding']=local_binding_report
 save(ROOT/'logs/build_latest.json',report); print(json.dumps(report,indent=2))
 return report


if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--source-workspace',required=True)
 p.add_argument('--stage-only',action='store_true'); p.add_argument('--local-code-archive',action='store_true')
 p.add_argument('--local-bindings',action='store_true')
 args=p.parse_args(); build(args.source_workspace,args.stage_only,args.local_code_archive,args.local_bindings)
