"""Metadata/byte verification CLI, and explicit caller-bound optional inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def resource(name):
    return json.loads((PACKAGE/'resources'/name).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def resolve_assets(mapping_path):
    mapping_path=Path(mapping_path).resolve()
    mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    result={}
    for key,value in mapping.get('assets',mapping).items():
        if not isinstance(value,str):
            raise ValueError('Each logical asset maps to one explicit local path')
        value=os.path.expandvars(os.path.expanduser(value))
        if '${' in value:
            raise FileNotFoundError('Unresolved asset placeholder: '+key)
        p=Path(value)
        result[key]=p if p.is_absolute() else mapping_path.parent/p
    return result


def verify(assets=None):
    manifest=ROOT/'manifests/public_files_v1.json'
    if not manifest.is_file():
        # Installed wheels need not carry the enclosing source distribution.
        bundle=dict(available=False,note='Run from unpacked source release for full bundle byte verification')
    else:
        doc=json.loads(manifest.read_text(encoding='utf-8')); errors=[]
        for row in doc['files']:
            p=ROOT/row['path']
            if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
                errors.append(row['path'])
        bundle=dict(available=True,qualified=not errors,checked_files=len(doc['files']),mismatches=errors)
    asset_result=dict(supplied=False)
    if assets:
        paths=resolve_assets(assets); errors=[]; checked=[]
        expected={r['asset_id']:r for r in resource('model_assets_v1.json')['assets']}
        for key,p in paths.items():
            if key not in expected or not p.is_file() or digest(p)!=expected[key]['sha256']:
                errors.append(key)
            else:
                checked.append(key)
        asset_result=dict(supplied=True,qualified=not errors,checked=checked,mismatches=errors,
            note='Byte identity only; no model deserialization or clinical certification')
    return dict(bundle=bundle,assets=asset_result,patient_arrays_read=0,model_forwards=0,fits=0)


def main(argv=None):
    p=argparse.ArgumentParser(prog='strokepointpp',description='StrokePoint++ research-only reproducibility toolbox')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('inspect'); sub.add_parser('summary')
    v=sub.add_parser('verify'); v.add_argument('--assets')
    i=sub.add_parser('infer')
    i.add_argument('--method',choices=('gate','3d','fuse'),required=True)
    i.add_argument('--assets',required=True); i.add_argument('--input',required=True)
    i.add_argument('--output',required=True); i.add_argument('--device',default='cpu')
    i.add_argument('--fold',type=int,choices=(0,1),default=0)
    a=p.parse_args(argv)
    if a.command=='inspect':
        out=resource('release_summary_v1.json')
    elif a.command=='summary':
        out=resource('aggregate_results_v1.json')
    elif a.command=='verify':
        out=verify(a.assets)
        if out['bundle'].get('qualified') is False or out['assets'].get('qualified') is False:
            print(json.dumps(out,indent=2,ensure_ascii=False)); return 2
    else:
        from .adapters import infer_npz
        out=infer_npz(a.method,a.assets,a.input,a.output,a.device,a.fold)
    print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0
