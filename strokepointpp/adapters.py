"""New parameterized RELEASE engineering adapter, not original sealed runner.

Fixed algorithm arithmetic is copied into gate_engine by the build script with
only import-namespace substitutions. Users bind authorized visible NPZ and exact
local SHA-matched fitted assets. No data/weights download, fitting or tuning.
"""
import json
from pathlib import Path

from .cli import resource,resolve_assets,digest


def _assets(mapping):
    expected={r['asset_id']:r for r in resource('model_assets_v1.json')['assets']}
    paths=resolve_assets(mapping)
    def get(key):
        if key not in paths:
            raise FileNotFoundError('Required asset not mapped: '+key+'; see model_assets.local.example.json')
        path=paths[key]
        if not path.is_file():
            raise FileNotFoundError('Required authorized local asset is absent: '+key)
        if key not in expected or digest(path)!=expected[key]['sha256']:
            raise ValueError('Asset SHA256 mismatch: '+key)
        return path
    return get


def _device(torch,device):
    d=torch.device(device)
    if d.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent device substitution')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return d


def load_gate(mapping,device='cpu'):
    import numpy as np
    import torch
    from ._frozen.full_information import make_model
    from ._frozen.gate_arrays import predict_serialized
    from .gate_engine import SOURCE_ORDER
    get=_assets(mapping); d=_device(torch,device)
    projection_path=get('gate_projection')
    with np.load(projection_path,allow_pickle=False) as z:
        projection={k:z[k] for k in ('global_projection','context_product','fine_product')}
    prior_doc=json.loads(get('gate_priors').read_text(encoding='utf-8'))['priors']
    priors={f:np.asarray(prior_doc[str(f)],np.float64) for f in (0,1)}
    heads={h:json.loads(get('gate_'+h+'_head').read_text(encoding='utf-8')) for h in ('pcom','a2')}
    for head in heads.values():
        if head['threshold']!=.05:
            raise ValueError('Fixed head threshold changed')
        predict_serialized(head,np.zeros((1,256),np.float64))
    sources=[]
    for fold,seed in SOURCE_ORDER:
        prefix=f'gate_f{fold}_s{seed}'
        with torch.random.fork_rng(devices=[]):
            model=make_model('info_euclidean',seed+1000*fold)
        state=torch.load(get(prefix+'_checkpoint'),map_location='cpu',weights_only=True)
        if (state['arm']!='info_euclidean' or state['fold']!=fold or state['seed']!=seed
                or state['protocol_sha256']!='10344c4da487caeb50961d738c424c4bae4983cb5667f92bbad0f8141f702f73'):
            raise ValueError('Original checkpoint lineage metadata mismatch')
        model.load_state_dict(state['state_dict'],strict=True)
        model.eval().requires_grad_(False).to(d)
        with np.load(get(prefix+'_normalization'),allow_pickle=False) as z:
            norm={k:z[k] for k in ('Cmean','Cstd','Fmean','Fstd','Lmean','Lstd')}
        with np.load(get(prefix+'_readout'),allow_pickle=False) as z:
            weight,bias=z['weight_float32'],z['bias_float32']
        sources.append(dict(fold=fold,seed=seed,model=model,normalization=norm,weight=weight,bias=bias))
    return dict(sources=sources,projection=projection,priors=priors,heads=heads,device=str(d),
        protocol_sha256=resource('model_assets_v1.json')['gate_recipe_protocol_sha256'])


def predict_gate(bundle,visible):
    from .gate_engine import predict_yz
    return predict_yz(bundle,visible)


def load_spatial(mapping,arm='BASE3D',fold=0,device='cpu'):
    import torch
    from ._frozen.spatial_models import create_model
    if arm not in ('BASE3D','CONCAT','JOINT','NO_FEEDBACK') or fold not in (0,1):
        raise ValueError('Unknown archived spatial arm/fold')
    get=_assets(mapping); d=_device(torch,device)
    saved=torch.load(get(f'spatial_f{fold}_{arm}'),map_location='cpu',weights_only=True)
    if saved['fit_id']!=f'fold{fold}__{arm}' or saved['step']!=2000:
        raise ValueError('Frozen short-pilot model lineage mismatch')
    model=create_model(arm,20261027+fold)
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval().requires_grad_(False).to(d)


def predict_spatial(model,visible):
    import numpy as np
    import torch
    from ._frozen.spatial_models import GEOMETRY_KEYS,LABELS
    expected={'volume_CDHW','local_NCDHW','point_grid_XYZ','point_xyz',*GEOMETRY_KEYS}
    if set(visible)!=expected:
        raise ValueError('Exact label-free prepared spatial NPZ keys required')
    d=next(model.parameters()).device
    def tensor(key): return torch.from_numpy(np.asarray(visible[key])[None]).to(d)
    with torch.inference_mode(),torch.autocast(device_type=d.type,dtype=torch.bfloat16,enabled=d.type=='cuda'):
        out=model(tensor('volume_CDHW'),tensor('local_NCDHW'),tensor('point_grid_XYZ'),tensor('point_xyz'),
            {k:tensor(k) for k in GEOMETRY_KEYS})
    logits=out['point_logits'][0].float().cpu().numpy()
    return dict(point_logits=logits,point_probs=torch.from_numpy(logits).softmax(-1).numpy(),
        point_labels=np.asarray(LABELS)[logits.argmax(-1)],
        presence_logits=out['presence_logits'][0].float().cpu().numpy())


def infer_npz(method,mapping,input_path,output_path,device='cpu',fold=0):
    import numpy as np
    input_path,output_path=Path(input_path),Path(output_path)
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite an existing prediction')
    if method=='gate':
        from .gate_engine import GEOMETRY_KEYS
        expected=set(GEOMETRY_KEYS)
    else:
        from ._frozen.spatial_models import GEOMETRY_KEYS
        expected={'volume_CDHW','local_NCDHW','point_grid_XYZ','point_xyz',*GEOMETRY_KEYS}
    with np.load(input_path,allow_pickle=False) as z:
        if set(z.files)!=expected:
            raise ValueError('Input NPZ must contain exactly the documented visible keys; labels/extra arrays are not decoded')
        visible={k:z[k] for k in z.files}
    if method=='gate':
        result=predict_gate(load_gate(mapping,device),visible)
    else:
        result=predict_spatial(load_spatial(mapping,'BASE3D' if method=='3d' else 'CONCAT',fold,device),visible)
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    return dict(method=method,output=str(output_path),output_sha256=digest(output_path),fits=0,
        new_release_adapter=True,raw_image_preprocessing_performed=False,
        caveat='Given-foreground research prediction; no clinical safety or exact CPU/CUDA replay certification')
