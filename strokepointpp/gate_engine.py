"""Release engineering extraction of six fixed numerical functions; import adaptation only."""
from typing import Mapping,Any
import numpy as np
LABELS=(1,2,3,4,5,6,7,8,9,10,11,12,15)
SOURCE_ORDER=((0,20261020),(0,20261021),(1,20261020),(1,20261021))
GEOMETRY_KEYS=('xyz','s1','s2','g1','g2','i21','w21','i10','w10','path_features','native_gd','euclidean_ids','euclidean_mask','euclidean_active')
def validate_visible(visible: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact label-free geometry contract; no hidden point subsampling/realignment."""
    if set(visible) != set(GEOMETRY_KEYS):
        raise ValueError('Exact visible geometry keys required; labels/identity metadata forbidden')
    v = {k: np.asarray(visible[k]) for k in GEOMETRY_KEYS}
    shapes = dict(xyz=(2048, 3), s1=(128,), s2=(32,), g1=(128, 32), g2=(32, 32),
        i21=(128, 3), w21=(128, 3), i10=(2048, 3), w10=(2048, 3),
        path_features=(32, 128, 2), native_gd=(32, 128), euclidean_active=(32,))
    for key, shape in shapes.items():
        if v[key].shape != shape:
            raise ValueError(key + ' shape changed: ' + str(v[key].shape))
    if v['euclidean_ids'].ndim != 2 or v['euclidean_ids'].shape[0] != 32:
        raise ValueError('Euclidean local support must be 32 by fixed K')
    if v['euclidean_mask'].shape != v['euclidean_ids'].shape:
        raise ValueError('Euclidean local support mask mismatch')
    for key in ('xyz', 'w21', 'w10', 'path_features'):
        if v[key].dtype != np.float32 or not np.isfinite(v[key]).all():
            raise ValueError('Original float32 finite visible operator required: ' + key)
    if not np.issubdtype(v['native_gd'].dtype, np.floating) or np.isnan(v['native_gd']).any() or np.any(v['native_gd'] < 0):
        raise ValueError('Native path distance must be nonnegative finite/+inf, no NaN')
    finite_gd = np.isfinite(v['native_gd'])
    encoded = np.stack([np.where(finite_gd, v['native_gd'] / 120., 1.), finite_gd.astype(float)], -1).astype(np.float32)
    if not np.array_equal(v['path_features'], encoded):
        raise ValueError('Visible path encoding differs from fixed raw-distance/120 plus finite-bit rule')
    limits = dict(s1=2048, s2=128, g1=2048, g2=128, i21=32, i10=128, euclidean_ids=128)
    for key, limit in limits.items():
        if v[key].dtype != np.int64 or np.any((v[key] < 0) | (v[key] >= limit)):
            raise ValueError('Original geometry integer index/range changed: ' + key)
    if any(v[k].dtype != np.bool_ for k in ('euclidean_mask', 'euclidean_active')):
        raise ValueError('Original local masks must be Boolean')
    for key in ('w21', 'w10'):
        if np.any(v[key] < 0) or not np.allclose(v[key].sum(-1), 1., atol=2e-6, rtol=0):
            raise ValueError('Invalid original interpolation weights: ' + key)
    return v


def build_batch(visible: Mapping[str, np.ndarray], device: str):
    """Exact R0V2 geodesic-interpolation override, preserving source dtypes/order."""
    import torch
    case = validate_visible(visible)
    geo = {k: torch.from_numpy(case[k][None]).to(device) for k in
           ('s1', 's2', 'g1', 'g2', 'i21', 'w21', 'i10', 'w10')}
    for name in ('ids', 'mask', 'active'):
        geo['local_' + name] = torch.from_numpy(case['euclidean_' + name][None]).to(device)
    geo['path_features'] = torch.from_numpy(case['path_features'][None]).to(device)
    gd = case['native_gd'].T
    ids, weights = case['i21'].copy(), case['w21'].copy()
    for row in range(len(gd)):
        valid = np.flatnonzero(np.isfinite(gd[row]))
        if not len(valid):
            continue
        picked = sorted(valid, key=lambda i: (float(gd[row, i]), int(i)))[:3]
        while len(picked) < 3:
            picked.append(picked[-1])
        ids[row] = picked
        w = 1 / np.maximum(gd[row, picked], 1e-8)
        weights[row] = w / w.sum()
    geo['i21'] = torch.from_numpy(ids[None]).to(device)
    geo['w21'] = torch.from_numpy(weights[None]).to(device)
    return torch.from_numpy(case['xyz'][None]).to(device), geo


def _extract(source: Mapping[str, Any], visible: Mapping[str, np.ndarray], device: str):
    import torch
    from ._frozen.pointnet import gather
    model = source['model']
    seen = {}
    handles = [module.register_forward_hook(lambda m, i, o, n=name: seen.update({n: o.detach()}))
        for name, module in (('sa1', model.sa1), ('global', model.global_net), ('final', model.fp1))]
    try:
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            xyz, geo = build_batch(visible, device)
            model(xyz, geo)
            fine = (gather(seen['sa1'].amax(2), geo['i10']) * geo['w10'][..., None]).sum(2)
            return {key: value.cpu().numpy() for key, value in
                dict(xyz=xyz, F=fine, G=seen['global'].amax(1), L=seen['final']).items()}
    finally:
        for handle in handles:
            handle.remove()


def calibrated_logits(packet, projection, norm, weight, bias, prior):
    """CPU NumPy archived144-design arithmetic including fixed fold log prior."""
    gp = packet['G'] @ projection['global_projection']
    c = np.concatenate((packet['xyz'], np.broadcast_to(gp[:, None, :], (*packet['xyz'].shape[:2], 61))), -1)
    zc = (c - norm['Cmean']) / norm['Cstd']
    zf = (packet['F'] - norm['Fmean']) / norm['Fstd']
    x = np.concatenate((zc, zf, (zc @ projection['context_product']) * (zf @ projection['fine_product'])), -1).astype(np.float32)
    return ((x @ weight.T + bias) + np.log(prior)[None, None, :]).astype(np.float32)[0]


def predict_from_packets(packets, visible_xyz, heads):
    """Label-free aggregation/gating; also supports CPU-only synthetic fixtures."""
    from scipy.special import softmax
    from ._frozen.gate_arrays import aggregate_scan_sources, scan_features, predict_serialized, apply_patient_gate
    aggregated = aggregate_scan_sources(packets, SOURCE_ORDER, expected_n_points=2048)
    features = scan_features(visible_xyz, **aggregated)
    presence = {name: float(predict_serialized(heads[name], features[name][None])['probability'][0]) for name in ('pcom', 'a2')}
    ungated_index = aggregated['cal'].argmax(-1)
    gated_index = apply_patient_gate(aggregated['cal'], presence['pcom'], presence['a2'])
    blocked = ([] if presence['pcom'] >= .05 else [7, 8]) + ([] if presence['a2'] >= .05 else [12])
    gated_cal = aggregated['cal'].copy()
    gated_cal[:, blocked] = -np.inf
    if not np.array_equal(gated_cal.argmax(-1), gated_index):
        raise RuntimeError('Diagnostic gated probability labels differ from original gate')
    return dict(cal_logits=aggregated['cal'], ungated_probs=softmax(aggregated['cal'], axis=-1),
        gated_logits=gated_cal, gated_probs=softmax(gated_cal, axis=-1),
        ungated_indices=ungated_index, gated_indices=gated_index,
        ungated_labels=np.asarray(LABELS)[ungated_index], gated_labels=np.asarray(LABELS)[gated_index],
        F=aggregated['F'], L=aggregated['L'], pcom_features=features['pcom'], a2_features=features['a2'],
        pcom_probability=presence['pcom'], a2_probability=presence['a2'], blocked_indices=np.asarray(blocked, dtype=np.int64),
        gated_probs_definition='Diagnostic renormalization after fixed class blocking; not a separately fitted or validated risk probability',
        single_modality='MRA', source_order=SOURCE_ORDER, source_model_packets=4, model_forward_calls=0,
        hidden_coordinate_alignment=False, target_use=False, fits=0)


def predict_yz(bundle: Mapping[str, Any], visible: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Exactly4 frozen forward passes, full2048 points; no label/patient selector."""
    case = validate_visible(visible)
    if tuple((s['fold'], s['seed']) for s in bundle['sources']) != SOURCE_ORDER:
        raise ValueError('Frozen deployment source order mismatch')
    packets = []
    for source in bundle['sources']:
        packet = _extract(source, case, bundle['device'])
        cal = calibrated_logits(packet, bundle['projection'], source['normalization'],
            source['weight'], source['bias'], bundle['priors'][source['fold']])
        packets.append(dict(fold=source['fold'], seed=source['seed'], cal=cal, F=packet['F'][0], L=packet['L'][0]))
    result = predict_from_packets(packets, case['xyz'], bundle['heads'])
    result['protocol_sha256'] = bundle['protocol_sha256']
    result['model_forward_calls'] = 4
    return result

