"""Prospective small-budget development learning; not an independent test."""
import copy
import numpy as np
import torch
from .pointnet import PointNetPlusPlus, PointNet, gather

ARMS = ('pointnet', 'pointnetpp', 'dual_euclidean', 'dual_graph')


class SeparatedPointNetPP(PointNetPlusPlus):
    """Independent local SA2 and Euclidean global SA2, shared original decoder.

    Mask/support decisions contain no labels. Insufficient local support routes
    to the global-SA2 features. No zero-initialized learned gate blocks gradients.
    """
    def __init__(self):
        super().__init__()
        self.local_sa2 = copy.deepcopy(self.sa2)

    def forward(self, xyz, geo):
        p1 = gather(xyz, geo['s1']); p2 = gather(p1, geo['s2'])
        f1 = self.sa1(gather(xyz, geo['g1']) - p1[:, :, None]).amax(2)
        global_f2 = self.sa2(torch.cat([
            gather(p1, geo['g2']) - p2[:, :, None], gather(f1, geo['g2'])], -1)).amax(2)
        context = self.global_net(torch.cat([p2, global_f2], -1)).amax(1, keepdim=True)
        ids, mask, active = geo['local_ids'], geo['local_mask'], geo['local_active']
        local = self.local_sa2(torch.cat([
            gather(p1, ids) - p2[:, :, None], gather(f1, ids)], -1))
        # Finite sentinel avoids all-invalid max/backward NaNs. Such rows fall back.
        local = local.masked_fill(~mask[..., None], torch.finfo(local.dtype).min).amax(2)
        active = active & mask.any(2)
        f2 = torch.where(active[..., None], local, global_f2)
        f21 = (gather(f2, geo['i21']) * geo['w21'][..., None]).sum(2)
        up1 = self.fp2(torch.cat([f1, f21, context.expand(-1, p1.shape[1], -1)], -1))
        up0 = (gather(up1, geo['i10']) * geo['w10'][..., None]).sum(2)
        return self.head(self.fp1(torch.cat([xyz, up0], -1)))


def make_model(arm, seed):
    torch.manual_seed(seed)
    return PointNet() if arm == 'pointnet' else PointNetPlusPlus() if arm == 'pointnetpp' else SeparatedPointNetPP()


def batch(cases, keys, arm):
    arrays = {k: torch.from_numpy(np.stack([cases[key][k] for key in keys]))
              for k in cases[keys[0]] if not k.startswith(('graph_', 'euclidean_'))}
    xyz, y = arrays.pop('xyz'), arrays.pop('y')
    if arm.startswith('dual_'):
        prefix = 'graph_' if arm == 'dual_graph' else 'euclidean_'
        for name in ('ids', 'mask', 'active'):
            arrays['local_' + name] = torch.from_numpy(np.stack([cases[key][prefix + name] for key in keys]))
    return xyz, arrays, y


def loss_fn(logits, target, weights):
    logp = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    w = weights[target]
    return -(logp * w).sum() / w.sum()


def dice(y, pred):
    cm = np.bincount(y * 13 + pred, minlength=169).reshape(13, 13)
    den = cm.sum(0) + cm.sum(1); valid = den > 0
    return float((2 * np.diag(cm)[valid] / den[valid]).mean())
