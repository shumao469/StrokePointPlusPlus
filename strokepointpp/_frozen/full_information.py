"""C1 full-input-information control and constrained local routing null."""
import numpy as np
import torch
from .pointnet import gather, mlp
from .local_global import SeparatedPointNetPP

ARMS = ('rep_euclidean', 'rep_graph', 'info_euclidean', 'info_graph', 'info_rewired')


class FullInformationPP(SeparatedPointNetPP):
    """Every info arm reads the identical FULL32x128 native path-distance matrix.

    Dense shared learned set aggregation allows the Euclidean local-routing
    control to exploit full-mask information, not just a lossy fixed histogram.
    This is graph-aware, not a topology-free comparator. The null scrambles only
    local routing; intact full-distance information remains in ALL info arms.
    """
    def __init__(self):
        super().__init__()
        self.info_net = mlp([69, 64, 128])

    def forward(self, xyz, geo):
        p = gather(xyz, geo['s1']); q = gather(p, geo['s2'])
        f = self.sa1(gather(xyz, geo['g1']) - p[:, :, None]).amax(2)
        base = self.sa2(torch.cat([gather(p, geo['g2']) - q[:, :, None], gather(f, geo['g2'])], -1)).amax(2)
        dense = self.info_net(torch.cat([
            p[:, None] - q[:, :, None], f[:, None].expand(-1, q.shape[1], -1, -1), geo['path_features']], -1)).amax(2)
        context = self.global_net(torch.cat([q, (base + dense) * .5], -1)).amax(1, keepdim=True)
        ids, mask = geo['local_ids'], geo['local_mask']
        local = self.local_sa2(torch.cat([gather(p, ids) - q[:, :, None], gather(f, ids)], -1))
        local = local.masked_fill(~mask[..., None], torch.finfo(local.dtype).min).amax(2)
        local = torch.where((geo['local_active'] & mask.any(2))[..., None], local, base)
        local = (local + dense) * .5
        interp = (gather(local, geo['i21']) * geo['w21'][..., None]).sum(2)
        up = self.fp2(torch.cat([f, interp, context.expand(-1, p.shape[1], -1)], -1))
        fine = (gather(up, geo['i10']) * geo['w10'][..., None]).sum(2)
        return self.head(self.fp1(torch.cat([xyz, fine], -1)))


def make_model(arm, seed):
    assert arm in ARMS
    torch.manual_seed(seed)
    return FullInformationPP() if arm.startswith('info_') else SeparatedPointNetPP()


def encode_paths(gd):
    finite = np.isfinite(gd)
    return np.stack([np.where(finite, gd / 120., 1.), finite.astype(float)], -1).astype(np.float32)


def rewired_pack(original, eu, seed, attempts=20000, bin_mm=5.):
    """Directed query-source anchor operator, NOT a native voxel-graph rewrite.

    Double-edge swaps preserve row/column degrees, self edges, support/fallback
    masks, and each query's exact5mm Euclidean-distance-bin histogram. No labels.
    """
    ids = original['ids'].copy(); mask = original['mask'].copy(); active = original['active'].copy()
    rows = np.broadcast_to(np.arange(len(ids))[:, None], ids.shape)
    nonself = mask & active[:, None] & (eu[rows, ids] > 1e-5)
    positions = np.argwhere(nonself)
    adjacency = np.zeros(eu.shape, bool); adjacency[rows[mask], ids[mask]] = True
    bins = np.floor(eu / bin_mm).astype(np.int64)
    rng = np.random.default_rng(seed); accepted = 0
    if len(positions) >= 2:
        choices = rng.integers(0, len(positions), (attempts, 2))
        for first, second in choices:
            a, ai = positions[first]; b, bi = positions[second]
            if a == b: continue
            u, v = ids[a, ai], ids[b, bi]
            if u == v or adjacency[a, v] or adjacency[b, u]: continue
            if eu[a, v] <= 1e-5 or eu[b, u] <= 1e-5: continue
            if bins[a, u] != bins[a, v] or bins[b, v] != bins[b, u]: continue
            ids[a, ai], ids[b, bi] = v, u
            adjacency[a, u] = adjacency[b, v] = False
            adjacency[a, v] = adjacency[b, u] = True
            accepted += 1
    final_adjacency = np.zeros(eu.shape, bool); final_adjacency[rows[mask], ids[mask]] = True
    old_adjacency = np.zeros(eu.shape, bool); old_adjacency[rows[mask], original['ids'][mask]] = True
    used_old = old_adjacency & active[:, None] & (eu > 1e-5)
    changed = float(np.count_nonzero(used_old & ~final_adjacency) / max(1, used_old.sum()))
    return dict(ids=ids, mask=mask, active=active), dict(accepted_swaps=accepted,
                changed_used_nonself_edge_fraction=changed, attempts=attempts, euclidean_bin_mm=bin_mm)


def check_rewire(original, rewired, eu):
    r = np.arange(len(eu))[:, None]; bins = np.floor(eu / 5.).astype(int)
    np.testing.assert_array_equal(original['mask'], rewired['mask'])
    np.testing.assert_array_equal(original['active'], rewired['active'])
    before = original['ids']; after = rewired['ids']; valid = original['mask']
    np.testing.assert_array_equal(np.bincount(before[valid], minlength=eu.shape[1]), np.bincount(after[valid], minlength=eu.shape[1]))
    for row in range(len(eu)):
        ok = valid[row]
        assert len(set(after[row, ok])) == np.count_nonzero(ok)
        np.testing.assert_array_equal(np.sort(bins[row, before[row, ok]]), np.sort(bins[row, after[row, ok]]))
        np.testing.assert_array_equal(eu[row, before[row]] <= 1e-5, eu[row, after[row]] <= 1e-5)
    np.testing.assert_array_equal(((eu[r, before] > 1e-5) & valid).sum(1), ((eu[r, after] > 1e-5) & valid).sum(1))


def batch(cases, keys, arm):
    first = cases[keys[0]]
    names = ('s1', 's2', 'g1', 'g2', 'i21', 'w21', 'i10', 'w10')
    geo = {k: torch.from_numpy(np.stack([cases[p][k] for p in keys])) for k in names}
    prefix = 'rewired' if arm == 'info_rewired' else 'graph' if arm.endswith('graph') else 'euclidean'
    for name in ('ids', 'mask', 'active'):
        geo['local_' + name] = torch.from_numpy(np.stack([cases[p][prefix + '_' + name] for p in keys]))
    if arm.startswith('info_'):
        geo['path_features'] = torch.from_numpy(np.stack([cases[p]['path_features'] for p in keys]))
    return torch.from_numpy(np.stack([cases[p]['xyz'] for p in keys])), geo, torch.from_numpy(np.stack([cases[p]['y'] for p in keys]))
