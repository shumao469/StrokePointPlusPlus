"""Point-cloud baselines for conditional CoW branch labeling.

PointNet++ SSG adaptation: farthest-point sampling, radius neighborhoods, shared
MLP/max set abstraction, global pooling and inverse-distance feature propagation.
Not the original TensorFlow implementation or a CTA-to-vessel segmentation model.
Point ordering and anatomical branch labels NEVER define neighborhood connectivity.
"""
import numpy as np
import torch
from torch import nn


def fps(xyz, n):
    n = min(n, len(xyz))
    selected = np.empty(n, np.int64)
    distance = np.full(len(xyz), np.inf)
    farthest = int(np.argmax(((xyz-xyz.mean(0))**2).sum(1)))
    for i in range(n):
        selected[i] = farthest
        distance = np.minimum(distance, ((xyz-xyz[farthest])**2).sum(1))
        # Exclude selected indices even for duplicate coordinates.
        distance[selected[:i+1]] = -1
        farthest = int(np.argmax(distance))
    return selected


def neighborhoods(xyz, centers, radius, k):
    d2 = ((xyz[centers, None]-xyz[None])**2).sum(-1)
    ids = np.argsort(d2, axis=1, kind='stable')[:, :min(k, len(xyz))]
    invalid = np.take_along_axis(d2, ids, 1) > radius**2
    ids[invalid] = np.broadcast_to(centers[:, None], ids.shape)[invalid]
    if ids.shape[1] < k:
        ids = np.pad(ids, ((0, 0), (0, k-ids.shape[1])), mode='edge')
    return ids.astype(np.int64)


def interpolation(fine, coarse):
    d2 = ((fine[:, None]-coarse[None])**2).sum(-1)
    idx = np.argsort(d2, axis=1, kind='stable')[:, :3]
    distance = np.sqrt(np.take_along_axis(d2, idx, 1))
    weight = 1.0/np.maximum(distance, 1e-8)
    weight /= weight.sum(1, keepdims=True)
    return idx.astype(np.int64), weight.astype(np.float32)


def geometry(xyz):
    """Input coordinates only; reused by every architecture and every split."""
    s1 = fps(xyz, 128); p1 = xyz[s1]
    s2 = fps(p1, 32); p2 = p1[s2]
    i21, w21 = interpolation(p1, p2)
    i10, w10 = interpolation(xyz, p1)
    return {'s1': s1, 'g1': neighborhoods(xyz, s1, .15, 32),
            's2': s2, 'g2': neighborhoods(p1, s2, .30, 32),
            'i21': i21, 'w21': w21, 'i10': i10, 'w10': w10}


def gather(x, idx):
    batch = torch.arange(x.shape[0], device=x.device).view(-1, *([1]*(idx.ndim-1)))
    return x[batch, idx]


def mlp(channels):
    layers=[]
    for a,b in zip(channels[:-1],channels[1:]):
        layers.extend([nn.Linear(a,b),nn.LayerNorm(b),nn.ReLU()])
    return nn.Sequential(*layers)


class PointwiseMLP(nn.Module):
    def __init__(self, classes=13):
        super().__init__(); self.net=nn.Sequential(mlp([3,64,128,128]),nn.Linear(128,classes))
    def forward(self, xyz, geo=None):return self.net(xyz)


class PointNet(nn.Module):
    """PointNet-style local/global baseline; no T-Net alignment to preserve laterality."""
    def __init__(self, classes=13):
        super().__init__(); self.local=mlp([3,64,128]); self.global_net=mlp([128,256])
        self.head=nn.Sequential(mlp([384,128,64]),nn.Linear(64,classes))
    def forward(self, xyz, geo=None):
        f=self.local(xyz); g=self.global_net(f).amax(1,keepdim=True).expand(-1,xyz.shape[1],-1)
        return self.head(torch.cat([f,g],-1))


class PointNetPlusPlus(nn.Module):
    def __init__(self, classes=13):
        super().__init__()
        self.sa1=mlp([3,32,64]);self.sa2=mlp([67,64,128]);self.global_net=mlp([131,128,256])
        self.fp2=mlp([448,128,128]);self.fp1=mlp([131,128,64]);self.head=nn.Linear(64,classes)
    def forward(self, xyz, geo):
        p1=gather(xyz,geo['s1']); p2=gather(p1,geo['s2'])
        relative1=gather(xyz,geo['g1'])-p1[:,:,None]
        f1=self.sa1(relative1).amax(2)
        relative2=gather(p1,geo['g2'])-p2[:,:,None]
        f2=self.sa2(torch.cat([relative2,gather(f1,geo['g2'])],-1)).amax(2)
        global_feature=self.global_net(torch.cat([p2,f2],-1)).amax(1,keepdim=True)
        f2_to_1=(gather(f2,geo['i21'])*geo['w21'][...,None]).sum(2)
        up1=self.fp2(torch.cat([f1,f2_to_1,global_feature.expand(-1,p1.shape[1],-1)],-1))
        up0=(gather(up1,geo['i10'])*geo['w10'][...,None]).sum(2)
        return self.head(self.fp1(torch.cat([xyz,up0],-1)))


MODELS={'pointwise_mlp':PointwiseMLP,'pointnet':PointNet,'pointnetpp':PointNetPlusPlus}
