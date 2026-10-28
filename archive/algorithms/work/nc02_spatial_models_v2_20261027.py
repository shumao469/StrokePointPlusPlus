"""NC02 v2 FP32-sensitive kernels for the unchanged fixed-pilot architecture.

V2 preserves all parameters, architecture, initialization and supervision. Only
numerically sensitive sampling/loss operations explicitly use FP32 under bf16
autocast; this is not an additional fit or changed model configuration. V1 is
retained unchanged.

No project-data IO, weight loading, optimizer, training, or patient forward
occurs on import. Backbones are constructed FROM SCRATCH. The official
PlainConvUNet architecture is not the entire nnUNet planning/training system;
short fixed-pilot results must not be called a fully trained nnUNet baseline.
ISLES infarct weights are not loaded or applicable to this anatomical task.

Every arm consumes the SAME MRA raw volume, physical local5cubed patches,
normalized volume point-grid and public point XYZ. BASE3D uses a coordinate
MLP, while CONCAT/JOINT/NO_FEEDBACK use the actual PointNet++ SSG network
from src.vascutime_pointnet, with its final classifier replaced by Identity.
Its neighborhoods/interpolation must be the fixed coordinate-only geometry,
not label-guided neighborhoods or unannounced native-geodesic substitutions.
BASE3D is an enhanced image baseline (U-Net + local patches + coordinates),
not an unmodified pure nnUNet. The other arms' same-information CONCAT
control is not weakened. Ordinary concatenation, FiLM and presence feedback
are not, by themselves, claimed to be novel mechanisms.

Axes: volume[B,C,D=K,H=J,W=I]; grid_xyz[B,N,3]=(I,J,K) normalized to[-1,1]
for that SAME volume using align_corners=True. A crop/affine/native-voxel
conversion belongs to the independently frozen data loader. Local patches
have D,H,W=5 and fixed physical offsets supplied by that loader; this module
does not silently reinterpret native5voxel patches as fixed-mm patches.

Dense channels14: [background] + canonical13 vascular labels. Point channels
13 contain vascular labels only. Complete-native presence targets are13
binary values. Candidate/patient labels are NEVER broadcast to point GT.
Given-GT foreground sampling or a GT-foreground crop remains conditional
anatomical labeling, not a validated end-to-end foreground extraction result.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from src.vascutime_pointnet import PointNetPlusPlus, geometry as coordinate_geometry


LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)
DENSE_LABELS = (0,) + LABELS
ARMS = ("BASE3D", "CONCAT", "JOINT", "NO_FEEDBACK")
GEOMETRY_KEYS = ("s1", "s2", "g1", "g2", "i21", "w21", "i10", "w10")
FEATURES = (32, 64, 128, 256, 320)
LOCAL_OFFSETS_MM = (-1.2, -.6, 0., .6, 1.2)
LOSS_WEIGHTS = {"dense_CE": 1., "foreground_Dice": 1., "point_CE": 1., "presence_BCE": 1.}


def voxel_indices_to_grid(indices_xyz: torch.Tensor, shape_dhw: tuple[int, int, int]) -> torch.Tensor:
    if indices_xyz.shape[-1] != 3 or not indices_xyz.is_floating_point():
        raise ValueError("Floating voxel indices with last dimension I,J,K required")
    if len(shape_dhw) != 3 or any(int(s) != s or s < 2 for s in shape_dhw):
        raise ValueError("D,H,W must be integers >=2")
    extent = indices_xyz.new_tensor((shape_dhw[2] - 1, shape_dhw[1] - 1, shape_dhw[0] - 1))
    if not torch.isfinite(indices_xyz).all() or torch.any(indices_xyz < 0) or torch.any(indices_xyz > extent):
        raise ValueError("Nonfinite/out-of-volume voxel index")
    return 2. * indices_xyz / extent - 1.


def sample_volume(volume: torch.Tensor, grid_xyz: torch.Tensor) -> torch.Tensor:
    """B,C,D,H,W and B,N,XYZ → B,N,C; fixed align_corners=True."""
    if volume.ndim != 5 or grid_xyz.ndim != 3 or grid_xyz.shape[-1] != 3 or volume.shape[0] != grid_xyz.shape[0]:
        raise ValueError("volume/grid dimensions mismatch")
    if not torch.isfinite(grid_xyz).all() or torch.any(grid_xyz < -1) or torch.any(grid_xyz > 1):
        raise ValueError("Sampling grid outside bound volume")
    # CUDA3D grid_sample support is not assumed for bf16. Keep both operands
    # explicitly FP32 even when the surrounding model uses bf16 autocast.
    with torch.autocast(device_type=volume.device.type, enabled=False):
        sampled = F.grid_sample(volume.float(), grid_xyz.float()[:, :, None, None, :],
                                mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, :, 0, 0].transpose(1, 2)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or mask.shape != values.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected B,N,C values and bool B,N mask")
    n = mask.sum(1)
    return (values * mask[:, :, None].to(values.dtype)).sum(1) / n.clamp_min(1)[:, None], n > 0


def build_coordinate_geometry(point_xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Pure coordinate-only preparation; caller controls data scope and time.

    Input B,N,3 or N,3; output always has the B dimension. The PointNet++
    radii are the archived .15/.30 in centered-world/100 units. This routine
    uses no labels and no voxel graph. It does not perform file IO.
    """
    xyz = np.asarray(point_xyz)
    if xyz.ndim == 2:
        xyz = xyz[None]
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[1] < 32 or not np.isfinite(xyz).all():
        raise ValueError("Finite B,N>=32,3 coordinates required")
    rows = [coordinate_geometry(row.astype(np.float32, copy=False)) for row in xyz]
    return {k: np.stack([row[k] for row in rows]) for k in GEOMETRY_KEYS}


def _mlp(nin: int, widths: tuple[int, ...]) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = nin
    for width in widths:
        modules.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.LeakyReLU(.01)))
        previous = width
    return nn.Sequential(*modules)


def _make_unet() -> PlainConvUNet:
    unet = PlainConvUNet(
        input_channels=1, n_stages=5, features_per_stage=FEATURES,
        conv_op=nn.Conv3d, kernel_sizes=(3,) * 5, strides=(1, 2, 2, 2, 2),
        n_conv_per_stage=(2,) * 5, num_classes=14,
        n_conv_per_stage_decoder=(2,) * 4, conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": .01, "inplace": True},
        deep_supervision=False)
    # Official decoder creates convenience-only auxiliary output convolutions
    # even when deep supervision=False, but never executes them. Remove their
    # unused parameters identically for every arm, preserving the exact forward.
    for index in range(len(unet.decoder.seg_layers) - 1):
        unet.decoder.seg_layers[index] = nn.Identity()
    unet.apply(PlainConvUNet.initialize)
    return unet


class SpatialAnatomicalModel(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown fixed NC02 arm")
        self.arm = arm
        # All common modules are initialized before arm-specific modules so
        # create_model(...,same_seed) gives exact common starting parameters.
        self.unet = _make_unet()
        self.local_encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3), nn.GroupNorm(4, 16), nn.LeakyReLU(.01),
            nn.Conv3d(16, 32, 3), nn.GroupNorm(8, 32), nn.LeakyReLU(.01))
        # Same local image32 + decoder32 + global decoder32 + XYZ3 + branch64.
        self.fusion = _mlp(163, (128, 64))
        self.point_head = nn.Linear(64, 13)
        self.presence_head = nn.Sequential(_mlp(96, (64,)), nn.Linear(64, 13))
        if arm in ("JOINT", "NO_FEEDBACK"):
            self.context_modulation = nn.Linear(96, 128)
        if arm == "JOINT":
            self.presence_feedback = nn.Parameter(torch.zeros(13))
        if arm == "BASE3D":
            self.coordinate_branch = _mlp(3, (64, 64))
        else:
            self.pointnet = PointNetPlusPlus(classes=13)
            self.pointnet.head = nn.Identity()

    def metadata(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        return dict(arm=self.arm, numerical_version=2,
                    explicit_FP32_kernels=["grid_sample", "dense_CE", "foreground_Dice", "point_CE", "presence_BCE"],
                    architecture="official PlainConvUNet plus common physical-patch/point readout",
                    official_backbone="dynamic_network_architectures.architectures.unet.PlainConvUNet",
                    features_per_stage=list(FEATURES), strides=[1, 2, 2, 2, 2],
                    convs_per_stage=2, decoder_convs_per_stage=2,
                    deep_supervision=False, unused_auxiliary_segmentation_parameters_removed=True,
                    dense_label_order=list(DENSE_LABELS), point_label_order=list(LABELS),
                    point_branch=("coordinate MLP64" if self.arm == "BASE3D" else "actual PointNetPlusPlus SSG64; classifier replaced by Identity"),
                    pointnet_source="src/vascutime_pointnet.py",
                    pointnet_geometry="coordinate-only FPS/radius/Euclidean interpolation, radii .15/.30",
                    parameter_count=total, from_scratch=True,
                    native_voxel_labels_not_point_indices=True,
                    local_patch_offsets_mm=list(LOCAL_OFFSETS_MM),
                    input_volume_axes="B,C,D=K,H=J,W=I", point_grid_order="I,J,K normalized[-1,1] for exact volume",
                    align_corners=True, presence_feedback_bound=.5,
                    NO_FEEDBACK_retains_presence_supervision=True,
                    loss_weights=dict(LOSS_WEIGHTS), foreground_dice_smooth=1.,
                    fully_trained_nnunet_certification=False, pretrained_ISLES_weights=False,
                    end_to_end_foreground_claim=False, novelty_or_efficacy_claim=False)

    def _validate_geometry(self, geo: Mapping[str, torch.Tensor], xyz: torch.Tensor) -> None:
        batch, n, _ = xyz.shape
        if set(GEOMETRY_KEYS) - set(geo):
            raise ValueError("Missing original PointNet++ geometry fields")
        s1, s2 = geo["s1"], geo["s2"]
        if s1.ndim != 2 or s2.ndim != 2 or s1.shape[0] != batch or s2.shape[0] != batch:
            raise ValueError("Batched PointNet++ support indices required")
        p1, p2 = s1.shape[1], s2.shape[1]
        required = {"g1": (batch, p1, 32), "g2": (batch, p2, 32),
                    "i21": (batch, p1, 3), "w21": (batch, p1, 3),
                    "i10": (batch, n, 3), "w10": (batch, n, 3)}
        for key, shape in required.items():
            if geo[key].shape != shape or geo[key].device != xyz.device:
                raise ValueError(f"Geometry field {key} shape/device mismatch")
        limits = {"s1": n, "s2": p1, "g1": n, "g2": p1, "i21": p2, "i10": p1}
        for key, upper in limits.items():
            a = geo[key]
            if a.dtype != torch.long or a.device != xyz.device or torch.any((a < 0) | (a >= upper)):
                raise ValueError(f"Geometry index {key} outside original coordinates")
        for key in ("w21", "w10"):
            a = geo[key]
            if not torch.isfinite(a).all() or torch.any(a < 0) or not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-6, rtol=0):
                raise ValueError("Interpolation weights must be finite normalized nonnegative")

    def forward(self, volume: torch.Tensor, local_patches: torch.Tensor,
                grid_xyz: torch.Tensor, point_xyz: torch.Tensor,
                geo: Mapping[str, torch.Tensor], point_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if volume.ndim != 5 or volume.shape[1] != 1 or any(s < 32 or s % 16 for s in volume.shape[2:]):
            raise ValueError("Volume must be B,1,D,H,W, dimensions >=32 and divisible16")
        b = volume.shape[0]
        if point_xyz.ndim != 3 or point_xyz.shape[0] != b or point_xyz.shape[-1] != 3 or point_xyz.shape[1] < 32:
            raise ValueError("point_xyz must be B,N>=32,3")
        n = point_xyz.shape[1]
        if grid_xyz.shape != (b, n, 3) or local_patches.shape != (b, n, 1, 5, 5, 5):
            raise ValueError("Bound grid and physical5cubed patch shape mismatch")
        for a in (volume, local_patches, grid_xyz, point_xyz):
            if a.dtype != volume.dtype or a.device != volume.device or not torch.isfinite(a).all():
                raise ValueError("Inputs must share finite dtype/device")
        mask = point_mask if point_mask is not None else torch.ones((b, n), dtype=torch.bool, device=volume.device)
        if mask.shape != (b, n) or mask.dtype != torch.bool or mask.device != volume.device:
            raise ValueError("point_mask must be bool B,N")
        self._validate_geometry(geo, point_xyz)
        captured: list[torch.Tensor] = []
        hook = self.unet.decoder.stages[-1].register_forward_hook(lambda m, i, o: captured.append(o))
        try:
            dense = self.unet(volume)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (b, 32, *volume.shape[2:]):
            raise RuntimeError("Official decoder fine-feature binding changed")
        image_features = captured[0]
        sampled_image = sample_volume(image_features, grid_xyz)
        sampled_dense = sample_volume(dense, grid_xyz)
        global_image = image_features.mean((2, 3, 4))
        local = self.local_encoder(local_patches.reshape(b * n, 1, 5, 5, 5)).reshape(b, n, 32)
        if self.arm == "BASE3D":
            branch = self.coordinate_branch(point_xyz)
        else:
            branch = self.pointnet(point_xyz, geo)
        hidden = self.fusion(torch.cat((sampled_image, local,
            global_image[:, None, :].expand(-1, n, -1), point_xyz, branch), -1))
        if self.arm in ("JOINT", "NO_FEEDBACK"):
            pooled_before, _ = masked_mean(hidden, mask)
            gamma, beta = self.context_modulation(torch.cat((global_image, pooled_before), -1)).chunk(2, -1)
            hidden = hidden * (1 + .25 * torch.tanh(gamma[:, None, :])) + .25 * torch.tanh(beta[:, None, :])
        pooled, aggregation_valid = masked_mean(hidden, mask)
        presence = self.presence_head(torch.cat((global_image, pooled), -1))
        # Identical direct semantic image baseline in every arm, plus residual
        # readout. Background is excluded ONLY for conditional foreground points.
        points = sampled_dense[:, :, 1:] + self.point_head(hidden)
        if self.arm == "JOINT":
            points = points + .5 * torch.tanh(self.presence_feedback)[None, None, :] * torch.tanh(presence[:, None, :])
        points = points.masked_fill(~mask[:, :, None], 0.)
        return dict(dense_logits=dense, point_logits=points, presence_logits=presence,
                    sampled_dense_logits=sampled_dense, point_mask=mask,
                    aggregation_valid=aggregation_valid)


def create_model(arm: str, seed: int = 20261027) -> SpatialAnatomicalModel:
    """No fitted parameters loaded; restore caller CPU RNG after initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return SpatialAnatomicalModel(arm)


def foreground_dice_loss(dense_logits: torch.Tensor, dense_target: torch.Tensor) -> torch.Tensor:
    if dense_logits.ndim != 5 or dense_logits.shape[1] != 14 or dense_target.shape != (dense_logits.shape[0], *dense_logits.shape[2:]):
        raise ValueError("Dense logits/target dimensions mismatch")
    valid = dense_target != -100
    probs = dense_logits.float().softmax(1)
    values = []
    for label_index in range(1, 14):
        prediction = probs[:, label_index] * valid.to(probs.dtype)
        target = (dense_target == label_index).to(probs.dtype)
        intersection = (prediction * target).flatten(1).sum(1, dtype=torch.float32)
        denominator = prediction.flatten(1).sum(1, dtype=torch.float32) + target.flatten(1).sum(1, dtype=torch.float32)
        values.append((2. * intersection + 1.) / (denominator + 1.))
    return 1. - torch.stack(values, 1).mean()


def loss_and_outputs(model: SpatialAnatomicalModel, volume: torch.Tensor,
                     local_patches: torch.Tensor, grid_xyz: torch.Tensor,
                     point_xyz: torch.Tensor, geo: Mapping[str, torch.Tensor],
                     dense_target: torch.Tensor, point_target: torch.Tensor,
                     presence_target: torch.Tensor, point_mask: torch.Tensor | None = None,
                     presence_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Same fixed four-term supervised objective in ALL four arms; no fit/step."""
    out = model(volume, local_patches, grid_xyz, point_xyz, geo, point_mask)
    if dense_target.shape != (volume.shape[0], *volume.shape[2:]) or dense_target.dtype != torch.long:
        raise ValueError("Dense targets must be long B,D,H,W indices0..13, not native value15")
    dense_valid = dense_target != -100
    if not dense_valid.any() or torch.any((dense_target[dense_valid] < 0) | (dense_target[dense_valid] >= 14)):
        raise ValueError("No dense supervision or invalid dense class index")
    if point_target.shape != point_xyz.shape[:2] or point_target.dtype != torch.long:
        raise ValueError("Point targets must be long B,N indices0..12")
    point_valid = out["point_mask"] & (point_target != -100)
    if not point_valid.any() or torch.any((point_target[point_valid] < 0) | (point_target[point_valid] >= 13)):
        raise ValueError("No point supervision or invalid point class index")
    if presence_target.shape != out["presence_logits"].shape or not torch.isfinite(presence_target).all() or torch.any((presence_target != 0) & (presence_target != 1)):
        raise ValueError("Complete-native presence supervision must be B,13 binary")
    pm = presence_mask if presence_mask is not None else torch.ones_like(presence_target, dtype=torch.bool)
    if pm.shape != presence_target.shape or pm.dtype != torch.bool:
        raise ValueError("Presence validity must be bool B,13")
    pm = pm & out["aggregation_valid"][:, None]
    if not pm.any():
        raise ValueError("No valid native presence supervision")
    dense_ce = F.cross_entropy(out["dense_logits"].float(), dense_target, ignore_index=-100)
    dense_dice = foreground_dice_loss(out["dense_logits"], dense_target)
    point_ce = F.cross_entropy(out["point_logits"][point_valid].float(), point_target[point_valid])
    presence_bce = F.binary_cross_entropy_with_logits(out["presence_logits"][pm].float(), presence_target[pm].float())
    total = dense_ce + dense_dice + point_ce + presence_bce
    out.update(dense_CE=dense_ce, foreground_Dice=dense_dice,
               point_CE=point_ce, presence_BCE=presence_bce, total_loss=total)
    return total, out


def synthetic_checks() -> dict[str, Any]:
    """CPU-only full forward/backward, with no optimizer steps or project IO."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    checks: dict[str, bool] = {}
    counts: dict[str, int] = {}
    try:
        generator = torch.Generator(device="cpu").manual_seed(20261027)
        volume = torch.randn(1, 1, 32, 32, 32, generator=generator)
        local = torch.randn(1, 48, 1, 5, 5, 5, generator=generator)
        xyz = torch.randn(1, 48, 3, generator=generator) * .1
        grid = torch.rand(1, 48, 3, generator=generator) * 2. - 1.
        geometry_np = build_coordinate_geometry(xyz.numpy())
        geo = {k: torch.from_numpy(v) for k, v in geometry_np.items()}
        dense_y = torch.randint(13, (1, 32, 32, 32), generator=generator)
        point_y = torch.randint(12, (1, 48), generator=generator)
        presence_y = torch.ones(1, 13); presence_y[:, -1] = 0
        common_initial = None
        for arm in ARMS:
            model = create_model(arm)
            counts[arm] = sum(p.numel() for p in model.parameters())
            initial = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("unet.", "local_encoder.", "fusion.", "point_head.", "presence_head."))}
            if common_initial is None:
                common_initial = initial
            else:
                checks[f"{arm}_common_initial_state_exact"] = set(initial) == set(common_initial) and all(torch.equal(initial[k], common_initial[k]) for k in initial)
            loss, out = loss_and_outputs(model, volume, local, grid, xyz, geo, dense_y, point_y, presence_y)
            loss.backward()
            checks[f"{arm}_finite_loss_and_all_gradients"] = bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            checks[f"{arm}_dense14_point13_presence13"] = out["dense_logits"].shape == (1, 14, 32, 32, 32) and out["point_logits"].shape == (1, 48, 13) and out["presence_logits"].shape == (1, 13)
            first_conv = next(m for m in model.unet.encoder.modules() if isinstance(m, nn.Conv3d))
            checks[f"{arm}_raw_volume_receives_gradient"] = bool(first_conv.weight.grad.abs().sum() > 0)
            checks[f"{arm}_physical_local_patch_gradient"] = bool(model.local_encoder[0].weight.grad.abs().sum() > 0)
            checks[f"{arm}_absent_presence_class_zero_supervised"] = bool(model.presence_head[-1].bias.grad[-1] > 0)
            checks[f"{arm}_unused_official_auxiliary_heads_removed"] = all(isinstance(m, nn.Identity) for m in list(model.unet.decoder.seg_layers)[:-1])
            if arm != "BASE3D":
                checks[f"{arm}_actual_PointNetPlusPlus_feature_branch"] = isinstance(model.pointnet, PointNetPlusPlus) and isinstance(model.pointnet.head, nn.Identity)
                checks[f"{arm}_pointnet_gradient"] = bool(model.pointnet.sa1[0].weight.grad.abs().sum() > 0)
            del model, out, loss, initial
        checks["all_effective_parameter_counts_within5percent"] = max(counts.values()) / min(counts.values()) <= 1.05
        z, y, x = torch.meshgrid(torch.arange(32.), torch.arange(32.), torch.arange(32.), indexing="ij")
        ramp = (10000. * z + 100. * y + x)[None, None]
        voxel = torch.tensor([[[3., 5., 7.], [0., 0., 0.], [31., 31., 31.]]])
        read = sample_volume(ramp, voxel_indices_to_grid(voxel, (32, 32, 32)))[..., 0]
        expected = 10000. * voxel[..., 2] + 100. * voxel[..., 1] + voxel[..., 0]
        checks["DHW_XYZ_grid_sample_index_alignment"] = torch.allclose(read, expected, atol=.05, rtol=0)
        mixed_volume = ramp.to(torch.bfloat16)
        mixed_grid = voxel_indices_to_grid(voxel, (32, 32, 32))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_read = sample_volume(mixed_volume, mixed_grid)
        float_read = sample_volume(mixed_volume.float(), mixed_grid.float())
        checks["mixed_bf16_volume_float32_grid_sample_explicit_FP32"] = mixed_read.dtype == torch.float32 and torch.equal(mixed_read, float_read)
        bf16_logits = torch.randn(1, 14, 4, 4, 4, generator=generator).to(torch.bfloat16).requires_grad_(True)
        mini_target = torch.randint(14, (1, 4, 4, 4), generator=generator)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mini_dice = foreground_dice_loss(bf16_logits, mini_target)
        mini_dice.backward()
        checks["bf16_Dice_explicit_FP32_finite_reduction_gradient"] = mini_dice.dtype == torch.float32 and bool(torch.isfinite(mini_dice)) and bool(torch.isfinite(bf16_logits.grad).all())
        amp_model = create_model("JOINT")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_loss, amp_output = loss_and_outputs(amp_model, volume, local, grid, xyz, geo, dense_y, point_y, presence_y)
        amp_loss.backward()
        checks["CPU_bf16_autocast_full_model_all_losses_FP32"] = all(amp_output[k].dtype == torch.float32 and bool(torch.isfinite(amp_output[k])) for k in ("dense_CE", "foreground_Dice", "point_CE", "presence_BCE", "total_loss")) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in amp_model.parameters())
        del amp_model, amp_output, amp_loss
        values = torch.randn(1, 4, 3, generator=generator)
        mean, valid = masked_mean(values, torch.zeros(1, 4, dtype=torch.bool))
        checks["empty_aggregate_explicit_invalid"] = not bool(valid.any()) and bool((mean == 0).all())
        checks["dense_last_class_maps_native15"] = DENSE_LABELS[13] == 15 and LABELS[12] == 15
    finally:
        torch.set_num_threads(old_threads)
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(qualified=all(checks.values()), checks=checks, parameter_counts=counts,
                model_fits=0, optimizer_steps=0, project_data_reads=0,
                CPU_only=True, same_task_full_nnunet_training_certified=False,
                caveat="Pure official-architecture/source/gradient checks; no patient efficacy, training sufficiency, or novelty result")
