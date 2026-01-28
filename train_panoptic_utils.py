#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoptic stereo training utilities (non-SDF).
"""
import argparse
import os
import platform
import math
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from la_loader.synthetic_data_loader import LASyntheticDataset3PerIns
from la_loader import la_transforms

from models.panoptic_stereo import PanopticStereoMultiHead, pos_mu_to_pointmap
from models.stereo_disparity import make_gn
from utils import dist_utils, rot_utils
from utils.logging_utils import draw_axes_on_images_bk, visualize_mono_torch
from utils.projection import SilhouetteDepthRenderer
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from losses import loss_functions
from train_utils import (
    DictMeters,
    _flush_train_window_to_tb,
    _init_window_meter,
    _reset_window_meter,
    _update_window_meter,
    build_lr_scheduler,
    disparity_nll_laplace_raft_style,
    load_model_state as _load_model_state,
    load_toml,
    log_meters_to_tb,
    seed_worker,
    set_global_seed,
    write_toml,
)

mp.set_start_method("spawn", force=True)
supported = getattr(mp, "get_all_sharing_strategies", lambda: ["file_system"])()
strategy = "file_descriptor" if "file_descriptor" in supported else "file_system"
mp.set_sharing_strategy(strategy)

_WARNED_SYMMETRY_KEYS = False  # Warn once when symmetry metadata is missing/empty.

_VARLEN_KEYS = {
    "class_ids",
    "diameters_list",
    "faces_list",
    "object_ids",
    "objs_in_left",
    "objs_in_right",
    "symmetry_axes",
    "symmetry_orders",
    "verts_list",
}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for variable-length fields."""
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        if key in _VARLEN_KEYS:
            out[key] = [b[key] for b in batch]
        else:
            out[key] = default_collate([b[key] for b in batch])
    return out


def _build_filtering_transforms(for_train: bool) -> la_transforms.LACompose:
    """Create filtering transforms for train/val datasets."""
    transforms = []
    if for_train:
        transforms.extend(
            [
                (la_transforms.LAToGray(), 0.5, ["input_image"]),
                (la_transforms.LAGaussianBlur(), 0.5, ["input_image"]),
                (la_transforms.LARandomNoise(sigma=3), 0.5, ["input_image"]),
                (la_transforms.LAColorSpaceTransform(), 0.5, ["input_image"]),
            ]
        )
    transforms.extend(
        [
            (la_transforms.LAToTensor(), 1.0, ["float_image", "input_image"]),
            (la_transforms.LASegToTensor(), 1.0, ["seg"]),
            (la_transforms.LANormalize(), 1.0, ["input_image"]),
        ]
    )
    return la_transforms.LACompose(transforms)


def _build_spatial_transforms() -> la_transforms.LACompose:
    """Create spatial transforms shared by train/val datasets."""
    return la_transforms.LACompose(
        [
            (la_transforms.LARandomCrop([[-30, -30], [30, 30]], [-20, 20]), 1.0, ["trans_matrix"]),
            (la_transforms.LARandomResize([[0.66, 0.66], [0.8, 0.8]]), 1.0, ["trans_matrix"]),
        ]
    )


def make_dataloaders(cfg: dict, distributed: bool):
    """Build train/val dataloaders and samplers."""
    filtering_trans_list = _build_filtering_transforms(for_train=True)
    spatial_trans_list = _build_spatial_transforms()
    out_list = (
        "stereo",
        "depth",
        "disparity",
        "semantic_seg",
        "instance_seg",
    )
    use_camera_list = ["ZED2", "D415", "ZEDmini"]
    out_size_wh = (cfg["data"]["width"],cfg["data"]["height"])
    train_ds = LASyntheticDataset3PerIns(
        out_list=out_list,
        with_data_path=True,
        use_camera_list=use_camera_list,
        with_camera_params=True,
        out_size_wh=out_size_wh,
        with_depro_matrix=True,
        target_scene_list=cfg["data"]["train_datasets"][0]["target_scene_list"],
        spatial_transform=spatial_trans_list,
        filtering_transform=filtering_trans_list,
    )
    n_classes = int(getattr(train_ds, "n_classes", 1))

    filtering_trans_list = _build_filtering_transforms(for_train=False)
    spatial_trans_list = _build_spatial_transforms()
    val_ds = LASyntheticDataset3PerIns(
        out_list=out_list,
        with_data_path=True,
        use_camera_list=use_camera_list,
        with_camera_params=True,
        out_size_wh=out_size_wh,
        with_depro_matrix=True,
        target_scene_list=cfg["data"]["val_datasets"][0]["target_scene_list"],
        spatial_transform=spatial_trans_list,
        filtering_transform=filtering_trans_list,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=(train_sampler is None),
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        sampler=val_sampler,
        worker_init_fn=seed_worker,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
    )
    return train_loader, val_loader, train_sampler, val_sampler, n_classes


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Build the panoptic stereo model."""
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {}) or {}
    point_map_norm_mean = dcfg.get("point_map_norm_mean")
    point_map_norm_std = dcfg.get("point_map_norm_std")
    if point_map_norm_mean is not None and point_map_norm_std is not None:
        point_map_norm_mean = [float(v) for v in point_map_norm_mean]
        point_map_norm_std = [float(v) for v in point_map_norm_std]
    else:
        point_map_norm_mean = None
        point_map_norm_std = None
    seg_cfg = cfg.get("seg_head", {}) or {}
    head_base_ch = int(seg_cfg.get("head_base_ch", seg_cfg.get("head_c4", 96)))
    if "head_ch_scale" in seg_cfg:
        head_ch_scale = float(seg_cfg.get("head_ch_scale", 1.35))
    elif "head_c4" in seg_cfg and "head_c8" in seg_cfg and float(seg_cfg["head_c4"]) > 0.0:
        head_ch_scale = float(seg_cfg["head_c8"]) / float(seg_cfg["head_c4"])
    else:
        head_ch_scale = 1.35
    head_downsample = int(seg_cfg.get("head_downsample", 4))

    return PanopticStereoMultiHead(
        levels=int(mcfg.get("levels", 4)),
        norm_layer=make_gn(16),
        l2_normalize_feature=bool(mcfg.get("l2_normalize_feature", True)),
        use_ctx_aspp=bool(mcfg.get("use_ctx_aspp", True)),
        lookup_mode=str(mcfg.get("lookup_mode", "1d")),
        radius_w=int(mcfg.get("radius_w", 4)),
        radius_h=int(mcfg.get("radius_h", 0)),
        hidden_ch=int(mcfg.get("hidden_ch", 96)),
        context_ch=int(mcfg.get("context_ch", 96)),
        num_classes=num_classes,
        rot_repr=str(mcfg.get("rot_repr", "r6d")),
        emb_dim=int(seg_cfg.get("emb_dim", 16)),
        use_dummy_head=bool(seg_cfg.get("use_dummy_head", False)),
        head_base_ch=head_base_ch,
        head_ch_scale=head_ch_scale,
        head_downsample=head_downsample,
        point_map_norm_mean=point_map_norm_mean,
        point_map_norm_std=point_map_norm_std,
    )


def _prepare_stereo_and_cam(batch: Dict[str, Any], device: torch.device):
    """Prepare stereo inputs and camera parameters."""
    stereo = batch["stereo"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True).unsqueeze(1) * 1000.0
    disp_gt = batch["disparity"].to(device, non_blocking=True).unsqueeze(1)

    left_k = batch["camera_params"]["left_k"].to(device, non_blocking=True).unsqueeze(1)
    right_k = batch["camera_params"]["right_k"].to(device, non_blocking=True).unsqueeze(1)
    k_pair = torch.cat([left_k, right_k], dim=1)
    baseline = batch["camera_params"]["base_dist_mm"].to(device, non_blocking=True)

    return stereo, depth, disp_gt, k_pair, baseline, left_k


def _downsample_label(label: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Downsample a label map with nearest interpolation."""
    return F.interpolate(label.unsqueeze(1).float(), size=size_hw, mode="nearest").squeeze(1).long()


def _prepare_pose_targets(
    batch: Dict[str, Any],
    sem_gt: torch.Tensor,
    size_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare pose targets with background masking."""
    pos_map = batch["pos_map"].to(device, non_blocking=True)
    if pos_map.dim() == 4 and pos_map.shape[1] != 3 and pos_map.shape[-1] == 3:
        pos_map = pos_map.permute(0, 3, 1, 2)
    pos_map = F.interpolate(pos_map, size=size_hw, mode="bilinear", align_corners=False)

    rot_map = batch["rot_map"].to(device, non_blocking=True)
    if rot_map.dim() == 5 and not (rot_map.shape[1] == 3 and rot_map.shape[2] == 3):
        rot_map = rot_map.permute(0, 3, 4, 1, 2)
    rot_map = F.interpolate(rot_map.flatten(1, 2), size=size_hw, mode="nearest")
    rot_map = rot_map.view(rot_map.size(0), 3, 3, size_hw[0], size_hw[1])

    sem_mask_hw = _downsample_label(sem_gt, size_hw) > 0
    sem_mask_hw = sem_mask_hw.unsqueeze(1)
    objs_in_left_list: List[torch.Tensor] = []
    B = len(batch["objs_in_left"])
    counts: List[int] = []
    for objs_in_left in batch["objs_in_left"]:
        objs_in_left = [torch.from_numpy(obj_in_left) for obj_in_left in objs_in_left]
        counts.append(len(objs_in_left))
        objs_in_left_list.extend(objs_in_left)
    objs_in_left_list = torch.stack(objs_in_left_list, dim=0)
    Kmax = max(counts) if counts else 0
    valid_k = torch.zeros((B, Kmax), dtype=torch.bool, device=device)
    objs_in_left = torch.zeros((B, Kmax, 4, 4), dtype=torch.float32, device=device)
    st_k = 0
    for b, k in enumerate(counts):
        valid_k[b, :k] = True
        objs_in_left[b, :k] = objs_in_left_list[st_k: st_k + k]
        st_k += counts[b]
    return pos_map, rot_map, sem_mask_hw, objs_in_left


def _build_objs_in_left_from_batch(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build objs_in_left tensor and valid mask from batch lists."""
    objs_in_left_raw = batch.get("objs_in_left", None)
    if objs_in_left_raw is None:
        return None, None

    objs_in_left_list: List[torch.Tensor] = []
    counts: List[int] = []
    for objs_in_left in objs_in_left_raw:
        objs_in_left = [torch.as_tensor(obj_in_left).float() for obj_in_left in objs_in_left]
        counts.append(len(objs_in_left))
        objs_in_left_list.extend(objs_in_left)
    if not counts:
        return None, None
    if len(objs_in_left_list) == 0:
        B = len(counts)
        return torch.zeros((B, 0, 4, 4), dtype=torch.float32, device=device), torch.zeros(
            (B, 0), dtype=torch.bool, device=device
        )

    objs_in_left_list = torch.stack(objs_in_left_list, dim=0).to(device)
    B = len(counts)
    Kmax = max(counts) if counts else 0
    valid_k = torch.zeros((B, Kmax), dtype=torch.bool, device=device)
    objs_in_left = torch.zeros((B, Kmax, 4, 4), dtype=torch.float32, device=device)
    st_k = 0
    for b, k in enumerate(counts):
        if k > 0:
            valid_k[b, :k] = True
            objs_in_left[b, :k] = objs_in_left_list[st_k: st_k + k]
        st_k += counts[b]
    return objs_in_left, valid_k




def _compute_pose_losses_and_metrics(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sem_gt: torch.Tensor,
    size_hw: Tuple[int, int],
    left_k: torch.Tensor,
    wks_inst: torch.Tensor,
    wfg_inst: torch.Tensor,
    valid_k_inst: torch.Tensor,
    image_size: Tuple[int, int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute pose losses and pose-related metrics."""
    pos_gt_map, rot_gt_map, pose_mask, _ = _prepare_pose_targets(batch, sem_gt, size_hw, device)
    gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(pos_gt_map, left_k[:, 0], downsample=1, use_logz=True)
    loss_pos = loss_functions.pos_loss_hetero_map(
        pred["pos_mu_norm"],
        pred["pos_logvar_norm"],
        gt_pos_mu_map,
        pose_mask,
    )
    pos_map_pred = pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
    pos_diff = pos_map_pred - pos_gt_map
    pos_l2_map = torch.sqrt((pos_diff * pos_diff).sum(dim=1, keepdim=True) + 1e-6)
    pos_map_l2 = (pos_l2_map * pose_mask.to(pos_l2_map.dtype)).sum() / pose_mask.sum().clamp_min(1.0)
    r_pred = pred.get("pose_R")
    t_pred = pred.get("pose_t")
    v_pred = pred.get("pose_valid")
    r_gt, t_gt, v_gt, _, _ = rot_utils.pose_from_maps_auto(
        rot_map=rot_gt_map,
        pos_map=pos_gt_map,
        Wk_1_4=wks_inst,
        wfg=wfg_inst,
        min_px=10,
        min_wsum=1e-6,
    )
    rot_gt_map_use = rot_gt_map
    r_gt_use = r_gt
    sym_axes, sym_orders = _prepare_symmetry_tensors(batch, device, valid_k_inst.shape[1])
    if sym_axes is not None and sym_orders is not None and pred["rot_mat"].numel() > 0:
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)
        inst_gt_hw = _downsample_label(inst_gt, size_hw)
        inst_id_map = (inst_gt_hw - 1).clamp_min(0)
        fg_mask = (inst_gt_hw > 0).unsqueeze(1)
        with torch.no_grad():
            r_gt_use, delta = rot_utils.align_pose_by_symmetry_min_rotation(
                r_pred,
                r_gt,
                sym_axes,
                sym_orders,
            )
            rot_gt_map_use = rot_utils.apply_symmetry_rotation_map(
                rot_gt_map_use,
                sym_axes,
                delta,
                inst_id_map,
                fg_mask=fg_mask,
            )
    loss_rot = loss_functions.rotation_loss_hetero_map(
        pred["rot_mat"],
        rot_gt_map_use,
        pred["rot_logvar_theta"],
        pose_mask,
    )
    rot_deg_map = _rot_geodesic_map_deg(pred["rot_mat"], rot_gt_map_use).unsqueeze(1)
    rot_map_deg = (rot_deg_map * pose_mask.to(rot_deg_map.dtype)).sum() / pose_mask.sum().clamp_min(1.0)
    origin_in = _origin_in_image_from_t(t_gt, left_k[:, 0], image_size)
    valid_inst = v_pred & v_gt & valid_k_inst & origin_in
    if valid_inst.any():
        t_diff = t_pred - t_gt
        t_l2 = torch.sqrt((t_diff * t_diff).sum(dim=-1) + 1e-6)
        pos_inst_l2 = (t_l2 * valid_inst.to(t_l2.dtype)).sum() / valid_inst.sum().clamp_min(1.0)
        r_deg = _rot_geodesic_deg(r_pred, r_gt_use)
        rot_inst_deg = (r_deg * valid_inst.to(r_deg.dtype)).sum() / valid_inst.sum().clamp_min(1.0)
    else:
        pos_inst_l2 = pos_map_l2.new_tensor(0.0)
        rot_inst_deg = pos_map_l2.new_tensor(0.0)

    model_points, _ = _build_model_points_from_batch(batch, device)
    if model_points is not None and valid_inst.any():
        adds = _adds_core_from_Rt_no_norm(
            r_pred,
            t_pred,
            r_gt_use,
            t_gt,
            model_points,
            use_symmetric=True,
            valid_mask=valid_inst,
        )
        add = _adds_core_from_Rt_no_norm(
            r_pred,
            t_pred,
            r_gt_use,
            t_gt,
            model_points,
            use_symmetric=False,
            valid_mask=valid_inst,
        )
    else:
        adds = pos_map_l2.new_tensor(0.0)
        add = pos_map_l2.new_tensor(0.0)

    return {
        "loss_pos": loss_pos,
        "loss_rot": loss_rot,
        "rot_map_deg": rot_map_deg,
        "pos_map_l2": pos_map_l2,
        "pos_inst_l2": pos_inst_l2,
        "rot_inst_deg": rot_inst_deg,
        "adds": adds,
        "add": add,
    }


def _compute_pos_losses_and_metrics(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sem_gt: torch.Tensor,
    size_hw: Tuple[int, int],
    left_k: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute translation losses and metrics without rotation outputs."""
    pos_gt_map, _, pose_mask, _ = _prepare_pose_targets(batch, sem_gt, size_hw, device)
    gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(pos_gt_map, left_k[:, 0], downsample=1, use_logz=True)
    loss_pos = loss_functions.pos_loss_hetero_map(
        pred["pos_mu_norm"],
        pred["pos_logvar_norm"],
        gt_pos_mu_map,
        pose_mask,
    )
    pos_map_pred = pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
    pos_diff = pos_map_pred - pos_gt_map
    pos_l2_map = torch.sqrt((pos_diff * pos_diff).sum(dim=1, keepdim=True) + 1e-6)
    pos_map_l2 = (pos_l2_map * pose_mask.to(pos_l2_map.dtype)).sum() / pose_mask.sum().clamp_min(1.0)
    return {
        "loss_pos": loss_pos,
        "pos_map_l2": pos_map_l2,
    }


def _prepare_pose_maps(
    batch: Dict[str, Any],
    size_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare pose maps (pos_map, rot_map) at the target resolution."""
    pos_map = batch.get("pos_map", None)
    rot_map = batch.get("rot_map", None)
    if pos_map is None or rot_map is None:
        return None, None

    pos_map = pos_map.to(device, non_blocking=True)
    if pos_map.dim() == 4 and pos_map.shape[1] != 3 and pos_map.shape[-1] == 3:
        pos_map = pos_map.permute(0, 3, 1, 2)
    pos_map = F.interpolate(pos_map, size=size_hw, mode="bilinear", align_corners=False)

    rot_map = rot_map.to(device, non_blocking=True)
    if rot_map.dim() == 5 and not (rot_map.shape[1] == 3 and rot_map.shape[2] == 3):
        rot_map = rot_map.permute(0, 3, 4, 1, 2)
    rot_map = F.interpolate(rot_map.flatten(1, 2), size=size_hw, mode="nearest")
    rot_map = rot_map.view(rot_map.size(0), 3, 3, size_hw[0], size_hw[1])
    return pos_map, rot_map


def _build_affinity_targets(inst_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build affinity targets and valid masks from instance IDs."""
    B, H, W = inst_hw.shape
    tgt = torch.zeros((B, 4, H, W), device=inst_hw.device, dtype=torch.float32)
    valid = torch.zeros_like(tgt, dtype=torch.bool)

    same = inst_hw[:, :, :-1] == inst_hw[:, :, 1:]
    both_fg = (inst_hw[:, :, :-1] > 0) & (inst_hw[:, :, 1:] > 0)
    tgt[:, 0, :, :-1] = same.float()
    valid[:, 0, :, :-1] = both_fg

    same = inst_hw[:, :, 1:] == inst_hw[:, :, :-1]
    both_fg = (inst_hw[:, :, 1:] > 0) & (inst_hw[:, :, :-1] > 0)
    tgt[:, 1, :, 1:] = same.float()
    valid[:, 1, :, 1:] = both_fg

    same = inst_hw[:, :-1, :] == inst_hw[:, 1:, :]
    both_fg = (inst_hw[:, :-1, :] > 0) & (inst_hw[:, 1:, :] > 0)
    tgt[:, 2, :-1, :] = same.float()
    valid[:, 2, :-1, :] = both_fg

    same = inst_hw[:, 1:, :] == inst_hw[:, :-1, :]
    both_fg = (inst_hw[:, 1:, :] > 0) & (inst_hw[:, :-1, :] > 0)
    tgt[:, 3, 1:, :] = same.float()
    valid[:, 3, 1:, :] = both_fg

    return tgt, valid


def _affinity_loss(
    aff_logits: torch.Tensor,
    aff_target: torch.Tensor,
    aff_valid: torch.Tensor,
    neg_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted BCE loss for affinity edges."""
    weights = torch.ones_like(aff_target)
    weights = torch.where(aff_target > 0.5, weights, weights * neg_weight)
    loss = F.binary_cross_entropy_with_logits(aff_logits, aff_target, weight=weights, reduction="none")
    valid_f = aff_valid.to(loss.dtype)
    loss = (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    return loss, aff_valid.sum()


@torch.no_grad()
def _sample_indices_per_instance(
    inst_flat: torch.Tensor,           # (N,) int64, bg=0
    max_per_inst: int = 64,
    min_pixels_per_inst: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sampled foreground indices per instance.

    Returns:
      fg_idx: (N_fg,) indices into flattened image where inst>0
      inv:    (N_fg,) instance-id mapped to [0..K-1] for each fg pixel
      sel:    (M,)   indices into fg_idx (and inv/emb_fg) selected by sampling
    """
    fg_idx = torch.nonzero(inst_flat > 0, as_tuple=False).squeeze(1)
    inst_fg = inst_flat[fg_idx]  # (N_fg,)

    uniq_ids, inv = torch.unique(inst_fg, sorted=True, return_inverse=True)  # inv in [0..K-1]
    K = int(uniq_ids.numel())

    if K == 0:
        return fg_idx, inv, fg_idx.new_empty((0,), dtype=torch.long)

    # count per instance in fg
    cnt = torch.bincount(inv, minlength=K)

    # sample up to max_per_inst per instance
    sel_chunks = []
    for k in range(K):
        if cnt[k].item() < min_pixels_per_inst:
            continue
        idx_k = torch.nonzero(inv == k, as_tuple=False).squeeze(1)  # indices into fg arrays
        if idx_k.numel() > max_per_inst:
            perm = torch.randperm(idx_k.numel(), device=inst_flat.device)[:max_per_inst]
            idx_k = idx_k[perm]
        sel_chunks.append(idx_k)

    if not sel_chunks:
        return fg_idx, inv, fg_idx.new_empty((0,), dtype=torch.long)

    sel = torch.cat(sel_chunks, dim=0)
    return fg_idx, inv, sel


def embedding_cosface_sampled(
    emb: torch.Tensor,                 # (B,C,H,W)
    inst_hw: torch.Tensor,             # (B,H,W) int, bg=0
    max_per_inst: int = 64,
    margin: float = 0.25,              # CosFace m
    scale: float = 32.0,               # CosFace s
    min_pixels_per_inst: int = 4,
    detach_proto: bool = True,
    topk_neg: Optional[int] = None,    # optional: restrict negatives to top-k most similar
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic-proxy CosFace for instance embeddings with random sampling.
    - Samples up to `max_per_inst` pixels per instance (foreground only).
    - Prototypes are computed from ALL fg pixels (more stable).
    - If topk_neg is set (e.g., 16), softmax is computed over {positive + top-k negatives}.

    Returns: (loss, num_samples)
    """
    assert emb.dim() == 4 and inst_hw.dim() == 3
    B, C, H, W = emb.shape
    assert inst_hw.shape[0] == B and inst_hw.shape[1] == H and inst_hw.shape[2] == W

    emb_n = F.normalize(emb, dim=1, eps=1e-6)
    loss_sum = emb.new_tensor(0.0)
    n_sum = emb.new_tensor(0.0)

    for b in range(B):
        inst = inst_hw[b]
        inst_flat = inst.reshape(-1).long()

        # fg indices & per-instance mapping (inv)
        fg_idx = torch.nonzero(inst_flat > 0, as_tuple=False).squeeze(1)
        if fg_idx.numel() == 0:
            continue

        # flattened normalized embeddings for fg pixels
        emb_flat = emb_n[b].permute(1, 2, 0).reshape(-1, C)      # (HW,C)
        emb_fg_all = emb_flat[fg_idx]                            # (N_fg,C)
        inst_fg = inst_flat[fg_idx]                              # (N_fg,)

        uniq_ids, inv = torch.unique(inst_fg, sorted=True, return_inverse=True)
        K = int(uniq_ids.numel())
        if K <= 1:
            continue

        # filter tiny instances (for this loss)
        cnt = torch.bincount(inv, minlength=K)
        keep = cnt >= min_pixels_per_inst
        if keep.sum().item() <= 1:
            continue

        # remap to compact [0..K2-1]
        keep_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
        remap = torch.full((K,), -1, device=emb.device, dtype=torch.long)
        remap[keep_ids] = torch.arange(keep_ids.numel(), device=emb.device)
        y_all = remap[inv]
        valid = y_all >= 0

        emb_fg_all = emb_fg_all[valid]
        y_all = y_all[valid]
        K2 = int(keep_ids.numel())
        if K2 <= 1:
            continue

        # sample up to max_per_inst per instance
        sel_chunks = []
        for k in range(K2):
            idx_k = torch.nonzero(y_all == k, as_tuple=False).squeeze(1)
            if idx_k.numel() == 0:
                continue
            if idx_k.numel() > max_per_inst:
                perm = torch.randperm(idx_k.numel(), device=emb.device)[:max_per_inst]
                idx_k = idx_k[perm]
            sel_chunks.append(idx_k)
        if not sel_chunks:
            continue
        sel = torch.cat(sel_chunks, dim=0)

        emb_s = emb_fg_all[sel]   # (M,C)
        y_s = y_all[sel]          # (M,)
        M = int(emb_s.shape[0])
        if M == 0:
            continue

        # prototypes from ALL pixels (stable)
        mu = torch.zeros((K2, C), device=emb.device, dtype=emb.dtype)
        cnt2 = torch.zeros((K2, 1), device=emb.device, dtype=emb.dtype)
        mu.index_add_(0, y_all, emb_fg_all)
        cnt2.index_add_(0, y_all, torch.ones((emb_fg_all.shape[0], 1), device=emb.device, dtype=emb.dtype))
        mu = mu / cnt2.clamp_min(1.0)
        mu = F.normalize(mu, dim=1, eps=1e-6)
        if detach_proto:
            mu = mu.detach()

        # cosine logits (M,K2)
        logits = scale * (emb_s @ mu.t())

        # restrict negatives to top-k (optional)
        if topk_neg is not None and 0 < topk_neg < K2:
            # keep: positive + topk most similar (excluding positive)
            with torch.no_grad():
                # topk over all classes, then ensure positive included
                vals, idxs = torch.topk(logits, k=topk_neg + 1, dim=1, largest=True, sorted=False)  # (M,topk+1)
                # add positive index explicitly then unique per row
                pos = y_s.view(-1, 1)
                idxs = torch.cat([idxs, pos], dim=1)  # (M, topk+2)
                # unique per-row (small k so brute force ok)
                new_logits = []
                new_targets = []
                for i in range(M):
                    row = idxs[i]
                    row = row.unique()
                    # build reduced logits
                    l = logits[i, row]
                    # target is position of positive class within row
                    t = (row == y_s[i]).nonzero(as_tuple=False).squeeze(1)
                    # should exist
                    new_logits.append(l)
                    new_targets.append(t)
                # pad to max length in batch
                maxL = max(l.numel() for l in new_logits)
                logits_red = logits.new_full((M, maxL), -1e4)  # very negative for padding
                target_red = torch.zeros((M,), device=emb.device, dtype=torch.long)
                for i in range(M):
                    l = new_logits[i]
                    logits_red[i, : l.numel()] = l
                    target_red[i] = new_targets[i].item()
            logits_use = logits_red
            y_use = target_red
        else:
            logits_use = logits
            y_use = y_s

        # CosFace margin: subtract from correct logit
        logits_use[torch.arange(M, device=emb.device), y_use] -= scale * margin

        loss = F.cross_entropy(logits_use, y_use, reduction="mean")
        loss_sum = loss_sum + loss * M
        n_sum = n_sum + M

    loss_out = loss_sum / n_sum.clamp_min(1.0)
    # Keep graph dependency on emb when no samples are selected.
    loss_out = loss_out + emb.sum() * 0.0 * (n_sum == 0).to(loss_out.dtype)
    return loss_out, n_sum


def _colorize_semantic(seg: torch.Tensor, num_classes: int, seed: int = 123) -> torch.Tensor:
    """Colorize semantic labels with a fixed palette."""
    if seg.dim() == 2:
        seg = seg.unsqueeze(0)
    B, H, W = seg.shape
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = 0
    seg_np = seg.cpu().numpy().astype(np.int64)
    out = np.zeros((B, H, W, 3), dtype=np.uint8)
    for b in range(B):
        out[b] = palette[seg_np[b] % num_classes]
    return torch.from_numpy(out).permute(0, 3, 1, 2)


def _colorize_instance_ids(inst: np.ndarray, seed: int = 999) -> torch.Tensor:
    """Colorize instance IDs with deterministic random colors."""
    H, W = inst.shape
    max_id = int(inst.max())
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, size=(max_id + 1, 3), dtype=np.uint8)
    palette[0] = 0
    color = palette[inst]
    return torch.from_numpy(color).permute(2, 0, 1)



def _infer_instance_from_heads(
    sem_logits: torch.Tensor,
    aff_logits: torch.Tensor,
    emb: torch.Tensor,
    tau_high: float,
    # embedding merge
    tau_emb_merge: float = 0.85,  # NOTE: emb_merge_metric="cos" uses cosine similarity threshold
    emb_merge_iters: int = 1,
    emb_merge_metric: Literal["cos", "l2"] = "cos",
    emb_merge_small_area: Optional[int] = None,
    # semantic
    use_gt_semantic: bool = False,
    semantic_gt_hw: Optional[torch.Tensor] = None,
    union_same_semantic_only: bool = True,
    eps: float = 1e-6,
) -> List[np.ndarray]:
    """
    Infer instance IDs from semantic, affinity, and embeddings.

    Steps:
      1) things mask from semantic prediction (or GT for debug)
      2) Conservative union-find with mutual-check affinity (right/down edges only)
      3) Optional component merge using embedding prototypes (CosFace-aligned: cosine)
      4) background is ID=0

    Args:
      tau_high:
        affinity sigmoid prob threshold for mutual-check union
      tau_emb_merge:
        - emb_merge_metric="cos": merge if cos_sim(mu_a, mu_c) >= tau_emb_merge
        - emb_merge_metric="l2" : merge if ||mu_a - mu_c|| <= tau_emb_merge
      emb_merge_small_area:
        if set, only allow merges when (area[a] <= th) or (area[c] <= th).
        Useful to remove tiny 1px islands without causing big merges.
    """
    # semantic prediction
    sem_pred = sem_logits.argmax(dim=1)
    if use_gt_semantic and semantic_gt_hw is not None:
        sem_pred = semantic_gt_hw

    # CPU numpy once
    sem_np = sem_pred.detach().cpu().numpy()                       # (B,H,W) int
    aff_np = torch.sigmoid(aff_logits).detach().cpu().numpy()      # (B,4,H,W) float

    # CosFace前提: inferenceでも normalize して cosine を扱いやすくする
    emb_t = emb.detach()
    emb_t = emb_t / (torch.linalg.norm(emb_t, dim=1, keepdim=True) + eps)
    emb_np = emb_t.cpu().permute(0, 2, 3, 1).numpy()               # (B,H,W,D) float (unit)

    out: List[np.ndarray] = []

    for b in range(sem_np.shape[0]):
        sem_b = sem_np[b]                   # (H,W)
        sem_mask = (sem_b != 0)             # things mask
        H, W = sem_mask.shape
        n_pix = H * W

        parent = np.arange(n_pix, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, c: int) -> None:
            ra, rb = find(a), find(c)
            if ra != rb:
                parent[rb] = ra

        def idx(y: int, x: int) -> int:
            return y * W + x

        # --- mutual-check edges (right & down only) ---
        if W > 1:
            right_ok = (
                sem_mask[:, :-1] & sem_mask[:, 1:]
                & (aff_np[b, 0, :, :-1] > tau_high)     # →
                & (aff_np[b, 1, :, 1:] > tau_high)      # ← (at neighbor)
            )
            if union_same_semantic_only:
                right_ok &= (sem_b[:, :-1] == sem_b[:, 1:])
            ry, rx = np.where(right_ok)
        else:
            ry, rx = np.empty((0,), np.int64), np.empty((0,), np.int64)

        if H > 1:
            down_ok = (
                sem_mask[:-1, :] & sem_mask[1:, :]
                & (aff_np[b, 2, :-1, :] > tau_high)     # ↓
                & (aff_np[b, 3, 1:, :] > tau_high)      # ↑ (at neighbor)
            )
            if union_same_semantic_only:
                down_ok &= (sem_b[:-1, :] == sem_b[1:, :])
            dy, dx = np.where(down_ok)
        else:
            dy, dx = np.empty((0,), np.int64), np.empty((0,), np.int64)

        # union only selected edges
        for y, x in zip(ry.tolist(), rx.tolist()):
            union(idx(y, x), idx(y, x + 1))
        for y, x in zip(dy.tolist(), dx.tolist()):
            union(idx(y, x), idx(y + 1, x))

        # --- label components (pixel UF -> compact IDs) ---
        inst = np.zeros((H, W), dtype=np.int32)
        root_to_id = {}
        next_id = 1
        ys, xs = np.where(sem_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            r = find(idx(y, x))
            if r not in root_to_id:
                root_to_id[r] = next_id
                next_id += 1
            inst[y, x] = root_to_id[r]

        # --- optional embedding-based merge (CosFace-aligned) ---
        if emb_merge_iters > 0:
            D = emb_np.shape[-1]
            flat_emb = emb_np[b].reshape(-1, D)

            for _ in range(emb_merge_iters):
                num_comp = int(inst.max())
                if num_comp <= 1:
                    break

                # component counts
                flat_inst = inst.ravel()
                cnt = np.bincount(flat_inst, minlength=num_comp + 1).astype(np.float32).reshape(-1, 1)

                # component prototype sums (emb is already unit per pixel)
                mu = np.zeros((num_comp + 1, D), dtype=np.float32)
                np.add.at(mu, flat_inst, flat_emb)
                mu = mu / np.clip(cnt, 1.0, None)

                # normalize prototypes (CosFace-like)
                mu_norm = np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12
                mu = mu / mu_norm

                # adjacency pairs (right/down)
                pairs = []

                if W > 1:
                    a = inst[:, :-1]
                    c = inst[:, 1:]
                    m = (a > 0) & (c > 0) & (a != c)
                    if union_same_semantic_only:
                        m &= (sem_b[:, :-1] == sem_b[:, 1:])
                    if np.any(m):
                        aa = a[m].astype(np.int32)
                        cc = c[m].astype(np.int32)
                        pairs.append(np.stack([np.minimum(aa, cc), np.maximum(aa, cc)], axis=1))

                if H > 1:
                    a = inst[:-1, :]
                    c = inst[1:, :]
                    m = (a > 0) & (c > 0) & (a != c)
                    if union_same_semantic_only:
                        m &= (sem_b[:-1, :] == sem_b[1:, :])
                    if np.any(m):
                        aa = a[m].astype(np.int32)
                        cc = c[m].astype(np.int32)
                        pairs.append(np.stack([np.minimum(aa, cc), np.maximum(aa, cc)], axis=1))

                if not pairs:
                    break

                adj = np.unique(np.concatenate(pairs, axis=0), axis=0)  # (E,2)

                # component-level union-find
                comp_parent = np.arange(num_comp + 1, dtype=np.int32)

                def cfind(x: int) -> int:
                    while comp_parent[x] != x:
                        comp_parent[x] = comp_parent[comp_parent[x]]
                        x = comp_parent[x]
                    return x

                def cunion(a_: int, c_: int) -> None:
                    ra, rb = cfind(a_), cfind(c_)
                    if ra != rb:
                        comp_parent[rb] = ra

                # merge decision
                for a_, c_ in adj.tolist():
                    if emb_merge_small_area is not None:
                        if (cnt[a_, 0] > emb_merge_small_area) and (cnt[c_, 0] > emb_merge_small_area):
                            continue  # only merge if at least one is small

                    if emb_merge_metric == "cos":
                        # cos similarity since mu is normalized
                        sim = float(np.dot(mu[a_], mu[c_]))
                        if sim >= tau_emb_merge:
                            cunion(a_, c_)
                    elif emb_merge_metric == "l2":
                        dist = float(np.linalg.norm(mu[a_] - mu[c_]))
                        if dist <= tau_emb_merge:
                            cunion(a_, c_)
                    else:
                        raise ValueError(f"Unknown emb_merge_metric={emb_merge_metric}")

                # vectorized relabel using component roots
                roots = np.arange(num_comp + 1, dtype=np.int32)
                for k in range(1, num_comp + 1):
                    roots[k] = cfind(k)

                uniq_roots = np.unique(roots[1:])  # exclude 0
                root_to_new = np.zeros((num_comp + 1,), dtype=np.int32)
                root_to_new[uniq_roots] = np.arange(1, uniq_roots.size + 1, dtype=np.int32)

                new_inst = root_to_new[roots[inst]]
                changed = np.any(new_inst != inst)
                inst = new_inst
                if not changed:
                    break

        inst[~sem_mask] = 0
        out.append(inst)

    return out



def _log_segmentation_visuals(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    sem_logits: torch.Tensor,
    aff_logits: torch.Tensor,
    emb: torch.Tensor,
    semantic_gt: torch.Tensor,
    instance_gt: torch.Tensor,
    cfg: dict,
    n_images: int,
) -> None:
    """Log semantic and instance segmentation visualizations."""
    if writer is None:
        return
    sem_logits = sem_logits[:n_images]
    aff_logits = aff_logits[:n_images]
    emb = emb[:n_images]
    semantic_gt = semantic_gt[:n_images]
    instance_gt = instance_gt[:n_images]

    size_hw = sem_logits.shape[-2:]
    sem_gt_hw = _downsample_label(semantic_gt, size_hw)

    inst_pred = _infer_instance_from_heads(
        sem_logits,
        aff_logits,
        emb,
        tau_high=float(cfg.get("instance", {}).get("tau_aff", 0.95)),
        tau_emb_merge=float(cfg.get("instance", {}).get("tau_emb_merge", 0.6)),
        emb_merge_iters=int(cfg.get("instance", {}).get("emb_merge_iters", 1)),
        use_gt_semantic=bool(cfg.get("instance", {}).get("use_gt_semantic", False)),
        semantic_gt_hw=sem_gt_hw,
    )

    num_classes = int(cfg.get("data", {}).get("n_classes", sem_logits.shape[1]))
    sem_pred = sem_logits.argmax(dim=1)
    sem_pred_color = _colorize_semantic(sem_pred, num_classes).float() / 255.0
    sem_gt_color = _colorize_semantic(sem_gt_hw, num_classes).float() / 255.0

    inst_pred_color = torch.stack([_colorize_instance_ids(inst) for inst in inst_pred], dim=0).float() / 255.0
    inst_gt_hw = _downsample_label(instance_gt, size_hw).cpu().numpy()
    inst_gt_color = torch.stack([_colorize_instance_ids(inst) for inst in inst_gt_hw], dim=0).float() / 255.0

    grid_sem = vutils.make_grid(
        torch.cat([sem_pred_color, sem_gt_color], dim=0),
        nrow=n_images,
        normalize=False,
    )
    writer.add_image(f"{prefix}/vis_semantic_pred_gt", grid_sem, step_value)

    grid_inst = vutils.make_grid(
        torch.cat([inst_pred_color, inst_gt_color], dim=0),
        nrow=n_images,
        normalize=False,
    )
    writer.add_image(f"{prefix}/vis_instance_pred_gt", grid_inst, step_value)


def _overlay_mask_rgb(img_bchw: torch.Tensor, mask_b1hw: torch.Tensor, color=(0.0, 1.0, 0.0), alpha: float = 0.4):
    """Overlay a single-channel mask on RGB images."""
    c = torch.tensor(color, device=img_bchw.device, dtype=img_bchw.dtype).view(1, 3, 1, 1)
    m = mask_b1hw.clamp(0, 1)
    return img_bchw * (1 - alpha * m) + c * (alpha * m)


def _build_meshes_from_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[Optional[Meshes], torch.Tensor]:
    """Build flat meshes and valid_k from batch lists."""
    verts_list = batch.get("verts_list", None)
    faces_list = batch.get("faces_list", None)
    if verts_list is None or faces_list is None:
        return None, torch.zeros((0, 0), dtype=torch.bool, device=device)

    verts_all: List[torch.Tensor] = []
    faces_all: List[torch.Tensor] = []
    counts: List[int] = []
    for vlist, flist in zip(verts_list, faces_list):
        v_t = [torch.from_numpy(v).float() for v in vlist]
        f_t = [torch.from_numpy(f).long() for f in flist]
        if len(v_t) != len(f_t):
            raise ValueError("verts_list and faces_list length mismatch")
        counts.append(len(v_t))
        verts_all.extend(v_t)
        faces_all.extend(f_t)

    B = len(counts)
    Kmax = max(counts) if counts else 0
    if Kmax == 0:
        return None, torch.zeros((B, 0), dtype=torch.bool, device=device)

    valid_k = torch.zeros((B, Kmax), dtype=torch.bool, device=device)
    for b, k in enumerate(counts):
        valid_k[b, :k] = True

    meshes_flat = Meshes(verts=verts_all, faces=faces_all).to(device)
    return meshes_flat, valid_k


def _build_model_points_from_batch(
    batch: Dict[str, Any],
    device: torch.device,
    max_points: int = 2048,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sample model points from meshes and pack into (B,K,N,3)."""
    verts_list = batch.get("verts_list", None)
    faces_list = batch.get("faces_list", None)
    diameters_list = batch.get("diameters_list", None)
    if verts_list is None or faces_list is None or diameters_list is None:
        return None, None

    verts_all: List[torch.Tensor] = []
    faces_all: List[torch.Tensor] = []
    diameters_all: List[torch.Tensor] = []
    counts: List[int] = []
    for vlist, flist, dlist in zip(verts_list, faces_list, diameters_list):
        v_t = [torch.as_tensor(v).float() for v in vlist]
        f_t = [torch.as_tensor(f).long() for f in flist]
        d_t = [float(torch.as_tensor(d).reshape(-1)[0].item()) for d in dlist]
        if len(v_t) != len(f_t) or len(v_t) != len(d_t):
            raise ValueError("verts_list, faces_list, and diameters_list length mismatch")
        counts.append(len(v_t))
        verts_all.extend(v_t)
        faces_all.extend(f_t)
        diameters_all.extend(d_t)

    B = len(counts)
    Kmax = max(counts) if counts else 0
    if Kmax == 0:
        return None, None

    meshes_flat = Meshes(verts=verts_all, faces=faces_all).to(device)
    pts_all = sample_points_from_meshes(meshes_flat, max_points)  # (sum_K, N, 3)

    model_points = pts_all.new_zeros(B, Kmax, pts_all.size(1), 3)
    diameters = pts_all.new_zeros(B, Kmax)
    idx = 0
    for b, k in enumerate(counts):
        if k > 0:
            model_points[b, :k] = pts_all[idx : idx + k]
            diameters[b, :k] = torch.as_tensor(
                diameters_all[idx : idx + k], device=device, dtype=pts_all.dtype
            )
        idx += k
    return model_points, diameters


def _adds_core_from_Rt_no_norm(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    model_points: torch.Tensor,
    *,
    use_symmetric: bool = True,
    max_points: int = 2048,
    valid_mask: Optional[torch.Tensor] = None,
    rot_only: bool = False,
) -> torch.Tensor:
    """Compute ADD/ADD-S without diameter normalization."""
    device = R_pred.device
    P = model_points if model_points.dim() == 4 else model_points.unsqueeze(1)
    N = P.size(2)
    if N > max_points:
        idx = torch.arange(N, device=device)[:max_points]
        P = P[:, :, idx, :]
    if rot_only:
        t_pred = t_gt

    def _xfm(R: torch.Tensor, t: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        return (R @ Q.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(2)

    Xp = _xfm(R_pred.to(torch.float32), t_pred.to(torch.float32), P.to(torch.float32))
    Xg = _xfm(R_gt.to(torch.float32), t_gt.to(torch.float32), P.to(torch.float32))

    if use_symmetric:
        D = torch.cdist(Xp, Xg, p=2)
        d1 = D.min(dim=3).values.mean(dim=2)
        d2 = D.min(dim=2).values.mean(dim=2)
        d = torch.minimum(d1, d2)
    else:
        d = (Xp - Xg).norm(dim=-1).mean(dim=-1)

    if valid_mask is not None:
        v = valid_mask.to(d.dtype)
        return (d * v).sum() / v.sum().clamp_min(1.0)
    return d.mean()


def _build_meshes_from_batch_filtered(
    batch: Dict[str, Any],
    valid_keep: torch.Tensor,
    device: torch.device,
) -> Optional[Meshes]:
    """Build flat meshes filtered by valid_keep."""
    verts_list = batch.get("verts_list", None)
    faces_list = batch.get("faces_list", None)
    if verts_list is None or faces_list is None:
        return None

    verts_all: List[torch.Tensor] = []
    faces_all: List[torch.Tensor] = []
    for b, (vlist, flist) in enumerate(zip(verts_list, faces_list)):
        for k, (v, f) in enumerate(zip(vlist, flist)):
            if k < valid_keep.size(1) and bool(valid_keep[b, k].item()):
                verts_all.append(torch.from_numpy(v).float())
                faces_all.append(torch.from_numpy(f).long())

    if not verts_all:
        return None
    return Meshes(verts=verts_all, faces=faces_all).to(device)


def _build_instance_weight_map(inst_ids: torch.Tensor, valid_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-instance weight maps from GT instance IDs."""
    B, H, W = inst_ids.shape
    Kmax = valid_k.shape[1]
    if Kmax == 0:
        wfg = (inst_ids > 0).unsqueeze(1).to(torch.float32)
        return inst_ids.new_zeros((B, 0, 1, H, W), dtype=torch.float32), wfg

    wfg = (inst_ids > 0).unsqueeze(1).to(torch.float32)
    wks = inst_ids.new_zeros((B, Kmax, 1, H, W), dtype=torch.float32)
    for k in range(Kmax):
        mask_k = (inst_ids == (k + 1)).unsqueeze(1).to(torch.float32)
        wks[:, k:k + 1] = mask_k.unsqueeze(2)
    wks = wks * valid_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(wks.dtype)
    return wks, wfg


def _build_valid_k_from_inst(inst_ids: torch.Tensor) -> torch.Tensor:
    """Build valid_k flags from instance ID maps."""
    B = inst_ids.size(0)
    max_id = int(inst_ids.max().item()) if inst_ids.numel() > 0 else 0
    if max_id == 0:
        return inst_ids.new_zeros((B, 0), dtype=torch.bool)

    valid_k = inst_ids.new_zeros((B, max_id), dtype=torch.bool)
    for k in range(max_id):
        valid_k[:, k] = (inst_ids == (k + 1)).any(dim=(1, 2))
    return valid_k


def _warn_missing_symmetry_keys(axes_list: Any, orders_list: Any) -> None:
    """Warn once if symmetry keys are missing or empty."""
    global _WARNED_SYMMETRY_KEYS
    if _WARNED_SYMMETRY_KEYS:
        return
    missing = axes_list is None or orders_list is None
    empty = False
    if not missing:
        empty_axes = True
        for axes_item in axes_list:
            if axes_item is None:
                continue
            if hasattr(axes_item, "__len__") and len(axes_item) > 0:
                empty_axes = False
                break
        empty_orders = True
        for orders_item in orders_list:
            if orders_item is None:
                continue
            if hasattr(orders_item, "__len__") and len(orders_item) > 0:
                empty_orders = False
                break
        empty = empty_axes or empty_orders
    if missing or empty:
        if dist_utils.is_main_process():
            print("[warn] symmetryキーが欠けている／空")
        _WARNED_SYMMETRY_KEYS = True


def _prepare_symmetry_tensors(
    batch: Dict[str, Any],
    device: torch.device,
    Kmax: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build symmetry axis/order tensors aligned with instance IDs."""
    axes_list = batch.get("symmetry_axes", None)
    orders_list = batch.get("symmetry_orders", None)
    _warn_missing_symmetry_keys(axes_list, orders_list)
    if axes_list is None or orders_list is None:
        return None, None
    B = len(axes_list)
    axes = torch.zeros((B, Kmax, 3), device=device, dtype=torch.float32)
    orders = torch.ones((B, Kmax), device=device, dtype=torch.float32)
    for b, (axes_item, orders_item) in enumerate(zip(axes_list, orders_list)):
        if axes_item is None or orders_item is None:
            continue
        if not isinstance(axes_item, (list, tuple)):
            axes_item = list(axes_item)
        if not isinstance(orders_item, (list, tuple)):
            orders_item = list(orders_item)
        k_lim = min(len(axes_item), Kmax)
        for k in range(k_lim):
            axes[b, k] = torch.as_tensor(axes_item[k], device=device, dtype=torch.float32).view(-1)[:3]
            orders[b, k] = float(torch.as_tensor(orders_item[k]).reshape(-1)[0].item())
    return axes, orders


def _rot_geodesic_map_deg(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pixel geodesic rotation error in degrees."""
    R_rel = torch.einsum("bijhw,bjkhw->bikhw", R_pred.transpose(1, 2), R_gt)
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_val = (tr - 1.0) * 0.5
    cos_clamped = cos_val.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_clamped)
    return theta * (180.0 / 3.141592653589793)


def _rot_geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Compute per-instance geodesic rotation error in degrees."""
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_val = (tr - 1.0) * 0.5
    cos_clamped = cos_val.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_clamped)
    return theta * (180.0 / 3.141592653589793)


def _origin_in_image_from_t(
    t_map: torch.Tensor,
    K_left: torch.Tensor,
    image_size: Tuple[int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Check if object origin projects inside the image."""
    H, W = image_size
    x = t_map[..., 0]
    y = t_map[..., 1]
    z = t_map[..., 2]
    z_safe = z.clamp_min(eps)

    fx = K_left[:, 0, 0].unsqueeze(1)
    fy = K_left[:, 1, 1].unsqueeze(1)
    cx = K_left[:, 0, 2].unsqueeze(1)
    cy = K_left[:, 1, 2].unsqueeze(1)

    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    in_img = (z > eps) & (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))
    return in_img


_CUBE24_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def _cube24_rotations(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate 24 cube rotations (det=+1) as rotation matrices."""
    key = (str(device), str(dtype))
    cached = _CUBE24_CACHE.get(key, None)
    if cached is not None:
        return cached
    mats: List[torch.Tensor] = []
    axes = [0, 1, 2]
    signs = [-1.0, 1.0]
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    R = torch.zeros((3, 3), dtype=torch.float32)
                    R[0, perm[0]] = sx
                    R[1, perm[1]] = sy
                    R[2, perm[2]] = sz
                    if torch.det(R) > 0.0:
                        mats.append(R)
    if len(mats) != 24:
        raise ValueError(f"cube24 rotations count mismatch: {len(mats)}")
    out = torch.stack(mats, dim=0).to(device=device, dtype=dtype)
    _CUBE24_CACHE[key] = out
    return out


def _axis_rotations(axis: torch.Tensor, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate rotations around a given axis (n steps)."""
    if n <= 0:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    axis = axis.to(device=device, dtype=dtype)
    axis = axis / axis.norm().clamp_min(1e-6)
    angles = torch.arange(n, device=device, dtype=dtype) * (2.0 * math.pi / float(n))
    rotvec = axis.view(1, 3) * angles.view(-1, 1)
    return rot_utils.so3_exp_batch(rotvec)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion (w,x,y,z) to rotation matrix."""
    w, x, y, z = q.unbind(dim=-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz
    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def _random_rotations(
    num: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample random rotations (approx uniform) using random quaternions."""
    if num <= 0:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    u1 = torch.rand(num, device=device, dtype=dtype, generator=generator)
    u2 = torch.rand(num, device=device, dtype=dtype, generator=generator)
    u3 = torch.rand(num, device=device, dtype=dtype, generator=generator)
    r1 = torch.sqrt(1.0 - u1)
    r2 = torch.sqrt(u1)
    t1 = 2.0 * math.pi * u2
    t2 = 2.0 * math.pi * u3
    q = torch.stack(
        [
            r2 * torch.cos(t2),
            r1 * torch.sin(t1),
            r1 * torch.cos(t1),
            r2 * torch.sin(t2),
        ],
        dim=-1,
    )
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return _quat_to_rotmat(q).to(dtype=dtype)


def _select_rotations(
    R: torch.Tensor,
    num: int,
    is_train: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select or pad rotations to fixed count."""
    if num <= 0:
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        return eye, torch.ones(1, device=device, dtype=torch.bool)
    if R.numel() == 0:
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        return eye.repeat(num, 1, 1), torch.zeros(num, device=device, dtype=torch.bool)

    H = R.size(0)
    if H >= num:
        if is_train and H > num:
            perm = torch.randperm(H, device=device)[:num]
            R_sel = R[perm]
        else:
            R_sel = R[:num]
        return R_sel, torch.ones(num, device=device, dtype=torch.bool)

    pad = num - H
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(pad, 1, 1)
    R_sel = torch.cat([R, eye], dim=0)
    valid = torch.zeros(num, device=device, dtype=torch.bool)
    valid[:H] = True
    return R_sel, valid


def _build_base_rotations(
    cfg: dict,
    num: int,
    device: torch.device,
    dtype: torch.dtype,
    is_train: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build base rotation hypotheses."""
    init_set = str(cfg.get("init_rot_set", "cube24")).lower()
    if init_set == "cube24":
        R = _cube24_rotations(device, dtype)
    elif init_set == "icosa60":
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        R = _random_rotations(60, device, dtype, generator=gen)
    elif init_set == "random":
        R = _random_rotations(num, device, dtype, generator=None if is_train else torch.Generator(device=device).manual_seed(0))
    elif init_set == "axisn":
        n = max(1, int(cfg.get("axisN_default", num)))
        axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        R = _axis_rotations(axis, n, device, dtype)
    elif init_set == "none":
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    else:
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return _select_rotations(R, num, is_train, device, dtype)


def _build_symmetry_rotations(
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    axis: Optional[torch.Tensor] = None,
    order: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build symmetry rotations for a single instance."""
    mode = str(cfg.get("symmetry_mode", "none")).lower()
    if mode == "none":
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        valid = torch.ones(1, device=device, dtype=torch.bool)
        return R, valid
    if mode == "cube24":
        R = _cube24_rotations(device, dtype)
        valid = torch.ones(R.size(0), device=device, dtype=torch.bool)
        return R, valid
    if mode == "axisn_from_batch":
        n_default = int(cfg.get("axisN_default", 16))
        n = n_default
        if order is not None:
            try:
                n = int(torch.as_tensor(order).item())
            except Exception:
                n = n_default
        if n <= 0:
            n = n_default
        if axis is None or not torch.isfinite(axis).all():
            axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        R = _axis_rotations(axis, n, device, dtype)
        valid = torch.ones(R.size(0), device=device, dtype=torch.bool)
        return R, valid
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    valid = torch.ones(1, device=device, dtype=torch.bool)
    return R, valid


def _sample_instance_points_from_point_map(
    point_map: torch.Tensor,
    inst_map: torch.Tensor,
    conf_map: Optional[torch.Tensor],
    n_points: int,
    conf_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample per-instance 3D points from a point map."""
    B, _, H, W = point_map.shape
    max_id = int(inst_map.max().item()) if inst_map.numel() > 0 else 0
    pts = point_map.new_zeros((B, max_id, n_points, 3))
    wts = point_map.new_zeros((B, max_id, n_points))
    valid_pts = torch.zeros((B, max_id, n_points), device=point_map.device, dtype=torch.bool)
    valid_inst = torch.zeros((B, max_id), device=point_map.device, dtype=torch.bool)
    if max_id <= 0:
        return pts, wts, valid_pts, valid_inst

    conf_mask = None
    if conf_map is not None:
        conf_mask = conf_map.squeeze(1) >= float(conf_thr)

    for b in range(B):
        for k in range(max_id):
            inst_id = k + 1
            mask = inst_map[b] == inst_id
            if conf_mask is not None:
                mask = mask & conf_mask[b]
            idx = torch.nonzero(mask, as_tuple=False)
            if idx.numel() == 0:
                continue
            valid_inst[b, k] = True
            if idx.size(0) > n_points:
                perm = torch.randperm(idx.size(0), device=point_map.device)[:n_points]
                idx = idx[perm]
            sel = point_map[b, :, idx[:, 0], idx[:, 1]].transpose(0, 1)
            pts[b, k, : sel.size(0)] = sel
            valid_pts[b, k, : sel.size(0)] = True
            if conf_map is not None:
                w = conf_map[b, 0, idx[:, 0], idx[:, 1]]
            else:
                w = point_map.new_ones((sel.size(0),))
            wts[b, k, : w.size(0)] = w
    return pts, wts, valid_pts, valid_inst


def pose_refine_implicit_sdf(
    *,
    z_inst: torch.Tensor,
    point_map_1x: torch.Tensor,
    point_conf_1x: Optional[torch.Tensor],
    inst_map: Optional[torch.Tensor] = None,
    wks_inst: Optional[torch.Tensor] = None,
    pos_map: Optional[torch.Tensor] = None,
    t0: Optional[torch.Tensor] = None,
    R0: Optional[torch.Tensor] = None,
    sym_axes: Optional[torch.Tensor] = None,
    sym_orders: Optional[torch.Tensor] = None,
    cfg: dict = None,
    implicit_sdf_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    is_train: bool = True,
) -> Dict[str, torch.Tensor]:
    """Refine pose using implicit SDF consistency with multi-hypotheses."""
    if cfg is None:
        cfg = {}
    pose_cfg = cfg.get("pose_refine", {}) or {}
    enabled = bool(pose_cfg.get("enabled", False))
    B, K, _ = z_inst.shape
    device = z_inst.device
    dtype = z_inst.dtype
    num_hyp = int(pose_cfg.get("num_hyp_train", 8 if is_train else 24))
    if not is_train:
        num_hyp = int(pose_cfg.get("num_hyp_test", num_hyp))
    if num_hyp <= 0:
        num_hyp = 1
    refine_steps = int(pose_cfg.get("refine_steps_train", 1 if is_train else 4))
    if not is_train:
        refine_steps = int(pose_cfg.get("refine_steps_test", refine_steps))
    n_points = int(pose_cfg.get("sample_M", 256))
    conf_thr = float(pose_cfg.get("conf_thr", 0.0))
    robust = str(pose_cfg.get("robust", "charbonnier")).lower()
    charb_eps = float(pose_cfg.get("charb_eps", 0.5))
    huber_tau = float(pose_cfg.get("huber_tau", 2.0))
    inlier_tau = float(pose_cfg.get("sdf_inlier_tau", 2.0))
    max_drot_deg = float(pose_cfg.get("max_drot_deg", 5.0))
    max_dt = float(pose_cfg.get("max_dt_mm", 10.0))
    lr_rot = float(pose_cfg.get("lr_rot", 0.2))
    lr_trans = float(pose_cfg.get("lr_trans", 0.5))
    min_px = int(pose_cfg.get("min_px", 30))
    min_wsum = float(pose_cfg.get("min_wsum", 1e-6))

    pose_R = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(B, K, 1, 1)
    pose_t = torch.zeros((B, K, 3), device=device, dtype=dtype)
    pose_valid = torch.zeros((B, K), device=device, dtype=torch.bool)
    best_hyp = torch.full((B, K), -1, device=device, dtype=torch.long)
    init_energy = 1e6
    E_hyp = torch.full((B, K, num_hyp), init_energy, device=device, dtype=torch.float32)
    E_hist = torch.full((B, K, num_hyp, max(1, refine_steps)), init_energy, device=device, dtype=torch.float32)
    inlier_hist = torch.zeros((B, K, num_hyp, max(1, refine_steps)), device=device, dtype=torch.float32)

    if not enabled or implicit_sdf_fn is None:
        return {
            "pose_R_refined": pose_R,
            "pose_t_refined": pose_t,
            "pose_valid": pose_valid,
            "best_hyp": best_hyp,
            "E_hyp": E_hyp,
            "E_hist": E_hist,
            "inlier_frac_hist": inlier_hist,
        }

    if inst_map is None:
        if wks_inst is None:
            return {
                "pose_R_refined": pose_R,
                "pose_t_refined": pose_t,
                "pose_valid": pose_valid,
                "best_hyp": best_hyp,
                "E_hyp": E_hyp,
                "E_hist": E_hist,
                "inlier_frac_hist": inlier_hist,
            }
        inst_map = wks_inst.squeeze(2).argmax(dim=1) + 1
    if inst_map.shape[-2:] != point_map_1x.shape[-2:]:
        inst_map = (
            F.interpolate(inst_map.unsqueeze(1).float(), size=point_map_1x.shape[-2:], mode="nearest")
            .squeeze(1)
            .to(inst_map.dtype)
        )

    p_cam, w_pts, valid_pts, valid_inst = _sample_instance_points_from_point_map(
        point_map_1x,
        inst_map,
        point_conf_1x,
        n_points=n_points,
        conf_thr=conf_thr,
    )
    if t0 is None:
        if pos_map is not None and wks_inst is not None:
            if pos_map.shape[-2:] != wks_inst.shape[-2:]:
                pos_map = F.interpolate(pos_map, size=wks_inst.shape[-2:], mode="bilinear", align_corners=False)
            w_sum = wks_inst.sum(dim=(3, 4)).clamp_min(min_wsum)
            t0 = (wks_inst * pos_map.unsqueeze(1)).sum(dim=(3, 4)) / w_sum
            valid_t0 = (wks_inst.sum(dim=(3, 4)) >= min_px) & torch.isfinite(t0).all(dim=-1)
        else:
            t0 = torch.zeros((B, K, 3), device=device, dtype=dtype)
            valid_t0 = torch.zeros((B, K), device=device, dtype=torch.bool)
    else:
        if wks_inst is not None:
            valid_t0 = (wks_inst.sum(dim=(2, 3, 4)) >= min_px) & torch.isfinite(t0).all(dim=-1)
        else:
            valid_t0 = torch.isfinite(t0).all(dim=-1)

    for b in range(B):
        for k in range(K):
            if not valid_inst[b, k]:
                continue
            if not valid_t0[b, k]:
                continue
            if valid_pts[b, k].sum().item() < max(1, min_px):
                continue

            z = z_inst[b, k].to(torch.float32)
            p = p_cam[b, k].to(torch.float32)
            w = w_pts[b, k].to(torch.float32)
            vmask = valid_pts[b, k]
            if vmask.sum().item() == 0:
                continue

            R_base, base_valid = _build_base_rotations(pose_cfg, num_hyp, device, torch.float32, is_train)
            if R0 is not None:
                R0_k = R0[b, k].to(torch.float32)
                R_base = torch.matmul(R0_k.unsqueeze(0), R_base)
            sym_axis = sym_axes[b, k] if sym_axes is not None else None
            sym_order = sym_orders[b, k] if sym_orders is not None else None
            G, _ = _build_symmetry_rotations(pose_cfg, device, torch.float32, sym_axis, sym_order)
            R_candidates = torch.matmul(R_base.unsqueeze(1), G.unsqueeze(0)).reshape(-1, 3, 3)
            R_candidates, hyp_valid = _select_rotations(
                R_candidates, num_hyp, is_train, device, torch.float32
            )
            if hyp_valid.sum().item() == 0:
                continue

            rotvec = rot_utils.so3_log_batch(R_candidates).detach()
            t = t0[b, k].view(1, 3).to(torch.float32).repeat(num_hyp, 1)
            w_mask = w * vmask.to(torch.float32)
            w_sum = w_mask.sum().clamp_min(1e-6)
            if w_sum.item() <= 0.0:
                continue

            n_steps = max(1, refine_steps)
            grad_ctx = contextlib.nullcontext() if torch.is_grad_enabled() else torch.enable_grad()
            with grad_ctx:
                for step in range(n_steps):
                    rotvec = rotvec.detach().requires_grad_(True)
                    t = t.detach().requires_grad_(True)
                    R = rot_utils.so3_exp_batch(rotvec)
                    p_rel = p.unsqueeze(0) - t.unsqueeze(1)
                    x_obj = torch.einsum("hij,hmj->hmi", R.transpose(-1, -2), p_rel)
                    z_h = z.view(1, 1, -1).expand(num_hyp, 1, -1)
                    x_h = x_obj.view(num_hyp, 1, -1, 3)
                    sdf = implicit_sdf_fn(z_h, x_h).view(num_hyp, -1)
                    if robust == "huber":
                        abs_r = sdf.abs()
                        loss = torch.where(
                            abs_r <= huber_tau,
                            0.5 * abs_r * abs_r,
                            huber_tau * (abs_r - 0.5 * huber_tau),
                        )
                    else:
                        loss = torch.sqrt(sdf * sdf + (charb_eps * charb_eps))
                    energy = (loss * w_mask.view(1, -1)).sum(dim=1) / w_sum
                    inlier = (sdf.abs() < inlier_tau) & (w_mask.view(1, -1) > 0.0)
                    denom = (w_mask.view(1, -1) > 0.0).sum(dim=1).clamp_min(1)
                    inlier_frac = inlier.float().sum(dim=1) / denom
                    E_hist[b, k, :, step] = energy.detach()
                    inlier_hist[b, k, :, step] = inlier_frac.detach()

                    if step < n_steps - 1:
                        grad_rot, grad_t = torch.autograd.grad(
                            energy.sum(),
                            [rotvec, t],
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )
                        if grad_rot is None or grad_t is None:
                            break
                        max_drot = math.radians(max_drot_deg)
                        drot = grad_rot
                        dt = grad_t
                        drot_norm = drot.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        dt_norm = dt.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        drot = drot * torch.clamp(max_drot / drot_norm, max=1.0)
                        dt = dt * torch.clamp(max_dt / dt_norm, max=1.0)
                        rotvec = (rotvec - lr_rot * drot).detach()
                        t = (t - lr_trans * dt).detach()

            E_hyp[b, k] = energy.detach()
            energy_valid = energy.clone()
            energy_valid[~hyp_valid] = float("inf")
            best = torch.argmin(energy_valid)
            best_hyp[b, k] = int(best.item())
            R_best = rot_utils.so3_exp_batch(rotvec)[best].to(dtype)
            t_best = t[best].to(dtype)
            pose_R[b, k] = R_best
            pose_t[b, k] = t_best
            pose_valid[b, k] = True

    return {
        "pose_R_refined": pose_R,
        "pose_t_refined": pose_t,
        "pose_valid": pose_valid,
        "best_hyp": best_hyp,
        "E_hyp": E_hyp,
        "E_hist": E_hist,
        "inlier_frac_hist": inlier_hist,
    }


def _log_pose_visuals(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    pos_map_gt: torch.Tensor,
    rot_map_gt: torch.Tensor,
    inst_gt: torch.Tensor,
    left_k: torch.Tensor,
    batch: Dict[str, Any],
    pos_gt,
    rot_gt,
    n_images: int,
) -> None:
    """Log pose and silhouette visualizations using GT instance masks."""
    if writer is None:
        return

    device = stereo.device
    meshes_flat, valid_k = _build_meshes_from_batch(batch, device)
    if meshes_flat is None or valid_k.numel() == 0:
        return

    size_hw = pred["pos_mu"].shape[-2:]
    inst_hw = _downsample_label(inst_gt, size_hw)
    wks, wfg = _build_instance_weight_map(inst_hw, valid_k)

    pos_map_pred = pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
    r_pred, t_pred, v_pred, _, _ = rot_utils.pose_from_maps_auto(
        rot_map=pred["rot_mat"],
        pos_map=pos_map_pred,
        Wk_1_4=wks,
        wfg=wfg,
        min_px=10,
        min_wsum=1e-6,
    )
    r_gt, t_gt, v_gt, _, _ = rot_utils.pose_from_maps_auto(
        rot_map=rot_map_gt,
        pos_map=pos_map_gt,
        Wk_1_4=wks,
        wfg=wfg,
        min_px=10,
        min_wsum=1e-6,
    )

    image_size = (stereo.shape[-2], stereo.shape[-1])
    origin_in = _origin_in_image_from_t(t_gt, left_k[:, 0], image_size)
    valid_render = valid_k & origin_in
    valid_pred = v_pred & valid_render
    valid_gt = v_gt & valid_render

    # T_pred = rot_utils.compose_T_from_Rt(r_pred, t_pred, valid_pred)
    # T_gt = rot_utils.compose_T_from_Rt(r_gt, t_gt, valid_gt)
    
    T_pred = rot_utils.compose_T_from_Rt(r_pred, t_pred, valid_pred)
    T_gt = rot_utils.compose_T_from_Rt(rot_gt, pos_gt, valid_gt)

    meshes_flat = _build_meshes_from_batch_filtered(batch, valid_render, device)
    if meshes_flat is None:
        return

    renderer = SilhouetteDepthRenderer().to(device)

    pred_r = renderer(
        meshes_flat=meshes_flat,
        T_cam_obj=T_pred,
        K_left=left_k[:, 0],
        valid_k=valid_render,
        image_size=image_size,
    )
    gt_r = renderer(
        meshes_flat=meshes_flat,
        T_cam_obj=T_gt,
        K_left=left_k[:, 0],
        valid_k=valid_render,
        image_size=image_size,
    )
    sil_pred = pred_r["silhouette"]
    sil_gt = gt_r["silhouette"]

    overlay_pred = _overlay_mask_rgb(stereo[:n_images, :3], sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45)
    overlay_gt = _overlay_mask_rgb(stereo[:n_images, :3], sil_gt[:n_images], color=(1.0, 0.0, 0.0), alpha=0.45)
    overlay_both = _overlay_mask_rgb(overlay_gt, sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45)

    axes_pred = draw_axes_on_images_bk(
        overlay_pred,
        left_k[:n_images, 0],
        T_pred[:n_images],
        axis_len=50,
        valid=valid_pred[:n_images],
    )
    axes_gt = draw_axes_on_images_bk(
        overlay_gt,
        left_k[:n_images, 0],
        T_gt[:n_images],
        axis_len=50,
        valid=valid_gt[:n_images],
    )

    grid = vutils.make_grid(
        torch.cat([stereo[:n_images, :3], axes_pred, axes_gt, overlay_both], dim=0),
        nrow=4,
        normalize=True,
        scale_each=True,
    )
    writer.add_image(f"{prefix}/vis_pose_sil", grid, step_value)


def _log_disp_and_depth(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    disp_pred: torch.Tensor,
    disp_gt: torch.Tensor,
    depth_pred_1x: torch.Tensor,
    depth_gt_mm: torch.Tensor,
    vis_mask: torch.Tensor,
    n_images: int,
    disp_scale: float = 1.0,
) -> None:
    """Log disparity/depth visualizations."""
    disp_vis = F.interpolate(
        disp_pred[:n_images],
        size=depth_gt_mm.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ) * float(disp_scale)
    depth_vis = F.interpolate(
        depth_pred_1x[:n_images],
        size=depth_gt_mm.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    vis_disp_pred, vis_disp_gt = visualize_mono_torch(
        disp_vis,
        vis_mask,
        disp_gt[:n_images],
        vis_mask,
    )
    vis_depth_pred, vis_depth_gt = visualize_mono_torch(
        depth_vis,
        vis_mask,
        depth_gt_mm[:n_images],
        vis_mask,
    )

    grid = vutils.make_grid(
        torch.cat(
            [stereo[:n_images, :3], stereo[:n_images, 3:], vis_disp_pred, vis_disp_gt],
            dim=0,
        ),
        nrow=4,
        normalize=True,
        scale_each=True,
    )
    writer.add_image(f"{prefix}/vis_disp", grid, step_value)

    grid = vutils.make_grid(
        torch.cat(
            [stereo[:n_images, :3], stereo[:n_images, 3:], vis_depth_pred, vis_depth_gt],
            dim=0,
        ),
        nrow=4,
        normalize=True,
        scale_each=True,
    )
    writer.add_image(f"{prefix}/vis_depth", grid, step_value)


def disparity_nll_laplace_scaled(
    preds: List[torch.Tensor],
    logvars: List[torch.Tensor],
    gt_disp_1x: torch.Tensor,
    gt_valid_1x: Optional[torch.Tensor],
    downsample: int,
    gamma: float = 0.9,
    charb_eps: float = 0.0,
) -> torch.Tensor:
    """Compute Laplace NLL loss for disparity predictions at a given scale."""
    assert len(preds) > 0
    assert len(preds) == len(logvars)

    h, w = preds[0].shape[-2:]
    gt_scaled = F.interpolate(gt_disp_1x, size=(h, w), mode="bilinear", align_corners=False) / float(downsample)

    if gt_valid_1x is None:
        valid_scaled = torch.ones((gt_disp_1x.size(0), 1, h, w), device=gt_disp_1x.device, dtype=gt_disp_1x.dtype)
    else:
        if gt_valid_1x.dim() == 3:
            v = (gt_valid_1x > 0).unsqueeze(1)
        else:
            v = (gt_valid_1x > 0)
        valid_scaled = F.interpolate(v.to(gt_disp_1x.dtype), size=(h, w), mode="nearest")

    total = gt_disp_1x.new_tensor(0.0)
    denom = gt_disp_1x.new_tensor(0.0)
    N = len(preds)

    for i in range(N):
        w_i = gamma ** (N - 1 - i)
        pred = preds[i]
        logv = logvars[i]
        r = pred - gt_scaled
        if charb_eps > 0.0:
            abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
        else:
            abs_r = r.abs()
        nll = abs_r * torch.exp(-0.5 * logv) + 0.5 * logv
        total = total + w_i * (nll * valid_scaled).sum()
        denom = denom + w_i * valid_scaled.sum()

    return total / denom.clamp_min(1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter],
    device: torch.device,
    scheduler=None,
    sched_step_when: Optional[str] = None,
    extra_loss_fn: Optional[Callable[[Dict[str, Any]], Dict[str, torch.Tensor]]] = None,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    l1 = torch.nn.L1Loss()
    log_every = cfg["train"]["log_interval"]

    window_sum, window_cnt = _init_window_meter()
    global_step = epoch * len(loader)

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = _prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        size_hw = stereo.shape[-2:]
        inst_gt_hw = _downsample_label(inst_gt, size_hw)
        valid_k_inst = _build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = _build_instance_weight_map(inst_gt_hw, valid_k_inst)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"]):
            pred = model(
                stereo,
                k_pair,
                baseline,
                iters=cfg["model"]["n_iter"],
                Wk_1_4=wks_inst,
                wfg_1_4=wfg_inst,
            )
            disp_preds = pred["disp_preds"]
            disp_logvar_preds = pred["disp_log_var_preds"]
            mask = (inst_gt > 0).unsqueeze(1) # without background
            loss_disp = disparity_nll_laplace_raft_style(
                disp_preds,
                disp_logvar_preds,
                disp_gt,
                mask,
            )
            loss_disp_1x = disparity_nll_laplace_scaled(
                [pred["disp_1x"]],
                [pred["disp_log_var_1x"]],
                disp_gt,
                mask,
                downsample=1,
            )

            size_hw = pred["sem_logits"].shape[-2:]
            sem_gt_hw = _downsample_label(sem_gt, size_hw)
            loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_hw, use_focal=False)
            loss_sem = loss_cls
            has_pos = all(k in pred for k in ("pos_mu", "pos_mu_norm", "pos_logvar_norm"))
            has_rot = all(k in pred for k in ("rot_mat", "rot_logvar_theta"))
            if has_pos and has_rot:
                pose_out = _compute_pose_losses_and_metrics(
                    pred=pred,
                    batch=batch,
                    sem_gt=sem_gt,
                    size_hw=size_hw,
                    left_k=left_k,
                    wks_inst=wks_inst,
                    wfg_inst=wfg_inst,
                    valid_k_inst=valid_k_inst,
                    image_size=(stereo.shape[-2], stereo.shape[-1]),
                    device=device,
                )
                loss_pos = pose_out["loss_pos"]
                loss_rot = pose_out["loss_rot"]
                rot_map_deg = pose_out["rot_map_deg"]
                pos_map_l2 = pose_out["pos_map_l2"]
                pos_inst_l2 = pose_out["pos_inst_l2"]
                rot_inst_deg = pose_out["rot_inst_deg"]
                adds = pose_out["adds"]
                add = pose_out["add"]
            elif has_pos:
                pos_out = _compute_pos_losses_and_metrics(
                    pred=pred,
                    batch=batch,
                    sem_gt=sem_gt,
                    size_hw=size_hw,
                    left_k=left_k,
                    device=device,
                )
                loss_pos = pos_out["loss_pos"]
                pos_map_l2 = pos_out["pos_map_l2"]
                zero = loss_pos.new_tensor(0.0)
                loss_rot = zero
                rot_map_deg = zero
                pos_inst_l2 = zero
                rot_inst_deg = zero
                adds = zero
                add = zero
            else:
                zero = loss_disp.new_tensor(0.0)
                loss_pos = zero
                loss_rot = zero
                rot_map_deg = zero
                pos_map_l2 = zero
                pos_inst_l2 = zero
                rot_inst_deg = zero
                adds = zero
                add = zero

            aff_tgt, aff_valid = _build_affinity_targets(inst_gt_hw)
            loss_aff, aff_valid_px = _affinity_loss(
                pred["aff_logits"],
                aff_tgt,
                aff_valid,
                neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 8.0)),
            )
            loss_emb, emb_pairs = embedding_cosface_sampled(
                pred["emb"],
                inst_gt_hw,
                max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
                margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
                scale=32.0,
                min_pixels_per_inst=4,
                detach_proto=True,
                topk_neg=None,           # 重いなら 16 とかに
            )
            logs = {
                "L_sem": loss_sem.detach(),
                "L_cls": loss_cls.detach(),
                "L_pos": loss_pos.detach(),
                "L_pos_map_l2": pos_map_l2.detach(),
                "L_pos_inst_l2": pos_inst_l2.detach(),
                "L_rot_map_deg": rot_map_deg.detach(),
                "L_rot_inst_deg": rot_inst_deg.detach(),
                "L_rot": loss_rot.detach(),
                "L_adds": adds.detach(),
                "L_add": add.detach(),
                "L_aff": loss_aff.detach(),
                "L_emb": loss_emb.detach(),
                "L_disp_1x": loss_disp_1x.detach(),
                "L_aff_valid_px": aff_valid_px.detach(),
                "L_emb_pairs": emb_pairs.detach(),
            }

            loss = cfg["loss"]["w_disp"] * loss_disp
            loss = loss + cfg.get("loss", {}).get("w_disp_1x", 1.0) * loss_disp_1x
            loss = loss + cfg.get("loss", {}).get("w_sem", 1.0) * loss_sem
            loss = loss + cfg.get("loss", {}).get("w_cls", 1.0) * loss_cls
            loss = loss + cfg.get("loss", {}).get("w_pos", 1.0) * loss_pos
            loss = loss + cfg.get("loss", {}).get("w_rot", 1.0) * loss_rot
            loss = loss + cfg.get("loss", {}).get("w_aff", 1.0) * loss_aff
            loss = loss + cfg.get("loss", {}).get("w_emb", 1.0) * loss_emb

            if extra_loss_fn is not None:
                extra_out = extra_loss_fn(
                    {
                        "model": model,
                        "pred": pred,
                        "batch": batch,
                        "cfg": cfg,
                        "device": device,
                        "stereo": stereo,
                        "sem_gt": sem_gt,
                        "inst_gt": inst_gt,
                        "inst_gt_hw": inst_gt_hw,
                        "valid_k_inst": valid_k_inst,
                        "wks_inst": wks_inst,
                        "wfg_inst": wfg_inst,
                        "left_k": left_k,
                        "size_hw": size_hw,
                        "writer": writer,
                        "epoch": epoch,
                        "it": it,
                        "global_step": global_step,
                        "log_interval": log_every,
                        "phase": "train",
                    }
                )
                if extra_out:
                    extra_loss = extra_out.get("loss", None)
                    if extra_loss is not None:
                        loss = loss + extra_loss
                    extra_logs = extra_out.get("logs", None)
                    if extra_logs:
                        logs.update(extra_logs)
                    pred_updates = extra_out.get("pred", None)
                    if pred_updates:
                        pred.update(pred_updates)

        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        depth_pred_1x = pred["point_map_1x"][:, 2:3]
        depth_mae = l1(depth_pred_1x[mask], depth[mask])
        depth_err = (depth_pred_1x - depth).abs()
        valid_mask = mask.bool()
        if valid_mask.any().item():
            depth_acc_1mm = (depth_err[valid_mask] < 1.0).float().mean()
            depth_acc_2mm = (depth_err[valid_mask] < 2.0).float().mean()
            depth_acc_4mm = (depth_err[valid_mask] < 4.0).float().mean()
        else:
            depth_acc_1mm = depth_err.new_tensor(0.0)
            depth_acc_2mm = depth_err.new_tensor(0.0)
            depth_acc_4mm = depth_err.new_tensor(0.0)
        logs["depth_acc_1mm"] = depth_acc_1mm
        logs["depth_acc_2mm"] = depth_acc_2mm
        logs["depth_acc_4mm"] = depth_acc_4mm

        did_optim_step = scaler.get_scale() >= prev_scale
        if scheduler is not None and sched_step_when == "step" and did_optim_step:
            scheduler.step()

        total_loss += loss.detach()
        window_cnt = _update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs)

        global_step = epoch * len(loader) + it
        if writer is not None and dist_utils.is_main_process() and (global_step % log_every == 0):
            _flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train")
            with torch.no_grad():
                _log_disp_and_depth(
                    writer,
                    global_step,
                    "train",
                    stereo,
                    pred["disp_1x"],
                    disp_gt,
                    depth_pred_1x,
                    depth,
                    mask[: min(4, stereo.size(0))].to(torch.float32),
                    n_images=min(4, stereo.size(0)),
                    disp_scale=1.0,
                )
                _log_segmentation_visuals(
                    writer,
                    global_step,
                    "train",
                    pred["sem_logits"],
                    pred["aff_logits"],
                    pred["emb"],
                    sem_gt,
                    inst_gt,
                    cfg,
                    n_images=min(4, stereo.size(0)),
                )
                if has_rot and has_pos:
                    pos_gt_map, rot_gt_map, _, objs_in_left = _prepare_pose_targets(
                        batch,
                        sem_gt,
                        pred["sem_logits"].shape[-2:],
                        device,
                    )
                    _log_pose_visuals(
                        writer,
                        global_step,
                        "train",
                        stereo,
                        pred,
                        pos_gt_map,
                        rot_gt_map,
                        inst_gt,
                        left_k,
                        batch,
                        objs_in_left[..., :3, 3],
                        objs_in_left[..., :3, :3],
                        n_images=min(4, stereo.size(0)),
                    )
            window_cnt = _reset_window_meter(window_sum, window_cnt)

    return total_loss.item() / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter],
    device: torch.device,
    extra_loss_fn: Optional[Callable[[Dict[str, Any]], Dict[str, torch.Tensor]]] = None,
) -> Tuple[float, float]:
    """Validate the model and return disparity and semantic losses."""
    model.eval()
    meters = DictMeters()
    meters.add_avg("L")
    meters.add_avg("L_sem")
    meters.add_avg("L_cls")
    meters.add_avg("L_pos")
    meters.add_avg("L_pos_map_l2")
    meters.add_avg("L_pos_inst_l2")
    meters.add_avg("L_rot")
    meters.add_avg("L_rot_map_deg")
    meters.add_avg("L_rot_inst_deg")
    meters.add_avg("L_adds")
    meters.add_avg("L_add")
    meters.add_avg("L_disp_1x")
    meters.add_avg("L_aff")
    meters.add_avg("L_emb")
    meters.add_sc("depth_acc_1mm")
    meters.add_sc("depth_acc_2mm")
    meters.add_sc("depth_acc_4mm")

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = _prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        size_hw = stereo.shape[-2:]
        inst_gt_hw = _downsample_label(inst_gt, size_hw)
        valid_k_inst = _build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = _build_instance_weight_map(inst_gt_hw, valid_k_inst)

        pred = model(
            stereo,
            k_pair,
            baseline,
            iters=cfg["model"]["n_iter"],
            Wk_1_4=wks_inst,
            wfg_1_4=wfg_inst,
        )
        disp_preds = pred["disp_preds"]
        disp_logvar_preds = pred["disp_log_var_preds"]
        mask = (inst_gt > 0).unsqueeze(1) # without background
        loss_disp = disparity_nll_laplace_raft_style(
            disp_preds,
            disp_logvar_preds,
            disp_gt,
            mask,
        )
        loss_disp_1x = disparity_nll_laplace_scaled(
            [pred["disp_1x"]],
            [pred["disp_log_var_1x"]],
            disp_gt,
            mask,
            downsample=1,
        )

        size_hw = pred["sem_logits"].shape[-2:]
        sem_gt_hw = _downsample_label(sem_gt, size_hw)
        loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_hw, use_focal=False)
        loss_sem = loss_cls
        has_pos = all(k in pred for k in ("pos_mu", "pos_mu_norm", "pos_logvar_norm"))
        has_rot = all(k in pred for k in ("rot_mat", "rot_logvar_theta"))
        if has_pos and has_rot:
            pose_out = _compute_pose_losses_and_metrics(
                pred=pred,
                batch=batch,
                sem_gt=sem_gt,
                size_hw=size_hw,
                left_k=left_k,
                wks_inst=wks_inst,
                wfg_inst=wfg_inst,
                valid_k_inst=valid_k_inst,
                image_size=(stereo.shape[-2], stereo.shape[-1]),
                device=device,
            )
            loss_pos = pose_out["loss_pos"]
            loss_rot = pose_out["loss_rot"]
            rot_map_deg = pose_out["rot_map_deg"]
            pos_map_l2 = pose_out["pos_map_l2"]
            pos_inst_l2 = pose_out["pos_inst_l2"]
            rot_inst_deg = pose_out["rot_inst_deg"]
            adds = pose_out["adds"]
            add = pose_out["add"]
        elif has_pos:
            pos_out = _compute_pos_losses_and_metrics(
                pred=pred,
                batch=batch,
                sem_gt=sem_gt,
                size_hw=size_hw,
                left_k=left_k,
                device=device,
            )
            loss_pos = pos_out["loss_pos"]
            pos_map_l2 = pos_out["pos_map_l2"]
            zero = loss_pos.new_tensor(0.0)
            loss_rot = zero
            rot_map_deg = zero
            pos_inst_l2 = zero
            rot_inst_deg = zero
            adds = zero
            add = zero
        else:
            zero = loss_disp.new_tensor(0.0)
            loss_pos = zero
            loss_rot = zero
            rot_map_deg = zero
            pos_map_l2 = zero
            pos_inst_l2 = zero
            rot_inst_deg = zero
            adds = zero
            add = zero

        aff_tgt, aff_valid = _build_affinity_targets(inst_gt_hw)
        loss_aff, _ = _affinity_loss(
            pred["aff_logits"],
            aff_tgt,
            aff_valid,
            neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 4.0)),
        )
        loss_emb, emb_pairs = embedding_cosface_sampled(
            pred["emb"],
            inst_gt_hw,
            max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
            margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
            scale=32.0,
            min_pixels_per_inst=4,
            detach_proto=True,
            topk_neg=None,           # 重いなら 16 とかに
        )

        loss = cfg["loss"]["w_disp"] * loss_disp
        loss = loss + cfg.get("loss", {}).get("w_disp_1x", 1.0) * loss_disp_1x
        loss = loss + cfg.get("loss", {}).get("w_sem", 1.0) * loss_sem
        loss = loss + cfg.get("loss", {}).get("w_cls", 1.0) * loss_cls
        loss = loss + cfg.get("loss", {}).get("w_pos", 1.0) * loss_pos
        loss = loss + cfg.get("loss", {}).get("w_rot", 1.0) * loss_rot
        loss = loss + cfg.get("loss", {}).get("w_aff", 1.0) * loss_aff
        loss = loss + cfg.get("loss", {}).get("w_emb", 1.0) * loss_emb

        if extra_loss_fn is not None:
            extra_out = extra_loss_fn(
                {
                    "model": model,
                    "pred": pred,
                    "batch": batch,
                    "cfg": cfg,
                    "device": device,
                    "stereo": stereo,
                    "sem_gt": sem_gt,
                    "inst_gt": inst_gt,
                    "inst_gt_hw": inst_gt_hw,
                    "valid_k_inst": valid_k_inst,
                    "wks_inst": wks_inst,
                    "wfg_inst": wfg_inst,
                    "left_k": left_k,
                    "size_hw": size_hw,
                    "writer": writer,
                    "epoch": epoch,
                    "it": it,
                    "global_step": epoch,
                    "log_interval": 1,
                    "phase": "val",
                }
            )
            if extra_out:
                extra_loss = extra_out.get("loss", None)
                if extra_loss is not None:
                    loss = loss + extra_loss
                extra_logs = extra_out.get("logs", None)
                if extra_logs:
                    for k, v in extra_logs.items():
                        if k not in meters.m:
                            meters.add_avg(k)
                        meters.update_avg(k, float(v.item()), n=stereo.size(0))
                pred_updates = extra_out.get("pred", None)
                if pred_updates:
                    pred.update(pred_updates)

        meters.update_avg("L", float(loss.item()), n=stereo.size(0))
        meters.update_avg("L_sem", float(loss_sem.item()), n=stereo.size(0))
        meters.update_avg("L_cls", float(loss_cls.item()), n=stereo.size(0))
        meters.update_avg("L_pos", float(loss_pos.item()), n=stereo.size(0))
        meters.update_avg("L_pos_map_l2", float(pos_map_l2.item()), n=stereo.size(0))
        meters.update_avg("L_pos_inst_l2", float(pos_inst_l2.item()), n=stereo.size(0))
        meters.update_avg("L_rot", float(loss_rot.item()), n=stereo.size(0))
        meters.update_avg("L_rot_map_deg", float(rot_map_deg.item()), n=stereo.size(0))
        meters.update_avg("L_rot_inst_deg", float(rot_inst_deg.item()), n=stereo.size(0))
        meters.update_avg("L_adds", float(adds.item()), n=stereo.size(0))
        meters.update_avg("L_add", float(add.item()), n=stereo.size(0))
        meters.update_avg("L_disp_1x", float(loss_disp_1x.item()), n=stereo.size(0))
        meters.update_avg("L_aff", float(loss_aff.item()), n=stereo.size(0))
        meters.update_avg("L_emb", float(loss_emb.item()), n=stereo.size(0))

        depth_pred_1x = pred["point_map_1x"][:, 2:3]
        depth_err = (depth_pred_1x - depth).abs()
        valid_mask = mask.bool()
        valid_count = float(valid_mask.sum().item())
        if valid_count > 0.0:
            acc_1mm = float((depth_err[valid_mask] < 1.0).sum().item())
            acc_2mm = float((depth_err[valid_mask] < 2.0).sum().item())
            acc_4mm = float((depth_err[valid_mask] < 4.0).sum().item())
        else:
            acc_1mm = 0.0
            acc_2mm = 0.0
            acc_4mm = 0.0
        meters.update_sc("depth_acc_1mm", acc_1mm, valid_count)
        meters.update_sc("depth_acc_2mm", acc_2mm, valid_count)
        meters.update_sc("depth_acc_4mm", acc_4mm, valid_count)

        if writer is not None and dist_utils.is_main_process() and it == 0:
            _log_disp_and_depth(
                writer,
                epoch,
                "val",
                stereo,
                pred["disp_1x"],
                disp_gt,
                depth_pred_1x,
                depth,
                mask[: min(4, stereo.size(0))].to(torch.float32),
                n_images=min(4, stereo.size(0)),
                disp_scale=1.0,
            )
            _log_segmentation_visuals(
                writer,
                epoch,
                "val",
                pred["sem_logits"],
                pred["aff_logits"],
                pred["emb"],
                sem_gt,
                inst_gt,
                cfg,
                n_images=min(4, stereo.size(0)),
            )
            if has_rot and has_pos:
                pos_gt_map, rot_gt_map, _, objs_in_left = _prepare_pose_targets(
                    batch,
                    sem_gt,
                    pred["sem_logits"].shape[-2:],
                    device,
                )
                _log_pose_visuals(
                    writer,
                    epoch,
                    "val",
                    stereo,
                    pred,
                    pos_gt_map,
                    rot_gt_map,
                    inst_gt,
                    left_k,
                    batch,
                    objs_in_left[..., :3, 3],
                    objs_in_left[..., :3, :3],
                    n_images=min(4, stereo.size(0)),
                )

    loss_disp_avg = meters.get("L").avg
    loss_sem_avg = meters.get("L_sem").avg
    if writer is not None and dist_utils.is_main_process():
        log_meters_to_tb(writer, meters, epoch, prefix="val")
    return loss_disp_avg, loss_sem_avg


def main() -> None:
    """Entry point for panoptic stereo training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config TOML", default="configs/small_config_panoptic.toml")
    parser.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    args = parser.parse_args()

    cfg = load_toml(args.config)
    cfg.setdefault("train", {})
    cfg["train"]["amp"] = True
    set_global_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.launcher != "none":
        distributed = True
        if platform.system() == "Windows":
            backend = "gloo"
        else:
            backend = "nccl"
        dist_utils.init_dist(args.launcher, backend)
    else:
        distributed = False
    dist_utils.setup_for_distributed(is_master=dist_utils.is_main_process())

    out_dir = Path(os.path.join(cfg["train"]["output_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")))
    if dist_utils.is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        eff_cfg_path = out_dir / "config_used.toml"
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("runtime", {})
        cfg_copy["runtime"]["start_time"] = datetime.now().isoformat()
        cfg_copy["runtime"]["world_size"] = dist_utils.get_world_size()
        write_toml(cfg_copy, eff_cfg_path)
    dist_utils.barrier()

    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if dist_utils.is_main_process() else None

    device = torch.device("cuda", dist_utils.get_rank()) if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader, train_sampler, val_sampler, n_classes = make_dataloaders(
        cfg, distributed=dist.is_initialized()
    )
    cfg.setdefault("data", {})
    cfg["data"]["n_classes"] = n_classes

    model = build_model(cfg, n_classes).to(device)
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_utils.get_rank()],
            output_device=dist_utils.get_rank(),
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler, sched_step_when = build_lr_scheduler(cfg, optimizer, steps_per_epoch, total_steps)

    model_path_cfg = str(cfg.get("train", {}).get("model_path", "") or "").strip()
    load_mode_cfg = str(cfg.get("train", {}).get("load_mode", "auto")).lower()
    strict_cfg = bool(cfg.get("train", {}).get("strict", False))

    if os.path.exists(model_path_cfg):
        ckpt = torch.load(model_path_cfg, map_location=device)
        if load_mode_cfg == "resume":
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = _load_model_state(model, state, strict=strict_cfg)
            if dist_utils.is_main_process():
                print(f"[config-load][resume] {model_path_cfg}")
                if missing:
                    print(f"  missing keys: {missing}")
                if unexpected:
                    print(f"  unexpected keys: {unexpected}")

            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(ckpt, dict) and "scaler" in ckpt and cfg["train"]["amp"]:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    if dist_utils.is_main_process():
                        print(f"[resume] AMP scaler load skipped: {e}")
            if isinstance(ckpt, dict) and "scheduler" in ckpt and ckpt["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    if dist_utils.is_main_process():
                        print(f"[resume] scheduler load skipped: {e}")

            if isinstance(ckpt, dict) and "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
        else:
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = _load_model_state(model, state, strict=strict_cfg)
            if dist_utils.is_main_process():
                print(f"[config-load][weights] {model_path_cfg}")
                if missing:
                    print(f"  missing keys: {missing}")
                if unexpected:
                    print(f"  unexpected keys: {unexpected}")

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            cfg,
            writer,
            device,
            scheduler,
            sched_step_when,
        )
        val_disp, val_sem = validate(model, val_loader, epoch, cfg, writer, device)

        if scheduler is not None:
            if sched_step_when == "epoch":
                scheduler.step()
            elif sched_step_when == "epoch_metric":
                scheduler.step(val_disp)

        if dist_utils.is_main_process():
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_disp={val_disp:.4f}  val_sem={val_sem:.4f}")
            ckpt = {
                "epoch": epoch,
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, out_dir / f"checkpoint_{epoch:03d}.pth")

    if writer is not None:
        writer.close()

    dist_utils.barrier()


if __name__ == "__main__":
    main()
