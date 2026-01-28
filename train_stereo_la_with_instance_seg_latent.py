
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo training for panoptic segmentation with latent map head and implicit SDF losses.
"""
import argparse
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import roi_align
import torchvision.utils as vutils

from la_loader.synthetic_data_loader import LASyntheticDataset3PerIns
from la_loader import la_transforms

from models.panoptic_stereo import PanopticStereoMultiHeadLatent, ImplicitSDFHead
from models.stereo_disparity import make_gn
from utils import dist_utils, rot_utils
from utils.logging_utils import draw_axes_on_images_bk
from utils.projection import SilhouetteDepthRenderer
import train_panoptic_utils as pan_utils
from train_utils import (
    build_lr_scheduler,
    load_model_state as _load_model_state,
    load_toml,
    seed_worker,
    set_global_seed,
    write_toml,
)

mp.set_start_method("spawn", force=True)
supported = getattr(mp, "get_all_sharing_strategies", lambda: ["file_system"])()
strategy = "file_descriptor" if "file_descriptor" in supported else "file_system"
mp.set_sharing_strategy(strategy)


_VARLEN_KEYS = {
    "class_ids",
    "diameters_list",
    "faces_list",
    "object_ids",
    "objs_in_left",
    "objs_in_right",
    "SDFs",
    "SDFs_meta",
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
    loss_cfg = cfg.get("loss", {}) or {}
    use_implicit = bool(loss_cfg.get("use_implicit_sdf", False))
    w_sdf_map = float(loss_cfg.get("w_sdf", 0.0))

    use_camera_list = ["ZED2", "D415", "ZEDmini"]
    out_size_wh = (cfg["data"]["width"], cfg["data"]["height"])
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


def _ensure_sdf_meta_in_batch(batch: dict) -> dict:
    """Ensure SDFs_meta exists even if dataset only returns SDFs."""
    if not isinstance(batch, dict):
        return batch
    if "SDFs" not in batch:
        return batch
    if batch.get("SDFs_meta", None) is not None:
        return batch

    sdf_list = batch.get("SDFs", None)
    if not isinstance(sdf_list, list):
        return batch

    meta_list = []
    for sdf_objs in sdf_list:
        if sdf_objs is None:
            meta_list.append(None)
            continue
        if not isinstance(sdf_objs, list):
            meta_list.append(None)
            continue
        meta_objs = []
        for entry in sdf_objs:
            meta = None
            if isinstance(entry, dict):
                if isinstance(entry.get("meta", None), dict):
                    meta = entry["meta"]
                elif "bbox_min" in entry and "bbox_max" in entry:
                    meta = {"bbox_min": entry["bbox_min"], "bbox_max": entry["bbox_max"]}
            if meta is None:
                meta = {}
            meta_objs.append(meta)
        meta_list.append(meta_objs)

    batch["SDFs_meta"] = meta_list
    return batch


class _BatchMetaLoader:
    """Wrap DataLoader to inject SDF meta when missing."""

    def __init__(self, loader):
        self._loader = loader

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for batch in self._loader:
            yield _ensure_sdf_meta_in_batch(batch)


def _extract_sdf_scale(meta, device, dtype) -> torch.Tensor:
    """Return scale factor to convert normalized SDF values to original units."""
    scale = 1.0
    if isinstance(meta, dict) and bool(meta.get("normalize_to_cube", False)):
        tr = meta.get("transform", {}) if isinstance(meta.get("transform", {}), dict) else {}
        raw = tr.get("scale", None)
        if raw is not None:
            try:
                raw = float(raw)
                if raw > 0.0:
                    scale = 1.0 / raw
            except Exception:
                pass
    return torch.tensor(scale, device=device, dtype=dtype)


def _aggregate_latent_per_instance(
    latent_map: torch.Tensor,
    wks_inst: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Aggregate latent map into per-instance vectors."""
    if latent_map.shape[-2:] != wks_inst.shape[-2:]:
        latent_map = F.interpolate(latent_map, size=wks_inst.shape[-2:], mode="bilinear", align_corners=False)
    wks = wks_inst.to(latent_map.dtype)
    denom = wks.sum(dim=(3, 4)).clamp_min(eps)
    lat_sum = (latent_map.unsqueeze(1) * wks).sum(dim=(3, 4))
    return lat_sum / denom


def _compute_t0_from_pos_map(
    pos_map: torch.Tensor,
    wks_inst: torch.Tensor,
    min_wsum: float = 1e-6,
    min_px: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-instance translation seeds from a point map."""
    if pos_map.shape[-2:] != wks_inst.shape[-2:]:
        pos_map = F.interpolate(pos_map, size=wks_inst.shape[-2:], mode="bilinear", align_corners=False)
    wks = wks_inst.to(pos_map.dtype)
    w_sum = wks.sum(dim=(2, 3, 4))
    t0 = (wks * pos_map.unsqueeze(1)).sum(dim=(3, 4)) / w_sum.clamp_min(min_wsum).unsqueeze(-1)
    valid = (w_sum >= float(min_px)) & torch.isfinite(t0).all(dim=-1)
    return t0, valid


def _log_pose_visuals_sdf(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: dict,
    inst_gt: torch.Tensor,
    left_k: torch.Tensor,
    batch: dict,
    pos_gt: torch.Tensor,
    rot_gt: torch.Tensor,
    n_images: int,
) -> None:
    """Log pose and silhouette visualizations using SDF delta-pose outputs."""
    if writer is None:
        return
    if "pose_R_refined" not in pred or "pose_t_refined" not in pred or "pose_valid" not in pred:
        return

    device = stereo.device
    meshes_flat, valid_k = pan_utils._build_meshes_from_batch(batch, device)
    if meshes_flat is None or valid_k.numel() == 0:
        return

    size_hw = pred["pos_mu"].shape[-2:]
    inst_hw = pan_utils._downsample_label(inst_gt, size_hw)
    wks, _ = pan_utils._build_instance_weight_map(inst_hw, valid_k)

    R_pred = pred["pose_R_refined"]
    t_pred = pred["pose_t_refined"]
    v_pred = pred["pose_valid"].to(dtype=torch.bool)

    image_size = (stereo.shape[-2], stereo.shape[-1])
    origin_in = pan_utils._origin_in_image_from_t(pos_gt, left_k[:, 0], image_size)
    valid_pred = v_pred & valid_k & origin_in
    valid_render = valid_pred
    valid_gt = valid_render

    T_pred = rot_utils.compose_T_from_Rt(R_pred, t_pred, valid_pred)
    T_gt = rot_utils.compose_T_from_Rt(rot_gt, pos_gt, valid_gt)

    meshes_flat = pan_utils._build_meshes_from_batch_filtered(batch, valid_render, device)
    if meshes_flat is None:
        return

    renderer = SilhouetteDepthRenderer().to(device)
    meshes_flat = meshes_flat.to(device)
    T_pred = T_pred.to(dtype=torch.float32)
    T_gt = T_gt.to(dtype=torch.float32)
    K_left = left_k[:, 0].to(dtype=torch.float32)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(autocast_device, enabled=False):
        pred_r = renderer(
            meshes_flat=meshes_flat,
            T_cam_obj=T_pred,
            K_left=K_left,
            valid_k=valid_render,
            image_size=image_size,
        )
        gt_r = renderer(
            meshes_flat=meshes_flat,
            T_cam_obj=T_gt,
            K_left=K_left,
            valid_k=valid_render,
            image_size=image_size,
        )
    sil_pred = pred_r["silhouette"]
    sil_gt = gt_r["silhouette"]

    overlay_pred = pan_utils._overlay_mask_rgb(
        stereo[:n_images, :3], sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45
    )
    overlay_gt = pan_utils._overlay_mask_rgb(
        stereo[:n_images, :3], sil_gt[:n_images], color=(1.0, 0.0, 0.0), alpha=0.45
    )
    overlay_both = pan_utils._overlay_mask_rgb(overlay_gt, sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45)

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
    writer.add_image(f"{prefix}/vis_pose_sil_sdf", grid, step_value)


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


def _extract_sdf_bounds(
    meta: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """Extract bbox min/max from SDF meta; return valid flag."""
    if not isinstance(meta, dict):
        return None, None, False

    bbox_min = meta.get("bbox_min", None)
    bbox_max = meta.get("bbox_max", None)
    if bbox_min is None or bbox_max is None:
        bbox_min = meta.get("grid_min", None)
        bbox_max = meta.get("grid_max", None)
    if bbox_min is None or bbox_max is None:
        return None, None, False

    bbox_min = torch.as_tensor(bbox_min, device=device, dtype=dtype).view(-1)[:3]
    bbox_max = torch.as_tensor(bbox_max, device=device, dtype=dtype).view(-1)[:3]
    if bbox_min.numel() != 3 or bbox_max.numel() != 3:
        return None, None, False
    if not torch.isfinite(bbox_min).all() or not torch.isfinite(bbox_max).all():
        return None, None, False
    if torch.any(bbox_max <= bbox_min):
        return None, None, False
    return bbox_min, bbox_max, True


def _sample_sdf_volume(
    sdf_vol: torch.Tensor,
    coords_norm: torch.Tensor,
) -> torch.Tensor:
    """Sample a 3D SDF volume at normalized coordinates in [-1, 1]."""
    if coords_norm.numel() == 0:
        return coords_norm.new_empty((0,))
    grid = coords_norm.view(1, -1, 1, 1, 3)
    vals = F.grid_sample(
        sdf_vol,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return vals.view(-1)


def _build_sdf_targets(
    batch: dict,
    inst_gt_hw: torch.Tensor,
    obj_coords: torch.Tensor,
    device: torch.device,
    cfg: dict,
):
    """Build per-pixel SDF target map with scale correction from meta."""
    sdf_list = batch.get("SDFs", None)
    sdf_meta_list = batch.get("SDFs_meta", None)
    if sdf_list is None or sdf_meta_list is None:
        return None, None
    if not isinstance(sdf_list, list) or not isinstance(sdf_meta_list, list):
        return None, None

    B, H, W = inst_gt_hw.shape
    if len(sdf_list) != B or len(sdf_meta_list) != B:
        return None, None

    max_points = int(cfg.get("loss", {}).get("sdf_max_points", 0))
    sdf_gt = obj_coords.new_zeros((B, 1, H, W))
    sdf_valid = torch.zeros((B, 1, H, W), device=device, dtype=torch.bool)

    for b in range(B):
        sdf_objs = sdf_list[b]
        meta_objs = sdf_meta_list[b]
        if sdf_objs is None or meta_objs is None:
            continue
        K = min(len(sdf_objs), len(meta_objs))
        if K <= 0:
            continue
        inst_map = inst_gt_hw[b]
        obj_coords_b = obj_coords[b].permute(1, 2, 0)
        sdf_gt_flat = sdf_gt[b, 0].view(-1)
        sdf_valid_flat = sdf_valid[b, 0].view(-1)

        for k in range(K):
            inst_id = k + 1
            mask = inst_map == inst_id
            if not mask.any():
                continue
            idx = torch.nonzero(mask, as_tuple=False)
            if idx.numel() == 0:
                continue
            if max_points > 0 and idx.size(0) > max_points:
                perm = torch.randperm(idx.size(0), device=device)[:max_points]
                idx = idx[perm]
            coords = obj_coords_b[idx[:, 0], idx[:, 1]]
            bbox_min, bbox_max, ok = _extract_sdf_bounds(meta_objs[k], device, coords.dtype)
            if not ok:
                continue
            denom = (bbox_max - bbox_min).clamp_min(1e-6)
            coords_norm = (coords - bbox_min) * (2.0 / denom) - 1.0
            valid = (coords_norm.abs() <= 1.0).all(dim=1)
            if not valid.any():
                continue
            coords_norm = coords_norm[valid]
            idx = idx[valid]

            sdf_entry = sdf_objs[k]
            if isinstance(sdf_entry, dict) and "sdf" in sdf_entry:
                sdf_entry = sdf_entry["sdf"]
            if sdf_entry is None:
                continue
            sdf_vol = torch.as_tensor(sdf_entry, device=device, dtype=obj_coords.dtype)
            if sdf_vol.dim() == 4 and sdf_vol.size(0) == 1:
                sdf_vol = sdf_vol.squeeze(0)
            if sdf_vol.dim() != 3:
                continue
            sdf_vol = sdf_vol.unsqueeze(0).unsqueeze(0)
            sdf_vals = _sample_sdf_volume(sdf_vol, coords_norm)

            scale = _extract_sdf_scale(meta_objs[k], device, sdf_vals.dtype)
            sdf_vals = (sdf_vals * scale).to(dtype=sdf_gt.dtype)

            finite = torch.isfinite(sdf_vals)
            if not finite.all():
                sdf_vals = sdf_vals[finite]
                idx = idx[finite]
            if sdf_vals.numel() == 0:
                continue

            flat_idx = idx[:, 0] * W + idx[:, 1]
            sdf_gt_flat[flat_idx] = sdf_vals
            sdf_valid_flat[flat_idx] = True

    return sdf_gt, sdf_valid


def _sdf_nll_laplace_map(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    charb_eps: float = 0.0,
) -> torch.Tensor:
    """Compute heteroscedastic Laplace NLL for SDF maps."""
    r = pred - target
    if charb_eps > 0.0:
        abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
    else:
        abs_r = r.abs()
    nll = abs_r * torch.exp(-0.5 * logvar) + 0.5 * logvar
    valid_f = valid.to(nll.dtype)
    return (nll * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def _build_sdf_hist_weights(
    sdf_gt: torch.Tensor,
    sdf_valid: torch.Tensor,
    cfg: dict,
) -> Optional[torch.Tensor]:
    """Build per-pixel weights to flatten |SDF| histogram."""
    n_bins = int(cfg.get("loss", {}).get("sdf_hist_bins", 0))
    if n_bins <= 0:
        return None
    abs_sdf = sdf_gt.abs()
    valid = sdf_valid
    abs_valid = abs_sdf[valid]
    if abs_valid.numel() == 0:
        return None

    max_dist = float(cfg.get("loss", {}).get("sdf_hist_max", 0.0))
    if max_dist <= 0.0:
        max_dist = float(abs_valid.max().item())
    max_dist = max(max_dist, 1e-6)

    clipped = abs_sdf.clamp(max=max_dist)
    bin_idx = torch.clamp((clipped / max_dist * n_bins).long(), max=n_bins - 1)
    bin_idx_valid = bin_idx[valid]

    counts = torch.bincount(bin_idx_valid, minlength=n_bins).to(dtype=abs_sdf.dtype)
    smooth = float(cfg.get("loss", {}).get("sdf_hist_smooth", 1.0))
    if smooth > 0.0:
        counts = counts + smooth
    inv = 1.0 / counts.clamp_min(1e-6)
    w_valid = inv[bin_idx_valid]
    w_mean = w_valid.mean().clamp_min(1e-6)
    inv = inv / w_mean

    weight_map = torch.zeros_like(abs_sdf)
    weight_map[valid] = inv[bin_idx_valid]
    return weight_map


def _sdf_nll_laplace_weighted(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    charb_eps: float = 0.0,
) -> torch.Tensor:
    """Compute weighted heteroscedastic Laplace NLL for SDF maps."""
    r = pred - target
    if charb_eps > 0.0:
        abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
    else:
        abs_r = r.abs()
    nll = abs_r * torch.exp(-0.5 * logvar) + 0.5 * logvar
    valid_f = valid.to(nll.dtype)
    w = weight.to(nll.dtype) * valid_f
    return (nll * w).sum() / w.sum().clamp_min(1.0)

def _compute_sdf_map_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    inst_gt_hw: torch.Tensor,
    size_hw: Tuple[int, int],
    device: torch.device,
    cfg: dict,
) -> Dict[str, torch.Tensor]:
    """Compute SDF loss using CAD SDF volumes and predicted sdf map/logvar."""
    zero = pred["disp_1x"].new_tensor(0.0)
    sdf_pred = pred.get("sdf_map", None)
    if sdf_pred is None:
        latent_map = pred.get("latent_map", None)
        if latent_map is None:
            return {"loss_sdf": zero, "sdf_valid_px": zero}
        if latent_map.size(1) != 1:
            raise ValueError("sdf_map missing and latent_map channel != 1")
        sdf_pred = latent_map
    sdf_logvar = pred.get("sdf_logvar", None)
    if sdf_logvar is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}

    pos_map, rot_map = _prepare_pose_maps(batch, size_hw, device)
    if pos_map is None or rot_map is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}
    point_map = pred.get("point_map_1x", None)
    if point_map is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}
    if point_map.shape[-2:] != size_hw:
        point_map = F.interpolate(point_map, size=size_hw, mode="bilinear", align_corners=False)
    if sdf_pred.shape[-2:] != size_hw:
        sdf_pred = F.interpolate(sdf_pred, size=size_hw, mode="bilinear", align_corners=False)
    if sdf_logvar.shape[-2:] != size_hw:
        sdf_logvar = F.interpolate(sdf_logvar, size=size_hw, mode="bilinear", align_corners=False)

    vec = point_map - pos_map
    R = rot_map.permute(0, 3, 4, 1, 2)
    vec = vec.permute(0, 2, 3, 1)
    obj_coords = torch.einsum("bhwij,bhwj->bhwi", R.transpose(-1, -2), vec)
    obj_coords = obj_coords.permute(0, 3, 1, 2)

    sdf_gt, sdf_valid = _build_sdf_targets(batch, inst_gt_hw, obj_coords, device, cfg)
    if sdf_gt is None or sdf_valid is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}
    sdf_valid = sdf_valid & torch.isfinite(sdf_gt)
    valid_px = sdf_valid.sum().to(dtype=zero.dtype)
    if valid_px.item() <= 0:
        return {"loss_sdf": zero, "sdf_valid_px": zero}

    charb_eps = float(cfg.get("loss", {}).get("sdf_charb_eps", 0.0))
    weight_map = _build_sdf_hist_weights(sdf_gt, sdf_valid, cfg)
    if weight_map is None:
        loss_sdf = _sdf_nll_laplace_map(sdf_pred, sdf_logvar, sdf_gt, sdf_valid, charb_eps=charb_eps)
    else:
        loss_sdf = _sdf_nll_laplace_weighted(
            sdf_pred, sdf_logvar, sdf_gt, sdf_valid, weight_map, charb_eps=charb_eps
        )
    return {"loss_sdf": loss_sdf, "sdf_valid_px": valid_px}

class InstanceLatentHead(nn.Module):
    """Small CNN head to aggregate ROI features into instance latent codes."""

    def __init__(self, in_ch: int, z_dim: int, hidden_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.SiLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_ch, z_dim, bias=True),
        )

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        return self.net(roi_feats)


def _get_model_attr(model: nn.Module, name: str):
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, name, None)


def _collect_instance_rois(
    inst_map: torch.Tensor,
    min_pixels: int,
) -> Tuple[List[List[float]], List[Tuple[int, int]], torch.Tensor]:
    B, H, W = inst_map.shape
    max_id = int(inst_map.max().item()) if inst_map.numel() > 0 else 0
    valid_inst = inst_map.new_zeros((B, max_id), dtype=torch.bool)
    rois: List[List[float]] = []
    roi_ids: List[Tuple[int, int]] = []
    if max_id <= 0:
        return rois, roi_ids, valid_inst

    for b in range(B):
        for k in range(max_id):
            inst_id = k + 1
            mask = inst_map[b] == inst_id
            if int(mask.sum().item()) < int(min_pixels):
                continue
            idx = torch.nonzero(mask, as_tuple=False)
            if idx.numel() == 0:
                continue
            y1 = int(idx[:, 0].min().item())
            y2 = int(idx[:, 0].max().item())
            x1 = int(idx[:, 1].min().item())
            x2 = int(idx[:, 1].max().item())
            if x2 <= x1 or y2 <= y1:
                continue
            rois.append([float(b), float(x1), float(y1), float(x2 + 1), float(y2 + 1)])
            roi_ids.append((b, k))
            valid_inst[b, k] = True
    return rois, roi_ids, valid_inst


def _extract_instance_latents(
    latent_map: torch.Tensor,
    inst_gt_hw: torch.Tensor,
    inst_latent_head: InstanceLatentHead,
    roi_size: int,
    min_pixels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latent_map.shape[-2:] != inst_gt_hw.shape[-2:]:
        inst_gt_hw = pan_utils._downsample_label(inst_gt_hw, latent_map.shape[-2:])

    rois, roi_ids, valid_inst = _collect_instance_rois(inst_gt_hw, min_pixels=min_pixels)
    B, C, _, _ = latent_map.shape
    Kmax = valid_inst.shape[1]
    z_dim = inst_latent_head.net[-1].out_features
    z_inst = latent_map.new_zeros((B, Kmax, z_dim))

    if len(rois) == 0:
        return z_inst, valid_inst

    roi_tensor = latent_map.new_tensor(rois)
    roi_feats = roi_align(latent_map, roi_tensor, output_size=(roi_size, roi_size), aligned=True)
    z_roi = inst_latent_head(roi_feats)
    for idx, (b, k) in enumerate(roi_ids):
        z_inst[b, k] = z_roi[idx]
    return z_inst, valid_inst


def _sample_bbox_points(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    u = torch.rand((n_points, 3), device=bbox_min.device, dtype=bbox_min.dtype)
    return bbox_min + u * (bbox_max - bbox_min)


def _compute_implicit_sdf_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    inst_gt_hw: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    cfg: dict,
) -> Dict[str, torch.Tensor]:
    """Compute implicit SDF loss from sampled CAD SDF points."""
    zero = pred["disp_1x"].new_tensor(0.0)
    latent_map = pred.get("latent_map", None)
    if latent_map is None:
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    inst_latent_head = _get_model_attr(model, "inst_latent_head")
    implicit_head = _get_model_attr(model, "implicit_sdf_head")
    if inst_latent_head is None or implicit_head is None:
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    loss_cfg = cfg.get("loss", {}) or {}
    n_points = int(loss_cfg.get("implicit_sdf_points", 512))
    roi_size = int(loss_cfg.get("implicit_sdf_roi_size", 8))
    min_pixels = int(loss_cfg.get("implicit_sdf_min_pixels", 4))
    if n_points <= 0:
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    z_inst, valid_inst = _extract_instance_latents(latent_map, inst_gt_hw, inst_latent_head, roi_size, min_pixels)
    if valid_inst.numel() == 0 or not valid_inst.any():
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    sdf_list = batch.get("SDFs", None)
    sdf_meta_list = batch.get("SDFs_meta", None)
    if sdf_list is None or sdf_meta_list is None:
        return {"loss_sdf_implicit": zero, "valid_inst": zero}
    if not isinstance(sdf_list, list) or not isinstance(sdf_meta_list, list):
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    B, Kmax = valid_inst.shape
    x_obj = latent_map.new_zeros((B, Kmax, n_points, 3))
    sdf_gt = latent_map.new_zeros((B, Kmax, n_points, 1))
    valid_pts = torch.zeros((B, Kmax, n_points), device=device, dtype=torch.bool)
    valid_use = valid_inst.clone()

    for b in range(B):
        sdf_objs = sdf_list[b] if b < len(sdf_list) else None
        meta_objs = sdf_meta_list[b] if b < len(sdf_meta_list) else None
        if sdf_objs is None or meta_objs is None:
            valid_use[b] = False
            continue
        kmax_b = min(len(sdf_objs), len(meta_objs), Kmax)
        if kmax_b < Kmax:
            valid_use[b, kmax_b:] = False
        for k in range(kmax_b):
            if not bool(valid_inst[b, k].item()):
                continue
            bbox_min, bbox_max, ok = _extract_sdf_bounds(meta_objs[k], device, latent_map.dtype)
            if not ok:
                valid_use[b, k] = False
                continue
            pts = _sample_bbox_points(bbox_min, bbox_max, n_points)
            denom = (bbox_max - bbox_min).clamp_min(1e-6)
            coords_norm = (pts - bbox_min) * (2.0 / denom) - 1.0

            sdf_entry = sdf_objs[k]
            if isinstance(sdf_entry, dict) and "sdf" in sdf_entry:
                sdf_entry = sdf_entry["sdf"]
            if sdf_entry is None:
                valid_use[b, k] = False
                continue
            sdf_vol = torch.as_tensor(sdf_entry, device=device, dtype=latent_map.dtype)
            if sdf_vol.dim() == 4 and sdf_vol.size(0) == 1:
                sdf_vol = sdf_vol.squeeze(0)
            if sdf_vol.dim() != 3:
                valid_use[b, k] = False
                continue
            sdf_vol = sdf_vol.unsqueeze(0).unsqueeze(0)
            sdf_vals = _sample_sdf_volume(sdf_vol, coords_norm)

            scale = _extract_sdf_scale(meta_objs[k], device, sdf_vals.dtype)
            sdf_vals = sdf_vals * scale

            finite = torch.isfinite(sdf_vals)
            if not finite.any():
                valid_use[b, k] = False
                continue
            x_obj[b, k] = pts
            sdf_gt[b, k, :, 0] = torch.nan_to_num(sdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
            valid_pts[b, k] = finite

    if not valid_use.any():
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    sdf_pred, sdf_logvar = implicit_head(z_inst, x_obj)
    if sdf_pred is None:
        return {"loss_sdf_implicit": zero, "valid_inst": zero}

    diff = sdf_pred - sdf_gt
    charb_eps = float(loss_cfg.get("sdf_charb_eps", 0.0))
    if charb_eps > 0.0:
        abs_r = torch.sqrt(diff * diff + (charb_eps * charb_eps))
    else:
        abs_r = diff.abs()

    if sdf_logvar is not None:
        nll = abs_r * torch.exp(-0.5 * sdf_logvar) + 0.5 * sdf_logvar
    else:
        nll = abs_r

    mask = (valid_use.unsqueeze(-1) & valid_pts).unsqueeze(-1)
    mask_f = mask.to(nll.dtype)
    loss = (nll * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    valid_inst_count = mask_f.sum()
    return {"loss_sdf_implicit": loss, "valid_inst": valid_inst_count}

def _sample_instance_points_from_point_map(
    point_map: torch.Tensor,
    inst_gt_hw: torch.Tensor,
    conf_map: Optional[torch.Tensor],
    n_points: int,
    conf_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample per-instance points from point_map using instance masks."""
    B, _, H, W = point_map.shape
    max_id = int(inst_gt_hw.max().item()) if inst_gt_hw.numel() > 0 else 0
    pts = point_map.new_zeros((B, max_id, n_points, 3))
    valid = torch.zeros((B, max_id, n_points), device=point_map.device, dtype=torch.bool)
    if max_id <= 0:
        return pts, valid

    conf_mask = None
    if conf_map is not None:
        conf_mask = conf_map.squeeze(1) >= float(conf_thresh)

    for b in range(B):
        for k in range(max_id):
            inst_id = k + 1
            mask = inst_gt_hw[b] == inst_id
            if conf_mask is not None:
                mask = mask & conf_mask[b]
            idx = torch.nonzero(mask, as_tuple=False)
            if idx.numel() == 0:
                continue
            if idx.size(0) > n_points:
                perm = torch.randperm(idx.size(0), device=point_map.device)[:n_points]
                idx = idx[perm]
            sel = point_map[b, :, idx[:, 0], idx[:, 1]].transpose(0, 1)
            pts[b, k, : sel.size(0)] = sel
            valid[b, k, : sel.size(0)] = True
    return pts, valid


def _compute_sdf_consistency_pose_loss(
    pred: Dict[str, torch.Tensor],
    inst_gt_hw: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    cfg: dict,
) -> Dict[str, torch.Tensor]:
    """Compute SDF consistency loss to refine pose with implicit SDF."""
    zero = pred["disp_1x"].new_tensor(0.0)
    implicit_head = _get_model_attr(model, "implicit_sdf_head")
    inst_latent_head = _get_model_attr(model, "inst_latent_head")
    if implicit_head is None or inst_latent_head is None:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    latent_map = pred.get("latent_map", None)
    if latent_map is None:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    loss_cfg = cfg.get("loss", {}) or {}
    n_points = int(loss_cfg.get("sdf_consistency_points", 256))
    roi_size = int(loss_cfg.get("implicit_sdf_roi_size", 8))
    min_pixels = int(loss_cfg.get("implicit_sdf_min_pixels", 4))
    conf_thresh = float(loss_cfg.get("sdf_consistency_conf_thresh", 0.0))
    if n_points <= 0:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    z_inst, valid_inst = _extract_instance_latents(latent_map, inst_gt_hw, inst_latent_head, roi_size, min_pixels)
    if valid_inst.numel() == 0 or not valid_inst.any():
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    R = pred.get("pose_R", None)
    t = pred.get("pose_t", None)
    v_pose = pred.get("pose_valid", None)
    if R is None or t is None or v_pose is None:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    point_map = pred.get("point_map_1x", None)
    if point_map is None:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}
    if point_map.shape[-2:] != inst_gt_hw.shape[-2:]:
        point_map = F.interpolate(point_map, size=inst_gt_hw.shape[-2:], mode="bilinear", align_corners=False)

    conf_map = pred.get("point_map_conf_1x", None)
    if conf_map is not None and conf_map.shape[-2:] != inst_gt_hw.shape[-2:]:
        conf_map = F.interpolate(conf_map, size=inst_gt_hw.shape[-2:], mode="bilinear", align_corners=False)

    p_cam, valid_pts = _sample_instance_points_from_point_map(
        point_map,
        inst_gt_hw,
        conf_map,
        n_points,
        conf_thresh,
    )
    if valid_pts.sum().item() == 0:
        return {"loss_sdf_consistency": zero, "valid_pts": zero}

    p_cam = p_cam.detach()
    R_t = R.transpose(-1, -2)
    p_rel = p_cam - t.unsqueeze(2)
    x_obj = torch.einsum("bkij,bkmj->bkmi", R_t, p_rel)

    sdf_pred, _ = implicit_head(z_inst, x_obj)
    charb_eps = float(loss_cfg.get("sdf_consistency_charb_eps", loss_cfg.get("sdf_charb_eps", 0.0)))
    if charb_eps > 0.0:
        abs_r = torch.sqrt(sdf_pred * sdf_pred + (charb_eps * charb_eps))
    else:
        abs_r = sdf_pred.abs()

    valid_pose = v_pose.to(torch.bool)
    mask = valid_pts & valid_pose.unsqueeze(-1) & valid_inst.unsqueeze(-1)
    mask_f = mask.to(abs_r.dtype)
    loss = (abs_r * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return {"loss_sdf_consistency": loss, "valid_pts": mask_f.sum()}


def _build_extra_loss_fn(cfg: dict):
    loss_cfg = cfg.get("loss", {}) or {}
    use_implicit = bool(loss_cfg.get("use_implicit_sdf", False))
    use_consistency = bool(loss_cfg.get("use_sdf_consistency", False))
    w_sdf_map = float(loss_cfg.get("w_sdf", 0.0))
    w_implicit = float(loss_cfg.get("w_implicit_sdf", 1.0))
    w_consistency = float(loss_cfg.get("w_sdf_consistency", 1.0))
    log_pos_adds = bool(loss_cfg.get("log_pos_adds", True))
    pose_cfg = cfg.get("pose_refine", {}) or {}
    pose_refine_enabled = bool(pose_cfg.get("enabled", False))
    use_r0_if_available = bool(pose_cfg.get("use_r0_if_available", True))
    w_pose_energy = float(loss_cfg.get("w_pose_refine_energy", 0.0))
    w_pose_gt_rot = float(loss_cfg.get("w_pose_gt_rot", 0.0))
    w_pose_gt_trans = float(loss_cfg.get("w_pose_gt_trans", 0.0))

    def _extra_loss(context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pred = context["pred"]
        batch = context["batch"]
        sem_gt = context["sem_gt"]
        inst_gt_hw = context["inst_gt_hw"]
        size_hw = context["size_hw"]
        left_k = context["left_k"]
        model = context["model"]
        device = context["device"]
        writer = context.get("writer", None)
        epoch = int(context.get("epoch", 0))
        it = int(context.get("it", 0))
        global_step = int(context.get("global_step", epoch))
        log_interval = int(context.get("log_interval", 1))
        phase = str(context.get("phase", "train"))
        stereo = context.get("stereo", None)
        inst_gt = context.get("inst_gt", None)
        wks_inst = context.get("wks_inst", None)
        wfg_inst = context.get("wfg_inst", None)

        total = pred["disp_1x"].new_tensor(0.0)
        logs: Dict[str, torch.Tensor] = {}
        has_pose_refine = False

        if w_sdf_map > 0.0:
            sdf_out = _compute_sdf_map_loss(pred, batch, inst_gt_hw, size_hw, device, cfg)
            loss_sdf = sdf_out["loss_sdf"]
            total = total + w_sdf_map * loss_sdf
            logs["L_sdf_map"] = loss_sdf.detach()
            logs["L_sdf_map_valid_px"] = sdf_out["sdf_valid_px"].detach()

        if pose_refine_enabled and wks_inst is not None:
            implicit_head = _get_model_attr(model, "implicit_sdf_head")
            if implicit_head is not None and pred.get("latent_map", None) is not None:
                latent_map = pred["latent_map"].to(torch.float32)
                z_inst = _aggregate_latent_per_instance(latent_map, wks_inst)
                pos_map_pred = pan_utils.pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
                t0, t0_valid = _compute_t0_from_pos_map(
                    pos_map_pred,
                    wks_inst,
                    min_wsum=float(pose_cfg.get("min_wsum", 1e-6)),
                    min_px=int(pose_cfg.get("min_px", 30)),
                )
                R0 = pred.get("pose_R", None) if use_r0_if_available else None
                sym_axes, sym_orders = pan_utils._prepare_symmetry_tensors(batch, device, wks_inst.shape[1])

                def _implicit_query(z_in: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
                    sdf, _ = implicit_head(z_in, x_in)
                    return sdf

                pose_out = pan_utils.pose_refine_implicit_sdf(
                    z_inst=z_inst,
                    point_map_1x=pred["point_map_1x"],
                    point_conf_1x=pred.get("point_map_conf_1x", None),
                    inst_map=inst_gt_hw,
                    wks_inst=wks_inst,
                    pos_map=pos_map_pred,
                    t0=t0,
                    R0=R0,
                    sym_axes=sym_axes,
                    sym_orders=sym_orders,
                    cfg=cfg,
                    implicit_sdf_fn=_implicit_query,
                    is_train=(phase == "train"),
                )

                pred.update(
                    {
                        "pose_R_refined": pose_out["pose_R_refined"],
                        "pose_t_refined": pose_out["pose_t_refined"],
                        "pose_valid": pose_out["pose_valid"],
                        "best_hyp": pose_out["best_hyp"],
                        "E_hyp": pose_out["E_hyp"],
                        "E_hist": pose_out["E_hist"],
                        "inlier_frac_hist": pose_out["inlier_frac_hist"],
                    }
                )
                if "rot_mat" not in pred:
                    pred.update(
                        {
                            "pose_R": pose_out["pose_R_refined"],
                            "pose_t": pose_out["pose_t_refined"],
                            "pose_valid": pose_out["pose_valid"],
                        }
                    )

                pose_valid = pose_out["pose_valid"]
                best_idx = pose_out["best_hyp"].clamp_min(0).unsqueeze(-1)
                E_best = pose_out["E_hyp"].gather(-1, best_idx).squeeze(-1)
                valid_f = pose_valid.to(E_best.dtype)
                if valid_f.sum().item() > 0:
                    loss_pose_energy = (E_best * valid_f).sum() / valid_f.sum().clamp_min(1.0)
                else:
                    loss_pose_energy = E_best.sum() * 0.0
                if w_pose_energy > 0.0:
                    total = total + w_pose_energy * loss_pose_energy
                logs["L_pose_refine_energy"] = loss_pose_energy.detach()
                logs["L_pose_refine_valid"] = valid_f.sum().detach()
                has_pose_refine = True

                if pose_out["inlier_frac_hist"].numel() > 0:
                    inlier_last = pose_out["inlier_frac_hist"][..., -1]
                    inlier_best = inlier_last.gather(-1, best_idx).squeeze(-1)
                    logs["L_pose_refine_inlier_frac"] = (
                        (inlier_best * valid_f).sum() / valid_f.sum().clamp_min(1.0)
                    ).detach()

                need_pose_gt = (w_pose_gt_rot > 0.0) or (w_pose_gt_trans > 0.0) or log_pos_adds
                if need_pose_gt:
                    pos_gt_map, rot_gt_map, _, _ = pan_utils._prepare_pose_targets(
                        batch, sem_gt, size_hw, device
                    )
                    if pos_gt_map is not None and rot_gt_map is not None:
                        r_gt, t_gt, v_gt, _, _ = rot_utils.pose_from_maps_auto(
                            rot_map=rot_gt_map,
                            pos_map=pos_gt_map,
                            Wk_1_4=wks_inst,
                            wfg=wfg_inst,
                            min_px=10,
                            min_wsum=1e-6,
                        )
                        r_pred = pose_out["pose_R_refined"]
                        t_pred = pose_out["pose_t_refined"]
                        sym_axes, sym_orders = pan_utils._prepare_symmetry_tensors(batch, device, wks_inst.shape[1])
                        r_gt_use = r_gt
                        if sym_axes is not None and sym_orders is not None:
                            r_gt_use, _ = rot_utils.align_pose_by_symmetry_min_rotation(
                                r_pred, r_gt, sym_axes, sym_orders
                            )
                        valid_inst = pose_valid & v_gt
                        if valid_inst.any():
                            if w_pose_gt_rot > 0.0 or w_pose_gt_trans > 0.0:
                                r_rel = torch.matmul(r_pred.transpose(-1, -2), r_gt_use)
                                rotvec = rot_utils.so3_log_batch(r_rel.reshape(-1, 3, 3)).reshape_as(r_pred[..., 0])
                                rot_err = rotvec.norm(dim=-1)
                                t_err = (t_pred - t_gt).abs().sum(dim=-1)
                                rot_loss = (rot_err * valid_inst.to(rot_err.dtype)).sum() / valid_inst.sum().clamp_min(
                                    1.0
                                )
                                trans_loss = (t_err * valid_inst.to(t_err.dtype)).sum() / valid_inst.sum().clamp_min(
                                    1.0
                                )
                                if w_pose_gt_rot > 0.0:
                                    total = total + w_pose_gt_rot * rot_loss
                                if w_pose_gt_trans > 0.0:
                                    total = total + w_pose_gt_trans * trans_loss
                                logs["L_pose_refine_rot_rad"] = rot_loss.detach()
                                logs["L_pose_refine_trans_l1"] = trans_loss.detach()

                            if log_pos_adds:
                                model_points, _ = pan_utils._build_model_points_from_batch(batch, device)
                                if model_points is not None:
                                    adds = pan_utils._adds_core_from_Rt_no_norm(
                                        r_pred,
                                        t_pred,
                                        r_gt_use,
                                        t_gt,
                                        model_points,
                                        use_symmetric=True,
                                        valid_mask=valid_inst,
                                    )
                                    add = pan_utils._adds_core_from_Rt_no_norm(
                                        r_pred,
                                        t_pred,
                                        r_gt_use,
                                        t_gt,
                                        model_points,
                                        use_symmetric=False,
                                        valid_mask=valid_inst,
                                    )
                                else:
                                    adds = zero
                                    add = zero
                                logs["L_adds"] = adds.detach()
                                logs["L_add"] = add.detach()
                                logs["L_adds_valid_inst"] = valid_inst.sum().detach()

                if writer is not None and dist_utils.is_main_process():
                    should_log = False
                    if phase == "train":
                        should_log = (log_interval > 0) and (it % log_interval == 0)
                    else:
                        should_log = it == 0
                    if should_log and stereo is not None and inst_gt is not None:
                        objs_in_left, _ = pan_utils._build_objs_in_left_from_batch(batch, device)
                        if objs_in_left is not None:
                            _log_pose_visuals_sdf(
                                writer,
                                global_step,
                                phase,
                                stereo,
                                pred,
                                inst_gt,
                                left_k,
                                batch,
                                objs_in_left[..., :3, 3],
                                objs_in_left[..., :3, :3],
                                n_images=min(4, stereo.size(0)),
                            )

        if use_implicit:
            imp_out = _compute_implicit_sdf_loss(pred, batch, inst_gt_hw, model, device, cfg)
            loss_imp = imp_out["loss_sdf_implicit"]
            total = total + w_implicit * loss_imp
            logs["L_sdf_implicit"] = loss_imp.detach()
            logs["L_sdf_implicit_valid_inst"] = imp_out["valid_inst"].detach()

        if use_consistency:
            cons_out = _compute_sdf_consistency_pose_loss(pred, inst_gt_hw, model, device, cfg)
            loss_cons = cons_out["loss_sdf_consistency"]
            total = total + w_consistency * loss_cons
            logs["L_sdf_consistency"] = loss_cons.detach()
            logs["L_sdf_consistency_valid_pts"] = cons_out["valid_pts"].detach()

        if total.item() == 0.0 and not logs:
            return {}
        return {"loss": total, "logs": logs}

    if log_pos_adds and not pose_refine_enabled:
        raise ValueError("log_pos_adds requires pose_refine.enabled in latent training.")
    if not (w_sdf_map > 0.0 or use_implicit or use_consistency or log_pos_adds or pose_refine_enabled):
        return None
    return _extra_loss

def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Build the panoptic stereo model with latent head."""
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
    latent_cfg = cfg.get("latent", {}) or {}
    head_base_ch = int(seg_cfg.get("head_base_ch", seg_cfg.get("head_c4", 96)))
    if "head_ch_scale" in seg_cfg:
        head_ch_scale = float(seg_cfg.get("head_ch_scale", 1.35))
    elif "head_c4" in seg_cfg and "head_c8" in seg_cfg and float(seg_cfg["head_c4"]) > 0.0:
        head_ch_scale = float(seg_cfg["head_c8"]) / float(seg_cfg["head_c4"])
    else:
        head_ch_scale = 1.35
    head_downsample = int(seg_cfg.get("head_downsample", 4))
    latent_dim = int(latent_cfg.get("latent_dim", seg_cfg.get("latent_dim", 16)))
    latent_l2_norm = bool(latent_cfg.get("latent_l2_norm", True))
    model = PanopticStereoMultiHeadLatent(
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
        latent_dim=latent_dim,
        latent_l2_norm=latent_l2_norm,
        head_base_ch=head_base_ch,
        head_ch_scale=head_ch_scale,
        head_downsample=head_downsample,
        point_map_norm_mean=point_map_norm_mean,
        point_map_norm_std=point_map_norm_std,
    )

    loss_cfg = cfg.get("loss", {}) or {}
    pose_cfg = cfg.get("pose_refine", {}) or {}
    pose_refine_enabled = bool(pose_cfg.get("enabled", False))
    use_implicit = (
        bool(loss_cfg.get("use_implicit_sdf", False))
        or bool(loss_cfg.get("use_sdf_consistency", False))
        or pose_refine_enabled
    )
    if use_implicit:
        implicit_cfg = cfg.get("implicit_sdf", {}) or {}
        z_dim = int(implicit_cfg.get("z_dim", latent_dim))
        inst_hidden = int(implicit_cfg.get("inst_hidden_ch", 64))
        mlp_hidden = int(implicit_cfg.get("hidden_dim", 128))
        n_layers = int(implicit_cfg.get("n_layers", 4))
        n_freqs = int(implicit_cfg.get("n_freqs", 6))
        use_film = bool(implicit_cfg.get("use_film", True))
        use_logvar = bool(implicit_cfg.get("use_logvar", False))
        logvar_min = float(implicit_cfg.get("logvar_min", -8.0))
        logvar_max = float(implicit_cfg.get("logvar_max", 4.0))

        model.inst_latent_head = InstanceLatentHead(in_ch=latent_dim, z_dim=z_dim, hidden_ch=inst_hidden)
        model.implicit_sdf_head = ImplicitSDFHead(
            z_dim=z_dim,
            hidden_dim=mlp_hidden,
            n_layers=n_layers,
            n_freqs=n_freqs,
            use_film=use_film,
            out_logvar=use_logvar,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    return model


def main() -> None:
    """Entry point for panoptic stereo training with latent map head."""
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
    train_loader = _BatchMetaLoader(train_loader)
    val_loader = _BatchMetaLoader(val_loader)

    model = build_model(cfg, n_classes).to(device)
    if dist.is_initialized():
        ddp_find_unused = bool(cfg.get("train", {}).get("ddp_find_unused", True))
        model = DDP(
            model,
            device_ids=[dist_utils.get_rank()],
            output_device=dist_utils.get_rank(),
            find_unused_parameters=ddp_find_unused,
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

    extra_loss_fn = _build_extra_loss_fn(cfg)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = pan_utils.train_one_epoch(
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
            extra_loss_fn=extra_loss_fn,
        )
        val_disp, val_sem = pan_utils.validate(
            model,
            val_loader,
            epoch,
            cfg,
            writer,
            device,
            extra_loss_fn=extra_loss_fn,
        )

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
