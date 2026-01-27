#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo training for panoptic segmentation with latent map head (no rotation outputs).
Based on train_stereo_la_with_instance_seg.py.
"""
import argparse
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.panoptic_stereo import PanopticStereoMultiHeadLatent
from models.stereo_disparity import make_gn
from utils import dist_utils
from utils import rot_utils
from losses import loss_functions
from utils.logging_utils import draw_axes_on_images_bk
from utils.projection import SilhouetteDepthRenderer
import torchvision.utils as vutils

import train_stereo_la_with_instance_seg as base
import train_stereo_la as core

_BASE_COMPUTE_POS = base._compute_pos_losses_and_metrics


def _ensure_sdf_meta_in_batch(batch: dict) -> dict:
    """
    Ensure SDFs_meta exists even if dataset only returns SDFs.
    """
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
    """
    Wrap DataLoader to inject SDF meta when missing.
    """
    def __init__(self, loader):
        self._loader = loader

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for batch in self._loader:
            yield _ensure_sdf_meta_in_batch(batch)


def _normalize_obj_coords_for_sdf(
    meta: dict,
    coords: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Normalize object coordinates when SDFs are stored in normalized space.
    """
    if isinstance(meta, dict) and bool(meta.get("normalize_to_cube", False)):
        tr = meta.get("transform", {}) if isinstance(meta.get("transform", {}), dict) else {}
        center = tr.get("center", None)
        scale = tr.get("scale", None)
        if center is not None:
            c = torch.as_tensor(center, device=device, dtype=dtype).view(1, 3)
            if c.numel() >= 3:
                coords = coords - c[:, :3]
        if scale is not None:
            try:
                s = float(scale)
                if s > 0.0:
                    coords = coords * s
            except Exception:
                pass
    return coords


def _extract_sdf_bounds(
    meta: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract bbox min/max from SDF meta, falling back to [-1, 1].
    """
    bbox_min = None
    bbox_max = None
    if isinstance(meta, dict):
        bbox_min = meta.get("bbox_min", None)
        bbox_max = meta.get("bbox_max", None)
    if bbox_min is None or bbox_max is None:
        bbox_min = torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=dtype)
        bbox_max = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    else:
        bbox_min = torch.as_tensor(bbox_min, device=device, dtype=dtype).view(-1)[:3]
        bbox_max = torch.as_tensor(bbox_max, device=device, dtype=dtype).view(-1)[:3]
    return bbox_min, bbox_max


def _sample_sdf_volume(sdf_vol: torch.Tensor, coords_norm: torch.Tensor) -> torch.Tensor:
    """
    Sample a 3D SDF volume at normalized coordinates in [-1, 1].
    """
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


def _sdf_nll_laplace_map(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    charb_eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute heteroscedastic Laplace NLL for SDF maps.
    """
    r = pred - target
    if charb_eps > 0.0:
        abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
    else:
        abs_r = r.abs()
    nll = abs_r * torch.exp(-0.5 * logvar) + 0.5 * logvar
    valid_f = valid.to(nll.dtype)
    return (nll * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def _get_sdf_pose_net(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Extract SDF pose network from model (handles DDP)."""
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "predict_sdf_pose_delta"):
        return model
    return getattr(model, "sdf_pose_net", None)


def _get_sdf_decoder(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Extract SDF decoder from model (handles DDP)."""
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "sdf_decoder", None)


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


def _build_sdf_pose_pairs(
    pred: dict,
    batch: dict,
    wks_inst: torch.Tensor,
    device: torch.device,
    sdf_decoder: torch.nn.Module,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build paired SDF volumes from predicted SDF and CAD for pose-delta estimation."""
    sdf_list = batch.get("SDFs", None)
    if sdf_list is None or not isinstance(sdf_list, list):
        return None, None, None, None, None
    latent_map = pred.get("latent_map", None)
    if latent_map is None:
        return None, None, None, None, None
    latent_map = latent_map.to(torch.float32)

    objs_in_left, valid_k = base._build_objs_in_left_from_batch(batch, device)
    if objs_in_left is None or valid_k is None:
        return None, None, None, None, None

    B, Kmax = valid_k.shape
    if B == 0 or Kmax == 0:
        return None, None, None, None, None

    latent_k = _aggregate_latent_per_instance(latent_map, wks_inst)
    valid_lat = (wks_inst.sum(dim=(2, 3, 4)) > 0)

    latents: List[torch.Tensor] = []
    cad_vols: List[torch.Tensor] = []
    bks: List[Tuple[int, int]] = []
    ref_shape: Optional[Tuple[int, int, int]] = None
    for b in range(B):
        sdf_objs = sdf_list[b] if b < len(sdf_list) else None
        if sdf_objs is None:
            continue
        kmax_b = min(len(sdf_objs), Kmax)
        for k in range(kmax_b):
            if not valid_k[b, k] or not valid_lat[b, k]:
                continue
            sdf_entry = sdf_objs[k]
            if isinstance(sdf_entry, dict) and "sdf" in sdf_entry:
                sdf_entry = sdf_entry["sdf"]
            sdf_cad = torch.as_tensor(sdf_entry, device=device, dtype=torch.float32)
            if sdf_cad.dim() == 4 and sdf_cad.size(0) == 1:
                sdf_cad = sdf_cad.squeeze(0)
            if sdf_cad.dim() != 3:
                raise ValueError("SDF volume must be 3D (D, H, W)")
            if ref_shape is None:
                ref_shape = (int(sdf_cad.size(0)), int(sdf_cad.size(1)), int(sdf_cad.size(2)))
            elif ref_shape != (int(sdf_cad.size(0)), int(sdf_cad.size(1)), int(sdf_cad.size(2))):
                raise ValueError("SDF volume shapes must be consistent within a batch")

            latents.append(latent_k[b, k])
            cad_vols.append(sdf_cad)
            bks.append((b, k))

    if not latents or ref_shape is None:
        return None, None, None, None, None

    latent_flat = torch.stack(latents, dim=0)
    pred_vol = sdf_decoder(latent_flat, out_shape=ref_shape)
    cad_vol = torch.stack(cad_vols, dim=0).unsqueeze(1)
    sdf_pairs = torch.cat([pred_vol, cad_vol], dim=1)

    valid_sdf = torch.zeros((B, Kmax), dtype=torch.bool, device=device)
    for b, k in bks:
        valid_sdf[b, k] = True

    R_gt = objs_in_left[..., :3, :3]
    t_gt = objs_in_left[..., :3, 3]
    return sdf_pairs, R_gt, t_gt, valid_k, valid_sdf


def _compute_sdf_pose_delta(
    sdf_pose_net: torch.nn.Module,
    sdf_decoder: torch.nn.Module,
    pred: dict,
    batch: dict,
    wks_inst: torch.Tensor,
    wfg_inst: torch.Tensor,
    left_k: torch.Tensor,
    device: torch.device,
    cfg: dict,
) -> dict:
    """Compute SDF-based rotation prediction and pose tensors."""
    zero = pred["disp_1x"].new_tensor(0.0)
    sdf_pairs, R_gt, t_gt, valid_k, valid_sdf = _build_sdf_pose_pairs(
        pred=pred,
        batch=batch,
        wks_inst=wks_inst,
        device=device,
        sdf_decoder=sdf_decoder,
    )
    if sdf_pairs is None or R_gt is None or t_gt is None or valid_k is None or valid_sdf is None:
        dummy = pred["disp_1x"].new_zeros((1, 2, 4, 4, 4), dtype=torch.float32)
        if hasattr(sdf_pose_net, "predict_sdf_pose_delta"):
            rot_dummy, trans_dummy = sdf_pose_net.predict_sdf_pose_delta(dummy)
        else:
            rot_dummy, trans_dummy = sdf_pose_net(dummy)
        loss_keep = (rot_dummy.sum() + trans_dummy.sum()) * 0.0
        latent_map = pred.get("latent_map", None)
        if latent_map is not None:
            lat_dummy = latent_map.mean(dim=(2, 3))[:1]
            sdf_dummy = sdf_decoder(lat_dummy, out_shape=(4, 4, 4))
            loss_keep = loss_keep + sdf_dummy.sum() * 0.0
        return {
            "loss_sdf_pose_rot": loss_keep,
            "loss_sdf_pose_trans": loss_keep,
            "sdf_pose_deg": zero,
            "sdf_pose_trans_l1": zero,
            "sdf_pose_adds": zero,
            "sdf_pose_add": zero,
            "sdf_pose_valid": zero,
            "obj_in_camera_sdf": torch.zeros((0, 0, 4, 4), device=device, dtype=zero.dtype),
            "pose_R": torch.zeros((0, 0, 3, 3), device=device, dtype=zero.dtype),
            "pose_t": torch.zeros((0, 0, 3), device=device, dtype=zero.dtype),
        }

    B, Kmax = valid_k.shape
    sdf_pairs = sdf_pairs.to(device=device, dtype=torch.float32)
    if hasattr(sdf_pose_net, "predict_sdf_pose_delta"):
        rotvec, t_delta = sdf_pose_net.predict_sdf_pose_delta(sdf_pairs)
    else:
        rotvec, t_delta = sdf_pose_net(sdf_pairs)
    R_pred_flat = rot_utils.so3_exp_batch(rotvec)
    R_pred = torch.eye(3, device=device, dtype=pred["disp_1x"].dtype).view(1, 1, 3, 3).expand(B, Kmax, 3, 3).clone()
    t_delta_map = torch.zeros((B, Kmax, 3), device=device, dtype=pred["disp_1x"].dtype)
    bk_idx = torch.nonzero(valid_sdf, as_tuple=False)
    for i, idx in enumerate(bk_idx):
        b = int(idx[0].item())
        k = int(idx[1].item())
        R_pred[b, k] = R_pred_flat[i].to(R_pred.dtype)
        t_delta_map[b, k] = t_delta[i].to(t_delta_map.dtype)

    pos_map_pred = base.pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
    H, W = pos_map_pred.shape[-2:]
    eye_rot = torch.eye(3, device=device, dtype=pred["disp_1x"].dtype).view(1, 3, 3, 1, 1).expand(B, 3, 3, H, W)
    _, t_pred, valid_t, _, _ = rot_utils.pose_from_maps_auto(
        rot_map=eye_rot,
        pos_map=pos_map_pred,
        Wk_1_4=wks_inst,
        wfg=wfg_inst,
        min_px=10,
        min_wsum=1e-6,
    )

    sym_axes, sym_orders = base._prepare_symmetry_tensors(batch, device, Kmax)
    R_gt_use = R_gt
    if sym_axes is not None and sym_orders is not None:
        R_gt_use, _ = rot_utils.align_pose_by_symmetry_min_rotation(
            R_pred,
            R_gt,
            sym_axes,
            sym_orders,
        )

    valid_pose = valid_k & valid_sdf & valid_t
    if valid_pose.any():
        rot_deg = base._rot_geodesic_deg(R_pred[valid_pose], R_gt_use[valid_pose]).mean()
        R_rel = torch.matmul(R_pred.transpose(-1, -2), R_gt_use)
        tr = (R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]).clamp(-1.0 + 1e-6, 3.0 - 1e-6)
        cos_t = (tr - 1.0) * 0.5
        ang = torch.acos(cos_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        loss_rot = (ang * valid_pose.to(ang.dtype)).sum() / valid_pose.sum().clamp_min(1)
        t_delta_gt = (t_gt - t_pred).to(t_delta_map.dtype)
        t_diff = t_delta_map - t_delta_gt
        t_l1 = t_diff.abs().sum(dim=-1)
        loss_trans = (t_l1 * valid_pose.to(t_l1.dtype)).sum() / valid_pose.sum().clamp_min(1)
        trans_l1 = loss_trans
    else:
        rot_deg = zero
        loss_rot = zero
        loss_trans = zero
        trans_l1 = zero

    t_new = t_pred + t_delta_map
    obj_in_camera = rot_utils.compose_T_from_Rt(R_pred, t_new, valid_pose)
    adds = zero
    add = zero
    with torch.no_grad():
        model_points, _ = base._build_model_points_from_batch(batch, device)
        if model_points is not None and valid_pose.any():
            adds = base._adds_core_from_Rt_no_norm(
                R_pred,
                t_new,
                R_gt_use,
                t_gt,
                model_points,
                use_symmetric=True,
                valid_mask=valid_pose,
            )
            add = base._adds_core_from_Rt_no_norm(
                R_pred,
                t_new,
                R_gt_use,
                t_gt,
                model_points,
                use_symmetric=False,
                valid_mask=valid_pose,
            )
    return {
        "loss_sdf_pose_rot": loss_rot,
        "loss_sdf_pose_trans": loss_trans,
        "sdf_pose_deg": rot_deg,
        "sdf_pose_trans_l1": trans_l1,
        "sdf_pose_adds": adds,
        "sdf_pose_add": add,
        "sdf_pose_valid": valid_pose.to(dtype=zero.dtype),
        "obj_in_camera_sdf": obj_in_camera,
        "pose_R": R_pred,
        "pose_t": t_new,
    }


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
    if "pose_R_sdf" not in pred or "pose_t_sdf" not in pred or "pose_valid_sdf" not in pred:
        return

    device = stereo.device
    meshes_flat, valid_k = base._build_meshes_from_batch(batch, device)
    if meshes_flat is None or valid_k.numel() == 0:
        return

    size_hw = pred["pos_mu"].shape[-2:]
    inst_hw = base._downsample_label(inst_gt, size_hw)
    wks, _ = base._build_instance_weight_map(inst_hw, valid_k)

    R_pred = pred["pose_R_sdf"]
    t_pred = pred["pose_t_sdf"]
    v_pred = pred["pose_valid_sdf"].to(dtype=torch.bool)

    image_size = (stereo.shape[-2], stereo.shape[-1])
    origin_in = base._origin_in_image_from_t(pos_gt, left_k[:, 0], image_size)
    valid_render = valid_k & origin_in
    valid_pred = v_pred & valid_render
    valid_gt = valid_render

    T_pred = rot_utils.compose_T_from_Rt(R_pred, t_pred, valid_pred)
    T_gt = rot_utils.compose_T_from_Rt(rot_gt, pos_gt, valid_gt)

    meshes_flat = base._build_meshes_from_batch_filtered(batch, valid_render, device)
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

    overlay_pred = base._overlay_mask_rgb(
        stereo[:n_images, :3], sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45
    )
    overlay_gt = base._overlay_mask_rgb(
        stereo[:n_images, :3], sil_gt[:n_images], color=(1.0, 0.0, 0.0), alpha=0.45
    )
    overlay_both = base._overlay_mask_rgb(overlay_gt, sil_pred[:n_images], color=(0.0, 1.0, 0.0), alpha=0.45)

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


def _build_sdf_targets_scaled(
    batch: dict,
    inst_gt_hw: torch.Tensor,
    obj_coords: torch.Tensor,
    device: torch.device,
    cfg: dict,
):
    """
    Build per-pixel SDF target map in normalized coordinate space.
    """
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
            coords = _normalize_obj_coords_for_sdf(meta_objs[k], coords, device, coords.dtype)
            bbox_min, bbox_max = _extract_sdf_bounds(meta_objs[k], device, coords.dtype)
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
            sdf_vol = torch.as_tensor(sdf_entry, device=device, dtype=obj_coords.dtype)
            if sdf_vol.dim() == 4 and sdf_vol.size(0) == 1:
                sdf_vol = sdf_vol.squeeze(0)
            if sdf_vol.dim() != 3:
                raise ValueError("SDF volume must be 3D (D, H, W)")
            sdf_vol = sdf_vol.unsqueeze(0).unsqueeze(0)
            sdf_vals = _sample_sdf_volume(sdf_vol, coords_norm).to(dtype=sdf_gt.dtype)

            flat_idx = idx[:, 0] * W + idx[:, 1]
            sdf_gt_flat[flat_idx] = sdf_vals
            sdf_valid_flat[flat_idx] = True

    return sdf_gt, sdf_valid


def _compute_pos_losses_and_metrics_with_adds(
    pred: dict,
    batch: dict,
    sem_gt: torch.Tensor,
    size_hw: tuple,
    left_k: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Compute translation losses and ADD/ADD-S metrics using GT rotations.
    """
    pos_out = _BASE_COMPUTE_POS(pred, batch, sem_gt, size_hw, left_k, device)
    adds = pos_out["pos_map_l2"].new_tensor(0.0)
    add = pos_out["pos_map_l2"].new_tensor(0.0)

    with torch.no_grad():
        pos_gt_map, rot_gt_map, _, _ = base._prepare_pose_targets(batch, sem_gt, size_hw, device)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)
        inst_gt_hw = base._downsample_label(inst_gt, size_hw)
        valid_k = base._build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = base._build_instance_weight_map(inst_gt_hw, valid_k)

        pos_map_pred = base.pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
        r_pred, t_pred, v_pred, _, _ = rot_utils.pose_from_maps_auto(
            rot_map=rot_gt_map,
            pos_map=pos_map_pred,
            Wk_1_4=wks_inst,
            wfg=wfg_inst,
            min_px=10,
            min_wsum=1e-6,
        )
        r_gt, t_gt, v_gt, _, _ = rot_utils.pose_from_maps_auto(
            rot_map=rot_gt_map,
            pos_map=pos_gt_map,
            Wk_1_4=wks_inst,
            wfg=wfg_inst,
            min_px=10,
            min_wsum=1e-6,
        )
        valid_inst = v_pred & v_gt & valid_k
        model_points, _ = base._build_model_points_from_batch(batch, device)
        if model_points is not None and valid_inst.any():
            adds = base._adds_core_from_Rt_no_norm(
                r_pred,
                t_pred,
                r_gt,
                t_gt,
                model_points,
                use_symmetric=True,
                valid_mask=valid_inst,
            )
            add = base._adds_core_from_Rt_no_norm(
                r_pred,
                t_pred,
                r_gt,
                t_gt,
                model_points,
                use_symmetric=False,
                valid_mask=valid_inst,
            )

    pos_out["adds"] = adds
    pos_out["add"] = add
    return pos_out


def _build_sdf_hist_weights(
    sdf_gt: torch.Tensor,
    sdf_valid: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    Build per-pixel weights to flatten |SDF| histogram.
    """
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
    """
    Compute weighted heteroscedastic Laplace NLL for SDF maps.
    """
    r = pred - target
    if charb_eps > 0.0:
        abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
    else:
        abs_r = r.abs()
    nll = abs_r * torch.exp(-0.5 * logvar) + 0.5 * logvar
    valid_f = valid.to(nll.dtype)
    w = weight.to(nll.dtype) * valid_f
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def _compute_sdf_loss_weighted(
    pred: dict,
    batch: dict,
    inst_gt_hw: torch.Tensor,
    size_hw: tuple,
    device: torch.device,
    cfg: dict,
) -> dict:
    """Compute SDF loss with histogram-flattening weights."""
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

    pos_map, rot_map = base._prepare_pose_maps(batch, size_hw, device)
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

    sdf_gt, sdf_valid = _build_sdf_targets_scaled(batch, inst_gt_hw, obj_coords, device, cfg)
    if sdf_gt is None or sdf_valid is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}
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


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    cfg: dict,
    writer: SummaryWriter,
    device: torch.device,
    scheduler=None,
    sched_step_when: Optional[str] = None,
) -> float:
    """Train the model for one epoch with SDF losses enabled."""
    model.train()
    total_loss = 0.0
    l1 = torch.nn.L1Loss()
    log_every = cfg["train"]["log_interval"]

    window_sum, window_cnt = core._init_window_meter()
    global_step = epoch * len(loader)

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = base._prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        size_hw = stereo.shape[-2:]
        inst_gt_hw = base._downsample_label(inst_gt, size_hw)
        valid_k_inst = base._build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = base._build_instance_weight_map(inst_gt_hw, valid_k_inst)

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
            mask = (inst_gt > 0).unsqueeze(1)
            loss_disp = core.disparity_nll_laplace_raft_style(
                disp_preds,
                disp_logvar_preds,
                disp_gt,
                mask,
            )
            loss_disp_1x = base.disparity_nll_laplace_scaled(
                [pred["disp_1x"]],
                [pred["disp_log_var_1x"]],
                disp_gt,
                mask,
                downsample=1,
            )

            size_hw = pred["sem_logits"].shape[-2:]
            sem_gt_hw = base._downsample_label(sem_gt, size_hw)
            loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_hw, use_focal=False)
            loss_sem = loss_cls
            has_pos = all(k in pred for k in ("pos_mu", "pos_mu_norm", "pos_logvar_norm"))
            has_rot = all(k in pred for k in ("rot_mat", "rot_logvar_theta"))
            if has_pos and has_rot:
                pose_out = base._compute_pose_losses_and_metrics(
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
                pos_out = base._compute_pos_losses_and_metrics(
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
                adds = pos_out.get("adds", zero)
                add = pos_out.get("add", zero)
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

            aff_tgt, aff_valid = base._build_affinity_targets(inst_gt_hw)
            loss_aff, aff_valid_px = base._affinity_loss(
                pred["aff_logits"],
                aff_tgt,
                aff_valid,
                neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 8.0)),
            )
            loss_emb, emb_pairs = base.embedding_cosface_sampled(
                pred["emb"],
                inst_gt_hw,
                max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
                margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
                scale=32.0,
                min_pixels_per_inst=4,
                detach_proto=True,
                topk_neg=None,
            )
            w_sdf = float(cfg.get("loss", {}).get("w_sdf", 0.0))
            if w_sdf > 0.0:
                inst_gt_hw_sdf = inst_gt_hw
                if inst_gt_hw_sdf.shape[-2:] != size_hw:
                    inst_gt_hw_sdf = base._downsample_label(inst_gt, size_hw)
                sdf_out = _compute_sdf_loss_weighted(
                    pred=pred,
                    batch=batch,
                    inst_gt_hw=inst_gt_hw_sdf,
                    size_hw=size_hw,
                    device=device,
                    cfg=cfg,
                )
                loss_sdf = sdf_out["loss_sdf"]
                sdf_valid_px = sdf_out["sdf_valid_px"]
            else:
                zero = loss_disp.new_tensor(0.0)
                loss_sdf = zero
                sdf_valid_px = zero
            w_sdf_pose = float(cfg.get("loss", {}).get("w_sdf_pose", 0.0))
            w_sdf_pose_rot = float(cfg.get("loss", {}).get("w_sdf_pose_rot", w_sdf_pose))
            w_sdf_pose_trans = float(cfg.get("loss", {}).get("w_sdf_pose_trans", w_sdf_pose))
            use_sdf_pose = max(w_sdf_pose, w_sdf_pose_rot, w_sdf_pose_trans) > 0.0
            sdf_pose_net = _get_sdf_pose_net(model)
            sdf_decoder = _get_sdf_decoder(model)
            if use_sdf_pose and sdf_pose_net is not None and sdf_decoder is not None:
                with torch.amp.autocast("cuda", enabled=False):
                    sdf_pose_out = _compute_sdf_pose_delta(
                        sdf_pose_net=sdf_pose_net,
                        sdf_decoder=sdf_decoder,
                        pred=pred,
                        batch=batch,
                        wks_inst=wks_inst,
                        wfg_inst=wfg_inst,
                        left_k=left_k,
                        device=device,
                        cfg=cfg,
                    )
                loss_sdf_pose = sdf_pose_out["loss_sdf_pose_rot"]
                loss_sdf_pose_trans = sdf_pose_out["loss_sdf_pose_trans"]
                sdf_pose_deg = sdf_pose_out["sdf_pose_deg"]
                sdf_pose_trans_l1 = sdf_pose_out["sdf_pose_trans_l1"]
                sdf_pose_adds = sdf_pose_out["sdf_pose_adds"]
                sdf_pose_add = sdf_pose_out["sdf_pose_add"]
                pred["obj_in_camera_sdf"] = sdf_pose_out["obj_in_camera_sdf"]
                pred["pose_R_sdf"] = sdf_pose_out["pose_R"]
                pred["pose_t_sdf"] = sdf_pose_out["pose_t"]
                pred["pose_valid_sdf"] = sdf_pose_out["sdf_pose_valid"]
            else:
                loss_sdf_pose = loss_disp.new_tensor(0.0)
                loss_sdf_pose_trans = loss_disp.new_tensor(0.0)
                sdf_pose_deg = loss_disp.new_tensor(0.0)
                sdf_pose_trans_l1 = loss_disp.new_tensor(0.0)
                sdf_pose_adds = loss_disp.new_tensor(0.0)
                sdf_pose_add = loss_disp.new_tensor(0.0)
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
                "L_sdf": loss_sdf.detach(),
                "L_sdf_pose_rot": loss_sdf_pose.detach(),
                "L_sdf_pose_deg": sdf_pose_deg.detach(),
                "L_sdf_pose_trans": loss_sdf_pose_trans.detach(),
                "L_sdf_pose_trans_l1": sdf_pose_trans_l1.detach(),
                "L_sdf_pose_adds": sdf_pose_adds.detach(),
                "L_sdf_pose_add": sdf_pose_add.detach(),
                "L_disp_1x": loss_disp_1x.detach(),
                "L_aff_valid_px": aff_valid_px.detach(),
                "L_emb_pairs": emb_pairs.detach(),
                "L_sdf_valid_px": sdf_valid_px.detach(),
            }
            logs["L_pose"] = (loss_pos + loss_rot).detach()
            logs["L_sdf_pose"] = (loss_sdf_pose + loss_sdf_pose_trans).detach()

            loss = cfg["loss"]["w_disp"] * loss_disp
            loss = loss + cfg.get("loss", {}).get("w_disp_1x", 1.0) * loss_disp_1x
            loss = loss + cfg.get("loss", {}).get("w_sem", 1.0) * loss_sem
            loss = loss + cfg.get("loss", {}).get("w_cls", 1.0) * loss_cls
            loss = loss + cfg.get("loss", {}).get("w_pos", 1.0) * loss_pos
            loss = loss + cfg.get("loss", {}).get("w_rot", 1.0) * loss_rot
            loss = loss + cfg.get("loss", {}).get("w_aff", 1.0) * loss_aff
            loss = loss + cfg.get("loss", {}).get("w_emb", 1.0) * loss_emb
            loss = loss + w_sdf * loss_sdf
            loss = loss + w_sdf_pose_rot * loss_sdf_pose
            loss = loss + w_sdf_pose_trans * loss_sdf_pose_trans

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
        window_cnt = core._update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs)

        global_step = epoch * len(loader) + it
        if writer is not None and dist_utils.is_main_process() and (global_step % log_every == 0):
            core._flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train")
            with torch.no_grad():
                base._log_disp_and_depth(
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
                base._log_segmentation_visuals(
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
                    pos_gt_map, rot_gt_map, _, objs_in_left = base._prepare_pose_targets(
                        batch,
                        sem_gt,
                        pred["sem_logits"].shape[-2:],
                        device,
                    )
                    base._log_pose_visuals(
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
                    _log_pose_visuals_sdf(
                        writer,
                        global_step,
                        "train",
                        stereo,
                        pred,
                        inst_gt,
                        left_k,
                        batch,
                        objs_in_left[..., :3, 3],
                        objs_in_left[..., :3, :3],
                        n_images=min(4, stereo.size(0)),
                    )
            window_cnt = core._reset_window_meter(window_sum, window_cnt)

    return total_loss.item() / max(1, len(loader))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    epoch: int,
    cfg: dict,
    writer: SummaryWriter,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model with SDF losses enabled."""
    model.eval()
    meters = core.DictMeters()
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
    meters.add_avg("L_sdf")
    meters.add_avg("L_pose")
    meters.add_avg("L_sdf_pose")
    meters.add_avg("L_sdf_pose_rot")
    meters.add_avg("L_sdf_pose_deg")
    meters.add_avg("L_sdf_pose_trans")
    meters.add_avg("L_sdf_pose_trans_l1")
    meters.add_avg("L_sdf_pose_adds")
    meters.add_avg("L_sdf_pose_add")
    meters.add_sc("depth_acc_1mm")
    meters.add_sc("depth_acc_2mm")
    meters.add_sc("depth_acc_4mm")

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = base._prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        size_hw = stereo.shape[-2:]
        inst_gt_hw = base._downsample_label(inst_gt, size_hw)
        valid_k_inst = base._build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = base._build_instance_weight_map(inst_gt_hw, valid_k_inst)

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
        mask = (inst_gt > 0).unsqueeze(1)
        loss_disp = core.disparity_nll_laplace_raft_style(
            disp_preds,
            disp_logvar_preds,
            disp_gt,
            mask,
        )
        loss_disp_1x = base.disparity_nll_laplace_scaled(
            [pred["disp_1x"]],
            [pred["disp_log_var_1x"]],
            disp_gt,
            mask,
            downsample=1,
        )

        size_hw = pred["sem_logits"].shape[-2:]
        sem_gt_hw = base._downsample_label(sem_gt, size_hw)
        loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_hw, use_focal=False)
        loss_sem = loss_cls
        has_pos = all(k in pred for k in ("pos_mu", "pos_mu_norm", "pos_logvar_norm"))
        has_rot = all(k in pred for k in ("rot_mat", "rot_logvar_theta"))
        if has_pos and has_rot:
            pose_out = base._compute_pose_losses_and_metrics(
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
            pos_out = base._compute_pos_losses_and_metrics(
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
            adds = pos_out.get("adds", zero)
            add = pos_out.get("add", zero)
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

        aff_tgt, aff_valid = base._build_affinity_targets(inst_gt_hw)
        loss_aff, _ = base._affinity_loss(
            pred["aff_logits"],
            aff_tgt,
            aff_valid,
            neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 4.0)),
        )
        loss_emb, emb_pairs = base.embedding_cosface_sampled(
            pred["emb"],
            inst_gt_hw,
            max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
            margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
            scale=32.0,
            min_pixels_per_inst=4,
            detach_proto=True,
            topk_neg=None,
        )
        w_sdf = float(cfg.get("loss", {}).get("w_sdf", 0.0))
        if w_sdf > 0.0:
            inst_gt_hw_sdf = inst_gt_hw
            if inst_gt_hw_sdf.shape[-2:] != size_hw:
                inst_gt_hw_sdf = base._downsample_label(inst_gt, size_hw)
            sdf_out = _compute_sdf_loss_weighted(
                pred=pred,
                batch=batch,
                inst_gt_hw=inst_gt_hw_sdf,
                size_hw=size_hw,
                device=device,
                cfg=cfg,
            )
            loss_sdf = sdf_out["loss_sdf"]
        else:
            loss_sdf = loss_disp.new_tensor(0.0)
        w_sdf_pose = float(cfg.get("loss", {}).get("w_sdf_pose", 0.0))
        w_sdf_pose_rot = float(cfg.get("loss", {}).get("w_sdf_pose_rot", w_sdf_pose))
        w_sdf_pose_trans = float(cfg.get("loss", {}).get("w_sdf_pose_trans", w_sdf_pose))
        use_sdf_pose = max(w_sdf_pose, w_sdf_pose_rot, w_sdf_pose_trans) > 0.0
        sdf_pose_net = _get_sdf_pose_net(model)
        sdf_decoder = _get_sdf_decoder(model)
        if use_sdf_pose and sdf_pose_net is not None and sdf_decoder is not None:
            with torch.amp.autocast("cuda", enabled=False):
                sdf_pose_out = _compute_sdf_pose_delta(
                    sdf_pose_net=sdf_pose_net,
                    sdf_decoder=sdf_decoder,
                    pred=pred,
                    batch=batch,
                    wks_inst=wks_inst,
                    wfg_inst=wfg_inst,
                    left_k=left_k,
                    device=device,
                    cfg=cfg,
                )
            loss_sdf_pose = sdf_pose_out["loss_sdf_pose_rot"]
            loss_sdf_pose_trans = sdf_pose_out["loss_sdf_pose_trans"]
            sdf_pose_deg = sdf_pose_out["sdf_pose_deg"]
            sdf_pose_trans_l1 = sdf_pose_out["sdf_pose_trans_l1"]
            sdf_pose_adds = sdf_pose_out["sdf_pose_adds"]
            sdf_pose_add = sdf_pose_out["sdf_pose_add"]
            pred["obj_in_camera_sdf"] = sdf_pose_out["obj_in_camera_sdf"]
            pred["pose_R_sdf"] = sdf_pose_out["pose_R"]
            pred["pose_t_sdf"] = sdf_pose_out["pose_t"]
            pred["pose_valid_sdf"] = sdf_pose_out["sdf_pose_valid"]
        else:
            loss_sdf_pose = loss_disp.new_tensor(0.0)
            loss_sdf_pose_trans = loss_disp.new_tensor(0.0)
            sdf_pose_deg = loss_disp.new_tensor(0.0)
            sdf_pose_trans_l1 = loss_disp.new_tensor(0.0)
            sdf_pose_adds = loss_disp.new_tensor(0.0)
            sdf_pose_add = loss_disp.new_tensor(0.0)

        loss = cfg["loss"]["w_disp"] * loss_disp
        loss = loss + cfg.get("loss", {}).get("w_disp_1x", 1.0) * loss_disp_1x
        loss = loss + cfg.get("loss", {}).get("w_sem", 1.0) * loss_sem
        loss = loss + cfg.get("loss", {}).get("w_cls", 1.0) * loss_cls
        loss = loss + cfg.get("loss", {}).get("w_pos", 1.0) * loss_pos
        loss = loss + cfg.get("loss", {}).get("w_rot", 1.0) * loss_rot
        loss = loss + cfg.get("loss", {}).get("w_aff", 1.0) * loss_aff
        loss = loss + cfg.get("loss", {}).get("w_emb", 1.0) * loss_emb
        loss = loss + w_sdf * loss_sdf
        loss = loss + w_sdf_pose_rot * loss_sdf_pose
        loss = loss + w_sdf_pose_trans * loss_sdf_pose_trans

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
        meters.update_avg("L_sdf", float(loss_sdf.item()), n=stereo.size(0))
        meters.update_avg("L_pose", float((loss_pos + loss_rot).item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose", float((loss_sdf_pose + loss_sdf_pose_trans).item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_rot", float(loss_sdf_pose.item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_deg", float(sdf_pose_deg.item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_trans", float(loss_sdf_pose_trans.item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_trans_l1", float(sdf_pose_trans_l1.item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_adds", float(sdf_pose_adds.item()), n=stereo.size(0))
        meters.update_avg("L_sdf_pose_add", float(sdf_pose_add.item()), n=stereo.size(0))

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
            base._log_disp_and_depth(
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
            base._log_segmentation_visuals(
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
                pos_gt_map, rot_gt_map, _, objs_in_left = base._prepare_pose_targets(
                    batch,
                    sem_gt,
                    pred["sem_logits"].shape[-2:],
                    device,
                )
                base._log_pose_visuals(
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
                _log_pose_visuals_sdf(
                    writer,
                    epoch,
                    "val",
                    stereo,
                    pred,
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
        core.log_meters_to_tb(writer, meters, epoch, prefix="val")
    return loss_disp_avg, loss_sem_avg


# Override base helpers for latent training.
base._compute_pos_losses_and_metrics = _compute_pos_losses_and_metrics_with_adds


def build_model(cfg: dict, num_classes: int) -> torch.nn.Module:
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
    sdf_pose_cfg = cfg.get("sdf_pose", {}) or {}

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
        sdf_pose_cfg=sdf_pose_cfg,
    )
    return model


def main() -> None:
    """Entry point for panoptic stereo training with latent map head."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config TOML", default="configs/small_config_panoptic.toml")
    parser.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    args = parser.parse_args()

    cfg = base.load_toml(args.config)
    cfg.setdefault("train", {})
    cfg["train"]["amp"] = True
    base.set_global_seed(42)
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
        base.write_toml(cfg_copy, eff_cfg_path)
    dist_utils.barrier()

    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if dist_utils.is_main_process() else None

    device = torch.device("cuda", dist_utils.get_rank()) if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader, train_sampler, val_sampler, n_classes = base.make_dataloaders(
        cfg, distributed=dist.is_initialized()
    )
    cfg.setdefault("data", {})
    cfg["data"]["n_classes"] = n_classes
    train_loader = _BatchMetaLoader(train_loader)
    val_loader = _BatchMetaLoader(val_loader)

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
    scheduler, sched_step_when = core.build_lr_scheduler(cfg, optimizer, steps_per_epoch, total_steps)

    model_path_cfg = str(cfg.get("train", {}).get("model_path", "") or "").strip()
    load_mode_cfg = str(cfg.get("train", {}).get("load_mode", "auto")).lower()
    strict_cfg = bool(cfg.get("train", {}).get("strict", False))

    if os.path.exists(model_path_cfg):
        ckpt = torch.load(model_path_cfg, map_location=device)
        if load_mode_cfg == "resume":
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = base._load_model_state(model, state, strict=strict_cfg)
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
            missing, unexpected = base._load_model_state(model, state, strict=strict_cfg)
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
