#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for SSCFlow2 on test/val split.

This script mirrors train_stereo_la.py data preparation and runs pose metrics:
ADD-S, ADD, per-instance position error (mm), and rotation error (deg).
sample
python eval_stereo_la.py --ckpt F:/repos/training_obj_observer/outputs/run_debug/checkpoint_119.pth --pose_update --save_images
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.utils as vutils

import train_stereo_la as tr
from la_loader.synthetic_data_loader import LASyntheticDataset3PerObj
from la_loader.real_data_loader import LARealDataset4PerObj
from utils import dist_utils, rot_utils
from utils.projection import SilhouetteDepthRenderer
from utils.logging_utils import visualize_mono_torch
from models.multi_head import pos_mu_to_pointmap


def _select_dataset_cfg(cfg: dict, split: str) -> Tuple[dict, str]:
    data_cfg = cfg["data"]
    key = f"{split}_datasets"
    datasets = data_cfg.get(key)
    if datasets:
        return datasets[0], split
    if split != "val":
        fallback = data_cfg.get("val_datasets")
        if fallback:
            return fallback[0], "val"
    raise KeyError(f"no dataset config for split={split}")


class _DatasetWithIndex(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if isinstance(sample, dict):
            sample["dataset_index"] = idx
        return sample

    def __getattr__(self, name: str):
        if name == "base":
            raise AttributeError("base is not initialized")
        base = object.__getattribute__(self, "base")
        return getattr(base, name)


def make_eval_dataloader(cfg: dict, split: str, distributed: bool, batch_size: Optional[int]):
    ds_cfg, split_used = _select_dataset_cfg(cfg, split)
    use_camera_list = ds_cfg.get("use_camera_list", ["ZED2", "D415", "ZEDmini"])
    target_scene_list = ds_cfg.get("target_scene_list", [])
    filtering_trans = tr._build_filtering_transforms(for_train=False)
    spatial_trans = tr._build_spatial_transforms()

    out_list = ("stereo", "depth", "disparity", "instance_seg")
    if ds_cfg['name'] ==  "LASyntheticDataset3PerObj": 
        dataset = LASyntheticDataset3PerObj(
            out_list=out_list,
            with_data_path=True,
            use_camera_list=use_camera_list,
            with_camera_params=True,
            out_size_wh=(cfg["data"]["width"], cfg["data"]["height"]),
            with_depro_matrix=True,
            target_scene_list=target_scene_list,
            spatial_transform=spatial_trans,
            filtering_transform=filtering_trans,
        )
    else:
        dataset = LARealDataset4PerObj(
            out_list=out_list,
            with_data_path=True,
            use_camera_list=use_camera_list,
            with_camera_params=True,
            out_size_wh=(cfg["data"]["width"], cfg["data"]["height"]),
            with_depro_matrix=True,
            target_scene_list=target_scene_list,
            spatial_transform=spatial_trans,
            filtering_transform=filtering_trans,
        )
    dataset = _DatasetWithIndex(dataset)  
    class_table = dataset.class_dict
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    g = torch.Generator()
    g.manual_seed(42)

    bs = batch_size if batch_size is not None else cfg["train"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=tr.seed_worker,
        generator=g,
        shuffle=False,
        drop_last=False,
        collate_fn=tr.collate,
    )
    return loader, sampler, class_table, split_used


def _normalize_rgb(img: torch.Tensor) -> torch.Tensor:
    vmin = img.amin()
    vmax = img.amax()
    if (vmax - vmin) < 1e-6:
        return torch.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def _safe_key(path_str: str) -> str:
    return (
        path_str.replace(":", "")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _geodesic_angle_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_rel = R_pred.transpose(-2, -1) @ R_gt
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0, 1.0)
    theta = torch.acos(cos)
    return theta * (180.0 / math.pi)


def _compute_adds_add_raw(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    model_points: torch.Tensor,
    max_points: int,
) -> Tuple[float, float]:
    pts = model_points
    if pts.numel() == 0:
        return float("nan"), float("nan")
    if pts.dim() != 2 or pts.size(-1) != 3:
        raise ValueError("model_points must be (N,3)")
    if pts.size(0) > max_points:
        pts = pts[:max_points]

    R_pred = R_pred.to(torch.float32)
    t_pred = t_pred.to(torch.float32)
    R_gt = R_gt.to(torch.float32)
    t_gt = t_gt.to(torch.float32)
    pts = pts.to(torch.float32)

    Xp = (R_pred @ pts.t()).t() + t_pred.view(1, 3)
    Xg = (R_gt @ pts.t()).t() + t_gt.view(1, 3)

    add = (Xp - Xg).norm(dim=-1).mean().item()
    D = torch.cdist(Xp, Xg, p=2)
    d1 = D.min(dim=1).values.mean()
    d2 = D.min(dim=0).values.mean()
    adds = torch.minimum(d1, d2).item()
    return adds, add


def _normalize_add_metrics(
    adds_mm: float, add_mm: float, diameter_mm: Optional[float]
) -> Tuple[float, float]:
    if diameter_mm is None or diameter_mm <= 0.0:
        return float("nan"), float("nan")
    scale = diameter_mm / 100.0
    if scale <= 0.0:
        return float("nan"), float("nan")
    return adds_mm / scale, add_mm / scale


def _select_first_valid_idx(valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = valid_mask.size(0)
    has_valid = valid_mask.any(dim=1)
    idx = torch.zeros(B, dtype=torch.long, device=valid_mask.device)
    if has_valid.any():
        idx[has_valid] = valid_mask[has_valid].to(torch.int64).argmax(dim=1)
    return idx, has_valid


def _select_top1_pred_idx(inst: Dict[str, torch.Tensor]) -> torch.Tensor:
    score = inst.get("score", None)
    if score is None:
        score = inst.get("area_px", None)
    if score is None:
        score = inst["valid"].to(torch.float32)
    return score.argmax(dim=1)


def _build_semantic_weight_map(
    cls_logits: Optional[torch.Tensor],
    mask_logits: Optional[torch.Tensor],
    mask_thresh: float = 0.5,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if cls_logits is None and mask_logits is None:
        return None, None, None, None

    if cls_logits is not None:
        cls_prob = F.softmax(cls_logits.to(torch.float32), dim=1)
        B, C, H, W = cls_prob.shape
    else:
        B, _, H, W = mask_logits.shape
        cls_prob = None

    mask = None
    if mask_logits is not None:
        mask = (torch.sigmoid(mask_logits) > mask_thresh).squeeze(1)

    cls_ids = torch.full(
        (B,),
        -1,
        device=mask_logits.device if mask_logits is not None else cls_logits.device,
        dtype=torch.long,
    )
    Wk = torch.zeros((B, 1, 1, H, W), device=cls_ids.device, dtype=torch.float32)
    valid = torch.zeros((B,), device=cls_ids.device, dtype=torch.bool)
    pred_mask = torch.zeros((B, 1, H, W), device=cls_ids.device, dtype=torch.float32)

    if cls_prob is None:
        for b in range(B):
            if mask is None:
                w = torch.ones((H, W), device=cls_ids.device, dtype=torch.float32)
            else:
                w = mask[b].to(torch.float32)
            wsum = w.sum()
            if wsum > 0:
                Wk[b, 0, 0] = w / wsum
                pred_mask[b, 0] = w
                valid[b] = True
        return Wk, cls_ids, valid, pred_mask

    cls_map = cls_prob.argmax(dim=1)
    for b in range(B):
        if mask is not None and mask[b].any():
            scores = cls_prob[b, :, mask[b]].mean(dim=1)
        else:
            scores = cls_prob[b].mean(dim=(1, 2))
        cls_id = int(scores.argmax().item())
        cls_ids[b] = cls_id

        region_cls = (cls_map[b] == cls_id)
        if mask is not None and mask[b].any():
            pred_mask[b, 0] = mask[b].to(torch.float32)
        else:
            pred_mask[b, 0] = region_cls.to(torch.float32)
        region = region_cls
        if mask is not None and mask[b].any():
            region = region & mask[b]
        w = region.to(torch.float32)
        wsum = w.sum()
        if wsum > 0:
            Wk[b, 0, 0] = w / wsum
            valid[b] = True
    return Wk, cls_ids, valid, pred_mask


def _infer_gt_class_ids(
    class_map: torch.Tensor,
    weight_map_inst: torch.Tensor,
    valid_k: torch.Tensor,
) -> torch.Tensor:
    if class_map.dim() == 4:
        class_map = class_map.squeeze(1)
    B, K = valid_k.shape
    H, W = class_map.shape[-2:]
    num_classes = int(class_map.max().item()) + 1 if class_map.numel() > 0 else 0
    class_ids = torch.full((B, K), -1, device=class_map.device, dtype=torch.long)
    if num_classes <= 0:
        return class_ids

    cls_flat = class_map.reshape(B, H * W)
    w_map = weight_map_inst.squeeze(2)
    for b in range(B):
        for k in range(K):
            if not bool(valid_k[b, k]):
                continue
            w = w_map[b, k].reshape(-1)
            if w.sum() <= 0:
                continue
            cls = cls_flat[b]
            valid = (cls >= 0) & (cls < num_classes) & (w > 0)
            if not valid.any():
                continue
            bins = torch.zeros(num_classes, device=class_map.device, dtype=torch.float32)
            bins.scatter_add_(0, cls[valid].long(), w[valid].to(torch.float32))
            class_ids[b, k] = int(torch.argmax(bins).item())
    return class_ids


def _merge_results(results_list: List[Dict[str, Dict[str, dict]]]):
    merged: Dict[str, Dict[str, dict]] = {}
    for res in results_list:
        for cls_id, idx_dict in res.items():
            dst = merged.setdefault(cls_id, {})
            for idx_key, item in idx_dict.items():
                if idx_key not in dst:
                    dst[idx_key] = item
                else:
                    prev = dst[idx_key]
                    if isinstance(prev, list):
                        prev.append(item)
                    else:
                        dst[idx_key] = [prev, item]
    return merged


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    save_images: bool,
    max_add_points: int,
    iters: int,
    amp_on: bool,
    limit_batches: int,
    use_gt_peaks: bool,
    enable_pose_update: bool,
) -> Dict[str, Dict[str, dict]]:
    model.eval()
    renderer = SilhouetteDepthRenderer().to(device)
    results: Dict[str, Dict[str, dict]] = {}

    for it, batch in enumerate(loader):
        if limit_batches > 0 and it >= limit_batches:
            break

        stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k = tr._prepare_stereo_and_cam(
            batch, device
        )
        gt_iter0, instance_weight_map, mask = tr._prepare_iter0_targets(batch, cfg, device)
        gt_iter0["K_left_1x"] = left_k[:, 0]
        meshes = batch["meshes"].to(device)
        valid_k = batch["valid_k"].to(device)

        gt_pos_map = gt_iter0["pos_1_4"]
        if gt_pos_map.size(1) > 3:
            gt_pos_map = gt_pos_map[:, 0:3]

        gt_center_1_4 = gt_iter0["weight_map"] if use_gt_peaks else None
        gt_Wk_1_4 = instance_weight_map if use_gt_peaks else None
        gt_mask_1_4 = gt_iter0["mask_1_4"] if use_gt_peaks else None

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=amp_on):
                pred = model(
                    stereo,
                    input_mask,
                    k_pair,
                    baseline,
                    meshes,
                    use_gt_peaks=use_gt_peaks,
                    gt_center_1_4=gt_center_1_4,
                    gt_Wk_1_4=gt_Wk_1_4,
                    gt_mask_1_4=gt_mask_1_4,
                    with_shape_constraint=True,
                    enable_pose_update=enable_pose_update,
                    iters=iters,
                )

        pred_inst = pred["instances"]
        pred_idx_peak = _select_top1_pred_idx(pred_inst)

        sem_Wk, sem_cls_ids, sem_valid, sem_mask_14 = _build_semantic_weight_map(
            pred.get("cls_logits", None),
            pred.get("mask_logits", None),
        )
        if sem_Wk is not None:
            pred_pos_map = pos_mu_to_pointmap(pred["pos_mu"], k_pair[:, 0], downsample=4)
            pred_rot_map = pred["rot_mat"]
            wfg = (sem_Wk.squeeze(2) > 0).to(pred_pos_map.dtype)
            pred_R_sem, pred_t_sem, pred_valid_sem, _, _ = rot_utils.pose_from_maps_auto(
                rot_map=pred_rot_map,
                pos_map=pred_pos_map,
                Wk_1_4=sem_Wk,
                wfg=wfg,
                peaks_yx=None,
                min_px=10,
                min_wsum=1e-6,
                tau_peak=0.0,
            )
        else:
            b_ix = torch.arange(valid_k.size(0), device=valid_k.device)
            pred_R_sem = pred_inst["R"][b_ix, pred_idx_peak].unsqueeze(1)
            pred_t_sem = pred_inst["t"][b_ix, pred_idx_peak].unsqueeze(1)
            pred_valid_sem = pred_inst["valid"][b_ix, pred_idx_peak].unsqueeze(1).to(torch.bool)

        gt_R, gt_t, gt_valid_pose, _, _ = rot_utils.pose_from_maps_auto(
            rot_map=gt_iter0["R_1_4"],
            pos_map=gt_pos_map,
            Wk_1_4=instance_weight_map,
            wfg=gt_iter0["mask_1_4"],
            peaks_yx=gt_iter0["peaks_yx"],
            min_px=10,
            min_wsum=1e-6,
            tau_peak=0.0,
        )

        if valid_k.size(1) != 1:
            raise ValueError("single-object evaluation expects K=1 in valid_k")

        gt_valid = valid_k & gt_valid_pose
        gt_idx, gt_has = _select_first_valid_idx(gt_valid)
        pred_idx = _select_top1_pred_idx(pred_inst)
        b_ix = torch.arange(valid_k.size(0), device=valid_k.device)

        pred_R_sel = pred_R_sem
        pred_t_sel = pred_t_sem
        pred_valid_sel = pred_valid_sem
        sym_axes = batch.get("symmetry_axes", None)
        sym_orders = batch.get("symmetry_orders", None)
        if sym_axes is not None and sym_orders is not None:
            sym_axes = sym_axes.to(device, non_blocking=True)
            sym_orders = sym_orders.to(device, non_blocking=True)
            sym_axes_sel = sym_axes[b_ix, gt_idx].unsqueeze(1)
            sym_orders_sel = sym_orders[b_ix, gt_idx].unsqueeze(1)
        gt_R_sel = gt_R[b_ix, gt_idx].unsqueeze(1)
        gt_t_sel = gt_t[b_ix, gt_idx].unsqueeze(1)
        gt_valid_sel = gt_valid[b_ix, gt_idx].unsqueeze(1)
        if sym_axes is not None and sym_orders is not None:
            with torch.no_grad():
                gt_R_sel, _ = rot_utils.align_pose_by_symmetry_min_rotation(
                    pred_R_sel,
                    gt_R_sel,
                    sym_axes_sel,
                    sym_orders_sel,
                )
        eval_mask = (gt_has & pred_valid_sel.squeeze(1))

        class_ids = _infer_gt_class_ids(gt_iter0["cls_target"], instance_weight_map, valid_k)
        model_points = gt_iter0["model_points"]
        diameters = gt_iter0.get("diameters", None)

        pred_T = rot_utils.compose_T_from_Rt(pred_R_sel, pred_t_sel, pred_valid_sel)
        gt_T = rot_utils.compose_T_from_Rt(gt_R_sel, gt_t_sel, gt_valid_sel)

        img_left = stereo[:, :3]
        H, W = img_left.shape[-2:]
        image_size = (H, W)

        sem_mask_1x = None
        if save_images and sem_mask_14 is not None:
            sem_mask_1x = F.interpolate(
                sem_mask_14.to(torch.float32),
                size=image_size,
                mode="nearest",
            ).clamp(0.0, 1.0)

        pred_r = None
        gt_r = None
        disp_pred_vis = None
        disp_gt_vis = None
        input_seg_vis = None
        if save_images:
            pred_r = renderer(
                meshes_flat=meshes,
                T_cam_obj=pred_T,
                K_left=k_pair[:, 0],
                valid_k=valid_k,
                image_size=image_size,
            )
            gt_r = renderer(
                meshes_flat=meshes,
                T_cam_obj=gt_T,
                K_left=k_pair[:, 0],
                valid_k=valid_k,
                image_size=image_size,
            )
            disp_pred_1x = F.interpolate(
                pred["disp_preds"][-1],
                size=disp_gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ) * 4.0
            disp_pred_u8, disp_gt_u8 = visualize_mono_torch(
                disp_pred_1x,
                mask,
                disp_gt,
                mask,
            )
            disp_pred_vis = disp_pred_u8.to(torch.float32) / 255.0
            disp_gt_vis = disp_gt_u8.to(torch.float32) / 255.0
            input_seg_vis = input_mask.clamp(0.0, 1.0)

        data_paths = batch.get("data_path", None)
        if data_paths is None:
            data_paths = [f"batch_{it:06d}_idx_{b:02d}" for b in range(stereo.size(0))]
        elif isinstance(data_paths, (tuple, list)):
            data_paths = [str(p) for p in data_paths]
        else:
            data_paths = [str(data_paths) for _ in range(stereo.size(0))]

        B = valid_k.shape[0]
        data_indices = batch.get("dataset_index", None)
        if data_indices is None:
            data_indices = batch.get("data_index", None)
        if data_indices is None:
            data_indices = batch.get("index", None)
        if data_indices is None:
            data_indices = batch.get("idx", None)
        if data_indices is None:
            data_indices = [it * B + b for b in range(B)]
        elif torch.is_tensor(data_indices):
            data_indices = data_indices.detach().cpu().tolist()
        elif isinstance(data_indices, (tuple, list)):
            data_indices = [int(x) for x in data_indices]
        else:
            data_indices = [int(data_indices) for _ in range(B)]
        for b in range(B):
            data_path = data_paths[b]
            sample_index = int(data_indices[b])
            if not bool(eval_mask[b]):
                continue

            k_gt = int(gt_idx[b].item())
            k_pred = int(pred_idx_peak[b].item())
            cls_id = int(class_ids[b, k_gt].item())
            cls_key = str(cls_id)
            idx_key = str(sample_index)

            adds, add = _compute_adds_add_raw(
                pred_R_sel[b, 0],
                pred_t_sel[b, 0],
                gt_R_sel[b, 0],
                gt_t_sel[b, 0],
                model_points[b, k_gt],
                max_add_points,
            )
            diameter_mm = None
            if diameters is not None:
                diameter_mm = float(diameters[b, k_gt].item())
            adds_norm, add_norm = _normalize_add_metrics(adds, add, diameter_mm)

            adds_rot_only, add_rot_only = _compute_adds_add_raw(
                pred_R_sel[b, 0],
                gt_t_sel[b, 0],
                gt_R_sel[b, 0],
                gt_t_sel[b, 0],
                model_points[b, k_gt],
                max_add_points,
            )
            adds_rot_only_norm, add_rot_only_norm = _normalize_add_metrics(
                adds_rot_only, add_rot_only, diameter_mm
            )
            pos_mm = torch.norm(pred_t_sel[b, 0] - gt_t_sel[b, 0]).item()
            cam_obj_dist_mm = torch.norm(gt_t_sel[b, 0]).item()
            rot_deg = _geodesic_angle_deg(pred_R_sel[b, 0], gt_R_sel[b, 0]).item()

            pred_peak_yx = pred_inst["yx"][b, k_pred].to(torch.int64).tolist()
            gt_peak_yx = gt_iter0["peaks_yx"][b, k_gt].to(torch.int64).tolist()
            pred_score = pred_inst.get("score", None)
            pred_score_val = None
            if pred_score is not None:
                pred_score_val = float(pred_score[b, k_pred].item())

            image_path = ""
            if save_images and pred_r is not None and gt_r is not None:
                img_l = _normalize_rgb(img_left[b])
                img_r = _normalize_rgb(stereo[b, 3:])
                pred_disp = disp_pred_vis[b]
                gt_disp = disp_gt_vis[b]
                input_seg = tr._overlay_mask_rgb(
                    img_l.unsqueeze(0), input_seg_vis[b:b+1], color=(0.0, 1.0, 0.0), alpha=0.45
                )[0]

                pred_mask_rend = pred_r["silhouette"][b:b+1]
                gt_mask = gt_r["silhouette"][b:b+1]
                if sem_mask_1x is not None:
                    pred_mask_sem = sem_mask_1x[b:b+1]
                    pred_mask_vis = tr._overlay_mask_rgb(
                        img_l.unsqueeze(0), pred_mask_sem, color=(0.0, 1.0, 0.0), alpha=0.45
                    )[0]
                else:
                    pred_mask_vis = tr._overlay_mask_rgb(
                        img_l.unsqueeze(0), pred_mask_rend, color=(0.0, 1.0, 0.0), alpha=0.45
                    )[0]
                pred_rend = tr._overlay_mask_rgb(
                    img_l.unsqueeze(0), pred_mask_rend, color=(0.0, 1.0, 0.0), alpha=0.45
                )[0]
                gt_rend = tr._overlay_mask_rgb(
                    img_l.unsqueeze(0), gt_mask, color=(1.0, 0.0, 0.0), alpha=0.45
                )[0]
                pred_and_gt_rend = tr._overlay_mask_rgb(
                    gt_rend.unsqueeze(0), pred_mask_rend, color=(0.0, 1.0, 0.0), alpha=0.45
                )[0]

                panel = torch.stack(
                    [
                        img_l,
                        img_r,
                        input_seg,
                        pred_disp,
                        gt_disp,
                        pred_mask_vis,
                        pred_rend,
                        gt_rend,
                        pred_and_gt_rend,
                    ],
                    dim=0,
                )
                grid = vutils.make_grid(panel, nrow=3, normalize=False)
                safe_name = _safe_key(data_path)
                image_name = f"{safe_name}_k{k_pred:02d}_c{cls_id:02d}_idx{sample_index}.png"
                image_path = str(output_dir / image_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                vutils.save_image(grid, image_path)

            entry = {
                "dataset_index": int(sample_index),
                "data_path": data_path,
                "instance_index": int(k_pred),
                "gt_index": int(k_gt),
                "pred_peak_yx": pred_peak_yx,
                "gt_peak_yx": gt_peak_yx,
                "pred_score": pred_score_val,
                "adds": float(adds),
                "add": float(add),
                "adds_norm": float(adds_norm),
                "add_norm": float(add_norm),
                "adds_rot_only": float(adds_rot_only),
                "add_rot_only": float(add_rot_only),
                "adds_rot_only_norm": float(adds_rot_only_norm),
                "add_rot_only_norm": float(add_rot_only_norm),
                "pos_mm": float(pos_mm),
                "gt_cam_obj_dist_mm": float(cam_obj_dist_mm),
                "rot_deg": float(rot_deg),
                "image_path": image_path,
            }
            if cls_key not in results:
                results[cls_key] = {}
            if idx_key not in results[cls_key]:
                results[cls_key][idx_key] = entry
            else:
                prev = results[cls_key][idx_key]
                if isinstance(prev, list):
                    prev.append(entry)
                else:
                    results[cls_key][idx_key] = [prev, entry]
    return results


def _load_ckpt_into_model(model, ckpt_path: str, device: torch.device, strict: bool):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/inf_config.toml")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--batch_size", type=int, default=-1, help="override batch size")
    ap.add_argument("--iters", type=int, default=-1, help="override cfg['model']['n_iter']")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--output", type=str, default="evals/results.json")
    ap.add_argument("--eval_dir", type=str, default="evals")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--max_add_points", type=int, default=2048)
    ap.add_argument("--limit_batches", type=int, default=0, help="0=full loader, else limit iterator length")
    ap.add_argument("--use_gt_peaks", action="store_true", help="debug-only: use GT peaks for inference")
    ap.add_argument("--pose_update", action="store_true", help="enable render-based pose refinement")
    args = ap.parse_args()

    cfg = tr.load_toml(args.config)
    tr.set_global_seed(42)

    if args.launcher != "none":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist_utils.init_dist(args.launcher, backend)
        distributed = True
    else:
        distributed = False
    dist_utils.setup_for_distributed(is_master=dist_utils.is_main_process())

    device = torch.device("cuda", dist_utils.get_rank()) if torch.cuda.is_available() else torch.device("cpu")

    pose_update = bool(args.pose_update)
    force_topk = bool(pose_update and not args.use_gt_peaks)

    batch_size = None if args.batch_size <= 0 else int(args.batch_size)
    loader, sampler, class_table, split_used = make_eval_dataloader(
        cfg, args.split, distributed, batch_size
    )

    model = tr.build_model(cfg, class_table).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_utils.get_rank()] if device.type == "cuda" else None,
            output_device=dist_utils.get_rank() if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    if args.ckpt.strip():
        missing, unexpected = _load_ckpt_into_model(model, args.ckpt.strip(), device, strict=args.strict)
        if dist_utils.is_main_process():
            print(f"[ckpt] loaded: {args.ckpt.strip()}")
            if missing:
                print(f"[ckpt] missing keys: {len(missing)}")
            if unexpected:
                print(f"[ckpt] unexpected keys: {len(unexpected)}")

    if force_topk:
        if dist_utils.is_main_process():
            print("[eval] pose_update without gt peaks: force model.topk=1")
        target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        target.topk = 1

    iters = int(cfg["model"]["n_iter"])
    if args.iters > 0:
        iters = int(args.iters)

    output_dir = Path(args.eval_dir)
    results = evaluate(
        model=model,
        loader=loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        save_images=args.save_images,
        max_add_points=args.max_add_points,
        iters=iters,
        amp_on=bool(args.amp),
        limit_batches=args.limit_batches,
        use_gt_peaks=bool(args.use_gt_peaks),
        enable_pose_update=pose_update,
    )

    if dist.is_available() and dist.is_initialized():
        gathered: List[Dict[str, Dict[str, dict]]] = [None for _ in range(dist_utils.get_world_size())]
        dist.all_gather_object(gathered, results)
        if dist_utils.is_main_process():
            results = _merge_results(gathered)
        dist.barrier()

    if dist_utils.is_main_process():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "config": args.config,
                "ckpt": args.ckpt.strip(),
                "split": split_used,
                "timestamp": datetime.now().isoformat(),
                "max_add_points": args.max_add_points,
                "iters": iters,
                "use_gt_peaks": bool(args.use_gt_peaks),
                "pose_update": bool(pose_update),
                "force_topk": bool(force_topk),
            },
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval] wrote: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
