#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo training for panoptic segmentation without input masks.
This entry point is independent from train_stereo_la.py.
"""
import argparse
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tomlkit

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

from models.panoptic_stereo import PanopticStereoMultiHead
from models.stereo_disparity import make_gn
from utils import dist_utils
from utils.logging_utils import visualize_mono_torch
from losses import loss_functions

import train_stereo_la as base

mp.set_start_method("spawn", force=True)
supported = getattr(mp, "get_all_sharing_strategies", lambda: ["file_system"])()
strategy = "file_descriptor" if "file_descriptor" in supported else "file_system"
mp.set_sharing_strategy(strategy)

_CFG_TEXT_CACHE = None

_VARLEN_KEYS = {
    "class_ids",
    "diameters_list",
    "faces_list",
    "object_ids",
    "objs_in_left",
    "objs_in_right",
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


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers consistently."""
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))


def load_toml(config_path: str) -> dict:
    """Load TOML config while caching its raw text."""
    global _CFG_TEXT_CACHE
    config_path = Path(config_path)
    _CFG_TEXT_CACHE = config_path.read_text(encoding="utf-8")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = tomlkit.load(f)
    return cfg


def write_toml(cfg: dict, out_path: Path) -> None:
    """Write config data to a TOML file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        tomlkit.dump(cfg, f)


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
    out_size_wh = (256, 256)
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
    seg_cfg = cfg.get("seg_head", {}) or {}
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
        head_c4=int(seg_cfg.get("head_c4", 80)),
        head_c8=int(seg_cfg.get("head_c8", 112)),
        head_fuse_ch=int(seg_cfg.get("head_fuse_ch", 160)),
        head_geo_ch=int(seg_cfg.get("head_geo_ch", 112)),
        head_sem_ch=int(seg_cfg.get("head_sem_ch", 80)),
        head_inst_ch=int(seg_cfg.get("head_inst_ch", 80)),
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
    pos_map = pos_map * 0.001

    rot_map = batch["rot_map"].to(device, non_blocking=True)
    if rot_map.dim() == 5 and not (rot_map.shape[1] == 3 and rot_map.shape[2] == 3):
        rot_map = rot_map.permute(0, 3, 4, 1, 2)
    rot_map = F.interpolate(rot_map.flatten(1, 2), size=size_hw, mode="nearest")
    rot_map = rot_map.view(rot_map.size(0), 3, 3, size_hw[0], size_hw[1])

    sem_mask_1_4 = _downsample_label(sem_gt, size_hw) != 0
    sem_mask_1_4 = sem_mask_1_4.unsqueeze(1)

    pos_map = pos_map * sem_mask_1_4
    rot_map = rot_map * sem_mask_1_4.unsqueeze(1)

    return pos_map, rot_map, sem_mask_1_4


def _build_affinity_targets(inst_1_4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build affinity targets and valid masks from instance IDs."""
    B, H, W = inst_1_4.shape
    tgt = torch.zeros((B, 4, H, W), device=inst_1_4.device, dtype=torch.float32)
    valid = torch.zeros_like(tgt, dtype=torch.bool)

    same = inst_1_4[:, :, :-1] == inst_1_4[:, :, 1:]
    both_fg = (inst_1_4[:, :, :-1] > 0) & (inst_1_4[:, :, 1:] > 0)
    tgt[:, 0, :, :-1] = same.float()
    valid[:, 0, :, :-1] = both_fg

    same = inst_1_4[:, :, 1:] == inst_1_4[:, :, :-1]
    both_fg = (inst_1_4[:, :, 1:] > 0) & (inst_1_4[:, :, :-1] > 0)
    tgt[:, 1, :, 1:] = same.float()
    valid[:, 1, :, 1:] = both_fg

    same = inst_1_4[:, :-1, :] == inst_1_4[:, 1:, :]
    both_fg = (inst_1_4[:, :-1, :] > 0) & (inst_1_4[:, 1:, :] > 0)
    tgt[:, 2, :-1, :] = same.float()
    valid[:, 2, :-1, :] = both_fg

    same = inst_1_4[:, 1:, :] == inst_1_4[:, :-1, :]
    both_fg = (inst_1_4[:, 1:, :] > 0) & (inst_1_4[:, :-1, :] > 0)
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


def _embedding_loss_local_pairs(
    emb: torch.Tensor,
    inst_1_4: torch.Tensor,
    margin: float,
    neg_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contrastive loss using local neighbor pairs inside foreground."""
    B, _, _, _ = emb.shape
    total_loss = emb.new_tensor(0.0)
    total_pairs = emb.new_tensor(0.0)

    for b in range(B):
        inst = inst_1_4[b]
        emb_b = emb[b]

        mask_r = (inst[:, :-1] > 0) & (inst[:, 1:] > 0)
        if mask_r.any():
            diff = emb_b[:, :, :-1] - emb_b[:, :, 1:]
            dist = torch.norm(diff, dim=0)
            same = inst[:, :-1] == inst[:, 1:]
            pos = dist.pow(2)
            neg = F.relu(margin - dist).pow(2)
            loss_r = torch.where(same, pos, neg * neg_weight)
            total_loss = total_loss + (loss_r * mask_r).sum()
            total_pairs = total_pairs + mask_r.sum()

        mask_d = (inst[:-1, :] > 0) & (inst[1:, :] > 0)
        if mask_d.any():
            diff = emb_b[:, :-1, :] - emb_b[:, 1:, :]
            dist = torch.norm(diff, dim=0)
            same = inst[:-1, :] == inst[1:, :]
            pos = dist.pow(2)
            neg = F.relu(margin - dist).pow(2)
            loss_d = torch.where(same, pos, neg * neg_weight)
            total_loss = total_loss + (loss_d * mask_d).sum()
            total_pairs = total_pairs + mask_d.sum()

    loss = total_loss / total_pairs.clamp_min(1.0)
    return loss, total_pairs


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
    tau_emb_merge: float,
    emb_merge_iters: int,
    use_gt_semantic: bool,
    semantic_gt_1_4: Optional[torch.Tensor] = None,
) -> List[np.ndarray]:
    """
    Infer instance IDs from semantic, affinity, and embeddings.

    Changes vs original:
      - Much faster: no per-pixel double for-loop for union; we enumerate only True edges.
      - Mutual check for affinity:
          (y,x)-(y,x+1): aff[0,y,x] AND aff[1,y,x+1]
          (y,x)-(y+1,x): aff[2,y,x] AND aff[3,y+1,x]
      - Only uses right+down edges (no redundant left/up passes).
      - Embedding-merge loop runs only if emb_merge_iters > 0 (0 means disabled).
    """
    # semantic prediction
    sem_pred = sem_logits.argmax(dim=1)
    if use_gt_semantic and semantic_gt_1_4 is not None:
        sem_pred = semantic_gt_1_4

    # move once to CPU numpy (still per-batch processing for union-find)
    sem_np = sem_pred.detach().cpu().numpy()                        # (B,H,W) int
    aff_np = torch.sigmoid(aff_logits).detach().cpu().numpy()       # (B,4,H,W) float
    emb_np = emb.detach().cpu().permute(0, 2, 3, 1).numpy()         # (B,H,W,D) float

    out: List[np.ndarray] = []

    for b in range(sem_np.shape[0]):
        sem_b = sem_np[b]               # (H,W)
        sem_mask = (sem_b != 0)         # "things" mask
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

        # --- build mutual-checked edges (right & down only) ---
        # Right edges: between (y,x) and (y,x+1) for x in [0..W-2]
        # condition: both sem_mask True AND aff0(y,x) > tau AND aff1(y,x+1) > tau
        if W > 1:
            right_ok = (
                sem_mask[:, :-1] & sem_mask[:, 1:]
                & (aff_np[b, 0, :, :-1] > tau_high)
                & (aff_np[b, 1, :, 1:] > tau_high)
            )
            ry, rx = np.where(right_ok)
        else:
            ry, rx = np.empty((0,), np.int64), np.empty((0,), np.int64)

        # Down edges: between (y,x) and (y+1,x) for y in [0..H-2]
        # condition: both sem_mask True AND aff2(y,x) > tau AND aff3(y+1,x) > tau
        if H > 1:
            down_ok = (
                sem_mask[:-1, :] & sem_mask[1:, :]
                & (aff_np[b, 2, :-1, :] > tau_high)
                & (aff_np[b, 3, 1:, :] > tau_high)
            )
            dy, dx = np.where(down_ok)
        else:
            dy, dx = np.empty((0,), np.int64), np.empty((0,), np.int64)

        # union only the selected edges
        # (loop count = number of true edges, typically far smaller than H*W)
        for y, x in zip(ry.tolist(), rx.tolist()):
            union(idx(y, x), idx(y, x + 1))
        for y, x in zip(dy.tolist(), dx.tolist()):
            union(idx(y, x), idx(y + 1, x))

        # --- assign compact instance IDs from union-find roots ---
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

        # --- optional embedding-based merge (only if enabled) ---
        if emb_merge_iters > 0:
            D = emb_np.shape[-1]
            for _ in range(emb_merge_iters):
                num_comp = int(inst.max())
                if num_comp <= 1:
                    break

                # component means mu[k] via bincount / add.at (fast)
                mu = np.zeros((num_comp + 1, D), dtype=np.float32)
                cnt = np.bincount(inst.ravel(), minlength=num_comp + 1).astype(np.float32).reshape(-1, 1)

                # sum embeddings per component
                flat_inst = inst.ravel()
                flat_emb = emb_np[b].reshape(-1, D)
                # ignore background id=0 in summation (still fine if included; we'll keep it, it won't be used)
                np.add.at(mu, flat_inst, flat_emb)
                mu = mu / np.clip(cnt, 1.0, None)

                # adjacency pairs from current inst (vectorized)
                pairs = []

                if W > 1:
                    a = inst[:, :-1]
                    c = inst[:, 1:]
                    m = (a > 0) & (c > 0) & (a != c)
                    if np.any(m):
                        aa = a[m].astype(np.int32)
                        cc = c[m].astype(np.int32)
                        lo = np.minimum(aa, cc)
                        hi = np.maximum(aa, cc)
                        pairs.append(np.stack([lo, hi], axis=1))

                if H > 1:
                    a = inst[:-1, :]
                    c = inst[1:, :]
                    m = (a > 0) & (c > 0) & (a != c)
                    if np.any(m):
                        aa = a[m].astype(np.int32)
                        cc = c[m].astype(np.int32)
                        lo = np.minimum(aa, cc)
                        hi = np.maximum(aa, cc)
                        pairs.append(np.stack([lo, hi], axis=1))

                if not pairs:
                    break

                adj = np.unique(np.concatenate(pairs, axis=0), axis=0)  # (E,2)

                comp_parent = np.arange(num_comp + 1, dtype=np.int32)

                def cfind(x: int) -> int:
                    while comp_parent[x] != x:
                        comp_parent[x] = comp_parent[comp_parent[x]]
                        x = comp_parent[x]
                    return x

                def cunion(a: int, c: int) -> None:
                    ra, rb = cfind(a), cfind(c)
                    if ra != rb:
                        comp_parent[rb] = ra

                # merge adjacent components if embedding distance is small
                for a, c in adj.tolist():
                    if np.linalg.norm(mu[a] - mu[c]) < tau_emb_merge:
                        cunion(a, c)

                # relabel components to compact ids
                new_inst = np.zeros_like(inst)
                root_to_id = {}
                next_id = 1
                changed = False

                ys, xs = np.where(inst > 0)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    r = cfind(int(inst[y, x]))
                    if r not in root_to_id:
                        root_to_id[r] = next_id
                        next_id += 1
                    nid = root_to_id[r]
                    new_inst[y, x] = nid
                    if nid != inst[y, x]:
                        changed = True

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
    sem_gt_1_4 = _downsample_label(semantic_gt, size_hw)

    inst_pred = _infer_instance_from_heads(
        sem_logits,
        aff_logits,
        emb,
        tau_high=float(cfg.get("instance", {}).get("tau_aff", 0.95)),
        tau_emb_merge=float(cfg.get("instance", {}).get("tau_emb_merge", 0.6)),
        emb_merge_iters=int(cfg.get("instance", {}).get("emb_merge_iters", 1)),
        use_gt_semantic=bool(cfg.get("instance", {}).get("use_gt_semantic", False)),
        semantic_gt_1_4=sem_gt_1_4,
    )

    num_classes = int(cfg.get("data", {}).get("n_classes", sem_logits.shape[1]))
    sem_pred = sem_logits.argmax(dim=1)
    sem_pred_color = _colorize_semantic(sem_pred, num_classes).float() / 255.0
    sem_gt_color = _colorize_semantic(sem_gt_1_4, num_classes).float() / 255.0

    inst_pred_color = torch.stack([_colorize_instance_ids(inst) for inst in inst_pred], dim=0).float() / 255.0
    inst_gt_1_4 = _downsample_label(instance_gt, size_hw).cpu().numpy()
    inst_gt_color = torch.stack([_colorize_instance_ids(inst) for inst in inst_gt_1_4], dim=0).float() / 255.0

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
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    l1 = torch.nn.L1Loss()
    log_every = cfg["train"]["log_interval"]

    window_sum, window_cnt = base._init_window_meter()
    global_step = epoch * len(loader)

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = _prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"]):
            pred = model(
                stereo,
                k_pair,
                baseline,
                iters=cfg["model"]["n_iter"],
            )
            disp_preds = pred["disp_preds"]
            disp_logvar_preds = pred["disp_log_var_preds"]
            mask = disp_gt > 0
            loss_disp = base.disparity_nll_laplace_raft_style(
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
            sem_gt_1_4 = _downsample_label(sem_gt, size_hw)
            loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_1_4, use_focal=False)
            loss_sem = loss_cls

            pos_gt_map, rot_gt_map, pose_mask = _prepare_pose_targets(batch, sem_gt, size_hw, device)
            gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(pos_gt_map, left_k[:, 0], downsample=1)
            loss_pos = loss_functions.pos_loss_hetero_map(
                pred["pos_mu"],
                pred["pos_logvar"],
                gt_pos_mu_map,
                pose_mask,
            )
            loss_rot = loss_functions.rotation_loss_hetero_map(
                pred["rot_mat"],
                rot_gt_map,
                pred["rot_logvar_theta"],
                pose_mask,
            )

            inst_gt_1_4 = _downsample_label(inst_gt, size_hw)
            aff_tgt, aff_valid = _build_affinity_targets(inst_gt_1_4)
            loss_aff, aff_valid_px = _affinity_loss(
                pred["aff_logits"],
                aff_tgt,
                aff_valid,
                neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 8.0)),
            )
            loss_emb, emb_pairs = _embedding_loss_local_pairs(
                pred["emb"],
                inst_gt_1_4,
                margin=float(cfg.get("loss", {}).get("emb_margin", 2.0)),
                neg_weight=float(cfg.get("loss", {}).get("emb_neg_weight", 2.0)),
            )

            logs = {
                "L_sem": loss_sem.detach(),
                "L_cls": loss_cls.detach(),
                "L_pos": loss_pos.detach(),
                "L_rot": loss_rot.detach(),
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

        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        depth_pred_1x = pred["point_map_1x"][:, 2:3]
        depth_m = depth * 0.001
        depth_mae = l1(depth_pred_1x[mask], depth_m[mask])

        did_optim_step = scaler.get_scale() >= prev_scale
        if scheduler is not None and sched_step_when == "step" and did_optim_step:
            scheduler.step()

        total_loss += loss.detach()
        window_cnt = base._update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs)

        global_step = epoch * len(loader) + it
        if writer is not None and dist_utils.is_main_process() and (global_step % log_every == 0):
            base._flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train")
            with torch.no_grad():
                _log_disp_and_depth(
                    writer,
                    global_step,
                    "train",
                    stereo,
                    pred["disp_1x"],
                    disp_gt,
                    depth_pred_1x * 1000.0,
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
            window_cnt = base._reset_window_meter(window_sum, window_cnt)

    return total_loss.item() / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter],
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model and return disparity and semantic losses."""
    model.eval()
    meters = base.DictMeters()
    meters.add_avg("L")
    meters.add_avg("L_sem")
    meters.add_avg("L_cls")
    meters.add_avg("L_pos")
    meters.add_avg("L_rot")
    meters.add_avg("L_disp_1x")
    meters.add_avg("L_aff")
    meters.add_avg("L_emb")

    for it, batch in enumerate(loader):
        stereo, depth, disp_gt, k_pair, baseline, left_k = _prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        pred = model(
            stereo,
            k_pair,
            baseline,
            iters=cfg["model"]["n_iter"],
        )
        disp_preds = pred["disp_preds"]
        disp_logvar_preds = pred["disp_log_var_preds"]
        mask = disp_gt > 0
        loss_disp = base.disparity_nll_laplace_raft_style(
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
        sem_gt_1_4 = _downsample_label(sem_gt, size_hw)
        loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_1_4, use_focal=False)
        loss_sem = loss_cls

        pos_gt_map, rot_gt_map, pose_mask = _prepare_pose_targets(batch, sem_gt, size_hw, device)
        gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(pos_gt_map, left_k[:, 0], downsample=1)
        loss_pos = loss_functions.pos_loss_hetero_map(
            pred["pos_mu"],
            pred["pos_logvar"],
            gt_pos_mu_map,
            pose_mask,
        )
        loss_rot = loss_functions.rotation_loss_hetero_map(
            pred["rot_mat"],
            rot_gt_map,
            pred["rot_logvar_theta"],
            pose_mask,
        )

        inst_gt_1_4 = _downsample_label(inst_gt, size_hw)
        aff_tgt, aff_valid = _build_affinity_targets(inst_gt_1_4)
        loss_aff, _ = _affinity_loss(
            pred["aff_logits"],
            aff_tgt,
            aff_valid,
            neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 4.0)),
        )
        loss_emb, _ = _embedding_loss_local_pairs(
            pred["emb"],
            inst_gt_1_4,
            margin=float(cfg.get("loss", {}).get("emb_margin", 1.0)),
            neg_weight=float(cfg.get("loss", {}).get("emb_neg_weight", 2.0)),
        )

        meters.update_avg("L", float(loss_disp.item()), n=stereo.size(0))
        meters.update_avg("L_sem", float(loss_sem.item()), n=stereo.size(0))
        meters.update_avg("L_cls", float(loss_cls.item()), n=stereo.size(0))
        meters.update_avg("L_pos", float(loss_pos.item()), n=stereo.size(0))
        meters.update_avg("L_rot", float(loss_rot.item()), n=stereo.size(0))
        meters.update_avg("L_disp_1x", float(loss_disp_1x.item()), n=stereo.size(0))
        meters.update_avg("L_aff", float(loss_aff.item()), n=stereo.size(0))
        meters.update_avg("L_emb", float(loss_emb.item()), n=stereo.size(0))

        if writer is not None and dist_utils.is_main_process() and it == 0:
            depth_pred_1x = pred["point_map_1x"][:, 2:3]
            _log_disp_and_depth(
                writer,
                epoch,
                "val",
                stereo,
                pred["disp_1x"],
                disp_gt,
                depth_pred_1x * 1000.0,
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

    loss_disp_avg = meters.get("L").avg
    loss_sem_avg = meters.get("L_sem").avg
    return loss_disp_avg, loss_sem_avg


def _load_model_state(model, state_dict, strict: bool = False):
    """Load model weights with DDP-awareness."""
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    return missing, unexpected


def main() -> None:
    """Entry point for panoptic stereo training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config TOML", default="configs/small_config.toml")
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
    scheduler, sched_step_when = base.build_lr_scheduler(cfg, optimizer, steps_per_epoch, total_steps)

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
