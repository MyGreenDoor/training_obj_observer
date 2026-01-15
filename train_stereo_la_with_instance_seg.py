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
from typing import Any, Dict, List, Optional, Tuple, Literal

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

from models.panoptic_stereo import PanopticStereoMultiHead, pos_mu_to_pointmap
from models.stereo_disparity import make_gn
from utils import dist_utils, rot_utils
from utils.logging_utils import draw_axes_on_images_bk, visualize_mono_torch
from utils.projection import SilhouetteDepthRenderer
from pytorch3d.structures import Meshes
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

    rot_map = batch["rot_map"].to(device, non_blocking=True)
    if rot_map.dim() == 5 and not (rot_map.shape[1] == 3 and rot_map.shape[2] == 3):
        rot_map = rot_map.permute(0, 3, 4, 1, 2)
    rot_map = F.interpolate(rot_map.flatten(1, 2), size=size_hw, mode="nearest")
    rot_map = rot_map.view(rot_map.size(0), 3, 3, size_hw[0], size_hw[1])

    sem_mask_1_4 = _downsample_label(sem_gt, size_hw) > 0
    sem_mask_1_4 = sem_mask_1_4.unsqueeze(1)
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
    return pos_map, rot_map, sem_mask_1_4, objs_in_left


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
    inst_1_4: torch.Tensor,            # (B,H,W) int, bg=0
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
    assert emb.dim() == 4 and inst_1_4.dim() == 3
    B, C, H, W = emb.shape
    assert inst_1_4.shape[0] == B and inst_1_4.shape[1] == H and inst_1_4.shape[2] == W

    emb_n = F.normalize(emb, dim=1, eps=1e-6)
    loss_sum = emb.new_tensor(0.0)
    n_sum = emb.new_tensor(0.0)

    for b in range(B):
        inst = inst_1_4[b]
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
    tau_emb_merge: float = 0.85,  # NOTE: emb_merge_metric="cos" のときは cosine類似度しきい値（大きいほど厳しい）
    emb_merge_iters: int = 1,
    emb_merge_metric: Literal["cos", "l2"] = "cos",
    emb_merge_small_area: Optional[int] = None,  # 例: 4 or 9; Noneならサイズ制限なし
    # semantic
    use_gt_semantic: bool = False,
    semantic_gt_1_4: Optional[torch.Tensor] = None,
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
    if use_gt_semantic and semantic_gt_1_4 is not None:
        sem_pred = semantic_gt_1_4

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
    inst_1x = _downsample_label(inst_gt, size_hw)
    wks, wfg = _build_instance_weight_map(inst_1x, valid_k)

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

    valid_pred = v_pred & valid_k
    valid_gt = v_gt & valid_k

    # T_pred = rot_utils.compose_T_from_Rt(r_pred, t_pred, valid_pred)
    # T_gt = rot_utils.compose_T_from_Rt(r_gt, t_gt, valid_gt)
    
    T_pred = rot_utils.compose_T_from_Rt(r_pred, t_pred, valid_pred)
    T_gt = rot_utils.compose_T_from_Rt(rot_gt, pos_gt, valid_gt)

    renderer = SilhouetteDepthRenderer().to(device)
    image_size = (stereo.shape[-2], stereo.shape[-1])

    pred_r = renderer(
        meshes_flat=meshes_flat,
        T_cam_obj=T_pred,
        K_left=left_k[:, 0],
        valid_k=valid_pred,
        image_size=image_size,
    )
    gt_r = renderer(
        meshes_flat=meshes_flat,
        T_cam_obj=T_gt,
        K_left=left_k[:, 0],
        valid_k=valid_gt,
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
        axis_len=0.05,
        valid=valid_pred[:n_images],
    )
    axes_gt = draw_axes_on_images_bk(
        overlay_gt,
        left_k[:n_images, 0],
        T_gt[:n_images],
        axis_len=0.05,
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
            mask = (inst_gt > 0).unsqueeze(1) # without background
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

            pos_gt_map, rot_gt_map, pose_mask, objs_in_left = _prepare_pose_targets(batch, sem_gt, size_hw, device)
            gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(
                pos_gt_map, left_k[:, 0], downsample=1, use_logz=True
            )
            loss_pos = loss_functions.pos_loss_hetero_map(
                pred["pos_mu_norm"],
                pred["pos_logvar_norm"],
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
            loss_emb, emb_pairs = embedding_cosface_sampled(
                pred["emb"],
                inst_gt_1_4,    
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
        depth_mae = l1(depth_pred_1x[mask], depth[mask])

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

        pos_gt_map, rot_gt_map, pose_mask, objs_in_left = _prepare_pose_targets(batch, sem_gt, size_hw, device)
        gt_pos_mu_map = loss_functions._pos_mu_gt_from_t_map(
            pos_gt_map, left_k[:, 0], downsample=1, use_logz=True
        )
        loss_pos = loss_functions.pos_loss_hetero_map(
            pred["pos_mu_norm"],
            pred["pos_logvar_norm"],
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
        loss_emb, emb_pairs = embedding_cosface_sampled(
            pred["emb"],
            inst_gt_1_4,    
            max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
            margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
            scale=32.0,
            min_pixels_per_inst=4,
            detach_proto=True,
            topk_neg=None,           # 重いなら 16 とかに
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
