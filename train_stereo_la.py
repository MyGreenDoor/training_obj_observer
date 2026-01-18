#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo training template with DDP (torchrun) + TOML configs + TensorBoard.
Single-node multi-GPU ready. model is dummy but wired as real training.
Usage (single node, 4 GPUs):
  torchrun --nproc_per_node=4 --master_port=29501 train_stereo_la.py \
      --config configs/example_config.toml --launcher pytorch
"""
import argparse
import math
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tomlkit

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from la_loader.synthetic_data_loader import LASyntheticDataset3PerObj
from la_loader import la_transforms

from models.sscflow2 import SSCFlow2, make_gn
from utils import dist_utils, flow_utils, rot_utils
from utils.logging_utils import (
    draw_axes_on_images_bk,
    make_center_overlay_grid,
    make_mask_overlay_grid,
    visualize_mono_torch,
)
from utils.projection import SilhouetteDepthRenderer
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from losses import loss_functions

mp.set_start_method("spawn", force=True)
supported = getattr(mp, "get_all_sharing_strategies", lambda: ["file_system"])()
strategy = "file_descriptor" if "file_descriptor" in supported else "file_system"
mp.set_sharing_strategy(strategy)

# --- TOML read/write helpers ---
_CFG_TEXT_CACHE = None  # original text of the config file


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for variable instance counts.

    Args:
        batch: List of sample dictionaries from the dataset.

    Returns:
        Merged batch dictionary with padded instance fields.
    """
    # まず可変Kテンソル
    Wk_list = [b.pop('instance_weight_map') for b in batch]  # list of (Ki,H4,W4)

    # インスタンスごとの mesh をフラット化
    verts_all, faces_all, bk_splits, diameters_all = [], [], [], []  # bk_splits: 画像ごとの Ki
    for b in batch:
        vlist = [torch.from_numpy(v).float() for v in b.pop('verts_list')]  # list of (Ni,3)
        flist = [torch.from_numpy(f).long() for f in b.pop('faces_list')]  # list of (Mi,3)
        diameters_list = [
            torch.from_numpy(f).float() for f in b.pop('diameters_list')
        ]  # list of (1, )
        assert len(vlist) == len(flist)
        bk_splits.append(len(vlist))
        verts_all += vlist
        faces_all += flist
        diameters_all += diameters_list
    diameters_all = torch.concatenate(diameters_all, dim=0)

    meshes_all = Meshes(verts=verts_all, faces=faces_all)

    out = default_collate(batch)

    # Wk を (B,Kmax,H4,W4) パディング
    Wk = pad_sequence(
        [torch.as_tensor(w).float() for w in Wk_list],
        batch_first=True,
        padding_value=0.0,
    )
    valid_k = (Wk.abs().sum(dim=(2,3)) > 0)

    out['instance_weight_map'] = Wk
    out['valid_k'] = valid_k

    # ---- フラットな全インスタンスから点群サンプル → 再パック ----
    N_pts = 4096
    meshes_all_mm = Meshes(
        verts=[v for v in meshes_all.verts_list()],
        faces=meshes_all.faces_list()
    )
    pts_all = sample_points_from_meshes(meshes_all_mm, N_pts)  # (sum_K, N, 3)

    # 画像ごとにスライスして (B, Ki, N, 3) → pad to (B, Kmax, N, 3)
    B = len(bk_splits)
    Kmax = Wk.size(1)
    model_points = pts_all.new_zeros(B, Kmax, N_pts, 3)
    diameters = torch.zeros((B, Kmax), dtype=torch.float32)
    idx = 0
    for bidx, Ki in enumerate(bk_splits):
        if Ki > 0:
            model_points[bidx, :Ki] = pts_all[idx: idx+Ki]
            diameters[bidx, :Ki] = diameters_all[idx: idx+Ki]
        idx += Ki

    out['model_points'] = model_points  # (B,Kmax,N,3)
    out['diameters'] = diameters
    out['meshes'] = meshes_all_mm        # optional

    return out


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 完全決定論モード（速度低下あり）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # さらに厳密にするなら（PyTorch 1.8+）
    # torch.use_deterministic_algorithms(True)



def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers consistently.

    Args:
        worker_id: Worker index provided by DataLoader.
    """
    # DataLoader が割り当てた「そのワーカー固有の seed」
    # （各 worker で異なる値になっている）
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed  # 64bit
    # Python/NumPy にも反映（NumPy は 32bit のため mod）
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    

def load_toml(config_path: str) -> dict:
    """Load TOML config while caching its raw text.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        Parsed config dictionary.
    """
    global _CFG_TEXT_CACHE
    config_path = Path(config_path)
    _CFG_TEXT_CACHE = config_path.read_text(encoding="utf-8")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = tomlkit.load(f)
    return cfg


def write_toml(cfg: dict, out_path: Path) -> None:
    """Write config data to a TOML file.

    Args:
        cfg: Config dictionary to write.
        out_path: Destination path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        tomlkit.dump(cfg, f)


def _build_filtering_transforms(for_train: bool) -> la_transforms.LACompose:
    """Create filtering transforms for train/val datasets.

    Args:
        for_train: Whether to include augmentation transforms.

    Returns:
        Composed filtering transform.
    """
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
    """Create spatial transforms shared by train/val datasets.

    Returns:
        Composed spatial transform.
    """
    return la_transforms.LACompose(
        [
            (la_transforms.LARandomCrop([[-30, -30], [30, 30]], [-20, 20]), 1.0, ["trans_matrix"]),
            (la_transforms.LARandomResize([[0.66, 0.66], [0.8, 0.8]]), 1.0, ["trans_matrix"]),
            # (la_transforms.LARandomVerticalFlip(), 0.5, ["trans_matrix"]),
        ]
    )


def make_dataloaders(cfg: dict, distributed: bool):
    """Build train/val dataloaders and samplers.

    Args:
        cfg: Training configuration dictionary.
        distributed: Whether to use DistributedSampler.

    Returns:
        Tuple of (train_loader, val_loader, train_sampler, val_sampler, class_table).
    """
    filtering_trans_list = _build_filtering_transforms(for_train=True)
    spatial_trans_list = _build_spatial_transforms()
    out_list = ("stereo", 'depth', 'disparity', 'instance_seg')
    use_camera_list = ['ZED2', 'D415', 'ZEDmini']
    # train dataset
    train_ds = LASyntheticDataset3PerObj(
        out_list=out_list,
        with_data_path=True,
        use_camera_list=use_camera_list,
        with_camera_params=True,
        out_size_wh=(cfg['data']['width'], cfg['data']['height']),
        with_depro_matrix=True,
        target_scene_list=cfg['data']['train_datasets'][0]['target_scene_list'],
        spatial_transform=spatial_trans_list,
        filtering_transform=filtering_trans_list,
    )
    class_table = train_ds.class_dict
    filtering_trans_list = _build_filtering_transforms(for_train=False)
    spatial_trans_list = _build_spatial_transforms()
    val_ds = LASyntheticDataset3PerObj(
        out_list=out_list,
        with_data_path=True,
        use_camera_list=use_camera_list,
        with_camera_params=True,
        out_size_wh=(cfg['data']['width'], cfg['data']['height']),
        with_depro_matrix=True,
        target_scene_list=cfg['data']['val_datasets'][0]['target_scene_list'],
        spatial_transform=spatial_trans_list,
        filtering_transform=filtering_trans_list,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    # --- 3) DataLoader 側の base 乱数を固定 ---
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
    return train_loader, val_loader, train_sampler, val_sampler, class_table


def build_model(cfg: dict, class_table: dict) -> SSCFlow2:
    """Build the SSCFlow2 model using the config.

    Args:
        cfg: Training configuration dictionary.
        class_table: Mapping from class names to IDs.

    Returns:
        Instantiated SSCFlow2 model.
    """
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
    net = SSCFlow2(
        levels=int(mcfg.get("levels", 4)),
        norm_layer=make_gn(16),  # 既存の norm をお使いなら適宜差し替え
        l2_normalize_feature=bool(mcfg.get("l2_normalize_feature", True)),
        use_ctx_aspp=bool(mcfg.get("use_ctx_aspp", True)),
        lookup_mode=str(mcfg.get("lookup_mode", "1d")),
        radius_w=int(mcfg.get("radius_w", 4)),
        radius_h=int(mcfg.get("radius_h", 0)),
        context_ch=int(mcfg.get("context_ch", 128)),
        hidden_ch=int(mcfg.get("hidden_ch", 128)),
        num_classes=len(class_table.keys()) + 1,
        topk=int(mcfg.get("topk", 20)),
        nms_radius=int(mcfg.get("nms_radius", 4)),
        center_thresh=float(mcfg.get("center_thresh", 0.3)),
        rot_repr=str(mcfg.get("rot_repr", "r6d")),
        faces_per_pixel=int(mcfg.get("faces_per_pixel", 8)),
        blur_sigma=float(mcfg.get("blur_sigma", 1e-4)),
        blur_gamma=float(mcfg.get("blur_gamma", 1e-4)),
        shape_constraint_ema=float(mcfg.get("shape_constraint_ema", 0.4)),
        use_gate=bool(mcfg.get("use_gate", False)),
        use_so3_log_xyz=bool(mcfg.get("use_so3_log_xyz", True)),  # 回転=so3ログ, 並進=Δx,Δy,Δz
        use_so3_log_ratio_normal=bool(mcfg.get("use_so3_log_ratio_normal", True)),
        raft_like_updater=bool(mcfg.get("raft_like_updater", True)),
        point_map_norm_mean=point_map_norm_mean,
        point_map_norm_std=point_map_norm_std,
    )
    return net


def build_lr_scheduler(cfg, optimizer, steps_per_epoch: int, total_steps: int):
    """Build LR scheduler from config.

    Args:
        cfg: Training configuration dictionary.
        optimizer: Optimizer instance.
        steps_per_epoch: Number of steps per epoch.
        total_steps: Total steps for the entire training run.

    Returns:
        Tuple of (scheduler, step_when) where step_when in
        {"step", "epoch", "epoch_metric"}.
    """
    scfg = cfg.get("lr_scheduler", {}) or {}
    typ = scfg.get("type", "none").lower()
    if typ in ("none", "", None):
        return None, None

    # 汎用値
    base_lr = cfg["train"]["lr"]

    if typ == "cosine":
        # Linear warmup -> cosine decay to min_lr
        warmup_steps = int(scfg.get("warmup_steps", 0) or scfg.get("warmup_ratio", 0.05) * total_steps)
        min_lr = float(scfg.get("min_lr", 1e-6))
        min_factor = min_lr / base_lr

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            t = min(max(t, 0.0), 1.0)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, "step"

    if typ == "onecycle":
        pct_start = float(scfg.get("pct_start", 0.05))
        anneal = str(scfg.get("anneal_strategy", "cos"))
        div_factor = float(scfg.get("div_factor", 25.0))
        final_div = float(scfg.get("final_div_factor", 1e4))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base_lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy=anneal,
            div_factor=div_factor, final_div_factor=final_div
        )
        return scheduler, "step"

    if typ == "multistep":
        # 反復番号で指定（エポックではなく step 数）
        # デフォは 60% / 85% の時点で減衰
        ms = scfg.get("milestones")
        if ms is None:
            ms = [int(0.6 * total_steps), int(0.85 * total_steps)]
        gamma = float(scfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=gamma)
        return scheduler, "step"

    if typ == "plateau":
        # val 指標が停滞したら縮小
        factor = float(scfg.get("factor", 0.5))
        patience = int(scfg.get("patience", 5))
        min_lr = float(scfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, min_lr=min_lr
        )
        return scheduler, "epoch_metric"

    # 未知タイプ
    return None, None


@torch.no_grad()
def downsample_disp_and_mask_1_4(
    disp_1x: torch.Tensor,             # (B,1,H,W)
    valid_1x: Optional[torch.Tensor],  # (B,1,H,W) in {0,1} or None
    min_valid_ratio: float = 0.3,      # ← しきい値は内部デフォルトで用意
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample disparity and mask to 1/4 resolution.

    Steps:
        1) Resize mask and disp*mask with bilinear interpolation.
        2) Divide to get a valid-only average.
        3) Binarize mask using the valid ratio threshold.
        4) Scale disparity by 1/4 to match pixel units.

    Args:
        disp_1x: Disparity map at full resolution (B, 1, H, W).
        valid_1x: Validity mask or None.
        min_valid_ratio: Threshold to mark valid pixels after downsampling.

    Returns:
        Tuple of (disp_1_4, valid_1_4).
    """
    B, _, H, W = disp_1x.shape
    target_hw = (H // 4, W // 4)  # 端数が嫌なら round でもOK

    # マスクが無ければ NaN でない所を有効に
    if valid_1x is None:
        valid_1x = (disp_1x == disp_1x).to(disp_1x.dtype)
    else:
        valid_1x = valid_1x.to(disp_1x.dtype)

    # 連続マスクと重み付き視差を bilinear でダウンサンプル
    mask_ds  = F.interpolate(valid_1x, size=target_hw, mode="bilinear", align_corners=False)
    dispw_ds = F.interpolate(disp_1x * valid_1x, size=target_hw, mode="bilinear", align_corners=False)

    # 重みで割り戻し（無効除外平均）→ まだ 1x ピクセル単位
    eps = 1e-6
    disp_mean_1x = dispw_ds / (mask_ds + eps)

    # 有効率で2値化
    valid_1_4 = (mask_ds >= min_valid_ratio).to(disp_1x.dtype)

    # 単位を 1/4 へ
    disp_1_4 = (disp_mean_1x / 4.0) * valid_1_4

    return disp_1_4, valid_1_4



def disparity_nll_laplace_raft_style(
    preds_1_4: List[torch.Tensor],        # [(B,1,64,64), ...]
    logvars_1_4: List[torch.Tensor],      # [(B,1,64,64), ...] log(sigma^2)
    gt_disp_1x: torch.Tensor,             # (B,1,256,256)
    gt_valid_1x: Optional[torch.Tensor]=None,
    gamma: float = 0.9,
    charb_eps: float = 0.0,               # >0 で |r| をCharb化（擬似L1）
) -> torch.Tensor:
    """Compute Laplace NLL loss for disparity predictions.

    Args:
        preds_1_4: List of disparity predictions at 1/4 resolution.
        logvars_1_4: List of log variance predictions matching preds_1_4.
        gt_disp_1x: Ground-truth disparity at full resolution.
        gt_valid_1x: Optional validity mask for GT disparity.
        gamma: RAFT-style weighting factor for iterations.
        charb_eps: Optional Charbonnier epsilon for robust loss.

    Returns:
        Scalar loss tensor.
    """
    assert len(preds_1_4) > 0
    assert len(preds_1_4) == len(logvars_1_4)

    h, w = preds_1_4[0].shape[-2:]
    gt_1_4 = F.interpolate(gt_disp_1x, size=(h, w), mode="bilinear", align_corners=False) / 4.0

    if gt_valid_1x is None:
        valid_1_4 = torch.ones((gt_disp_1x.size(0), 1, h, w), device=gt_disp_1x.device, dtype=gt_disp_1x.dtype)
    else:
        if gt_valid_1x.dim() == 3:
            v = (gt_valid_1x > 0).unsqueeze(1)
        else:
            v = (gt_valid_1x > 0)
        valid_1_4 = F.interpolate(v.to(gt_disp_1x.dtype), size=(h, w), mode="nearest")

    N = len(preds_1_4)
    total = gt_disp_1x.new_tensor(0.0)
    denom = gt_disp_1x.new_tensor(0.0)

    for i in range(N):
        w_i = gamma ** (N - 1 - i)
        pred = preds_1_4[i]
        logv = logvars_1_4[i]

        r = pred - gt_1_4

        # |r| or Charb(|r|)
        if charb_eps > 0.0:
            abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
        else:
            abs_r = r.abs()

        # Laplace NLL with log_var = log(sigma^2):
        # b = exp(0.5*logv)
        # nll = |r|/b + log b = |r|*exp(-0.5*logv) + 0.5*logv
        nll_map = abs_r * torch.exp(-0.5 * logv) + 0.5 * logv
        nll_map = nll_map * valid_1_4

        num = valid_1_4.sum().clamp_min(1e-6)
        step = nll_map.sum() / num

        total = total + w_i * step
        denom = denom + w_i

    return total / denom

def flow_loss_raft_style(
    preds_flow: torch.Tensor,               # (B,T,2,H,W)
    gt_flow: torch.Tensor,                  # (B,2,H,W) or (B,T,2,H,W)
    gt_valid: Optional[torch.Tensor] = None,# (B,1,H,W) or (B,H,W) or (B,T,1,H,W) or (B,T,H,W) or None
    gamma: float = 0.9,
    loss: str = "charb",                    # "charb" | "smoothl1" | "l1"
    charb_eps: float = 1e-3,
    invalid_value: float = 400.0,
    logs: Optional[dict] = None,
) -> torch.Tensor:
    """Compute RAFT-style optical flow loss with validity masking.

    Args:
        preds_flow: Predicted flow tensor (B, T, 2, H, W).
        gt_flow: Ground-truth flow tensor.
        gt_valid: Optional validity mask.
        gamma: Iteration weighting factor.
        loss: Loss type ("charb", "smoothl1", "l1").
        charb_eps: Epsilon for Charbonnier loss.
        invalid_value: Value threshold for auto-validity.
        logs: Optional dict to populate with per-iter metrics.

    Returns:
        Scalar loss tensor.
    """
    # shape check
    assert preds_flow.dim() == 5, "preds_flow must be (B,T,2,H,W)"
    B, T, C, H, W = preds_flow.shape
    assert C == 2, "preds_flow must be (B,T,2,H,W)"

    assert gt_flow.dim() in (4, 5), "gt_flow must be (B,2,H,W) or (B,T,2,H,W)"
    assert gt_flow.shape[0] == B and gt_flow.shape[-3:] == (2, H, W), "gt_flow shape mismatch"

    dtype = gt_flow.dtype
    device = gt_flow.device

    # broadcast / align gt_flow to (B,T,2,H,W)
    if gt_flow.dim() == 4:
        gt = gt_flow[:, None].expand(B, T, 2, H, W)
    else:
        assert gt_flow.shape[1] == T, "gt_flow has T mismatch with preds_flow"
        gt = gt_flow

    # valid mask: make it (B,T,1,H,W) float
    if gt_valid is not None:
        if gt_valid.dim() == 3:
            # (B,H,W)
            v = (gt_valid > 0).to(dtype).unsqueeze(1).unsqueeze(2)  # (B,1,1,H,W)
            valid = v.expand(B, T, 1, H, W)
        elif gt_valid.dim() == 4:
            # (B,1,H,W) or (B,T,H,W)
            if gt_valid.size(1) == 1:
                v = (gt_valid > 0).to(dtype).unsqueeze(1)           # (B,1,1,H,W)
                valid = v.expand(B, T, 1, H, W)
            elif gt_valid.size(1) == T:
                valid = (gt_valid > 0).to(dtype).unsqueeze(2)       # (B,T,1,H,W)
            else:
                raise AssertionError("gt_valid dim=4 must be (B,1,H,W) or (B,T,H,W)")
        elif gt_valid.dim() == 5:
            # (B,T,1,H,W)
            assert gt_valid.size(1) == T, "gt_valid has T mismatch with preds_flow"
            assert gt_valid.size(2) == 1, "gt_valid dim=5 must have channel=1 (B,T,1,H,W)"
            valid = (gt_valid > 0).to(dtype)
        else:
            raise AssertionError("gt_valid must be (B,H,W)/(B,1,H,W)/(B,T,H,W)/(B,T,1,H,W)")
    else:
        fx = gt[:, :, 0:1]
        fy = gt[:, :, 1:2]
        ok = (
            (fx.abs() < invalid_value * 0.99)
            & (fy.abs() < invalid_value * 0.99)
            & (~torch.isnan(fx))
            & (~torch.isnan(fy))
        )
        valid = ok.to(dtype)  # (B,T,1,H,W)

    # error map: (B,T,1,H,W)
    if loss == "charb":
        diff = preds_flow - gt
        err = torch.sqrt(diff.square().sum(dim=2, keepdim=True) + (charb_eps * charb_eps))
    elif loss == "smoothl1":
        comp = F.smooth_l1_loss(preds_flow, gt, reduction="none")  # (B,T,2,H,W)
        err = comp.sum(dim=2, keepdim=True)
    elif loss == "l1":
        err = (preds_flow - gt).abs().sum(dim=2, keepdim=True)
    else:
        raise RuntimeError("unknown loss: " + loss)

    # per-iter mean over valid pixels: step_loss[T]
    reduce_dims = (0, 2, 3, 4)  # sum over B, channel(1), H, W
    valid_sum = valid.sum(dim=reduce_dims).clamp_min(1e-6)         # (T,)
    step_loss = (err * valid).sum(dim=reduce_dims) / valid_sum     # (T,)

    # ---- logging: per-iter optical flow error ----
    if logs is not None:
        # key例: flow/err_t00, flow/err_t01, ...
        # detachしてfloat化（tensorのまま保持したいなら .item() を外す）
        for i in range(T):
            logs[f"L_flow/err_t{i:02d}"] = step_loss[i].detach()

        # ついでに最終iterや平均も欲しければ（任意）
        logs["L_flow/err_last"] = step_loss[-1].detach()
        logs["L_flow/err_mean"] = step_loss.mean().detach()

    # RAFT-style weights: gamma^(T-1-i)
    exps = torch.arange(T - 1, -1, -1, device=device, dtype=step_loss.dtype)
    w = torch.pow(torch.as_tensor(gamma, device=device, dtype=step_loss.dtype), exps)  # (T,)

    total = (step_loss * w).sum()
    denom = w.sum().clamp_min(1e-6)
    return total / denom

@torch.no_grad()
def disparity_epe_sums(
    pred_1_4: Union[torch.Tensor, List[torch.Tensor]],  # (B,1,H/4,W/4) もしくはリスト
    gt_disp_1x: torch.Tensor,                           # (B,1,H,W)
    gt_valid_1x: Optional[torch.Tensor] = None,         # (B,1,H,W) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute summed EPE for disparity.

    Returns:
        sum_err: Sum of absolute errors under the valid mask.
        sum_valid: Sum of valid pixels.
    """
    # 反復予測なら最後を使う
    pred = pred_1_4[-1] if isinstance(pred_1_4, (list, tuple)) else pred_1_4
    

    # 形状を合わせる
    if pred.shape[-2:] != gt_disp_1x.shape[-2:]:
        pred = F.interpolate(pred, size=gt_disp_1x.shape[-2:], mode="bilinear", align_corners=False) * 4.0
    err = (pred - gt_disp_1x).abs()
    sum_err   = (err * gt_valid_1x.to(gt_disp_1x.dtype)).sum()
    sum_valid = gt_valid_1x.sum()
    return sum_err, sum_valid


def down4_like(x: torch.Tensor, ref: torch.Tensor, mode="bilinear") -> torch.Tensor:
    """Downsample tensor to match reference 1/4 resolution.

    Args:
        x: Input tensor (B, C, H, W).
        ref: Reference tensor at 1/4 resolution.
        mode: Interpolation mode.

    Returns:
        Downsampled tensor.
    """
    _, _, H4, W4 = ref.shape
    if mode == "nearest":
        return F.interpolate(x, size=(H4, W4), mode="nearest")
    if mode == "area":
        return F.adaptive_avg_pool2d(x, (H4, W4))
    # bilinear (デフォルト)
    return F.interpolate(x, size=(H4, W4), mode="bilinear", align_corners=False)


def check_yx(yx, H, W, name="yx"):
    """Validate yx coordinates against image bounds.

    Args:
        yx: Tensor of y/x coordinates.
        H: Image height.
        W: Image width.
        name: Label used in error messages.

    Raises:
        RuntimeError: If any coordinate is out of bounds.
    """
    y = yx[..., 0]
    x = yx[..., 1]
    bad = (y < 0) | (y >= H) | (x < 0) | (x >= W)
    if bad.any():
        i = torch.nonzero(bad, as_tuple=False)[0]
        raise RuntimeError(f"{name} OOB: y={int(y[i])}, x={int(x[i])}, H={H}, W={W}")


def log_scalars(writer, logs: dict, step: int, prefix: str = "train"):
    """Log scalar tensors to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        logs: Dictionary of scalar tensors.
        step: Global step value.
        prefix: Tag prefix for scalar names.
    """
    # logs の値は tensor なので .item() で Python float に
    for k, v in logs.items():
        writer.add_scalar(f"{prefix}/{k}", v.item(), step)


class _BaseMeter:
    """Base interface for metric meters."""
    def reset(self): ...
    def all_reduce_(self): ...


class AverageMeter(_BaseMeter):
    """Weighted average meter: sum(val * n) / sum(n)."""
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0.0
    def update(self, val: float, n: float = 1.0):
        self.sum += float(val) * float(n)
        self.cnt += float(n)
    @property
    def avg(self) -> float:
        return self.sum / max(self.cnt, 1e-12)
    def reset(self):
        self.sum = 0.0; self.cnt = 0.0
    def all_reduce_(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor(
            [self.sum, self.cnt],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum, self.cnt = float(t[0].item()), float(t[1].item())


class SumCountMeter(_BaseMeter):
    """Sum/count meter where avg = sum / count."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0
    def update_sc(self, sum_val: float, count: float):
        self.sum += float(sum_val)
        self.count += float(count)
    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1e-12)
    def reset(self):
        self.sum = 0.0; self.count = 0.0
    def all_reduce_(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor(
            [self.sum, self.count],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum, self.count = float(t[0].item()), float(t[1].item())

class DictMeters:
    """Container for multiple meters keyed by name."""
    def __init__(self):
        self.m: Dict[str, _BaseMeter] = {}
    def add_avg(self, name: str):
        self.m[name] = AverageMeter()
    def add_sc(self, name: str):
        self.m[name] = SumCountMeter()
    def update_avg(self, name: str, val: float, n: float = 1.0):
        cast = self.m[name]; assert isinstance(cast, AverageMeter)
        cast.update(val, n)
    def update_sc(self, name: str, sum_val: float, count: float):
        cast = self.m[name]; assert isinstance(cast, SumCountMeter)
        cast.update_sc(sum_val, count)
    def get(self, name: str) -> _BaseMeter:
        return self.m[name]
    def all_reduce_(self):
        for v in self.m.values(): v.all_reduce_()
    def averages(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.m.items()}


def log_meters_to_tb(writer, meters: DictMeters, step: int, prefix: str):
    """Write meter averages to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        meters: Meter container.
        step: Global step value.
        prefix: Tag prefix for metric names.
    """
    for k, v in meters.m.items():
        writer.add_scalar(f"{prefix}/{k}", v.avg, step)
        

@torch.no_grad()
def _ensure_avg_meters(meters: "DictMeters", logs: dict):
    """Ensure AverageMeter exists for each L_* key in logs.

    Args:
        meters: Meter container.
        logs: Dictionary of logged loss tensors.
    """
    # logs 内の "L_" で始まるキーが meters に無ければ追加
    for k in logs.keys():
        if k.startswith("L_"):
            try:
                # 既にあればスキップ（AverageMeter を持っている前提）
                meters.avg[k]
            except Exception:
                meters.add_avg(k)
        

def _slice_meshes_flat(meshes_flat, valid_k, b_start, b_end):
    """Slice a flat Meshes list by sub-batch range.

    Args:
        meshes_flat: Meshes with length sum(valid_k).
        valid_k: Tensor of shape (B, K) with valid flags.
        b_start: Start batch index.
        b_end: End batch index.

    Returns:
        Sliced Meshes object.
    """
    # 累積して先頭から必要数だけ数える
    counts = valid_k.sum(dim=1).tolist()  # 各バッチの有効インスタンス数 (list[int])
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + int(c))
    start = offs[b_start]
    end   = offs[b_end]
    return meshes_flat[start:end]

def _overlay_mask_rgb(img_bchw, mask_b1hw, color=(0.0, 1.0, 0.0), alpha=0.4):
    """Overlay a single-channel mask on RGB images.

    Args:
        img_bchw: RGB images in (B, 3, H, W) with range [0, 1].
        mask_b1hw: Mask tensor in (B, 1, H, W) with range [0, 1].
        color: Overlay color as RGB tuple.
        alpha: Blend ratio.

    Returns:
        Overlayed RGB images.
    """
    B, _, H, W = img_bchw.shape
    c = torch.tensor(color, device=img_bchw.device, dtype=img_bchw.dtype).view(1,3,1,1)
    m = mask_b1hw.clamp(0,1)
    return img_bchw * (1 - alpha*m) + c * (alpha*m)
        
# ==== 共通ヘルパ（重複除去；振る舞いは既存通り） ===================================

def _init_window_meter():
    """Initialize windowed loss accumulators.

    Returns:
        Tuple of (window_sum dict, window_count int).
    """
    return {
        "loss": 0.0,
        "loss_disp":  0.0,
        "depth_mae":  0.0,
        "L_ctr":  0.0,
        "L_mask":  0.0,
        "L_pos":  0.0,
        "L_rot":  0.0,
        "L_cls":  0.0,
        "L_adds": 0.0,
    }, 0

def _reset_window_meter(window_sum, window_cnt_ref):
    """Reset windowed sums.

    Args:
        window_sum: Dictionary of accumulated values.
        window_cnt_ref: Placeholder for count (kept for API compatibility).

    Returns:
        Reset count value.
    """
    for k in window_sum:
        window_sum[k] = 0.0
    return 0  # new count

def _update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs):
    """Update windowed sums with new loss values.

    Args:
        window_sum: Dictionary of accumulated values.
        window_cnt: Current count.
        loss: Total loss tensor.
        loss_disp: Disparity loss tensor.
        depth_mae: Depth MAE tensor.
        logs: Dictionary of additional loss components.

    Returns:
        Updated window count.
    """
    window_sum["loss"]      += float(loss.detach().item())
    window_sum["loss_disp"] += float(loss_disp.detach().item())
    window_sum["depth_mae"] += float(depth_mae.detach().item())
    # logs に来たものをすべて動的に反映（"L_" で始まる Tensor のみ集計）
    for k, v in logs.items():
        if torch.is_tensor(v):
            if k not in window_sum:
                window_sum[k] = 0.0
            window_sum[k] += float(v.detach().item())

    return window_cnt + 1

def _flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train"):
    """Flush windowed training metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        window_sum: Dictionary of accumulated values.
        window_cnt: Number of steps accumulated.
        optimizer: Optimizer with learning rate state.
        global_step: Global step value.
        prefix: Tag prefix for metric names.
    """
    if writer is None or window_cnt <= 0:
        return
    denom = max(window_cnt, 1)
    writer.add_scalar(f"{prefix}/loss",       window_sum["loss"]      / denom, global_step)
    writer.add_scalar(f"{prefix}/loss_disp",  window_sum["loss_disp"] / denom, global_step)
    writer.add_scalar(f"{prefix}/depth_mae",  window_sum["depth_mae"] / denom, global_step)
    writer.add_scalar(f"{prefix}/lr", optimizer.param_groups[0]["lr"], global_step)
    
    # 追加/動的な loss 群（"L_" で始まるキーをすべて出力）
    for k, s in window_sum.items():
        if k.startswith("L_"):
            writer.add_scalar(f"{prefix}/{k}", s / denom, global_step)

@torch.no_grad()
def _prepare_iter0_targets(batch, cfg, device):
    """Build iter0 targets shared by train/val.

    Args:
        batch: Input batch dictionary.
        cfg: Training configuration dictionary.
        device: Target device.

    Returns:
        Tuple of (gt_iter0, instance_weight_map, mask).
    """
    gt_iter0 = {}
    mask = batch['instance_seg'].to(device, non_blocking=True).unsqueeze(1) > 0
    H4 = cfg['data']['height'] // 4
    W4 = cfg['data']['width']  // 4

    gt_iter0['mask_1_4']    = F.interpolate(mask.to(torch.float32), size=(H4, W4), mode="nearest")
    gt_iter0['center_heat'] = batch['center_mask'].to(device, non_blocking=True).unsqueeze(1)
    gt_iter0['weight_map']  = batch['weight_map'].to(device, non_blocking=True).unsqueeze(1)
    gt_iter0['pos_1_4']     = batch['pos_map'].to(device, non_blocking=True) # mm
    gt_iter0['cls_target']  = batch['class_map'].to(device, non_blocking=True)
    gt_iter0['R_1_4']       = batch['rot_map'].to(device, non_blocking=True)
    gt_iter0['diameters']   = batch['diameters'].to(device, non_blocking=True)

    instance_weight_map = batch['instance_weight_map'].to(device, non_blocking=True).unsqueeze(2)  # (B,K,1,H4,W4)
    B, K, _, H4m, W4m = instance_weight_map.shape
    flat = instance_weight_map.view(B, K, -1)
    argm = flat.argmax(-1)  # (B,K)
    ys, xs = argm // (W4m), argm % (W4m)
    gt_iter0['peaks_yx'] = torch.stack([ys, xs], dim=-1)  # (B,K,2)

    # 追加で必要なもの
    model_points = batch['model_points'].to(device, non_blocking=True)
    gt_iter0['model_points'] = model_points
    gt_iter0['weight_map_inst'] = instance_weight_map
    return gt_iter0, instance_weight_map, mask

@torch.no_grad()
def _prepare_stereo_and_cam(batch, device):
    """Prepare stereo inputs and camera parameters.

    Args:
        batch: Input batch dictionary.
        device: Target device.

    Returns:
        Tuple of tensors (stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k).
    """
    stereo  = batch["stereo"].to(device, non_blocking=True)
    depth   = batch["depth"].to(device, non_blocking=True).unsqueeze(1) * 1000.0  # m -> mm
    disp_gt = batch['disparity'].to(device, non_blocking=True).unsqueeze(1)

    left_k  = batch['camera_params']['left_k'].to(device, non_blocking=True).unsqueeze(1)
    right_k = batch['camera_params']['right_k'].to(device, non_blocking=True).unsqueeze(1)
    k_pair  = torch.cat([left_k, right_k], dim=1)  # (B,2,3,3)
    baseline = batch['camera_params']['base_dist_mm'].to(device, non_blocking=True)

    input_mask = batch['input_seg'].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    return stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k


def _log_disp_and_depth(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    disp_preds: List[torch.Tensor],
    disp_gt: torch.Tensor,
    depth_pred_1x: torch.Tensor,
    depth_gt_mm: torch.Tensor,
    vis_mask: torch.Tensor,
    n_images: int,
) -> torch.Tensor:
    """Log disparity/depth visualizations and return GT depth viz.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        disp_preds: List of predicted disparity maps.
        disp_gt: Ground-truth disparity.
        depth_pred_1x: Predicted depth at full resolution.
        depth_gt_mm: Ground-truth depth in mm.
        vis_mask: Valid mask for visualization.
        n_images: Number of images to visualize.

    Returns:
        Depth visualization tensor for GT.
    """
    disp_vis = (
        F.interpolate(
            disp_preds[-1][:n_images],
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        * 4.0
    )
    depth_vis = F.interpolate(
        depth_pred_1x[:n_images],
        size=(256, 256),
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
    return vis_depth_gt


def _log_pose_and_silhouette(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    k_pair: torch.Tensor,
    gt_iter0: Dict[str, torch.Tensor],
    meshes: Meshes,
    valid_k: torch.Tensor,
    renderer: torch.nn.Module,
    n_images: int,
) -> Tuple[torch.Tensor, torch.Tensor, Meshes, Tuple[int, int]]:
    """Log pose axes, silhouette overlays, and return artifacts.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        pred: Prediction dictionary.
        k_pair: Camera intrinsics for left/right.
        gt_iter0: Ground-truth dictionary for iter0.
        meshes: Meshes object for rendering.
        valid_k: Valid instance flags.
        renderer: Silhouette renderer module.
        n_images: Number of images to visualize.

    Returns:
        Tuple of (overlay_gt, projected_gt, meshes_flat_n, image_size).
    """
    H, W = stereo.shape[-2], stereo.shape[-1]
    image_size = (H, W)

    pred_instances = pred['instances']
    pred_pose = rot_utils.compose_T_from_Rt(
        pred_instances['R'],
        pred_instances['t'],
        pred_instances['valid'],
    )
    vis_pred_pose = pred_pose[:n_images].detach().clone()

    _, t_gt, R_gt = loss_functions.pick_representatives_mask_only(
        inst_mask=gt_iter0["pos_1_4"][:, -1:] > 0,
        valid_map=gt_iter0["pos_1_4"][:, -1:] > 0,
        pos_map=gt_iter0["pos_1_4"],
        rot_map=gt_iter0["R_1_4"],
        n_erode=1,
    )
    gt_pose = rot_utils.compose_T_from_Rt(R_gt, t_gt, pred_instances['valid'])
    vis_gt_pose = gt_pose[:n_images].detach().clone()

    meshes_flat_n = _slice_meshes_flat(meshes, valid_k, 0, n_images)

    pred_r = renderer(
        meshes_flat=meshes_flat_n,
        T_cam_obj=vis_pred_pose,
        K_left=k_pair[:n_images, 0],
        valid_k=valid_k[:n_images],
        image_size=image_size,
    )
    gt_r = renderer(
        meshes_flat=meshes_flat_n,
        T_cam_obj=vis_gt_pose,
        K_left=k_pair[:n_images, 0],
        valid_k=valid_k[:n_images],
        image_size=image_size,
    )
    sil_pred = pred_r["silhouette"]
    sil_gt = gt_r["silhouette"]

    overlay_pred = _overlay_mask_rgb(
        stereo[:n_images, :3],
        sil_pred,
        color=(0.0, 1.0, 0.0),
        alpha=0.45,
    )
    overlay_gt = _overlay_mask_rgb(
        stereo[:n_images, :3],
        sil_gt,
        color=(1.0, 0.0, 0.0),
        alpha=0.45,
    )
    overlay_both = _overlay_mask_rgb(
        _overlay_mask_rgb(stereo[:n_images, :3], sil_gt, color=(1.0, 0.0, 0.0), alpha=0.45),
        sil_pred,
        color=(0.0, 1.0, 0.0),
        alpha=0.45,
    )

    init_projected = draw_axes_on_images_bk(
        overlay_pred,
        k_pair[:4, 0],
        vis_pred_pose[0:4],
        axis_len=30.0,
        valid=valid_k[0:4],
    )
    projected_gt = draw_axes_on_images_bk(
        overlay_gt,
        k_pair[:4, 0],
        vis_gt_pose[0:4],
        axis_len=30.0,
        valid=valid_k[0:4],
    )

    grid = vutils.make_grid(
        torch.cat([stereo[:4, :3], init_projected, projected_gt, overlay_both], dim=0),
        nrow=4,
        normalize=True,
        scale_each=True,
    )
    writer.add_image(f"{prefix}/vis_init_pose", grid, step_value)

    grid_mask = make_mask_overlay_grid(
        stereo[:4, :3],
        pred["mask_logits"][:4],
        gt_iter0["mask_1_4"][:n_images],
        nrow=4,
        alpha=0.45,
    )
    writer.add_image(f"{prefix}/vis_mask_overlay", grid_mask, step_value)

    grid_ctr = make_center_overlay_grid(
        stereo[:4, :3],
        pred["center_logits"][:4],
        gt_iter0["center_heat"][:4],
        nrow=4,
        alpha=0.60,
    )
    writer.add_image(f"{prefix}/vis_center_overlay", grid_ctr, step_value)
    return overlay_gt, projected_gt, meshes_flat_n, image_size


def _log_final_pose(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    k_pair: torch.Tensor,
    gt_iter0: Dict[str, torch.Tensor],
    valid_k: torch.Tensor,
    renderer: torch.nn.Module,
    overlay_gt: torch.Tensor,
    projected_gt: torch.Tensor,
    meshes_flat_n: Meshes,
    image_size: Tuple[int, int],
    n_images: int,
) -> None:
    """Log final pose update overlays when present.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        pred: Prediction dictionary.
        k_pair: Camera intrinsics for left/right.
        gt_iter0: Ground-truth dictionary for iter0.
        valid_k: Valid instance flags.
        renderer: Silhouette renderer module.
        overlay_gt: GT silhouette overlay.
        projected_gt: GT axis overlay.
        meshes_flat_n: Subset of meshes for visualization.
        image_size: Image size tuple.
        n_images: Number of images to visualize.
    """
    if (
        ("rot_maps" not in pred)
        or ("pos_maps" not in pred)
        or len(pred["rot_maps"]) == 0
        or len(pred["pos_maps"]) == 0
    ):
        return

    R_map_final = pred["rot_maps"][-1][:n_images]
    t_map_final = pred["pos_maps"][-1][:n_images]

    wfg = torch.ones_like(gt_iter0["mask_1_4"][:n_images])
    r_final, t_final, is_valid, _, _ = rot_utils.pose_from_maps_auto(
        rot_map=R_map_final,
        pos_map=t_map_final,
        wfg=wfg,
        Wk_1_4=gt_iter0['weight_map_inst'][:n_images],
        min_px=10,
        min_wsum=1e-6,
    )
    T_final = rot_utils.compose_T_from_Rt(r_final, t_final, is_valid)
    T_final_vis = T_final[:n_images].detach().clone()

    final_r = renderer(
        meshes_flat=meshes_flat_n,
        T_cam_obj=T_final_vis,
        K_left=k_pair[:n_images, 0],
        valid_k=valid_k[:n_images],
        image_size=image_size,
    )
    sil_final = final_r["silhouette"]

    overlay_final = _overlay_mask_rgb(
        stereo[:n_images, :3],
        sil_final,
        color=(0.0, 1.0, 0.0),
        alpha=0.45,
    )
    final_axes = draw_axes_on_images_bk(
        overlay_final,
        k_pair[:n_images, 0],
        T_final_vis[:n_images],
        axis_len=30.0,
        valid=valid_k[:n_images],
    )
    overlay_both = _overlay_mask_rgb(overlay_gt, sil_final, color=(0.0, 1.0, 0.0), alpha=0.45)

    grid_final = vutils.make_grid(
        torch.cat([stereo[:n_images, :3], final_axes, projected_gt, overlay_both], dim=0),
        nrow=4,
        normalize=True,
        scale_each=True,
    )
    writer.add_image(f"{prefix}/vis_final_pose", grid_final, step_value)


def _log_depth_progress(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    depth_gt_mm: torch.Tensor,
    vis_mask: torch.Tensor,
    n_images: int,
) -> None:
    """Log depth prediction progress across iterations.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        pred: Prediction dictionary.
        depth_gt_mm: Ground-truth depth in mm.
        vis_mask: Valid mask for visualization.
        n_images: Number of images to visualize.
    """
    viz_z_list = []
    for iter_z in pred["Z_mm_list"]:
        resized_z = F.interpolate(
            iter_z[:n_images],
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        ones = torch.ones_like(vis_mask)
        viz_z_list.append(visualize_mono_torch(resized_z, ones, depth_gt_mm[:n_images], vis_mask))

    show_iters = [0, 1, 2, -1]
    imgs = [stereo[:n_images, :3]]
    for i in show_iters:
        if i < len(viz_z_list):
            imgs.append(viz_z_list[i][0])
    imgs.append(viz_z_list[-1][1])

    grid = vutils.make_grid(torch.cat(imgs, dim=0), nrow=4, normalize=True, scale_each=True)
    writer.add_image(f"{prefix}/depth_progress", grid, step_value)


def _log_flow_visuals(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    gt_iter0: Dict[str, torch.Tensor],
    depth_gt_mm: torch.Tensor,
    vis_depth_gt: torch.Tensor,
    n_images: int,
) -> None:
    """Log optical flow visualizations.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        pred: Prediction dictionary.
        gt_iter0: Ground-truth dictionary for iter0.
        depth_gt_mm: Ground-truth depth in mm.
        vis_depth_gt: Depth visualization tensor for GT.
        n_images: Number of images to visualize.
    """
    flow_pred, flow_gt = None, None
    for key in ["flow64", "flow", "flow_pred", "flow_list"]:
        if key in pred:
            if key == "flow_list" and isinstance(pred[key], (list, tuple)) and len(pred[key]) > 0:
                flow_pred = pred[key][-1][:n_images]
            else:
                flow_pred = pred[key][:n_images]
            break

    if "flow_gt" in pred:
        flow_gt = pred["flow_gt"][:n_images]
    elif "flow_gt" in gt_iter0:
        flow_gt = gt_iter0["flow_gt"][:n_images]

    if (flow_pred is not None) and (flow_gt is not None):
        rgb_pred, rgb_gt, vmax_used = flow_utils.visualize_flow_pair_same_scale(
            flow_pred,
            flow_gt,
            invalid_value=400.0,
            vmax=None,
            quantile=0.95,
        )
        rgb_pred = F.interpolate(rgb_pred, size=(256, 256), mode="bilinear", align_corners=False)
        rgb_gt = F.interpolate(rgb_gt, size=(256, 256), mode="bilinear", align_corners=False)

        grid_flow = vutils.make_grid(
            torch.cat([stereo[:n_images, :3], rgb_pred, rgb_gt], dim=0),
            nrow=4,
            normalize=True,
            scale_each=True,
        )
        writer.add_image(f"{prefix}/vis_flow_pred_gt_same_scale", grid_flow, step_value)
        writer.add_scalar(f"{prefix}/flow_vmax_used", float(vmax_used.item()), step_value)
        return

    if flow_pred is not None:
        rgb_pred = flow_utils.visualize_flow_torch(flow_pred, invalid_value=400.0)
        rgb_pred = F.interpolate(rgb_pred, size=(256, 256), mode="bilinear", align_corners=False)
        grid_flow_pred = vutils.make_grid(
            torch.cat([stereo[:n_images, :3], rgb_pred], dim=0),
            nrow=4,
            normalize=True,
            scale_each=True,
        )
        writer.add_image(f"{prefix}/vis_flow_pred", grid_flow_pred, step_value)
    if flow_gt is not None:
        rgb_gt = flow_utils.visualize_flow_torch(flow_gt, invalid_value=400.0)
        rgb_gt = F.interpolate(rgb_gt, size=(256, 256), mode="bilinear", align_corners=False)
        warped_depth = flow_utils.warp_depth_with_forward_flow(
            depth_gt_mm[:n_images],
            F.interpolate(flow_gt, size=(256, 256), mode="bilinear", align_corners=False) * 4,
        )
        flow_mask = warped_depth > 1e-3
        vis_warped_depth, _ = visualize_mono_torch(warped_depth, flow_mask, warped_depth, flow_mask)
        grid_flow_gt = vutils.make_grid(
            torch.cat([stereo[:n_images, :3], rgb_gt, vis_warped_depth, vis_depth_gt], dim=0),
            nrow=4,
            normalize=True,
            scale_each=True,
        )
        writer.add_image(f"{prefix}/vis_flow_gt", grid_flow_gt, step_value)


def _log_images_common(
    writer: SummaryWriter,
    step_value: int,
    prefix: str,
    stereo: torch.Tensor,
    disp_gt: torch.Tensor,
    depth_gt_mm: torch.Tensor,
    mask: torch.Tensor,
    disp_preds: List[torch.Tensor],
    depth_pred_1x: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    k_pair: torch.Tensor,
    gt_iter0: Dict[str, torch.Tensor],
    meshes: Meshes,
    valid_k: torch.Tensor,
    renderer: torch.nn.Module,
) -> None:
    """Log common training/validation visualizations.

    Args:
        writer: TensorBoard SummaryWriter.
        step_value: Global step or epoch.
        prefix: Tag prefix for logs.
        stereo: Stereo pair tensor.
        disp_gt: Ground-truth disparity.
        depth_gt_mm: Ground-truth depth in mm.
        mask: Valid mask.
        disp_preds: List of predicted disparity maps.
        depth_pred_1x: Predicted depth at full resolution.
        pred: Prediction dictionary.
        k_pair: Camera intrinsics for left/right.
        gt_iter0: Ground-truth dictionary for iter0.
        meshes: Meshes object for rendering.
        valid_k: Valid instance flags.
        renderer: Silhouette renderer module.
    """
    n_images = min(4, stereo.size(0))
    vis_mask = mask[:n_images].to(torch.float32)

    vis_depth_gt = _log_disp_and_depth(
        writer=writer,
        step_value=step_value,
        prefix=prefix,
        stereo=stereo,
        disp_preds=disp_preds,
        disp_gt=disp_gt,
        depth_pred_1x=depth_pred_1x,
        depth_gt_mm=depth_gt_mm,
        vis_mask=vis_mask,
        n_images=n_images,
    )

    overlay_gt, projected_gt, meshes_flat_n, image_size = _log_pose_and_silhouette(
        writer=writer,
        step_value=step_value,
        prefix=prefix,
        stereo=stereo,
        pred=pred,
        k_pair=k_pair,
        gt_iter0=gt_iter0,
        meshes=meshes,
        valid_k=valid_k,
        renderer=renderer,
        n_images=n_images,
    )

    _log_final_pose(
        writer=writer,
        step_value=step_value,
        prefix=prefix,
        stereo=stereo,
        pred=pred,
        k_pair=k_pair,
        gt_iter0=gt_iter0,
        valid_k=valid_k,
        renderer=renderer,
        overlay_gt=overlay_gt,
        projected_gt=projected_gt,
        meshes_flat_n=meshes_flat_n,
        image_size=image_size,
        n_images=n_images,
    )

    _log_depth_progress(
        writer=writer,
        step_value=step_value,
        prefix=prefix,
        stereo=stereo,
        pred=pred,
        depth_gt_mm=depth_gt_mm,
        vis_mask=vis_mask,
        n_images=n_images,
    )

    _log_flow_visuals(
        writer=writer,
        step_value=step_value,
        prefix=prefix,
        stereo=stereo,
        pred=pred,
        gt_iter0=gt_iter0,
        depth_gt_mm=depth_gt_mm,
        vis_depth_gt=vis_depth_gt,
        n_images=n_images,
    )


@torch.no_grad()
def _build_flow_gts(
    pred: Dict[str, torch.Tensor],
    gt_point_map_14: torch.Tensor,
    left_k_14: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
) -> torch.Tensor:
    """Build flow ground truth tensors for each predicted pose update.

    Args:
        pred: Prediction dictionary containing Rk_list and tk_list.
        gt_point_map_14: Ground-truth point map at 1/4 resolution.
        left_k_14: Left camera intrinsics at 1/4 resolution.
        R_gt: Ground-truth rotation matrices.
        t_gt: Ground-truth translation vectors in mm.

    Returns:
        Flow tensors stacked as (B, T, 2, H/4, W/4).
    """
    flow_gts = []
    for Rk, tk in zip(pred['Rk_list'][:-1], pred['tk_list'][:-1]):
        flow_gt = flow_utils.gt_flow_from_pointmap_and_poses(
            point_map=gt_point_map_14,
            K=left_k_14,
            R_src=R_gt,
            t_src_mm=t_gt,
            R_dst=Rk,
            t_dst_mm=tk,
            point_frame="camera",
            point_unit_m=False,
            invalid_value=400.0,
        )
        flow_gts.extend([flow_gt, flow_gt, flow_gt, flow_gt])
    return torch.stack(flow_gts, dim=1)


def train_one_epoch(
    model: SSCFlow2,
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
    """Train the model for one epoch.

    Args:
        model: Training model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        scaler: AMP gradient scaler.
        epoch: Current epoch index.
        cfg: Training configuration dictionary.
        writer: TensorBoard SummaryWriter.
        device: Target device.
        scheduler: Optional LR scheduler.
        sched_step_when: Scheduler step mode.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    l1 = torch.nn.L1Loss()
    log_every = cfg["train"]["log_interval"]

    window_sum, window_cnt = _init_window_meter()
    renderer = SilhouetteDepthRenderer(
        # faces_per_pixel=cfg['model']['faces_per_pixel'],
        # blur_sigma=cfg['model']['blur_sigma'],
        # blur_gamma=cfg['model']['blur_gamma'],
    ).to(device)
    global_step = epoch * len(loader)
    for it, batch in enumerate(loader):
        # ---- 入力・GT 準備（共通化） ----
        stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k = _prepare_stereo_and_cam(
            batch, device
        )
        gt_iter0, instance_weight_map, mask = _prepare_iter0_targets(batch, cfg, device)
        gt_iter0["K_left_1x"] = left_k[:, 0]
        meshes = batch['meshes'].to(device)
        valid  = batch['valid_k'].to(device)
        gt_iter0['valid_k'] = valid

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=cfg['train']['amp']):
            with_shape_constraint = global_step > (cfg['loss'].get('update_warmup_steps', 3000) * 10)
            pred = model(
                stereo, input_mask, k_pair, baseline, meshes, use_gt_peaks=True,
                gt_center_1_4=gt_iter0['weight_map'], gt_Wk_1_4=instance_weight_map,
                gt_mask_1_4=gt_iter0['mask_1_4'],
                with_shape_constraint=with_shape_constraint,
                iters=cfg['model']['n_iter'],
            )
            with torch.no_grad():
                left_k_14 = rot_utils.scale_K(k_pair[:, 0], 4)
                depth_14 = F.interpolate(
                    depth,
                    size=(gt_iter0["pos_1_4"].shape[-2], gt_iter0["pos_1_4"].shape[-1]),
                    mode="nearest",
                )
                gt_point_map_14 = rot_utils.depth_to_pointmap_from_K(depth_14, left_k_14)
                _, t_gt, R_gt = loss_functions.pick_representatives_mask_only(
                    inst_mask=gt_iter0["pos_1_4"][:, -1:] > 0,
                    valid_map=gt_iter0["pos_1_4"][:, -1:] > 0,
                    pos_map=gt_iter0["pos_1_4"],
                    rot_map=gt_iter0["R_1_4"],
                    n_erode=1,
                )
                gt_point_map_14 = gt_point_map_14 * (gt_iter0['mask_1_4'] > 0)
                flow_gts = _build_flow_gts(pred, gt_point_map_14, left_k_14, R_gt, t_gt)
            gt_iter0["flow_gts"] = flow_gts  # (B,2,H/4,W/4)
            gt_iter0['flow_gt'] = flow_gts[:, -1]
            disp_pred   = pred["disp_preds"]
            disp_log_var_preds   = pred["disp_log_var_preds"]
            depth_pred  = pred["depth_1_4"]
            loss_disp = disparity_nll_laplace_raft_style(
                disp_pred,
                disp_log_var_preds,
                disp_gt,
                mask,
            )
            loss_mh, logs = loss_functions.loss_step_iter0(
                pred,
                gt_iter0,
                cfg["loss"]["w_ctr"],
                cfg["loss"]["w_mask"],
                cfg["loss"]["w_pos"],
                cfg["loss"]["w_rot"],
                cfg["loss"]["w_cls"],
                cfg["loss"]["w_adds"],
                cfg["loss"]['w_rot_update'],
                cfg["loss"]['w_pos_update'],
                cfg["loss"]['w_adds_update'],
                cfg["loss"]['update_gamma'],
            )
            loss = cfg["loss"]["w_disp"] * loss_disp + loss_mh
            if 'flow_preds' in pred:
                loss_flow = flow_loss_raft_style(pred['flow_preds'], flow_gts, logs=logs)
                loss = loss + cfg["loss"]["w_flow"] * loss_flow

        pred_instances = pred['instances']

        # ---- 逆伝播 / 最適化 ----
        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        # ---- メトリクス ----
        depth_pred_1x = F.interpolate(depth_pred, size=(256, 256), mode="bilinear", align_corners=False)
        depth_mae = l1(depth_pred_1x[mask], depth[mask])

        # ---- スケジューラ（AMPのスキップ検知） ----
        did_optim_step = scaler.get_scale() >= prev_scale
        if scheduler is not None and sched_step_when == "step" and did_optim_step:
            scheduler.step()

        total_loss += loss.detach()
        window_cnt = _update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs)

        # ---- ログ（スカラーと画像） ----
        global_step = epoch * len(loader) + it
        if writer is not None and dist_utils.is_main_process() and (global_step % log_every == 0):
            _flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train")
            with torch.no_grad():
                _log_images_common(
                    writer,
                    global_step,
                    "train",
                    stereo,
                    disp_gt,
                    depth,
                    mask,
                    disp_pred,
                    depth_pred_1x,
                    pred,
                    k_pair,
                    gt_iter0,
                    meshes,
                    pred_instances['valid'],
                    renderer,
                )
            # 窓リセット
            window_cnt = _reset_window_meter(window_sum, window_cnt)

    return total_loss.item() / max(1, len(loader))


@torch.no_grad()
def validate(
    model: SSCFlow2,
    loader: DataLoader,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter],
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation and return metrics.

    Args:
        model: Training model.
        loader: Validation DataLoader.
        epoch: Current epoch index.
        cfg: Training configuration dictionary.
        writer: TensorBoard SummaryWriter.
        device: Target device.

    Returns:
        Tuple of (disparity EPE, pose metric placeholder).
    """
    model.eval()
    meters = DictMeters()
    meters.add_sc("disp_epe")
    meters.add_sc("depth_mae")
    for loss_str in ["L", "L_ctr", "L_mask", "L_pos", "L_rot", "L_cls", "L_adds"]:
        meters.add_avg(loss_str)
    renderer = SilhouetteDepthRenderer(
        # faces_per_pixel=cfg['model']['faces_per_pixel'],
        # blur_sigma=cfg['model']['blur_sigma'],
        # blur_gamma=cfg['model']['blur_gamma'],
    ).to(device)
    for it, batch in enumerate(loader):
        # ---- 入力・GT 準備（共通化） ----
        stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k = _prepare_stereo_and_cam(
            batch, device
        )
        gt_iter0, instance_weight_map, mask = _prepare_iter0_targets(batch, cfg, device)
        gt_iter0["K_left_1x"] = left_k[:, 0]
        meshes = batch['meshes'].to(device)   # 未使用だが元コードのまま保持
        valid  = batch['valid_k'].to(device)  # 〃

        # ---- 推論 & 損失 ----
        pred = model(
                stereo, input_mask, k_pair, baseline, meshes, use_gt_peaks=True,
                gt_center_1_4=gt_iter0['weight_map'], gt_Wk_1_4=instance_weight_map,
                gt_mask_1_4=gt_iter0['mask_1_4'], with_shape_constraint=True, iters=cfg['model']['n_iter']
        )
        pred_instances = pred['instances']
        loss_mh, logs = loss_functions.loss_step_iter0(
            pred,
            gt_iter0,
            cfg["loss"]["w_ctr"],
            cfg["loss"]["w_mask"],
            cfg["loss"]["w_pos"],
            cfg["loss"]["w_rot"],
            cfg["loss"]["w_cls"],
            cfg["loss"]['w_adds'],
            cfg["loss"]['w_rot_update'],
            cfg["loss"]['w_pos_update'],
            cfg["loss"]['w_adds_update'],
            cfg["loss"]['update_gamma'],
        )
        _ensure_avg_meters(meters, logs)              # ★追加：未知の L_* をメータに生やす
        for loss_str, v in logs.items():
            meters.update_avg(loss_str, float(v.item()), n=stereo.size(0))

        # ---- EPE / Depth-MAE ----
        disp_pred_last = pred["disp_preds"][-1]
        sum_err, sum_valid = disparity_epe_sums(disp_pred_last, disp_gt, mask.float())
        meters.update_sc("disp_epe", float(sum_err.item()), float(sum_valid.item()))

        depth_pred = pred["depth_1_4"]
        depth_pred_1x = F.interpolate(depth_pred, size=(256, 256), mode="bilinear", align_corners=False)
        abs_err_sum = (depth_pred_1x - depth).abs()[mask].sum().item()
        pix_count   = mask.sum().item()
        meters.update_sc("depth_mae", float(abs_err_sum), float(pix_count))

        # ---- 画像ログ（最初のバッチのみ） ----
        if writer is not None and dist_utils.is_main_process() and it == 0:
            _log_images_common(
                writer,
                epoch,
                "val",
                stereo,
                disp_gt,
                depth,
                mask,
                [disp_pred_last],
                depth_pred,
                pred,
                k_pair,
                gt_iter0,
                meshes,
                pred_instances['valid'],
                renderer,
            )

    # ---- DDP 集約 / 平均 ----
    meters.all_reduce_()
    avgs = meters.averages()

    if writer is not None and dist_utils.is_main_process():
        log_meters_to_tb(writer, meters, epoch, prefix="val")

    return avgs["disp_epe"], 0.0


def _load_model_state(model, state_dict, strict=False):
    """Load model weights with DDP-awareness.

    Args:
        model: Model or DDP-wrapped model.
        state_dict: State dict to load.
        strict: Whether to enforce strict key matching.

    Returns:
        Tuple of (missing_keys, unexpected_keys).
    """
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    return missing, unexpected


def main():
    """Entry point for stereo training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config TOML", default="configs/small_config.toml")
    parser.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    args = parser.parse_args()

    cfg = load_toml(args.config)
    set_global_seed(42)
    # DDP init
    if args.launcher != 'none':
        distributed = True
        if platform.system() == 'Windows':
            backend = 'gloo'
        else:
            backend = 'nccl'
        dist_utils.init_dist(args.launcher, backend)
    else:
        distributed = False
    dist_utils.setup_for_distributed(is_master=dist_utils.is_main_process())

    # Output / TB
    out_dir = Path(os.path.join(cfg["train"]["output_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")))
    if dist_utils.is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save an "effective config"
        eff_cfg_path = out_dir / "config_used.toml"
        cfg_copy = dict(cfg)  # shallow copy is fine here
        cfg_copy.setdefault("runtime", {})
        cfg_copy["runtime"]["start_time"] = datetime.now().isoformat()
        cfg_copy["runtime"]["world_size"] = dist_utils.get_world_size()
        write_toml(cfg_copy, eff_cfg_path)
    dist_utils.barrier()
    # TensorBoard
    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if dist_utils.is_main_process() else None

    # Device
    device = torch.device("cuda", dist_utils.get_rank()) if torch.cuda.is_available() else torch.device("cpu")

    # Data
    train_loader, val_loader, train_sampler, val_sampler, class_table = make_dataloaders(
        cfg,
        distributed=dist.is_initialized(),
    )
    # Model/opt
    model = build_model(cfg, class_table).to(device)
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_utils.get_rank()],
            output_device=dist_utils.get_rank(),
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['train']['amp'])

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * epochs)
    # --- LR Scheduler ---
    scheduler, sched_step_when = build_lr_scheduler(cfg, optimizer, steps_per_epoch, total_steps)
    
    model_path_cfg = str(cfg.get("train", {}).get("model_path", "") or "").strip()
    load_mode_cfg = str(cfg.get("train", {}).get("load_mode", "auto")).lower()
    strict_cfg = bool(cfg.get("train", {}).get("strict", False))

    if os.path.exists(model_path_cfg):
        # rankごとにロード（DDP下でも各rank同じ処理でOK）
        ckpt = torch.load(model_path_cfg, map_location=device)
        if load_mode_cfg == "resume":
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = _load_model_state(model, state, strict=strict_cfg)
            if dist_utils.is_main_process():
                print(f"[config-load][resume] {model_path_cfg}")
                if missing:   print(f"  missing keys: {missing}")
                if unexpected:print(f"  unexpected keys: {unexpected}")

            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(ckpt, dict) and "scaler" in ckpt and cfg['train']['amp']:
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

        else:  # weights
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            missing, unexpected = _load_model_state(model, state, strict=strict_cfg)
            if dist_utils.is_main_process():
                print(f"[config-load][weights] {model_path_cfg}")
                if missing:   print(f"  missing keys: {missing}")
                if unexpected:print(f"  unexpected keys: {unexpected}")

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])

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
        val_disp, val_pose = validate(model, val_loader, epoch, cfg, writer, device)
        # エポック単位/指標連動型スケジューラ
        if scheduler is not None:
            if sched_step_when == "epoch":
                scheduler.step()
            elif sched_step_when == "epoch_metric":
                # ここでは「視差 EPE」が小さいほど良いので、その値を渡す
                scheduler.step(val_disp)

        if dist_utils.is_main_process():
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_disp={val_disp:.4f}  val_pose={val_pose:.4f}")
            # Save checkpoint
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
    if dist_utils.is_main_process():
        print("Training finished. Logs in:", out_dir)

if __name__ == "__main__":
    main()
