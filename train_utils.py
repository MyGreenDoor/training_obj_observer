#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared training utilities across stereo training entry points.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tomlkit

import torch
import torch.distributed as dist
import torch.nn.functional as F


_CFG_TEXT_CACHE: Optional[str] = None


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


def build_lr_scheduler(cfg, optimizer, steps_per_epoch: int, total_steps: int):
    """Build LR scheduler from config."""
    scfg = cfg.get("lr_scheduler", {}) or {}
    typ = scfg.get("type", "none").lower()
    if typ in ("none", "", None):
        return None, None

    base_lr = cfg["train"]["lr"]

    if typ == "cosine":
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
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal,
            div_factor=div_factor,
            final_div_factor=final_div,
        )
        return scheduler, "step"

    if typ == "multistep":
        ms = scfg.get("milestones")
        if ms is None:
            ms = [int(0.6 * total_steps), int(0.85 * total_steps)]
        gamma = float(scfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=gamma)
        return scheduler, "step"

    if typ == "plateau":
        factor = float(scfg.get("factor", 0.5))
        patience = int(scfg.get("patience", 5))
        min_lr = float(scfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, min_lr=min_lr
        )
        return scheduler, "epoch_metric"

    return None, None


def disparity_nll_laplace_raft_style(
    preds_1_4,
    logvars_1_4,
    gt_disp_1x: torch.Tensor,
    gt_valid_1x: Optional[torch.Tensor] = None,
    gamma: float = 0.9,
    charb_eps: float = 0.0,
) -> torch.Tensor:
    """Compute Laplace NLL loss for disparity predictions."""
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

    n_steps = len(preds_1_4)
    total = gt_disp_1x.new_tensor(0.0)
    denom = gt_disp_1x.new_tensor(0.0)

    for i in range(n_steps):
        w_i = gamma ** (n_steps - 1 - i)
        pred = preds_1_4[i]
        logv = logvars_1_4[i]
        r = pred - gt_1_4

        if charb_eps > 0.0:
            abs_r = torch.sqrt(r * r + (charb_eps * charb_eps))
        else:
            abs_r = r.abs()

        nll_map = abs_r * torch.exp(-0.5 * logv) + 0.5 * logv
        nll_map = nll_map * valid_1_4
        num = valid_1_4.sum().clamp_min(1e-6)
        step = nll_map.sum() / num
        total = total + w_i * step
        denom = denom + w_i

    return total / denom


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
        self.sum = 0.0
        self.cnt = 0.0
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
        self.sum = 0.0
        self.count = 0.0
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
        cast = self.m[name]
        assert isinstance(cast, AverageMeter)
        cast.update(val, n)
    def update_sc(self, name: str, sum_val: float, count: float):
        cast = self.m[name]
        assert isinstance(cast, SumCountMeter)
        cast.update_sc(sum_val, count)
    def get(self, name: str) -> _BaseMeter:
        return self.m[name]
    def all_reduce_(self):
        for v in self.m.values():
            v.all_reduce_()
    def averages(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.m.items()}


def log_meters_to_tb(writer, meters: DictMeters, step: int, prefix: str):
    """Write meter averages to TensorBoard."""
    for k, v in meters.m.items():
        writer.add_scalar(f"{prefix}/{k}", v.avg, step)


def _init_window_meter():
    """Initialize windowed loss accumulators."""
    return {
        "loss": 0.0,
        "loss_disp": 0.0,
        "depth_mae": 0.0,
        "depth_acc_1mm": 0.0,
        "depth_acc_2mm": 0.0,
        "depth_acc_4mm": 0.0,
        "L_ctr": 0.0,
        "L_mask": 0.0,
        "L_pos": 0.0,
        "L_rot": 0.0,
        "L_cls": 0.0,
        "L_adds": 0.0,
    }, 0


def _reset_window_meter(window_sum, window_cnt_ref):
    """Reset windowed sums."""
    for k in window_sum:
        window_sum[k] = 0.0
    return 0


def _update_window_meter(window_sum, window_cnt, loss, loss_disp, depth_mae, logs):
    """Update windowed sums with new loss values."""
    window_sum["loss"] += float(loss.detach().item())
    window_sum["loss_disp"] += float(loss_disp.detach().item())
    window_sum["depth_mae"] += float(depth_mae.detach().item())
    for k, v in logs.items():
        if torch.is_tensor(v):
            if k not in window_sum:
                window_sum[k] = 0.0
            window_sum[k] += float(v.detach().item())
    return window_cnt + 1


def _flush_train_window_to_tb(writer, window_sum, window_cnt, optimizer, global_step, prefix="train"):
    """Flush windowed training metrics to TensorBoard."""
    if writer is None or window_cnt <= 0:
        return
    denom = max(window_cnt, 1)
    writer.add_scalar(f"{prefix}/loss", window_sum["loss"] / denom, global_step)
    writer.add_scalar(f"{prefix}/loss_disp", window_sum["loss_disp"] / denom, global_step)
    writer.add_scalar(f"{prefix}/depth_mae", window_sum["depth_mae"] / denom, global_step)
    if "depth_acc_1mm" in window_sum:
        writer.add_scalar(f"{prefix}/depth_acc_1mm", window_sum["depth_acc_1mm"] / denom, global_step)
    if "depth_acc_2mm" in window_sum:
        writer.add_scalar(f"{prefix}/depth_acc_2mm", window_sum["depth_acc_2mm"] / denom, global_step)
    if "depth_acc_4mm" in window_sum:
        writer.add_scalar(f"{prefix}/depth_acc_4mm", window_sum["depth_acc_4mm"] / denom, global_step)
    writer.add_scalar(f"{prefix}/lr", optimizer.param_groups[0]["lr"], global_step)
    for k, s in window_sum.items():
        if k.startswith("L_"):
            writer.add_scalar(f"{prefix}/{k}", s / denom, global_step)


def load_model_state(model, state_dict, strict: bool = False):
    """Load model weights with DDP-awareness, skipping shape mismatches."""
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target_state = target.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k not in target_state:
            continue
        if target_state[k].shape != v.shape:
            continue
        filtered[k] = v
    missing, unexpected = target.load_state_dict(filtered, strict=False)
    return missing, unexpected
