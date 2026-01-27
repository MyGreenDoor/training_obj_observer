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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.panoptic_stereo import PanopticStereoMultiHeadLatent
from models.sdf_pose_net import SDFPoseDeltaNet, SDFVolumeDecoder
from models.stereo_disparity import make_gn
from utils import dist_utils

import train_stereo_la_with_instance_seg as base
import train_stereo_la as core


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
            coords = base._normalize_obj_coords_for_sdf(meta_objs[k], coords, device, coords.dtype)
            bbox_min, bbox_max = base._extract_sdf_bounds(meta_objs[k], device, coords.dtype)
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
            sdf_vals = base._sample_sdf_volume(sdf_vol, coords_norm)

            flat_idx = idx[:, 0] * W + idx[:, 1]
            sdf_gt_flat[flat_idx] = sdf_vals
            sdf_valid_flat[flat_idx] = True

    return sdf_gt, sdf_valid


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

    sdf_gt, sdf_valid = base._build_sdf_targets(batch, inst_gt_hw, obj_coords, device, cfg)
    if sdf_gt is None or sdf_valid is None:
        return {"loss_sdf": zero, "sdf_valid_px": zero}
    valid_px = sdf_valid.sum().to(dtype=zero.dtype)
    if valid_px.item() <= 0:
        return {"loss_sdf": zero, "sdf_valid_px": zero}

    charb_eps = float(cfg.get("loss", {}).get("sdf_charb_eps", 0.0))
    weight_map = _build_sdf_hist_weights(sdf_gt, sdf_valid, cfg)
    if weight_map is None:
        loss_sdf = base._sdf_nll_laplace_map(sdf_pred, sdf_logvar, sdf_gt, sdf_valid, charb_eps=charb_eps)
    else:
        loss_sdf = _sdf_nll_laplace_weighted(
            sdf_pred, sdf_logvar, sdf_gt, sdf_valid, weight_map, charb_eps=charb_eps
        )
    return {"loss_sdf": loss_sdf, "sdf_valid_px": valid_px}


# Override base SDF target builder to keep normalized SDF targets in latent training.
base._build_sdf_targets = _build_sdf_targets_scaled
base._compute_sdf_loss = _compute_sdf_loss_weighted


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
    sdf_pose_cfg = cfg.get("sdf_pose", {}) or {}
    w_sdf_pose = float(cfg.get("loss", {}).get("w_sdf_pose", 1.0))
    w_sdf_pose_rot = float(cfg.get("loss", {}).get("w_sdf_pose_rot", w_sdf_pose))
    w_sdf_pose_trans = float(cfg.get("loss", {}).get("w_sdf_pose_trans", w_sdf_pose))
    if max(w_sdf_pose, w_sdf_pose_rot, w_sdf_pose_trans) > 0.0:
        model.sdf_pose_net = SDFPoseDeltaNet(
            in_ch=int(sdf_pose_cfg.get("in_ch", 2)),
            base_ch=int(sdf_pose_cfg.get("base_ch", 16)),
            num_down=int(sdf_pose_cfg.get("num_down", 4)),
            hidden_ch=int(sdf_pose_cfg.get("hidden_ch", 128)),
            out_scale_rot=float(sdf_pose_cfg.get("out_scale_rot", 0.5)),
            out_scale_trans=float(sdf_pose_cfg.get("out_scale_trans", 0.02)),
        )
        model.sdf_decoder = SDFVolumeDecoder(
            latent_dim=latent_dim,
            base_ch=int(sdf_pose_cfg.get("decoder_base_ch", 32)),
            base_res=int(sdf_pose_cfg.get("decoder_base_res", 4)),
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

        train_loss = base.train_one_epoch(
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
        val_disp, val_sem = base.validate(model, val_loader, epoch, cfg, writer, device)

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
