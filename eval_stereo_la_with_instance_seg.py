#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for PanopticStereoMultiHead on val/test split.

This script mirrors train_stereo_la_with_instance_seg.py data preparation and metrics.
python eval_stereo_la_with_instance_seg.py --ckpt F:/repos/training_obj_observer/outputs/run_debug/checkpoint_021.pth --save_images
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.utils as vutils

import train_stereo_la_with_instance_seg as tr
import train_stereo_la as base
from la_loader.synthetic_data_loader import LASyntheticDataset3PerIns
from la_loader.real_data_loader import LARealDataset4PerIns
from losses import loss_functions
from utils import dist_utils, rot_utils
from utils.logging_utils import visualize_mono_torch
from utils.projection import SilhouetteDepthRenderer


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


def _normalize_rgb(img: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB tensor to [0, 1] for visualization."""
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


def make_eval_dataloader(
    cfg: dict,
    split: str,
    distributed: bool,
    batch_size: Optional[int],
):
    ds_cfg, split_used = _select_dataset_cfg(cfg, split)
    use_camera_list = ds_cfg.get("use_camera_list", ["ZED2", "D415", "ZEDmini"])
    target_scene_list = ds_cfg.get("target_scene_list", [])
    filtering_trans = tr._build_filtering_transforms(for_train=False)
    spatial_trans = tr._build_spatial_transforms()

    out_list = (
        "stereo",
        "depth",
        "disparity",
        "semantic_seg",
        "instance_seg",
    )
    w_sdf = float(cfg.get("loss", {}).get("w_sdf", 0.0))
    w_sdf_pose = float(cfg.get("loss", {}).get("w_sdf_pose", 0.0))
    w_sdf_pose_rot = float(cfg.get("loss", {}).get("w_sdf_pose_rot", w_sdf_pose))
    w_sdf_pose_trans = float(cfg.get("loss", {}).get("w_sdf_pose_trans", w_sdf_pose))
    if w_sdf > 0.0 or w_sdf_pose > 0.0 or w_sdf_pose_rot > 0.0 or w_sdf_pose_trans > 0.0:
        out_list = out_list + ("SDFs", "SDFs_meta")
    if ds_cfg['name'] ==  "LASyntheticDataset3PerObj": 
        dataset = LASyntheticDataset3PerIns(
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
        dataset = LARealDataset4PerIns(
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
    n_classes = int(getattr(dataset, "n_classes", 1))
    return loader, sampler, n_classes, split_used


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    save_images: bool,
    iters: int,
    amp_on: bool,
    limit_batches: int,
) -> Dict[str, float]:
    model.eval()
    meters = base.DictMeters()
    renderer = SilhouetteDepthRenderer().to(device) if save_images else None
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
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
        if limit_batches > 0 and it >= limit_batches:
            break

        stereo, depth, disp_gt, k_pair, baseline, left_k = tr._prepare_stereo_and_cam(batch, device)
        sem_gt = batch["semantic_seg"].to(device, non_blocking=True)
        inst_gt = batch["instance_seg"].to(device, non_blocking=True)

        size_hw = stereo.shape[-2:]
        inst_gt_hw = tr._downsample_label(inst_gt, size_hw)
        valid_k_inst = tr._build_valid_k_from_inst(inst_gt_hw)
        wks_inst, wfg_inst = tr._build_instance_weight_map(inst_gt_hw, valid_k_inst)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=amp_on):
                pred = model(
                    stereo,
                    k_pair,
                    baseline,
                    iters=iters,
                    Wk_1_4=wks_inst,
                    wfg_1_4=wfg_inst,
                )

        disp_preds = pred["disp_preds"]
        disp_logvar_preds = pred["disp_log_var_preds"]
        mask = (inst_gt > 0).unsqueeze(1)
        loss_disp = base.disparity_nll_laplace_raft_style(
            disp_preds,
            disp_logvar_preds,
            disp_gt,
            mask,
        )
        loss_disp_1x = tr.disparity_nll_laplace_scaled(
            [pred["disp_1x"]],
            [pred["disp_log_var_1x"]],
            disp_gt,
            mask,
            downsample=1,
        )

        size_hw = pred["sem_logits"].shape[-2:]
        sem_gt_hw = tr._downsample_label(sem_gt, size_hw)
        loss_cls = loss_functions.classification_loss(pred["cls_logits"], sem_gt_hw, use_focal=False)
        loss_sem = loss_cls

        pose_out = tr._compute_pose_losses_and_metrics(
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

        aff_tgt, aff_valid = tr._build_affinity_targets(inst_gt_hw)
        loss_aff, _ = tr._affinity_loss(
            pred["aff_logits"],
            aff_tgt,
            aff_valid,
            neg_weight=float(cfg.get("loss", {}).get("aff_neg_weight", 4.0)),
        )
        loss_emb, _ = tr.embedding_cosface_sampled(
            pred["emb"],
            inst_gt_hw,
            max_per_inst=int(cfg.get("loss", {}).get("emb_max_per_inst", 64)),
            margin=float(cfg.get("loss", {}).get("emb_margin", 0.25)),
            scale=32.0,
            min_pixels_per_inst=4,
            detach_proto=True,
            topk_neg=None,
        )

        loss = cfg["loss"]["w_disp"] * loss_disp
        loss = loss + cfg.get("loss", {}).get("w_disp_1x", 1.0) * loss_disp_1x
        loss = loss + cfg.get("loss", {}).get("w_sem", 1.0) * loss_sem
        loss = loss + cfg.get("loss", {}).get("w_cls", 1.0) * loss_cls
        loss = loss + cfg.get("loss", {}).get("w_pos", 1.0) * loss_pos
        loss = loss + cfg.get("loss", {}).get("w_rot", 1.0) * loss_rot
        loss = loss + cfg.get("loss", {}).get("w_aff", 1.0) * loss_aff
        loss = loss + cfg.get("loss", {}).get("w_emb", 1.0) * loss_emb

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

        if save_images:
            B = stereo.size(0)
            image_size = (stereo.shape[-2], stereo.shape[-1])
            left_vis = torch.stack([_normalize_rgb(stereo[b, :3]) for b in range(B)], dim=0)
            right_vis = torch.stack([_normalize_rgb(stereo[b, 3:]) for b in range(B)], dim=0)

            seg_size_hw = pred["sem_logits"].shape[-2:]
            sem_gt_hw = tr._downsample_label(sem_gt, seg_size_hw)
            inst_gt_hw = tr._downsample_label(inst_gt, seg_size_hw)
            inst_pred = tr._infer_instance_from_heads(
                pred["sem_logits"],
                pred["aff_logits"],
                pred["emb"],
                tau_high=float(cfg.get("instance", {}).get("tau_aff", 0.95)),
                tau_emb_merge=float(cfg.get("instance", {}).get("tau_emb_merge", 0.6)),
                emb_merge_iters=int(cfg.get("instance", {}).get("emb_merge_iters", 1)),
                use_gt_semantic=bool(cfg.get("instance", {}).get("use_gt_semantic", False)),
                semantic_gt_hw=sem_gt_hw,
            )

            num_classes = int(cfg.get("data", {}).get("n_classes", pred["sem_logits"].shape[1]))
            sem_pred = pred["sem_logits"].argmax(dim=1)
            sem_pred_color = tr._colorize_semantic(sem_pred, num_classes).float() / 255.0
            sem_gt_color = tr._colorize_semantic(sem_gt_hw, num_classes).float() / 255.0

            inst_pred_color = torch.stack(
                [tr._colorize_instance_ids(inst) for inst in inst_pred], dim=0
            ).float() / 255.0
            inst_gt_color = torch.stack(
                [tr._colorize_instance_ids(inst) for inst in inst_gt_hw.cpu().numpy()], dim=0
            ).float() / 255.0

            sem_pred_color = F.interpolate(
                sem_pred_color.to(device), size=image_size, mode="nearest"
            )
            sem_gt_color = F.interpolate(
                sem_gt_color.to(device), size=image_size, mode="nearest"
            )
            inst_pred_color = F.interpolate(
                inst_pred_color.to(device), size=image_size, mode="nearest"
            )
            inst_gt_color = F.interpolate(
                inst_gt_color.to(device), size=image_size, mode="nearest"
            )

            disp_pred_1x = pred["disp_1x"]
            if disp_pred_1x.shape[-2:] != disp_gt.shape[-2:]:
                disp_pred_1x = F.interpolate(
                    disp_pred_1x,
                    size=disp_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            disp_pred_u8, disp_gt_u8 = visualize_mono_torch(
                disp_pred_1x,
                mask,
                disp_gt,
                mask,
            )
            disp_pred_vis = disp_pred_u8.to(torch.float32) / 255.0
            disp_gt_vis = disp_gt_u8.to(torch.float32) / 255.0

            pred_render = torch.zeros_like(left_vis)
            gt_render = torch.zeros_like(left_vis)
            pred_and_gt_render = torch.zeros_like(left_vis)
            if renderer is not None:
                meshes_flat, valid_k = tr._build_meshes_from_batch(batch, device)
                if meshes_flat is not None and valid_k.numel() > 0:
                    pose_size_hw = pred["pos_mu"].shape[-2:]
                    inst_hw = tr._downsample_label(inst_gt, pose_size_hw)
                    wks, wfg = tr._build_instance_weight_map(inst_hw, valid_k)
                    pos_gt_map, rot_gt_map, _, objs_in_left = tr._prepare_pose_targets(
                        batch,
                        sem_gt,
                        pose_size_hw,
                        device,
                    )
                    pos_map_pred = tr.pos_mu_to_pointmap(pred["pos_mu"], left_k[:, 0], downsample=1)
                    r_pred, t_pred, v_pred, _, _ = rot_utils.pose_from_maps_auto(
                        rot_map=pred["rot_mat"],
                        pos_map=pos_map_pred,
                        Wk_1_4=wks,
                        wfg=wfg,
                        min_px=10,
                        min_wsum=1e-6,
                    )
                    r_gt, t_gt, v_gt, _, _ = rot_utils.pose_from_maps_auto(
                        rot_map=rot_gt_map,
                        pos_map=pos_gt_map,
                        Wk_1_4=wks,
                        wfg=wfg,
                        min_px=10,
                        min_wsum=1e-6,
                    )
                    origin_in = tr._origin_in_image_from_t(t_gt, left_k[:, 0], image_size)
                    valid_render = valid_k & origin_in
                    valid_pred = v_pred & valid_render
                    valid_gt = v_gt & valid_render

                    pos_gt = objs_in_left[..., :3, 3]
                    rot_gt = objs_in_left[..., :3, :3]
                    T_pred = rot_utils.compose_T_from_Rt(r_pred, t_pred, valid_pred)
                    T_gt = rot_utils.compose_T_from_Rt(rot_gt, pos_gt, valid_gt)

                    meshes_use = tr._build_meshes_from_batch_filtered(batch, valid_render, device)
                    if meshes_use is not None:
                        pred_r = renderer(
                            meshes_flat=meshes_use,
                            T_cam_obj=T_pred,
                            K_left=left_k[:, 0],
                            valid_k=valid_render,
                            image_size=image_size,
                        )
                        gt_r = renderer(
                            meshes_flat=meshes_use,
                            T_cam_obj=T_gt,
                            K_left=left_k[:, 0],
                            valid_k=valid_render,
                            image_size=image_size,
                        )
                        sil_pred = pred_r["silhouette"]
                        sil_gt = gt_r["silhouette"]
                        pred_render = tr._overlay_mask_rgb(
                            left_vis, sil_pred, color=(0.0, 1.0, 0.0), alpha=0.45
                        )
                        gt_render = tr._overlay_mask_rgb(
                            left_vis, sil_gt, color=(1.0, 0.0, 0.0), alpha=0.45
                        )
                        pred_and_gt_render = tr._overlay_mask_rgb(
                            gt_render, sil_pred, color=(0.0, 1.0, 0.0), alpha=0.45
                        )

            data_paths = batch.get("data_path", None)
            if data_paths is None:
                data_paths = [f"batch_{it:06d}_idx_{b:02d}" for b in range(B)]
            elif isinstance(data_paths, (tuple, list)):
                data_paths = [str(p) for p in data_paths]
            else:
                data_paths = [str(data_paths) for _ in range(B)]

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
                safe_name = _safe_key(data_path)
                image_name = f"{safe_name}_idx{sample_index}.png"
                image_path = output_dir / image_name

                blank = torch.zeros_like(left_vis[b])
                panel = torch.stack(
                    [
                        left_vis[b],
                        right_vis[b],
                        inst_pred_color[b],
                        inst_gt_color[b],
                        sem_pred_color[b],
                        sem_gt_color[b],
                        disp_pred_vis[b],
                        disp_gt_vis[b],
                        pred_render[b],
                        gt_render[b],
                        pred_and_gt_render[b],
                        blank,
                    ],
                    dim=0,
                ).clamp(0.0, 1.0)
                vutils.save_image(panel, str(image_path), nrow=4)

    meters.all_reduce_()
    return meters.averages()


def _load_ckpt_into_model(model, ckpt_path: str, device: torch.device, strict: bool):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    return missing, unexpected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/inf_config_panoptic.toml")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=-1, help="override batch size")
    ap.add_argument("--iters", type=int, default=-1, help="override cfg['model']['n_iter']")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--output", type=str, default="evals_panoptic/imgs/panoptic_results.json")
    ap.add_argument("--eval_dir", type=str, default="evals_panoptic")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--limit_batches", type=int, default=0, help="0=full loader, else limit iterator length")
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

    batch_size = None if args.batch_size <= 0 else int(args.batch_size)
    loader, sampler, n_classes, split_used = make_eval_dataloader(
        cfg, args.split, distributed, batch_size
    )
    if sampler is not None:
        sampler.set_epoch(0)

    model = tr.build_model(cfg, n_classes).to(device)
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
        save_images=bool(args.save_images),
        iters=iters,
        amp_on=bool(args.amp),
        limit_batches=args.limit_batches,
    )

    if dist.is_available() and dist.is_initialized():
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
                "iters": iters,
            },
            "metrics": results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval] wrote: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
