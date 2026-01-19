#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference speed benchmark for PanopticStereoMultiHead.

Notes:
  - Default mode uses the dataset loader defined in train_stereo_la_with_instance_seg.py,
    which depends on la_loader. Use --synthetic to avoid dataset dependency.
  - Batch size is fixed to 1 for this benchmark.

Examples:
  python3 bench_panoptic_infer.py --config configs/config_full.toml --steps 200 --warmup 50 --amp --pin_batch
  python3 bench_panoptic_infer.py --config configs/config_full.toml --synthetic --steps 200 --warmup 50 --amp
  python3 bench_panoptic_infer.py --config configs/config_full.toml --ckpt path/to/checkpoint.pth --steps 200 --warmup 50 --amp
"""

import argparse
import time
from typing import Optional, Tuple

import torch

import train_stereo_la_with_instance_seg as tr


def _is_cuda(dev: torch.device) -> bool:
    return dev.type == "cuda"


@torch.no_grad()
def _sync(dev: torch.device) -> None:
    if _is_cuda(dev):
        torch.cuda.synchronize(dev)


def _make_events(dev: torch.device):
    if not _is_cuda(dev):
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _elapsed_ms(ev0, ev1) -> float:
    return float(ev0.elapsed_time(ev1))


def _bytes_to_mib(x: float) -> float:
    return float(x) / (1024.0 * 1024.0)


def _load_ckpt_into_model(model, ckpt_path: str, device: torch.device, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    return missing, unexpected


def _build_synthetic_inputs(
    device: torch.device,
    height: int,
    width: int,
    with_inst: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch_size = 1
    stereo = torch.rand(batch_size, 6, height, width, device=device)

    fx = float(max(height, width))
    fy = float(max(height, width))
    cx = float(width - 1) * 0.5
    cy = float(height - 1) * 0.5
    k_pair = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(batch_size, 2, 1, 1)
    k_pair[:, :, 0, 0] = fx
    k_pair[:, :, 1, 1] = fy
    k_pair[:, :, 0, 2] = cx
    k_pair[:, :, 1, 2] = cy

    baseline = torch.full((batch_size,), 50.0, device=device)

    if not with_inst:
        return stereo, k_pair, baseline, None, None

    wfg = torch.ones((batch_size, 1, height, width), device=device)
    wks = torch.ones((batch_size, 1, 1, height, width), device=device)
    return stereo, k_pair, baseline, wks, wfg


def _prep_inputs_from_batch(
    batch,
    device: torch.device,
    with_inst: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    stereo, _, _, k_pair, baseline, _ = tr._prepare_stereo_and_cam(batch, device)
    if not with_inst:
        return stereo, k_pair, baseline, None, None

    inst_gt = batch["instance_seg"].to(device, non_blocking=True)
    size_hw = stereo.shape[-2:]
    inst_hw = tr._downsample_label(inst_gt, size_hw)
    valid_k = tr._build_valid_k_from_inst(inst_hw)
    wks, wfg = tr._build_instance_weight_map(inst_hw, valid_k)
    return stereo, k_pair, baseline, wks, wfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/inf_config.toml", help="Path to config TOML")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint path (optional)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pin_batch", action="store_true", help="reuse 1 prepared batch to measure pure forward")
    ap.add_argument("--iters", type=int, default=-1, help="override cfg['model']['n_iter'] if >0")
    ap.add_argument("--cudnn_bench", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic inputs instead of dataset")
    ap.add_argument("--num_workers", type=int, default=0, help="override data.num_workers when using dataset")
    ap.add_argument("--height", type=int, default=0, help="synthetic height (0=use config)")
    ap.add_argument("--width", type=int, default=0, help="synthetic width (0=use config)")
    ap.add_argument("--num_classes", type=int, default=0, help="synthetic num_classes (0=use config or 1)")
    ap.add_argument("--with_inst", action="store_true", help="include instance weights for pose aggregation")
    args = ap.parse_args()

    cfg = tr.load_toml(args.config)
    tr.set_global_seed(42)

    if args.cudnn_bench:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Force batch size to 1 for this benchmark.
    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = 1
    cfg.setdefault("data", {})
    cfg["data"]["num_workers"] = int(args.num_workers)

    if args.synthetic:
        if args.num_classes > 0:
            num_classes = int(args.num_classes)
        else:
            num_classes = int(cfg.get("data", {}).get("n_classes", 1))
        model = tr.build_model(cfg, num_classes).to(device)
        train_loader = None
    else:
        train_loader, _, _, _, num_classes = tr.make_dataloaders(cfg, distributed=False)
        model = tr.build_model(cfg, num_classes).to(device)

    if args.ckpt.strip():
        missing, unexpected = _load_ckpt_into_model(model, args.ckpt.strip(), device, strict=args.strict)
        print(f"[ckpt] loaded: {args.ckpt.strip()}")
        if missing:
            print(f"[ckpt] missing keys: {len(missing)}")
        if unexpected:
            print(f"[ckpt] unexpected keys: {len(unexpected)}")

    model.eval()

    iters = int(cfg["model"]["n_iter"])
    if args.iters > 0:
        iters = int(args.iters)

    pinned = None
    if args.pin_batch:
        if args.synthetic:
            height = int(args.height or cfg["data"].get("height", 256))
            width = int(args.width or cfg["data"].get("width", 256))
            pinned = _build_synthetic_inputs(device, height, width, args.with_inst)
        else:
            loader_it = iter(train_loader)
            batch = next(loader_it)
            pinned = _prep_inputs_from_batch(batch, device, args.with_inst)

    ev = _make_events(device)
    times_ms = []
    total_steps = args.warmup + args.steps
    amp_on = bool(args.amp and _is_cuda(device))

    loader_it = None
    if train_loader is not None:
        loader_it = iter(train_loader)

    with torch.inference_mode():
        for step in range(total_steps):
            if pinned is not None:
                stereo, k_pair, baseline, wks, wfg = pinned
            else:
                if args.synthetic:
                    height = int(args.height or cfg["data"].get("height", 256))
                    width = int(args.width or cfg["data"].get("width", 256))
                    stereo, k_pair, baseline, wks, wfg = _build_synthetic_inputs(
                        device, height, width, args.with_inst
                    )
                else:
                    try:
                        batch = next(loader_it)
                    except StopIteration:
                        loader_it = iter(train_loader)
                        batch = next(loader_it)
                    stereo, k_pair, baseline, wks, wfg = _prep_inputs_from_batch(batch, device, args.with_inst)

            if _is_cuda(device) and ev is not None:
                _sync(device)
                ev[0].record()
                with torch.autocast("cuda", enabled=amp_on):
                    model(stereo, k_pair, baseline, iters=iters, Wk_1_4=wks, wfg_1_4=wfg)
                ev[1].record()
                _sync(device)
                elapsed = _elapsed_ms(ev[0], ev[1])
            else:
                t0 = time.perf_counter()
                with torch.autocast("cpu", enabled=False):
                    model(stereo, k_pair, baseline, iters=iters, Wk_1_4=wks, wfg_1_4=wfg)
                t1 = time.perf_counter()
                elapsed = (t1 - t0) * 1000.0

            if step >= args.warmup:
                if step == args.warmup and _is_cuda(device):
                    torch.cuda.reset_peak_memory_stats(device)
                times_ms.append(elapsed)

    if not times_ms:
        print("[bench] no measured steps")
        return

    times_ms = torch.tensor(times_ms)
    avg_ms = float(times_ms.mean().item())
    p50 = float(times_ms.kthvalue(max(1, int(0.50 * len(times_ms)))).values.item())
    p90 = float(times_ms.kthvalue(max(1, int(0.90 * len(times_ms)))).values.item())
    fps = 1000.0 / avg_ms

    print(f"[bench] steps={len(times_ms)} batch=1 iters={iters} amp={amp_on}")
    print(f"[bench] avg_ms={avg_ms:.3f} p50_ms={p50:.3f} p90_ms={p90:.3f} fps={fps:.2f}")
    if _is_cuda(device):
        max_alloc = _bytes_to_mib(torch.cuda.max_memory_allocated(device))
        max_reserved = _bytes_to_mib(torch.cuda.max_memory_reserved(device))
        print(f"[bench] max_mem_alloc_mib={max_alloc:.1f} max_mem_reserved_mib={max_reserved:.1f}")


# // RTX 4080 1280 * 720
# [bench] steps=200 batch=1 iters=4 amp=True
# [bench] avg_ms=66.307 p50_ms=66.249 p90_ms=66.556 fps=15.08
# [bench] max_mem_alloc_mib=1517.1 max_mem_reserved_mib=2000.0
# // RTX 4080 256 * 256
# [bench] steps=200 batch=1 iters=4 amp=True
# [bench] avg_ms=17.843 p50_ms=16.934 p90_ms=17.893 fps=56.04
# [bench] max_mem_alloc_mib=204.5 max_mem_reserved_mib=258.0

if __name__ == "__main__":
    main()
