#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference speed benchmark for SSCFlow2 forward.

Examples:
  # もっとも「純 forward」に近い：1バッチだけ作って使い回す（pin_batch）
  python3 bench_sscflow2_infer.py --config configs/small_config.toml --steps 300 --warmup 50 --amp --pin_batch

  # DataLoader を回して毎回別バッチ（collate含む）で測る
  python3 bench_sscflow2_infer.py --config configs/small_config.toml --steps 300 --warmup 50 --amp

  # checkpoint を読み込み（推奨：実モデルの重みで計測）
  python3 bench_sscflow2_infer.py --config configs/small_config.toml --ckpt path/to/checkpoint_099.pth --steps 300 --warmup 50 --amp --pin_batch

  # iters を上書き（推論反復回数を変えて比較）
  python3 bench_sscflow2_infer.py --config configs/small_config.toml --iters 6 --steps 300 --warmup 50 --amp --pin_batch
"""

import argparse
import time
from typing import Optional

import torch
import torch.distributed as dist

import train_stereo_la as tr
from utils import dist_utils


def _is_cuda(dev: torch.device) -> bool:
    return dev.type == "cuda"


@torch.no_grad()
def _sync(dev: torch.device):
    if _is_cuda(dev):
        torch.cuda.synchronize(dev)


def _make_events(dev: torch.device):
    if not _is_cuda(dev):
        return None
    return {
        "h2d0": torch.cuda.Event(enable_timing=True),
        "h2d1": torch.cuda.Event(enable_timing=True),
        "fw0":  torch.cuda.Event(enable_timing=True),
        "fw1":  torch.cuda.Event(enable_timing=True),
    }


def _elapsed_ms(ev0, ev1) -> float:
    return float(ev0.elapsed_time(ev1))


def _load_ckpt_into_model(model, ckpt_path: str, device: torch.device, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    return missing, unexpected


def _allreduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


@torch.no_grad()
def _prep_inputs_from_batch(batch, cfg, device):
    # training と同じ前処理関数を流用
    stereo, depth, disp_gt, k_pair, baseline, input_mask, left_k = tr._prepare_stereo_and_cam(batch, device)

    # 推論 forward に必要そうなもの
    meshes = batch["meshes"].to(device)
    # valid_k は model 内で使う可能性があるので一応残す（SSCFlow2 側が参照するなら）
    valid_k = batch["valid_k"].to(device)

    # 推論では gt を使わない想定
    return stereo, input_mask, k_pair, baseline, meshes, valid_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to config TOML", default="configs/inf_config.toml")
    ap.add_argument("--launcher", type=str, choices=["none", "pytorch"], default="none")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint path (optional)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pin_batch", action="store_true",
                    help="reuse 1 prepared batch to measure pure forward (recommended)")
    ap.add_argument("--iters", type=int, default=-1, help="override cfg['model']['n_iter'] if >0")
    ap.add_argument("--cudnn_bench", action="store_true")
    ap.add_argument("--limit_batches", type=int, default=0, help="0=full loader, else limit iterator length")
    args = ap.parse_args()

    cfg = tr.load_toml(args.config)
    tr.set_global_seed(42)

    # DDP init（必要な人向け，基本は none 推奨）
    if args.launcher != "none":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist_utils.init_dist(args.launcher, backend)
        distributed = True
    else:
        distributed = False
    dist_utils.setup_for_distributed(is_master=dist_utils.is_main_process())

    if args.cudnn_bench:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda", dist_utils.get_rank()) if torch.cuda.is_available() else torch.device("cpu")

    # data / model（build は training と同じ）
    train_loader, _, train_sampler, _, class_table = tr.make_dataloaders(cfg, distributed=distributed)
    model = tr.build_model(cfg, class_table).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_utils.get_rank()] if device.type == "cuda" else None,
            output_device=dist_utils.get_rank() if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # checkpoint load
    if args.ckpt.strip():
        missing, unexpected = _load_ckpt_into_model(model, args.ckpt.strip(), device, strict=args.strict)
        if dist_utils.is_main_process():
            print(f"[ckpt] loaded: {args.ckpt.strip()}")
            if missing:
                print(f"[ckpt] missing keys: {len(missing)}")
            if unexpected:
                print(f"[ckpt] unexpected keys: {len(unexpected)}")

    model.eval()

    # iters
    iters = int(cfg["model"]["n_iter"])
    if args.iters > 0:
        iters = int(args.iters)

    # loader iterator
    def _iter_loader():
        it = 0
        for b in train_loader:
            yield b
            it += 1
            if args.limit_batches > 0 and it >= args.limit_batches:
                break

    loader_it = iter(_iter_loader())

    # pin_batch: 1回だけ「バッチ→GPU入力」まで作って使い回す
    pinned = None
    if args.pin_batch:
        b0 = next(loader_it)
        pinned = _prep_inputs_from_batch(b0, cfg, device)

    if distributed:
        dist.barrier()

    ev = _make_events(device)

    # sums across ranks
    sums = {
        "dl_s": 0.0,     # batch fetch time（pin_batch ならほぼ 0）
        "h2d_ms": 0.0,   # H2D time（pin_batch ならほぼ 0）
        "fw_ms": 0.0,    # forward time
        "iters": 0.0,
        "imgs": 0.0,
    }

    total_steps = args.warmup + args.steps

    # inference_mode は推論最適化が強いので，推論速度を見るならこれが正しい
    amp_on = bool(args.amp and device.type == "cuda")

    for step in range(total_steps):
        # ---- dataloader ----
        t0 = time.perf_counter()
        if pinned is None:
            try:
                batch = next(loader_it)
            except StopIteration:
                loader_it = iter(_iter_loader())
                batch = next(loader_it)
        t1 = time.perf_counter()

        if _is_cuda(device) and ev is not None:
            ev["h2d0"].record()

        # ---- prepare inputs (H2D 含む) ----
        if pinned is None:
            stereo, input_mask, k_pair, baseline, meshes, valid_k = _prep_inputs_from_batch(batch, cfg, device)
        else:
            stereo, input_mask, k_pair, baseline, meshes, valid_k = pinned

        if _is_cuda(device) and ev is not None:
            ev["h2d1"].record()
            ev["fw0"].record()

        # ---- forward ----
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=amp_on):
                # 推論：gt は与えない（use_gt_peaks=False）
                # ※ SSCFlow2 側で追加引数が必須なら，ここだけあなたの forward に合わせて調整して下さい
                _pred = model(
                    stereo,
                    input_mask,
                    k_pair,
                    baseline,
                    meshes,
                    use_gt_peaks=False,
                    with_shape_constraint=True,
                    iters=iters,
                )

        if _is_cuda(device) and ev is not None:
            ev["fw1"].record()

        _sync(device)

        # ---- accumulate (after warmup) ----
        if step >= args.warmup:
            B = int(stereo.size(0))
            sums["dl_s"] += (t1 - t0)
            sums["iters"] += 1.0
            sums["imgs"] += float(B)

            if _is_cuda(device) and ev is not None:
                sums["h2d_ms"] += _elapsed_ms(ev["h2d0"], ev["h2d1"])
                sums["fw_ms"]  += _elapsed_ms(ev["fw0"],  ev["fw1"])

    pack = torch.tensor(
        [sums["dl_s"], sums["h2d_ms"], sums["fw_ms"], sums["iters"], sums["imgs"]],
        dtype=torch.float64,
        device=device if device.type == "cuda" else "cpu",
    )
    pack = _allreduce_sum(pack).cpu()
    dl_s, h2d_ms, fw_ms, iters_cnt, imgs_cnt = pack.tolist()
    iters_cnt = max(iters_cnt, 1.0)

    dl_ms_avg  = (dl_s / iters_cnt) * 1000.0
    h2d_ms_avg = (h2d_ms / iters_cnt)
    fw_ms_avg  = (fw_ms / iters_cnt)

    # forward のみを FPS 換算（pin_batch の場合は純 forward に近い）
    fps = imgs_cnt / (iters_cnt * (fw_ms_avg / 1000.0)) if fw_ms_avg > 1e-9 else 0.0

    if dist_utils.is_main_process():
        print("\n========== SSCFLOW2 INFER BENCH ==========")
        print(f"config        : {args.config}")
        print(f"ckpt          : {args.ckpt.strip() if args.ckpt.strip() else '(none)'}")
        print(f"world_size    : {dist_utils.get_world_size()}")
        print(f"device        : {device}")
        print(f"amp           : {amp_on}")
        print(f"pin_batch     : {args.pin_batch}")
        print(f"iters         : {iters}")
        print("------------------------------------------")
        print(f"dataloader    : {dl_ms_avg:8.3f} ms/iter (CPU)")
        if torch.cuda.is_available():
            print(f"H2D(prep)     : {h2d_ms_avg:8.3f} ms/iter (GPU)")
            print(f"forward       : {fw_ms_avg:8.3f} ms/iter (GPU)")
            print("------------------------------------------")
            print(f"throughput    : {fps:8.2f} images/sec (forward-only)")
        else:
            print("CUDA not available，CPU only．")
        print("==========================================\n")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
