#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downsample voxel SDF (.npz) by factor 2 with minabs/mean pooling.

Example:
  python downsample_sdf_npz.py --root_dir /mnt/ssd2tb/new_format_sdf \
    --dst_dir /mnt/ssd2tb/new_format_sdf_half --method minabs
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_npz(npz_path: Path) -> Tuple[np.ndarray, dict]:
    z = np.load(npz_path, allow_pickle=False)
    sdf = z["sdf"]
    meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    return sdf, meta


def save_npz(out_path: Path, sdf: np.ndarray, meta: dict):
    meta_bytes = np.bytes_(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    np.savez_compressed(
        out_path,
        sdf=sdf,
        meta=meta_bytes,
        bbox_min=np.asarray(meta["bbox_min"], dtype=np.float32),
        bbox_max=np.asarray(meta["bbox_max"], dtype=np.float32),
        voxel_size=np.asarray(meta["voxel_size"], dtype=np.float32),
        res=np.asarray(meta["res"], dtype=np.int32),
        signed=np.asarray(int(meta.get("signed_used", False)), dtype=np.int8),
    )


def downsample_minabs(sdf: np.ndarray) -> np.ndarray:
    r = sdf.shape[0]
    r2 = r // 2
    v = sdf.reshape(r2, 2, r2, 2, r2, 2)
    v = v.transpose(0, 2, 4, 1, 3, 5).reshape(r2, r2, r2, 8)
    idx = np.argmin(np.abs(v), axis=-1)
    out = np.take_along_axis(v, idx[..., None], axis=-1)[..., 0]
    return out


def downsample_mean(sdf: np.ndarray) -> np.ndarray:
    r = sdf.shape[0]
    r2 = r // 2
    v = sdf.reshape(r2, 2, r2, 2, r2, 2)
    out = v.mean(axis=(1, 3, 5))
    return out


def update_meta(meta: dict, res_in: int, res_out: int, voxel_size_in: float, method: str) -> dict:
    voxel_size_out = float(voxel_size_in) * 2.0
    bbox_min = np.asarray(meta["bbox_min"], dtype=np.float64)
    bbox_max = bbox_min + voxel_size_out * float(res_out - 1)
    grid_center = 0.5 * (bbox_min + bbox_max)

    meta = dict(meta)
    meta["res"] = int(res_out)
    meta["voxel_size"] = float(voxel_size_out)
    meta["bbox_min"] = bbox_min.astype(np.float32).tolist()
    meta["bbox_max"] = bbox_max.astype(np.float32).tolist()
    meta["grid_center"] = grid_center.astype(np.float32).tolist()
    meta["center_index"] = float((res_out - 1) * 0.5)
    meta["center_is_voxel"] = bool((res_out % 2) == 1)
    meta["downsample"] = {
        "factor": 2,
        "method": str(method),
        "res_in": int(res_in),
        "res_out": int(res_out),
    }
    return meta


def process_one(npz_path: Path, out_path: Path, method: str, overwrite: bool) -> Dict:
    try:
        if out_path.exists() and not overwrite:
            return {"status": "skip", "npz": str(npz_path), "out": str(out_path)}

        sdf, meta = load_npz(npz_path)
        if sdf.ndim != 3 or sdf.shape[0] != sdf.shape[1] or sdf.shape[1] != sdf.shape[2]:
            raise ValueError(f"Invalid SDF shape: {sdf.shape}")

        res_in = int(sdf.shape[0])
        res_even = res_in - (res_in % 2)
        if res_even != res_in:
            sdf = sdf[:res_even, :res_even, :res_even]
        if res_even < 2:
            raise ValueError(f"SDF resolution too small: {res_in}")

        if method == "minabs":
            sdf_out = downsample_minabs(sdf)
        elif method == "mean":
            sdf_out = downsample_mean(sdf)
        else:
            raise ValueError(f"Unknown method: {method}")

        voxel_size_in = float(meta["voxel_size"])
        res_out = int(sdf_out.shape[0])
        meta_out = update_meta(meta, res_in=res_even, res_out=res_out, voxel_size_in=voxel_size_in, method=method)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(out_path, sdf_out.astype(sdf.dtype, copy=False), meta_out)
        return {"status": "ok", "npz": str(npz_path), "out": str(out_path)}
    except Exception as e:
        return {"status": "fail", "npz": str(npz_path), "out": str(out_path), "msg": f"{type(e).__name__}: {e}"}


def iter_npz_files(root_dir: Path):
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".npz":
            yield p


def main():
    ap = argparse.ArgumentParser(description="Downsample SDF npz files by factor 2 with pooling.")
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--dst_dir", type=str, required=True)
    ap.add_argument("--method", choices=["minabs", "mean"], default="minabs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    npz_list = list(iter_npz_files(root_dir))
    if not args.quiet:
        print(f"Found {len(npz_list)} NPZ files under: {root_dir}")
        print(f"method={args.method}, workers={args.workers}")

    n_ok = n_skip = n_fail = 0

    if args.workers <= 1:
        for i, p in enumerate(npz_list, start=1):
            rel = p.relative_to(root_dir)
            out_path = (dst_dir / rel).with_suffix(".npz")
            r = process_one(p, out_path, method=args.method, overwrite=bool(args.overwrite))
            if r["status"] == "ok":
                n_ok += 1
                if not args.quiet:
                    print(f"[{i}/{len(npz_list)}] OK: {rel}")
            elif r["status"] == "skip":
                n_skip += 1
                if not args.quiet:
                    print(f"[{i}/{len(npz_list)}] SKIP: {rel}")
            else:
                n_fail += 1
                print(f"[{i}/{len(npz_list)}] FAIL: {rel}\n{r['msg']}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = []
            for p in npz_list:
                rel = p.relative_to(root_dir)
                out_path = (dst_dir / rel).with_suffix(".npz")
                futs.append(ex.submit(process_one, p, out_path, args.method, bool(args.overwrite)))
            for i, fut in enumerate(as_completed(futs), start=1):
                r = fut.result()
                rel = Path(r["npz"]).relative_to(root_dir)
                if r["status"] == "ok":
                    n_ok += 1
                    if not args.quiet:
                        print(f"[{i}/{len(futs)}] OK: {rel}")
                elif r["status"] == "skip":
                    n_skip += 1
                    if not args.quiet:
                        print(f"[{i}/{len(futs)}] SKIP: {rel}")
                else:
                    n_fail += 1
                    print(f"[{i}/{len(futs)}] FAIL: {rel}\n{r['msg']}")

    print(f"Done. ok={n_ok}, skip={n_skip}, fail={n_fail}, dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
