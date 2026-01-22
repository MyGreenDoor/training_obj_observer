#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _ensure_trimesh(mesh):
    """
    trimesh.load() が Scene を返すことがあるので，Trimesh に寄せる．
    """
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        # Scene -> 一つのメッシュに結合（STLなら通常Trimeshだが保険）
        if len(mesh.geometry) == 0:
            raise ValueError("Empty trimesh.Scene (no geometry).")
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError("Scene contains no Trimesh geometry.")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not trimesh.Trimesh: {type(mesh)}")
    return mesh


def mesh_to_voxel_sdf_trimesh(
    verts: np.ndarray,
    faces: np.ndarray,
    res: int = 128,
    padding: float = 0.05,
    normalize_to_cube: bool = False,
    trunc: Optional[float] = None,
    signed: bool = True,
    batch: int = 200_000,
    out_dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, dict]:
    """
    Triangle mesh -> voxel SDF grid.

    Returns:
        sdf: (res,res,res) out_dtype
        meta: dict (JSON化可能な要素中心，数値も含む)
    """
    import trimesh

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (V,3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")

    transform = None
    if normalize_to_cube:
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        center = 0.5 * (vmin + vmax)
        extent = float((vmax - vmin).max())
        if extent <= 0:
            raise ValueError("Degenerate mesh bbox extent.")
        scale = 1.0 / extent  # max edge length -> 1
        verts = (verts - center) * scale
        transform = {"center": center.tolist(), "scale": float(scale)}

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    bbox_min = mesh.bounds[0].copy()
    bbox_max = mesh.bounds[1].copy()
    extent = bbox_max - bbox_min
    pad = padding * float(extent.max() if extent.max() > 0 else 1.0)
    bbox_min -= pad
    bbox_max += pad

    xs = np.linspace(bbox_min[0], bbox_max[0], res, dtype=np.float64)
    ys = np.linspace(bbox_min[1], bbox_max[1], res, dtype=np.float64)
    zs = np.linspace(bbox_min[2], bbox_max[2], res, dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    N = pts.shape[0]

    pq = trimesh.proximity.ProximityQuery(mesh)
    out = np.empty((N,), dtype=np.float64)

    if signed:
        for s in range(0, N, batch):
            e = min(N, s + batch)
            out[s:e] = pq.signed_distance(pts[s:e])
    else:
        for s in range(0, N, batch):
            e = min(N, s + batch)
            out[s:e] = pq.distance(pts[s:e])

    sdf = out.reshape(res, res, res).astype(out_dtype, copy=False)

    if trunc is not None:
        t = float(trunc)
        np.clip(sdf, -t, t, out=sdf)

    voxel_size = float((bbox_max - bbox_min).max() / (res - 1))

    meta = {
        "version": 1,
        "res": int(res),
        "indexing": "ij",
        "signed": bool(signed),
        "sign_convention": "inside_negative",  # trimesh convention
        "padding": float(padding),
        "normalize_to_cube": bool(normalize_to_cube),
        "trunc": None if trunc is None else float(trunc),
        "bbox_min": bbox_min.astype(np.float32).tolist(),
        "bbox_max": bbox_max.astype(np.float32).tolist(),
        "voxel_size": float(voxel_size),
    }
    if transform is not None:
        meta["transform"] = transform

    return sdf, meta


def iter_stl_files(root_dir: Path):
    # 大文字拡張子にも対応
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".stl":
            yield p


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert STL meshes to voxel SDF and save as NPZ with mirrored folder structure."
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Input root directory to search STL files.")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output root directory to save NPZ files.")
    parser.add_argument("--res", type=int, default=128, help="Voxel resolution (default: 128).")
    parser.add_argument("--padding", type=float, default=0.05, help="BBox padding ratio (default: 0.05).")
    parser.add_argument(
        "--normalize_to_cube",
        action="store_true",
        help="Normalize mesh to have max bbox edge length = 1 and centered at origin before voxelization.",
    )
    parser.add_argument(
        "--signed_policy",
        choices=["auto", "force", "off"],
        default="auto",
        help="Signed SDF policy. auto: signed only if watertight; force: always signed; off: always unsigned.",
    )
    parser.add_argument("--trunc", type=float, default=None, help="Optional truncation value for TSDF (clamp).")
    parser.add_argument("--batch", type=int, default=200_000, help="Query batch size (default: 200000).")
    parser.add_argument(
        "--dtype",
        choices=["f16", "f32"],
        default="f32",
        help="Output SDF dtype (default: f32).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npz outputs.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    out_dtype = np.float16 if args.dtype == "f16" else np.float32

    # Lazy import (heavy)
    import trimesh

    stl_list = list(iter_stl_files(root_dir))
    if not args.quiet:
        print(f"Found {len(stl_list)} STL files under: {root_dir}")

    n_ok, n_skip, n_fail = 0, 0, 0

    for i, stl_path in enumerate(stl_list, start=1):
        rel = stl_path.relative_to(root_dir)
        out_path = (dst_dir / rel).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            if not args.quiet:
                print(f"[{i}/{len(stl_list)}] SKIP exists: {out_path}")
            continue

        try:
            mesh = trimesh.load(str(stl_path), force="mesh")
            mesh = _ensure_trimesh(mesh)

            # なるべく素の形状を使う（process=Trueだと修復が入ることがある）
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            is_watertight = bool(mesh.is_watertight)

            if args.signed_policy == "off":
                use_signed = False
            elif args.signed_policy == "force":
                use_signed = True
            else:
                use_signed = is_watertight  # auto

            sdf, meta = mesh_to_voxel_sdf_trimesh(
                verts=verts,
                faces=faces,
                res=args.res,
                padding=args.padding,
                normalize_to_cube=args.normalize_to_cube,
                trunc=args.trunc,
                signed=use_signed,
                batch=args.batch,
                out_dtype=out_dtype,
            )

            # 追加の追跡情報（npzに入れる）
            meta["source_relpath"] = str(rel).replace(os.sep, "/")
            meta["source_abspath"] = str(stl_path)
            meta["watertight"] = is_watertight
            meta["signed_policy"] = args.signed_policy
            meta_str = json.dumps(meta, ensure_ascii=False)

            np.savez_compressed(
                out_path,
                sdf=sdf,
                meta=np.string_(meta_str),
                # よく使うものはトップレベルにも置く（読込を楽にする）
                bbox_min=np.asarray(meta["bbox_min"], dtype=np.float32),
                bbox_max=np.asarray(meta["bbox_max"], dtype=np.float32),
                voxel_size=np.asarray(meta["voxel_size"], dtype=np.float32),
                res=np.asarray(meta["res"], dtype=np.int32),
                signed=np.asarray(int(meta["signed"]), dtype=np.int8),
            )

            n_ok += 1
            if not args.quiet:
                print(
                    f"[{i}/{len(stl_list)}] OK: {rel} -> {out_path.relative_to(dst_dir)} "
                    f"(signed={use_signed}, watertight={is_watertight})"
                )

        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(stl_list)}] FAIL: {stl_path}\n  {type(e).__name__}: {e}")

    print(f"Done. ok={n_ok}, skip={n_skip}, fail={n_fail}, dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
