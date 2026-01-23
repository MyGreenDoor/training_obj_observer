#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python sdf_npz_to_stl.py --npz xxx.npz --out yyy.stl --restore_size

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh


def load_npz(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    sdf = z["sdf"]
    meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    return sdf, meta


def load_stl_as_trimesh(stl_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"Empty trimesh.Scene: {stl_path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not trimesh.Trimesh: {type(mesh)}")
    return mesh


def compute_normalize_to_cube_transform_from_mesh(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """
    SDF生成側の normalize_to_cube と同じ定義:
      center = 0.5*(vmin+vmax)
      scale  = 1.0 / max_extent
      v_norm = (v - center) * scale
    逆変換:
      v = v_norm / scale + center
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = float((vmax - vmin).max())
    if extent <= 0:
        raise ValueError("Degenerate reference mesh bbox extent.")
    scale = 1.0 / extent
    return center, float(scale)


def get_inverse_transform(meta: dict, ref_stl: Optional[Path]) -> Tuple[np.ndarray, float]:
    """
    逆変換に使う (center, scale) を返す．
    優先順位:
      1) meta["transform"]
      2) ref_stl から算出
    """
    if meta.get("normalize_to_cube", False) and ("transform" in meta):
        center = np.asarray(meta["transform"]["center"], dtype=np.float64)
        scale = float(meta["transform"]["scale"])
        return center, scale

    if ref_stl is not None:
        ref_mesh = load_stl_as_trimesh(ref_stl)
        center, scale = compute_normalize_to_cube_transform_from_mesh(ref_mesh)
        return center, scale

    raise RuntimeError(
        "restore_size requested but transform is unavailable. "
        "Either (a) generate SDF with meta['transform'] stored, or (b) pass --ref_stl."
    )


def main():
    ap = argparse.ArgumentParser(description="Convert voxel SDF(.npz) to STL via marching cubes.")
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument(
        "--level",
        type=float,
        default=None,
        help="Isosurface level for marching cubes. Default: 0.0 if signed_used else 0.5*voxel_size.",
    )
    ap.add_argument(
        "--method",
        choices=["lewiner", "lorensen"],
        default="lewiner",
        help="marching cubes method (skimage).",
    )
    ap.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for marching cubes. Larger is faster but lower quality.",
    )

    # 互換のため残す（内部的に restore_size と同義にする）
    ap.add_argument(
        "--to_original_scale",
        action="store_true",
        help="(Deprecated) Same as --restore_size. Invert normalize_to_cube using meta['transform'].",
    )

    # 新しい「サイズ復元」系
    ap.add_argument(
        "--restore_size",
        action="store_true",
        help="Restore original scale (and translation) if SDF was generated with normalize_to_cube. "
             "Uses meta['transform'] if present, else requires --ref_stl.",
    )
    ap.add_argument(
        "--ref_stl",
        type=str,
        default=None,
        help="Reference STL to recover normalize_to_cube transform when meta lacks 'transform'.",
    )
    ap.add_argument(
        "--restore_scale_only",
        action="store_true",
        help="Only restore scale (keep current origin). Useful if you only care about size.",
    )

    ap.add_argument(
        "--process",
        action="store_true",
        help="Let trimesh process the mesh (can fix normals etc., but may alter mesh).",
    )

    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sdf, meta = load_npz(npz_path)

    bbox_min = np.asarray(meta["bbox_min"], dtype=np.float64)
    voxel_size = float(meta["voxel_size"])
    signed_used = bool(meta.get("signed_used", False))

    # level の既定
    if args.level is None:
        level = 0.0 if signed_used else 0.5 * voxel_size
    else:
        level = float(args.level)

    # marching cubes
    try:
        from skimage import measure
    except Exception as e:
        raise RuntimeError("scikit-image が必要です．pip install scikit-image") from e

    verts, faces, normals, values = measure.marching_cubes(
        volume=sdf,
        level=level,
        spacing=(voxel_size, voxel_size, voxel_size),
        method=args.method,
        step_size=int(args.step_size),
        allow_degenerate=False,
    )

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError(
            f"Empty mesh from marching cubes (level={level}). "
            f"Try --level {0.5*voxel_size:.6g} or smaller/larger."
        )

    # bbox_min を足して world 座標へ（spacing 済みなので index->metric は済んでいる）
    verts_world = verts.astype(np.float64, copy=False) + bbox_min[None, :]

    # サイズ復元（normalize_to_cube の逆変換）
    do_restore = bool(args.restore_size or args.to_original_scale)
    if do_restore:
        if not meta.get("normalize_to_cube", False):
            print("note: meta.normalize_to_cube is False, so --restore_size has no effect.")
        else:
            ref_stl = Path(args.ref_stl).expanduser().resolve() if args.ref_stl else None
            center, scale = get_inverse_transform(meta, ref_stl)

            if args.restore_scale_only:
                verts_world = verts_world / scale
            else:
                verts_world = verts_world / scale + center

    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=bool(args.process))

    # STL 出力（拡張子で自動判定）
    mesh.export(str(out_path))

    print(f"OK: {npz_path} -> {out_path}")
    print(f"  verts={len(mesh.vertices)} faces={len(mesh.faces)} level={level} signed_used(meta)={signed_used}")
    print(f"  bbox_min={bbox_min.tolist()} voxel_size={voxel_size}")
    if do_restore and meta.get("normalize_to_cube", False):
        src = "meta.transform" if ("transform" in meta) else ("ref_stl" if args.ref_stl else "unknown")
        print(f"  restore_size: True (source={src}, scale_only={bool(args.restore_scale_only)})")


if __name__ == "__main__":
    main()
