#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python stl_to_sdf_npz.py  --root_dir /mnt/ssd2tb/new_format --dst_dir /mnt/ssd2tb/new_format_sdf --normalize_to_cube \
--repair --merge_digits 6 --morph_close_iters 1 --res 128 --workers 10 --distance_method mesh --dtype f16
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

MAX_POINTS_PER_CHUNK = 1_000_000

def _ensure_trimesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Empty trimesh.Scene (no geometry).")
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError("Scene contains no Trimesh geometry.")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not trimesh.Trimesh: {type(mesh)}")
    return mesh


def iter_stl_files(root_dir: Path):
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".stl":
            yield p


def watertight_report(mesh: trimesh.Trimesh) -> Dict:
    """
    trimesh 4.x で安定して取れる情報を使って report を作る．
    edges_unique_inverse を bincount して boundary/nonmanifold を数える．
    """
    inv = np.asarray(mesh.edges_unique_inverse)
    m = int(np.asarray(mesh.edges_unique).shape[0])
    counts = np.bincount(inv, minlength=m)

    n_boundary = int((counts == 1).sum())
    n_nonmanifold = int((counts > 2).sum())

    return {
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "n_boundary_edges": n_boundary,
        "n_nonmanifold_edges": n_nonmanifold,
    }

def drop_faces_touching_nonmanifold(mesh: trimesh.Trimesh) -> int:
    """
    non-manifold edge（count>2）に触れている face を削除する．
    戻り値は削除face数．
    """
    inv = np.asarray(mesh.edges_unique_inverse)
    edges_u = np.asarray(mesh.edges_unique)
    counts = np.bincount(inv, minlength=int(edges_u.shape[0]))

    nm_edges = edges_u[counts > 2]
    if nm_edges.shape[0] == 0:
        return 0

    nm_set = {tuple(e) for e in np.sort(nm_edges, axis=1)}

    f = np.asarray(mesh.faces, dtype=np.int64)
    e0 = np.sort(f[:, [0, 1]], axis=1)
    e1 = np.sort(f[:, [1, 2]], axis=1)
    e2 = np.sort(f[:, [2, 0]], axis=1)

    bad = np.fromiter((tuple(a) in nm_set for a in e0), count=f.shape[0], dtype=bool)
    bad |= np.fromiter((tuple(a) in nm_set for a in e1), count=f.shape[0], dtype=bool)
    bad |= np.fromiter((tuple(a) in nm_set for a in e2), count=f.shape[0], dtype=bool)

    n_bad = int(bad.sum())
    if n_bad > 0:
        mesh.update_faces(~bad)
        mesh.remove_unreferenced_vertices()
    return n_bad


def _sanitize_vertices_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    area_eps_rel: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    warns: List[str] = []

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    V = v.shape[0]
    F = f.shape[0]
    if V == 0 or F == 0:
        raise ValueError("Empty vertices/faces.")

    # (1) face index が範囲内か
    in_range = (f >= 0).all(axis=1) & (f < V).all(axis=1)
    if not in_range.all():
        bad = int((~in_range).sum())
        warns.append(f"sanitize: drop {bad}/{F} faces (out-of-range indices)")
        f = f[in_range]
        if f.shape[0] == 0:
            raise ValueError("All faces dropped (out-of-range).")

    # (2) NaN/Inf 頂点を除去（それに触れるfaceも落とす）
    finite_v = np.isfinite(v).all(axis=1)
    if not finite_v.all():
        badv = int((~finite_v).sum())
        warns.append(f"sanitize: found {badv}/{V} non-finite vertices (NaN/Inf). Removing.")
        old_to_new = -np.ones((V,), dtype=np.int64)
        keep = np.where(finite_v)[0]
        old_to_new[keep] = np.arange(keep.shape[0], dtype=np.int64)

        f2 = old_to_new[f]
        okf = (f2 >= 0).all(axis=1)
        dropf = int((~okf).sum())
        if dropf > 0:
            warns.append(f"sanitize: drop {dropf}/{f.shape[0]} faces touching non-finite vertices")

        f = f2[okf]
        v = v[finite_v]
        if f.shape[0] == 0:
            raise ValueError("All faces dropped (non-finite vertices).")

    # (3) 同一頂点を含む退化faceを落とす
    deg = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    if deg.any():
        drop = int(deg.sum())
        warns.append(f"sanitize: drop {drop}/{f.shape[0]} degenerate faces (repeated indices)")
        f = f[~deg]
        if f.shape[0] == 0:
            raise ValueError("All faces dropped (degenerate).")

    # (4) 面積ほぼ0の三角形を落とす（ゼロ割の温床）
    tri = v[f]  # (F,3,3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)  # 2*area

    scale = float(np.ptp(v, axis=0).max() if v.shape[0] > 0 else 1.0)
    area_eps = area_eps_rel * max(scale * scale, 1e-30)
    ok = area2 > area_eps
    if not ok.all():
        drop = int((~ok).sum())
        warns.append(f"sanitize: drop {drop}/{f.shape[0]} near-zero-area faces (eps={area_eps:g})")
        f = f[ok]
        if f.shape[0] == 0:
            raise ValueError("All faces dropped (zero-area).")

    return v, f, warns


def repair_mesh_inplace(mesh: trimesh.Trimesh, fill_holes: bool = True, merge_digits: Optional[int] = None) -> List[str]:
    """
    mesh.process(validate=True) を使わない安全修復．
    戻り値：warning文字列リスト（metaに積む用）
    """
    warns: List[str] = []

    v, f, w = _sanitize_vertices_faces(mesh.vertices, mesh.faces)
    warns += w

    # キャッシュ不整合を避けるため作り直す
    mesh.vertices = v
    mesh.faces = f

    # いったん完全に新規Trimeshに置換する方がさらに安全
    tmp = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.vertices = tmp.vertices
    mesh.faces = tmp.faces

    # STL割れ対策
    try:
        if merge_digits is not None:
            mesh.merge_vertices(digits_vertex=int(merge_digits))
        else:
            mesh.merge_vertices()
    except Exception as e:
        warns.append(f"repair: merge_vertices failed ({type(e).__name__}: {e})")

    # 小穴埋め（効く範囲は限定的）
    if fill_holes:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception as e:
            warns.append(f"repair: fill_holes failed ({type(e).__name__}: {e})")

    # ★重要：fix_normals / process(validate=True) は呼ばない
    return warns


def compute_cube_bounds(mesh: trimesh.Trimesh, padding: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    等方な立方体bboxを作る（最大辺長基準）．
    cube_size = max_extent + 2*pad，pad = padding*max_extent
    """
    bmin, bmax = mesh.bounds
    center = 0.5 * (bmin + bmax)
    extent = (bmax - bmin)
    max_extent = float(extent.max() if extent.max() > 0 else 1.0)
    pad = float(padding) * max_extent
    half = 0.5 * max_extent + pad
    cube_min = center - half
    cube_max = center + half
    cube_size = float((cube_max - cube_min).max())
    return cube_min, cube_max, cube_size


def compute_grid_from_mesh(
    mesh: trimesh.Trimesh,
    res: int,
    padding: float,
    center_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Build a uniform grid centered at mesh bbox center (or override).
    Returns (grid_min, grid_max, voxel_size, grid_center, max_extent).
    """
    if res < 2:
        raise ValueError("res must be >= 2 to build a grid.")

    bmin, bmax = mesh.bounds
    center = 0.5 * (bmin + bmax)
    if center_override is not None:
        center = np.asarray(center_override, dtype=np.float64)

    extent = (bmax - bmin)
    max_extent = float(extent.max() if extent.max() > 0 else 1.0)
    pad = float(padding) * max_extent
    half = 0.5 * max_extent + pad

    voxel_size = float((2.0 * half) / float(res - 1))
    grid_min = center - half
    grid_max = grid_min + voxel_size * float(res - 1)
    grid_center = 0.5 * (grid_min + grid_max)
    return grid_min, grid_max, voxel_size, grid_center, max_extent


def build_grid_axes(grid_min: np.ndarray, voxel_size: float, res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(res, dtype=np.float64)
    xs = grid_min[0] + idx * float(voxel_size)
    ys = grid_min[1] + idx * float(voxel_size)
    zs = grid_min[2] + idx * float(voxel_size)
    return xs, ys, zs


def iter_grid_chunks(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_points: int = MAX_POINTS_PER_CHUNK,
):
    plane = int(ys.size) * int(zs.size)
    if plane <= 0:
        return
    chunk = max(1, int(max_points // plane))
    for i0 in range(0, int(xs.size), chunk):
        i1 = min(int(xs.size), i0 + chunk)
        xi = xs[i0:i1]
        xg, yg, zg = np.meshgrid(xi, ys, zs, indexing="ij")
        pts = np.stack((xg, yg, zg), axis=-1).reshape(-1, 3)
        block_shape = (int(i1 - i0), int(ys.size), int(zs.size))
        yield i0, i1, pts, block_shape


def voxel_surface_mask_from_trimesh(
    mesh: trimesh.Trimesh,
    cube_min: np.ndarray,
    voxel_size: float,
    res: int,
) -> np.ndarray:
    """
    mesh.voxelized(pitch) の surface voxel centers を，こちらの (res,res,res) グリッドにマップして surface mask を作る．
    """
    vox = mesh.voxelized(pitch=float(voxel_size))
    pts = np.asarray(vox.points)  # (K,3) voxel centers（surface寄り）

    # index = round((p - cube_min) / voxel_size)
    ijk = np.rint((pts - cube_min) / float(voxel_size)).astype(np.int32)

    valid = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < res) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < res) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < res)
    )
    ijk = ijk[valid]

    surf = np.zeros((res, res, res), dtype=bool)
    if ijk.shape[0] > 0:
        surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return surf


def signed_mask_via_propagation(surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    surface（障壁）から outside を境界floodで求める（SciPyのC実装で高速）．
    inside = (~surface) & (~outside)
    """
    from scipy.ndimage import binary_propagation, generate_binary_structure

    free = ~surface
    seed = np.zeros_like(surface, dtype=bool)

    # 境界面からfree領域を種にする
    seed[0, :, :] = free[0, :, :]
    seed[-1, :, :] = free[-1, :, :]
    seed[:, 0, :] = free[:, 0, :]
    seed[:, -1, :] = free[:, -1, :]
    seed[:, :, 0] = free[:, :, 0]
    seed[:, :, -1] = free[:, :, -1]

    struct = generate_binary_structure(rank=3, connectivity=1)  # 6近傍
    outside = binary_propagation(seed, structure=struct, mask=free)
    inside = free & (~outside)
    return inside, outside


def compute_sdf_mesh_distance_open3d(
    mesh: trimesh.Trimesh,
    grid_min: np.ndarray,
    voxel_size: float,
    res: int,
    want_signed: bool,
    out_dtype: np.dtype,
    out_sdf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, List[str]]:
    """
    Compute distance using Open3D RaycastingScene (often more memory stable).
    """
    warns: List[str] = []
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for mesh_backend=open3d.") from e

    if out_sdf is None:
        sdf = np.empty((res, res, res), dtype=out_dtype)
    else:
        sdf = out_sdf

    # Build Open3D triangle mesh
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v.astype(np.float64)),
        o3d.utility.Vector3iVector(f),
    )
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    if "positions" in mesh_t.vertex:
        mesh_t.vertex["positions"] = mesh_t.vertex["positions"].to(o3d.core.Dtype.Float32)
    if "indices" in mesh_t.triangle and mesh_t.triangle["indices"].dtype != o3d.core.Dtype.Int32:
        mesh_t.triangle["indices"] = mesh_t.triangle["indices"].to(o3d.core.Dtype.Int32)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    has_signed = hasattr(scene, "compute_signed_distance")
    has_distance = hasattr(scene, "compute_distance")
    if want_signed and not has_signed:
        warns.append("open3d: compute_signed_distance unavailable. Fallback to unsigned.")
        want_signed = False
    if not has_signed and not has_distance:
        raise RuntimeError("open3d RaycastingScene has no distance API.")

    xs, ys, zs = build_grid_axes(grid_min, voxel_size, res)
    neg_count = 0
    pos_count = 0

    for i0, i1, pts, block_shape in iter_grid_chunks(xs, ys, zs, max_points=MAX_POINTS_PER_CHUNK):
        t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        if want_signed and has_signed:
            vals = scene.compute_signed_distance(t).numpy()
        else:
            if has_distance:
                vals = scene.compute_distance(t).numpy()
            else:
                vals = np.abs(scene.compute_signed_distance(t).numpy())

        neg_count += int((vals < 0).sum())
        pos_count += int((vals > 0).sum())
        sdf[i0:i1, :, :] = vals.reshape(block_shape).astype(out_dtype, copy=False)
        del t, pts, vals

    signed_used = bool(want_signed and (neg_count > 0) and (pos_count > 0))
    if want_signed and not signed_used:
        warns.append("open3d: Signed labeling failed. Fallback to unsigned.")
        np.abs(sdf, out=sdf)
    return sdf, signed_used, warns


def compute_sdf_edt(
    mesh: trimesh.Trimesh,
    res: int,
    grid_min: np.ndarray,
    voxel_size: float,
    want_signed: bool,
    out_dtype: np.dtype,
    morph_close_iters: int,
) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    EDT-based SDF using surface voxel mask.
    """
    from scipy.ndimage import distance_transform_edt, binary_closing, generate_binary_structure

    warns: List[str] = []

    # surface mask
    surf = voxel_surface_mask_from_trimesh(mesh, cube_min=grid_min, voxel_size=voxel_size, res=res)

    # 低解像度でshellが漏れる対策（必要なら）
    if morph_close_iters > 0:
        struct = generate_binary_structure(3, 1)
        for _ in range(int(morph_close_iters)):
            surf = binary_closing(surf, structure=struct)

    # EDT（surfaceが0，その他が1で距離を計算）
    dist = distance_transform_edt(~surf, sampling=(voxel_size, voxel_size, voxel_size)).astype(out_dtype, copy=False)

    if want_signed:
        inside, outside = signed_mask_via_propagation(surf)

        if inside.sum() == 0:
            # watertight なら fill で占有（内部）を作って符号付けに使う
            try:
                vox_filled = mesh.voxelized(pitch=float(voxel_size), method="subdivide").fill()
                occ_pts = np.asarray(vox_filled.points)  # filled voxel centers
                ijk = np.rint((occ_pts - grid_min) / float(voxel_size)).astype(np.int32)

                valid = (
                    (ijk[:, 0] >= 0) & (ijk[:, 0] < res) &
                    (ijk[:, 1] >= 0) & (ijk[:, 1] < res) &
                    (ijk[:, 2] >= 0) & (ijk[:, 2] < res)
                )
                ijk = ijk[valid]
                occ = np.zeros((res, res, res), dtype=bool)
                if ijk.shape[0] > 0:
                    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

                if occ.sum() == 0:
                    warns.append("Signed labeling failed: fill() produced empty occupancy in target grid. Fallback to unsigned.")
                    sdf = dist
                    signed_used = False
                else:
                    sdf = dist.copy()
                    sdf[occ] = -sdf[occ]  # inside negative
                    signed_used = True
                    warns.append("Signed labeling: fallback used voxelized(...).fill() occupancy.")
            except Exception as e:
                warns.append(f"Signed labeling failed: fill() fallback error ({type(e).__name__}: {e}). Fallback to unsigned.")
                sdf = dist
                signed_used = False
        else:
            sign = np.ones_like(dist, dtype=np.int8)
            sign[inside] = -1
            sdf = dist * sign.astype(dist.dtype)
            signed_used = True
    else:
        sdf = dist
        signed_used = False

    return sdf, {"signed_used": bool(signed_used)}, warns


def compute_sdf(
    mesh: trimesh.Trimesh,
    res: int,
    padding: float,
    normalize_to_cube: bool,
    trunc: Optional[float],
    signed_policy: str,  # auto, force, off
    out_dtype: np.dtype,
    morph_close_iters: int,
    repair: bool,
    merge_digits: Optional[int],
    distance_method: str,  # mesh, edt
    sdf_out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Build SDF volume and metadata.
    """
    warns: List[str] = []

    # normalize_to_cube（最大辺長を1にして中心原点へ）
    transform = None
    if normalize_to_cube:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        center = 0.5 * (vmin + vmax)
        extent = float((vmax - vmin).max())
        if extent <= 0:
            raise ValueError("Degenerate mesh bbox extent.")
        scale = 1.0 / extent
        v2 = (v - center) * scale
        mesh = trimesh.Trimesh(vertices=v2, faces=np.asarray(mesh.faces), process=False)
        transform = {"center": center.tolist(), "scale": float(scale)}

    if repair:
        warns += repair_mesh_inplace(mesh, fill_holes=True, merge_digits=merge_digits)

    rep = watertight_report(mesh)
    print(rep)

    # decide signed policy
    if signed_policy == "off":
        want_signed = False
    elif signed_policy == "force":
        want_signed = True
    else:
        want_signed = bool(rep["is_watertight"])

    if signed_policy == "auto" and not rep["is_watertight"]:
        warns.append(
            f"Non-watertight by trimesh check (auto->unsigned) "
            f"(boundary_edges={rep['n_boundary_edges']}, nonmanifold_edges={rep['n_nonmanifold_edges']})"
        )

    # grid (centered; normalize_to_cube -> center at 0)
    center_override = np.zeros(3, dtype=np.float64) if normalize_to_cube else None
    grid_min, grid_max, voxel_size, grid_center, _ = compute_grid_from_mesh(
        mesh, res=res, padding=padding, center_override=center_override
    )

    vol_bytes = int(res) ** 3 * np.dtype(out_dtype).itemsize
    if vol_bytes >= (1 << 30):
        warns.append(
            f"SDF volume size ≈ {vol_bytes / (1 << 30):.2f} GiB. "
            "Consider lower res or dtype=f16."
        )

    if (res % 2) == 0:
        warns.append("res is even: grid center falls between voxels. Use odd res to place a voxel center at origin.")

    if distance_method == "mesh":
        if morph_close_iters > 0:
            warns.append("morph_close_iters ignored for distance_method=mesh.")
        try:
            sdf, signed_used, w = compute_sdf_mesh_distance_open3d(
                mesh=mesh,
                grid_min=grid_min,
                voxel_size=voxel_size,
                res=res,
                want_signed=want_signed,
                out_dtype=out_dtype,
                out_sdf=sdf_out,
            )
            warns += w
            surface_meta = {"source": "open3d_distance", "morph_close_iters": int(morph_close_iters)}
        except Exception as e:
            warns.append(f"open3d distance failed ({type(e).__name__}: {e}). Fallback to edt.")
            sdf, info, w = compute_sdf_edt(
                mesh=mesh,
                res=res,
                grid_min=grid_min,
                voxel_size=voxel_size,
                want_signed=want_signed,
                out_dtype=out_dtype,
                morph_close_iters=morph_close_iters,
            )
            warns += w
            signed_used = bool(info.get("signed_used", False))
            surface_meta = {"source": "trimesh.voxelized", "morph_close_iters": int(morph_close_iters)}
    else:
        sdf, info, w = compute_sdf_edt(
            mesh=mesh,
            res=res,
            grid_min=grid_min,
            voxel_size=voxel_size,
            want_signed=want_signed,
            out_dtype=out_dtype,
            morph_close_iters=morph_close_iters,
        )
        warns += w
        signed_used = bool(info.get("signed_used", False))
        surface_meta = {"source": "trimesh.voxelized", "morph_close_iters": int(morph_close_iters)}

    if trunc is not None:
        t = float(trunc)
        np.clip(sdf, -t, t, out=sdf)

    meta = {
        "version": 2,
        "method": "mesh_distance" if distance_method == "mesh" else "edt_voxel",
        "distance_method": str(distance_method),
        "res": int(res),
        "indexing": "ij",
        "padding": float(padding),
        "normalize_to_cube": bool(normalize_to_cube),
        "trunc": None if trunc is None else float(trunc),
        "bbox_min": grid_min.astype(np.float32).tolist(),
        "bbox_max": grid_max.astype(np.float32).tolist(),
        "voxel_size": float(voxel_size),
        "grid_center": grid_center.astype(np.float32).tolist(),
        "center_index": float((res - 1) * 0.5),
        "center_is_voxel": bool((res % 2) == 1),
        "watertight_report": rep,
        "signed_policy": signed_policy,
        "signed_used": bool(signed_used),
        "sign_convention": "inside_negative" if signed_used else "unsigned",
        "surface_voxel": surface_meta,
        "repair": {
            "enabled": bool(repair),
            "merge_digits": None if merge_digits is None else int(merge_digits),
        },
    }
    if transform is not None:
        meta["transform"] = transform

    return sdf, meta, warns


def replace_dirname(path: Path, src: str = "stl", dst: str = "sdf") -> Path:
    parts = [dst if p == src else p for p in path.parts]
    return Path(*parts)


def process_one_stl(stl_path_str: str, root_dir_str: str, dst_dir_str: str, params: Dict) -> Dict:
    """
    worker側．返り値は main が表示するための情報．
    """
    stl_path = Path(stl_path_str)
    root_dir = Path(root_dir_str)
    dst_dir = Path(dst_dir_str)

    rel = stl_path.relative_to(root_dir)
    rel_out = replace_dirname(rel, src="stl", dst="sdf")
    out_path = (dst_dir / rel_out).with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and (not params["overwrite"]):
        return {"status": "skip", "rel": str(rel), "out": str(out_path), "warns": [], "msg": "exists"}

    mesh = None
    sdf = None
    meta = None
    try:
        mesh = trimesh.load(str(stl_path), force="mesh", process=False)
        mesh = _ensure_trimesh(mesh)

        pre_warns: List[str] = []

        distance_method = params["distance_method"]

        sdf, meta, warns = compute_sdf(
            mesh=mesh,
            res=params["res"],
            padding=params["padding"],
            normalize_to_cube=params["normalize_to_cube"],
            trunc=params["trunc"],
            signed_policy=params["signed_policy"],
            out_dtype=params["out_dtype"],
            morph_close_iters=params["morph_close_iters"],
            repair=params["repair"],
            merge_digits=params["merge_digits"],
            distance_method=distance_method,
            sdf_out=None,
        )
        warns = pre_warns + warns

        meta["source_relpath"] = str(rel).replace(os.sep, "/")
        meta["source_abspath"] = str(stl_path)

        meta_bytes = np.bytes_(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        # signed を強制するなら，signed_used=False は失敗扱い
        if params["signed_policy"] == "force" and (not meta.get("signed_used", False)):
            raise RuntimeError(
                "signed_policy=force but signed_used=False "
                f"(likely labeling failed). watertight_report={meta.get('watertight_report')}"
            )
        # 主要値はトップレベルにも置く（読むとき楽）
        np.savez_compressed(
            out_path,
            sdf=sdf,
            meta=meta_bytes,
            bbox_min=np.asarray(meta["bbox_min"], dtype=np.float32),
            bbox_max=np.asarray(meta["bbox_max"], dtype=np.float32),
            voxel_size=np.asarray(meta["voxel_size"], dtype=np.float32),
            res=np.asarray(meta["res"], dtype=np.int32),
            signed=np.asarray(int(meta["signed_used"]), dtype=np.int8),
        )

        return {"status": "ok", "rel": str(rel), "out": str(out_path), "warns": warns, "msg": ""}

    except Exception as e:
        tb = traceback.format_exc(limit=10)
        return {"status": "fail", "rel": str(rel), "out": str(out_path), "warns": [], "msg": f"{type(e).__name__}: {e}\n{tb}"}
    finally:
        mesh = None
        sdf = None
        meta = None


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert STL meshes to voxel SDF using EDT, and save as NPZ with mirrored folder structure. Supports multiprocessing."
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)

    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--normalize_to_cube", action="store_true")

    parser.add_argument("--signed_policy", choices=["auto", "force", "off"], default="auto")
    parser.add_argument("--trunc", type=float, default=None)

    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # EDT用補助
    parser.add_argument(
        "--distance_method",
        choices=["mesh", "edt"],
        default="mesh",
        help="Distance computation method: mesh (direct distance) or edt (voxel surface + EDT).",
    )
    # memory-related tuning arguments removed for speed-first workflow
    parser.add_argument(
        "--morph_close_iters",
        type=int,
        default=0,
        help="Optional morphological closing iterations on surface voxels to reduce leaks (default: 0).",
    )

    # 修復（watertight改善用）
    parser.add_argument("--repair", action="store_true", help="Apply light mesh repair (process+merge+fill_holes).")
    parser.add_argument(
        "--merge_digits",
        type=int,
        default=None,
        help="Strengthen vertex welding by rounding digits (scale-dependent). Example: 6.",
    )

    # multiprocessing
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1).")

    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    out_dtype = np.float16 if args.dtype == "f16" else np.float32

    stl_list = list(iter_stl_files(root_dir))
    if not args.quiet:
        print(f"Found {len(stl_list)} STL files under: {root_dir}")
        print(
            f"workers={args.workers}, res={args.res}, signed_policy={args.signed_policy}, "
            f"dtype={args.dtype}, distance_method={args.distance_method}"
        )

    params = {
        "res": int(args.res),
        "padding": float(args.padding),
        "normalize_to_cube": bool(args.normalize_to_cube),
        "signed_policy": str(args.signed_policy),
        "trunc": None if args.trunc is None else float(args.trunc),
        "out_dtype": out_dtype,
        "overwrite": bool(args.overwrite),
        "distance_method": str(args.distance_method),
        "morph_close_iters": int(args.morph_close_iters),
        "repair": bool(args.repair),
        "merge_digits": None if args.merge_digits is None else int(args.merge_digits),
    }

    n_ok = n_skip = n_fail = 0

    if args.workers <= 1:
        for i, p in enumerate(stl_list, start=1):
            r = process_one_stl(str(p), str(root_dir), str(dst_dir), params)
            if r["status"] == "ok":
                n_ok += 1
                if not args.quiet:
                    print(f"[{i}/{len(stl_list)}] OK: {r['rel']} -> {Path(r['out']).relative_to(dst_dir)}")
                for w in r["warns"]:
                    print(f"  warning: {r['rel']}: {w}")
            elif r["status"] == "skip":
                n_skip += 1
                if not args.quiet:
                    print(f"[{i}/{len(stl_list)}] SKIP: {r['rel']} (exists)")
            else:
                n_fail += 1
                print(f"[{i}/{len(stl_list)}] FAIL: {r['rel']}\n{r['msg']}")
    else:
        # multiprocessing
        with ProcessPoolExecutor(max_workers=int(max(1, args.workers))) as ex:
            futs = []
            for p in stl_list:
                futs.append(ex.submit(process_one_stl, str(p), str(root_dir), str(dst_dir), params))

            for i, fut in enumerate(as_completed(futs), start=1):
                r = fut.result()
                if r["status"] == "fail":
                    print(f"[{i}/{len(futs)}] FAIL: {r['rel']}\n{r['msg']}")

                    # まだ開始していないジョブをキャンセル
                    for f in futs:
                        f.cancel()

                    # Python 3.9+ なら，未開始futureをまとめて破棄できる
                    ex.shutdown(wait=False, cancel_futures=True)

                    sys.exit(1)
                if r["status"] == "ok":
                    n_ok += 1
                    if not args.quiet:
                        print(f"[{i}/{len(futs)}] OK: {r['rel']}")
                    for w in r["warns"]:
                        print(f"  warning: {r['rel']}: {w}")
                elif r["status"] == "skip":
                    n_skip += 1
                    if not args.quiet:
                        print(f"[{i}/{len(futs)}] SKIP: {r['rel']} (exists)")
                else:
                    n_fail += 1
                    print(f"[{i}/{len(futs)}] FAIL: {r['rel']}\n{r['msg']}")

    print(f"Done. ok={n_ok}, skip={n_skip}, fail={n_fail}, dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
    
