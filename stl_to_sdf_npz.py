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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

MAX_POINTS_PER_CHUNK = 1_000_000


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


def is_trimesh(mesh) -> bool:
    return isinstance(mesh, trimesh.Trimesh)


def get_vertices_faces(mesh) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, MeshData):
        return mesh.vertices, mesh.faces
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def make_mesh(vertices: np.ndarray, faces: np.ndarray, like=None):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if is_trimesh(like):
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    return MeshData(vertices=v, faces=f)

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


def load_mesh_open3d(stl_path: Path) -> MeshData:
    """
    Load STL using Open3D and return MeshData.
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for --mesh_backend open3d.") from e
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh is None:
        raise ValueError(f"Open3D failed to load: {stl_path}")
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        raise ValueError(f"Open3D loaded empty mesh: {stl_path}")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    return MeshData(vertices=v, faces=f)


def build_open3d_legacy_mesh(vertices: np.ndarray, faces: np.ndarray):
    """
    Build Open3D legacy TriangleMesh from vertices/faces.
    """
    import open3d as o3d
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f),
    )


def open3d_remesh_for_stability(
    mesh,
    subdivide_iters: int,
) -> Tuple[object, List[str]]:
    """
    Apply light Open3D cleanup and optional midpoint subdivision.
    This does not increase geometric accuracy; it improves numerical stability.
    """
    warns: List[str] = []
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for --o3d_clean/--o3d_subdivide.") from e

    v, f = get_vertices_faces(mesh)
    m = build_open3d_legacy_mesh(v, f)

    before_faces = int(len(m.triangles))
    if hasattr(m, "remove_degenerate_triangles"):
        m.remove_degenerate_triangles()
    if hasattr(m, "remove_duplicated_triangles"):
        m.remove_duplicated_triangles()
    if hasattr(m, "remove_duplicated_vertices"):
        m.remove_duplicated_vertices()
    if hasattr(m, "remove_non_manifold_edges"):
        m.remove_non_manifold_edges()
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()
    after_faces = int(len(m.triangles))
    if after_faces != before_faces:
        warns.append(f"open3d cleanup: faces {before_faces} -> {after_faces}")

    iters = int(max(0, subdivide_iters))
    if iters > 0:
        for _ in range(iters):
            m = m.subdivide_midpoint(number_of_iterations=1)
        warns.append(f"open3d subdivide_midpoint: iters={iters}")

    v2 = np.asarray(m.vertices, dtype=np.float64)
    f2 = np.asarray(m.triangles, dtype=np.int64)
    mesh2 = make_mesh(vertices=v2, faces=f2, like=mesh)
    return mesh2, warns


def meshfix_watertight(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, List[str]]:
    """
    Apply pymeshfix to enforce watertightness. This may alter geometry.
    """
    warns: List[str] = []
    try:
        from pymeshfix import MeshFix
    except Exception as e:
        raise RuntimeError("pymeshfix is required for --meshfix.") from e

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    before_faces = int(f.shape[0])
    mf = MeshFix(v, f)
    try:
        mf.repair(verbose=False)
    except TypeError:
        mf.repair()

    v2 = getattr(mf, "v", None)
    f2 = getattr(mf, "f", None)
    if v2 is None or f2 is None:
        raise RuntimeError("pymeshfix returned empty mesh.")
    f2 = np.asarray(f2, dtype=np.int64)
    after_faces = int(f2.shape[0])
    if after_faces == 0:
        warns.append("meshfix: produced empty mesh, fallback to original.")
        return mesh, warns
    if before_faces >= 200 and after_faces < max(50, int(0.1 * before_faces)):
        warns.append(
            f"meshfix: faces {before_faces} -> {after_faces} (too small), fallback to original."
        )
        return mesh, warns
    mesh2 = trimesh.Trimesh(vertices=np.asarray(v2), faces=f2, process=False)
    warns.append(f"meshfix: applied (faces {before_faces} -> {after_faces})")
    return mesh2, warns


def iter_stl_files(root_dir: Path):
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".stl":
            yield p


def edge_stats_from_faces(faces: np.ndarray) -> Tuple[int, int]:
    """
    Count boundary and non-manifold edges from faces.
    """
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return 0, 0
    e0 = f[:, [0, 1]]
    e1 = f[:, [1, 2]]
    e2 = f[:, [2, 0]]
    edges = np.concatenate([e0, e1, e2], axis=0)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_boundary = int((counts == 1).sum())
    n_nonmanifold = int((counts > 2).sum())
    return n_boundary, n_nonmanifold


def watertight_report(mesh) -> Dict:
    """
    Build watertight report from mesh connectivity.
    """
    v, f = get_vertices_faces(mesh)
    n_boundary, n_nonmanifold = edge_stats_from_faces(f)
    is_watertight = bool((n_boundary == 0) and (n_nonmanifold == 0))

    if is_trimesh(mesh):
        return {
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "is_volume": bool(mesh.is_volume),
            "n_boundary_edges": n_boundary,
            "n_nonmanifold_edges": n_nonmanifold,
        }

    return {
        "is_watertight": is_watertight,
        "is_winding_consistent": bool(is_watertight),
        "is_volume": bool(is_watertight),
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


def face_components_from_faces(faces: np.ndarray) -> List[np.ndarray]:
    """
    Compute connected components (face adjacency) from faces.
    """
    f = np.asarray(faces, dtype=np.int64)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return []

    e0 = f[:, [0, 1]]
    e1 = f[:, [1, 2]]
    e2 = f[:, [2, 0]]
    edges = np.concatenate([e0, e1, e2], axis=0)
    edges = np.sort(edges, axis=1)
    face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    face_sorted = face_ids[order]

    parent = np.arange(n_faces, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    start = 0
    while start < edges_sorted.shape[0]:
        end = start + 1
        while end < edges_sorted.shape[0] and (edges_sorted[end] == edges_sorted[start]).all():
            end += 1
        if end - start >= 2:
            base = int(face_sorted[start])
            for k in range(start + 1, end):
                union(base, int(face_sorted[k]))
        start = end

    comps: Dict[int, List[int]] = {}
    for i in range(n_faces):
        r = int(find(i))
        comps.setdefault(r, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in comps.values()]


def filter_mesh_components(
    mesh,
    min_faces: int = 0,
    keep_largest: bool = False,
) -> Tuple[object, List[str]]:
    """
    Remove small disconnected components by face count.
    """
    warns: List[str] = []
    v, f = get_vertices_faces(mesh)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return mesh, warns
    if int(min_faces) <= 0 and not bool(keep_largest):
        return mesh, warns
    if is_trimesh(mesh):
        try:
            from trimesh.graph import connected_components
            comps = connected_components(mesh.face_adjacency, nodes=np.arange(n_faces))
        except Exception as e:
            warns.append(f"component filter fallback: {type(e).__name__}: {e}")
            comps = face_components_from_faces(f)
    else:
        comps = face_components_from_faces(f)
    if len(comps) <= 1:
        return mesh, warns

    sizes = [len(c) for c in comps]
    keep_faces = np.zeros((n_faces,), dtype=bool)
    if bool(keep_largest):
        idx = int(np.argmax(sizes))
        keep_faces[np.asarray(comps[idx], dtype=np.int64)] = True
        if int(min_faces) > 0 and sizes[idx] < int(min_faces):
            warns.append(
                f"component filter: largest has {sizes[idx]} faces < min_faces={int(min_faces)}, kept anyway."
            )
    else:
        for comp, size in zip(comps, sizes):
            if int(size) >= int(min_faces):
                keep_faces[np.asarray(comp, dtype=np.int64)] = True
        if not keep_faces.any():
            warns.append("component filter: no component meets min_faces; keeping original mesh.")
            return mesh, warns

    if keep_faces.all():
        return mesh, warns

    if is_trimesh(mesh):
        new_mesh = mesh.copy()
        new_mesh.update_faces(keep_faces)
        new_mesh.remove_unreferenced_vertices()
        warns.append(f"component filter: faces {n_faces} -> {int(len(new_mesh.faces))}")
        return new_mesh, warns

    new_faces = f[keep_faces]
    used = np.unique(new_faces)
    index_map = np.full((v.shape[0],), -1, dtype=np.int64)
    index_map[used] = np.arange(used.shape[0], dtype=np.int64)
    new_vertices = v[used]
    new_faces = index_map[new_faces]
    warns.append(f"component filter: faces {n_faces} -> {int(new_faces.shape[0])}")
    return MeshData(vertices=new_vertices, faces=new_faces), warns


def compute_cube_bounds(mesh, padding: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    等方な立方体bboxを作る（最大辺長基準）．
    cube_size = max_extent + 2*pad，pad = padding*max_extent
    """
    v, _ = get_vertices_faces(mesh)
    bmin = v.min(axis=0)
    bmax = v.max(axis=0)
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
    mesh,
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

    v, _ = get_vertices_faces(mesh)
    bmin = v.min(axis=0)
    bmax = v.max(axis=0)
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


def build_supersample_offsets(voxel_size: float, supersample: int) -> Optional[np.ndarray]:
    """
    Build sub-voxel offsets for supersampling (centered within each voxel).
    """
    ss = int(max(1, supersample))
    if ss <= 1:
        return None
    offs_1d = (np.arange(ss, dtype=np.float32) + 0.5) / float(ss) - 0.5
    offsets = np.stack(np.meshgrid(offs_1d, offs_1d, offs_1d, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets *= float(voxel_size)
    return offsets


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


def reconstruct_sign_from_voxels(
    mesh: trimesh.Trimesh,
    grid_min: np.ndarray,
    voxel_size: float,
    res: int,
    morph_close_iters: int,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Reconstruct inside mask from voxelized surface and flood fill.
    """
    warns: List[str] = []
    try:
        from scipy.ndimage import binary_closing, generate_binary_structure
    except Exception as e:
        warns.append(f"sign reconstruct failed: scipy unavailable ({type(e).__name__}: {e}).")
        return None, warns

    surf = voxel_surface_mask_from_trimesh(mesh, cube_min=grid_min, voxel_size=voxel_size, res=res)
    if morph_close_iters > 0:
        struct = generate_binary_structure(3, 1)
        for _ in range(int(morph_close_iters)):
            surf = binary_closing(surf, structure=struct)

    inside, _ = signed_mask_via_propagation(surf)
    if int(inside.sum()) > 0:
        return inside, warns

    try:
        vox_filled = mesh.voxelized(pitch=float(voxel_size), method="subdivide").fill()
        occ_pts = np.asarray(vox_filled.points)
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
        if int(occ.sum()) == 0:
            warns.append("sign reconstruct failed: fill() produced empty occupancy.")
            return None, warns
        warns.append("sign reconstruct: used voxelized(...).fill() occupancy.")
        return occ, warns
    except Exception as e:
        warns.append(f"sign reconstruct failed: fill() error ({type(e).__name__}: {e}).")
        return None, warns


def cleanup_negative_islands(
    sdf: np.ndarray,
    voxel_size: float,
    near_surface_thresh_vox: float = 1.5,
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Flip negative components that do not touch the surface (|sdf| > threshold).
    """
    warns: List[str] = []
    try:
        from scipy.ndimage import generate_binary_structure, label, minimum
    except Exception as e:
        warns.append(f"sign cleanup skipped: scipy unavailable ({type(e).__name__}: {e}).")
        return sdf, 0, warns

    neg = sdf < 0
    if not np.any(neg):
        return sdf, 0, warns

    struct = generate_binary_structure(3, 1)
    labels, num = label(neg, structure=struct)
    if num == 0:
        return sdf, 0, warns

    abs_sdf = np.abs(sdf, dtype=sdf.dtype)
    mins = minimum(abs_sdf, labels=labels, index=np.arange(1, num + 1))
    thresh = float(near_surface_thresh_vox) * float(voxel_size)
    keep = mins <= thresh
    remove_labels = np.where(~keep)[0] + 1
    if remove_labels.size == 0:
        return sdf, 0, warns

    mask_remove = np.isin(labels, remove_labels)
    sdf = sdf.copy()
    sdf[mask_remove] = np.abs(sdf[mask_remove])
    return sdf, int(remove_labels.size), warns


def open3d_occupancy_sign(
    mesh,
    grid_min: np.ndarray,
    voxel_size: float,
    res: int,
    supersample: int = 1,
    mode: str = "center",
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Compute inside mask using Open3D RaycastingScene.compute_occupancy.
    """
    warns: List[str] = []
    try:
        import open3d as o3d
    except Exception as e:
        warns.append(f"occupancy sign failed: open3d unavailable ({type(e).__name__}: {e}).")
        return None, warns

    v, f = get_vertices_faces(mesh)
    legacy = build_open3d_legacy_mesh(v, f)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    if "positions" in mesh_t.vertex:
        mesh_t.vertex["positions"] = mesh_t.vertex["positions"].to(o3d.core.Dtype.Float32)
    if "indices" in mesh_t.triangle and mesh_t.triangle["indices"].dtype != o3d.core.Dtype.Int32:
        mesh_t.triangle["indices"] = mesh_t.triangle["indices"].to(o3d.core.Dtype.Int32)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    if not hasattr(scene, "compute_occupancy"):
        warns.append("occupancy sign failed: compute_occupancy not available.")
        return None, warns

    ss = int(max(1, supersample))
    mode = str(mode).lower()
    if mode not in ("center", "any", "majority"):
        warns.append(f"occupancy sign: unknown mode '{mode}', fallback to 'center'.")
        mode = "center"
    offsets = build_supersample_offsets(voxel_size, ss) if (ss > 1 and mode != "center") else None

    xs, ys, zs = build_grid_axes(grid_min, voxel_size, res)
    inside = np.zeros((res, res, res), dtype=bool)
    for i0, i1, pts, block_shape in iter_grid_chunks(xs, ys, zs, max_points=MAX_POINTS_PER_CHUNK):
        if offsets is None:
            t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
            occ = scene.compute_occupancy(t).numpy().reshape(block_shape).astype(bool)
            inside[i0:i1, :, :] = occ
            del t, pts, occ
        else:
            count = None
            for off in offsets:
                t = o3d.core.Tensor(pts + off[None, :], dtype=o3d.core.Dtype.Float32)
                occ = scene.compute_occupancy(t).numpy().reshape(block_shape).astype(np.uint16, copy=False)
                if count is None:
                    count = occ
                else:
                    count += occ
                del t, occ
            if mode == "any":
                inside[i0:i1, :, :] = count > 0
            else:
                thresh = int(offsets.shape[0] // 2) + 1
                inside[i0:i1, :, :] = count >= thresh
            del pts, count
    return inside, warns




def build_watertight_mesh_via_occupancy(
    mesh: trimesh.Trimesh,
    grid_min: np.ndarray,
    voxel_size: float,
    res: int,
    scale: int,
    occupancy_mode: str,
) -> Tuple[Optional[trimesh.Trimesh], List[str]]:
    """
    Build a watertight proxy mesh by voxel occupancy + marching cubes.
    This trades geometric fidelity for topological robustness.
    """
    warns: List[str] = []
    scale = int(max(1, scale))
    res_wt = int(res) * scale
    voxel_size_wt = float(voxel_size) / float(scale)
    if res_wt < 2:
        warns.append("watertight voxelize skipped: res*scale < 2.")
        return None, warns
    try:
        from skimage import measure
    except Exception as e:
        warns.append(f"watertight voxelize failed: scikit-image unavailable ({type(e).__name__}: {e}).")
        return None, warns

    inside, w = open3d_occupancy_sign(
        mesh=mesh,
        grid_min=grid_min,
        voxel_size=voxel_size_wt,
        res=res_wt,
        supersample=1,
        mode=occupancy_mode,
    )
    warns += w
    if inside is None:
        warns.append("watertight voxelize failed: occupancy is None.")
        return None, warns
    if not np.any(inside):
        warns.append("watertight voxelize failed: occupancy is empty.")
        return None, warns

    try:
        verts, faces, normals, values = measure.marching_cubes(
            volume=inside.astype(np.float32, copy=False),
            level=0.5,
            spacing=(voxel_size_wt, voxel_size_wt, voxel_size_wt),
            method="lewiner",
            step_size=1,
            allow_degenerate=False,
        )
    except Exception as e:
        warns.append(f"watertight voxelize failed: marching_cubes error ({type(e).__name__}: {e}).")
        return None, warns

    verts_world = verts.astype(np.float64, copy=False) + grid_min[None, :]
    mesh_wt = make_mesh(vertices=verts_world, faces=faces, like=mesh)
    warns.append(f"watertight voxelize: res={res_wt}, scale={scale}, faces={int(len(faces))}")
    return mesh_wt, warns


def compute_sdf_mesh_distance_open3d(
    mesh,
    grid_min: np.ndarray,
    voxel_size: float,
    res: int,
    want_signed: bool,
    out_dtype: np.dtype,
    out_sdf: Optional[np.ndarray] = None,
    supersample: int = 1,
    supersample_mode: str = "mean",
    downsample: int = 1,
    downsample_mode: str = "minabs",
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
    v, f = get_vertices_faces(mesh)
    legacy = build_open3d_legacy_mesh(v, f)
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

    down = int(max(1, downsample))
    if down > 1 and supersample > 1:
        warns.append("supersample with downsample>1 is expensive; consider supersample=1.")
    if down > 1 and str(downsample_mode).lower() not in ("minabs",):
        warns.append(f"downsample_mode '{downsample_mode}' unsupported. Fallback to 'minabs'.")
        downsample_mode = "minabs"

    xs, ys, zs = build_grid_axes(grid_min, voxel_size, res)
    neg_count = 0
    pos_count = 0

    ss = int(max(1, supersample))
    mode = str(supersample_mode).lower()
    if mode not in ("mean", "minabs"):
        warns.append(f"supersample_mode '{mode}' is unsupported. Fallback to 'mean'.")
        mode = "mean"
    offsets = build_supersample_offsets(voxel_size, ss)

    def eval_distance(pts: np.ndarray) -> np.ndarray:
        if offsets is None:
            t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
            if want_signed and has_signed:
                vals = scene.compute_signed_distance(t).numpy()
            else:
                if has_distance:
                    vals = scene.compute_distance(t).numpy()
                else:
                    vals = np.abs(scene.compute_signed_distance(t).numpy())
            return vals
        vals_acc = None
        for off in offsets:
            t = o3d.core.Tensor(pts + off[None, :], dtype=o3d.core.Dtype.Float32)
            if want_signed and has_signed:
                v = scene.compute_signed_distance(t).numpy()
            else:
                if has_distance:
                    v = scene.compute_distance(t).numpy()
                else:
                    v = np.abs(scene.compute_signed_distance(t).numpy())
            if vals_acc is None:
                vals_acc = v
            else:
                if mode == "mean":
                    vals_acc += v
                else:
                    mask = np.abs(v) < np.abs(vals_acc)
                    vals_acc[mask] = v[mask]
            del t, v
        if mode == "mean":
            return vals_acc / float(offsets.shape[0])
        return vals_acc

    if down <= 1:
        for i0, i1, pts, block_shape in iter_grid_chunks(xs, ys, zs, max_points=MAX_POINTS_PER_CHUNK):
            vals = eval_distance(pts)
            neg_count += int((vals < 0).sum())
            pos_count += int((vals > 0).sum())
            sdf[i0:i1, :, :] = vals.reshape(block_shape).astype(out_dtype, copy=False)
            del pts, vals
    else:
        res_hi = int(res) * int(down)
        voxel_hi = float(voxel_size) / float(down)
        xs_hi, ys_hi, zs_hi = build_grid_axes(grid_min, voxel_hi, res_hi)
        if res_hi != ys_hi.size or res_hi != zs_hi.size:
            raise RuntimeError("downsample grid size mismatch.")
        for i in range(int(res)):
            xi0 = i * down
            xi1 = xi0 + down
            xs_block = xs_hi[xi0:xi1]
            slab = np.empty((down, res_hi, res_hi), dtype=np.float32)
            for j0, j1, pts, block_shape in iter_grid_chunks(xs_block, ys_hi, zs_hi, max_points=MAX_POINTS_PER_CHUNK):
                vals = eval_distance(pts)
                slab[j0:j1, :, :] = vals.reshape(block_shape).astype(np.float32, copy=False)
                del pts, vals
            # minabs pooling
            if downsample_mode == "minabs":
                reshaped = slab.reshape(down, res, down, res, down)
                pooled = (
                    reshaped.transpose(0, 2, 4, 1, 3)
                    .reshape(down * down * down, res, res)
                )
                abs_vals = np.abs(pooled)
                idx = np.argmin(abs_vals, axis=0)
                out = np.take_along_axis(pooled, idx[None, ...], axis=0)[0]
            else:
                out = slab[0, ::down, ::down]
            neg_count += int((out < 0).sum())
            pos_count += int((out > 0).sum())
            sdf[i, :, :] = out.astype(out_dtype, copy=False)
            del slab, out

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
    if not is_trimesh(mesh):
        raise RuntimeError("EDT SDF requires trimesh backend.")
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
    mesh,
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
    sign_occupancy: bool = False,
    sdf_supersample: int = 1,
    sdf_supersample_mode: str = "mean",
    sign_occupancy_mode: str = "center",
    watertight_voxelize: bool = False,
    wt_voxel_scale: int = 1,
    sdf_downsample: int = 1,
    sdf_downsample_mode: str = "minabs",
) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Build SDF volume and metadata.
    """
    warns: List[str] = []

    # normalize_to_cube（最大辺長を1にして中心原点へ）
    transform = None
    if normalize_to_cube:
        v, f = get_vertices_faces(mesh)
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        center = 0.5 * (vmin + vmax)
        extent = float((vmax - vmin).max())
        if extent <= 0:
            raise ValueError("Degenerate mesh bbox extent.")
        scale = 1.0 / extent
        v2 = (v - center) * scale
        mesh = make_mesh(vertices=v2, faces=f, like=mesh)
        transform = {"center": center.tolist(), "scale": float(scale)}

    if repair:
        if not is_trimesh(mesh):
            warns.append("repair skipped: trimesh backend required.")
        else:
            warns += repair_mesh_inplace(mesh, fill_holes=True, merge_digits=merge_digits)

    if distance_method == "edt" and not is_trimesh(mesh):
        raise RuntimeError("distance_method=edt requires trimesh backend.")

    # grid (centered; normalize_to_cube -> center at 0)
    center_override = np.zeros(3, dtype=np.float64) if normalize_to_cube else None
    grid_min, grid_max, voxel_size, grid_center, _ = compute_grid_from_mesh(
        mesh, res=res, padding=padding, center_override=center_override
    )

    if bool(watertight_voxelize):
        mesh_wt, w = build_watertight_mesh_via_occupancy(
            mesh=mesh,
            grid_min=grid_min,
            voxel_size=voxel_size,
            res=res,
            scale=int(wt_voxel_scale),
            occupancy_mode=str(sign_occupancy_mode),
        )
        warns += w
        if mesh_wt is not None:
            mesh = mesh_wt
        else:
            warns.append("watertight voxelize: fallback to original mesh.")

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
            warns.append("morph_close_iters is used only when voxel sign fallback is used.")
        try:
            dual_pass = int(sdf_supersample) > 1 and int(sdf_downsample) > 1
            if dual_pass:
                warns.append(
                    "sdf: dual-pass enabled (supersample>1 and downsample>1). "
                    "Compute two volumes and combine by minabs."
                )
                sdf_down, signed_used_down, w = compute_sdf_mesh_distance_open3d(
                    mesh=mesh,
                    grid_min=grid_min,
                    voxel_size=voxel_size,
                    res=res,
                    want_signed=want_signed,
                    out_dtype=out_dtype,
                    out_sdf=sdf_out,
                    supersample=1,
                    supersample_mode=str(sdf_supersample_mode),
                    downsample=int(sdf_downsample),
                    downsample_mode=str(sdf_downsample_mode),
                )
                warns += w
                sdf_ss, signed_used_ss, w2 = compute_sdf_mesh_distance_open3d(
                    mesh=mesh,
                    grid_min=grid_min,
                    voxel_size=voxel_size,
                    res=res,
                    want_signed=want_signed,
                    out_dtype=out_dtype,
                    out_sdf=None,
                    supersample=int(sdf_supersample),
                    supersample_mode=str(sdf_supersample_mode),
                    downsample=1,
                    downsample_mode=str(sdf_downsample_mode),
                )
                warns += w2
                # Combine by minabs to preserve thin features while reducing noise.
                abs_down = np.abs(sdf_down)
                abs_ss = np.abs(sdf_ss)
                use_ss = abs_ss < abs_down
                sdf_down[use_ss] = sdf_ss[use_ss]
                sdf_open3d = sdf_down
                neg_count = int((sdf_open3d < 0).sum())
                pos_count = int((sdf_open3d > 0).sum())
                signed_used_open3d = bool(want_signed and (neg_count > 0) and (pos_count > 0))
                sdf_ss = None
            else:
                sdf_open3d, signed_used_open3d, w = compute_sdf_mesh_distance_open3d(
                    mesh=mesh,
                    grid_min=grid_min,
                    voxel_size=voxel_size,
                    res=res,
                    want_signed=want_signed,
                    out_dtype=out_dtype,
                    out_sdf=sdf_out,
                    supersample=int(sdf_supersample),
                    supersample_mode=str(sdf_supersample_mode),
                    downsample=int(sdf_downsample),
                    downsample_mode=str(sdf_downsample_mode),
                )
                warns += w
            sign_source = "open3d"
            if want_signed:
                if bool(sign_occupancy):
                    inside, w2 = open3d_occupancy_sign(
                        mesh=mesh,
                        grid_min=grid_min,
                        voxel_size=voxel_size,
                        res=res,
                        supersample=int(sdf_supersample),
                        mode=str(sign_occupancy_mode),
                    )
                    warns += w2
                    if inside is not None:
                        sdf = sdf_open3d.copy()
                        np.abs(sdf_open3d, out=sdf)
                        sdf[inside] = -sdf[inside]
                        sdf, removed, w3 = cleanup_negative_islands(
                            sdf,
                            voxel_size=voxel_size,
                            near_surface_thresh_vox=1.5,
                        )
                        warns += w3
                        signed_used = True
                        if removed > 0:
                            warns.append(f"sign cleanup: removed {removed} negative islands.")
                            sign_source = "open3d_occupancy_clean"
                        else:
                            sign_source = "open3d_occupancy"
                    else:
                        sdf = sdf_open3d
                        signed_used = bool(signed_used_open3d)
                elif bool(signed_used_open3d):
                    sdf, removed, w2 = cleanup_negative_islands(
                        sdf_open3d,
                        voxel_size=voxel_size,
                        near_surface_thresh_vox=1.5,
                    )
                    warns += w2
                    signed_used = True
                    if removed > 0:
                        warns.append(f"sign cleanup: removed {removed} negative islands.")
                        sign_source = "open3d_clean"
                    else:
                        sign_source = "open3d"
                else:
                    if not is_trimesh(mesh):
                        warns.append("sign reconstruct skipped: trimesh backend required.")
                        sdf = sdf_open3d
                        signed_used = False
                        sign_source = "unsigned"
                    else:
                        inside, w2 = reconstruct_sign_from_voxels(
                            mesh=mesh,
                            grid_min=grid_min,
                            voxel_size=voxel_size,
                            res=res,
                            morph_close_iters=morph_close_iters,
                        )
                        warns += w2
                        if inside is not None:
                            sdf = sdf_open3d.copy()
                            np.abs(sdf_open3d, out=sdf)
                            sdf[inside] = -sdf[inside]
                            signed_used = True
                            sign_source = "voxel_flood"
                        else:
                            sdf = sdf_open3d
                            signed_used = False
            else:
                sdf = sdf_open3d
                signed_used = False
                sign_source = "unsigned"
            surface_meta = {
                "source": "open3d_distance",
                "morph_close_iters": int(morph_close_iters),
                "sign_source": sign_source,
                "sdf_supersample": int(sdf_supersample),
                "sdf_supersample_mode": str(sdf_supersample_mode),
                "sdf_downsample": int(sdf_downsample),
                "sdf_downsample_mode": str(sdf_downsample_mode),
                "sdf_dual_pass": bool(int(sdf_supersample) > 1 and int(sdf_downsample) > 1),
                "sign_occupancy_mode": str(sign_occupancy_mode),
            }
        except Exception as e:
            if not is_trimesh(mesh):
                raise RuntimeError(f"open3d distance failed ({type(e).__name__}: {e}).") from e
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
        mesh_backend = str(params.get("mesh_backend", "open3d"))
        pre_warns: List[str] = []
        if mesh_backend == "open3d":
            if params.get("distance_method", "mesh") != "mesh":
                raise RuntimeError("mesh_backend=open3d requires distance_method=mesh.")
            mesh = load_mesh_open3d(stl_path)
            if params.get("o3d_clean", False) or int(params.get("o3d_subdivide", 0)) > 0:
                mesh, w = open3d_remesh_for_stability(
                    mesh,
                    subdivide_iters=int(params.get("o3d_subdivide", 0)),
                )
                pre_warns += w
            if params.get("meshfix", False):
                pre_warns.append("meshfix skipped: trimesh backend required.")
            if params.get("repair", False):
                pre_warns.append("repair skipped: trimesh backend required.")
            if params.get("merge_digits") is not None:
                pre_warns.append("merge_digits skipped: trimesh backend required.")
        else:
            mesh = trimesh.load(str(stl_path), force="mesh", process=False)
            mesh = _ensure_trimesh(mesh)
            if params.get("o3d_clean", False) or int(params.get("o3d_subdivide", 0)) > 0:
                mesh, w = open3d_remesh_for_stability(
                    mesh,
                    subdivide_iters=int(params.get("o3d_subdivide", 0)),
                )
                pre_warns += w
            if params.get("meshfix", False):
                mesh, w = meshfix_watertight(mesh)
                pre_warns += w

        if int(params.get("min_component_faces", 0)) > 0 or bool(params.get("keep_largest_component", False)):
            mesh, w = filter_mesh_components(
                mesh,
                min_faces=int(params.get("min_component_faces", 0)),
                keep_largest=bool(params.get("keep_largest_component", False)),
            )
            pre_warns += w

        distance_method = params["distance_method"]
        repair_flag = bool(params.get("repair", False)) if mesh_backend != "open3d" else False
        merge_digits = params.get("merge_digits", None) if mesh_backend != "open3d" else None

        sdf, meta, warns = compute_sdf(
            mesh=mesh,
            res=params["res"],
            padding=params["padding"],
            normalize_to_cube=params["normalize_to_cube"],
            trunc=params["trunc"],
            signed_policy=params["signed_policy"],
            out_dtype=params["out_dtype"],
            morph_close_iters=params["morph_close_iters"],
            repair=repair_flag,
            merge_digits=merge_digits,
            distance_method=distance_method,
            sdf_out=None,
            sign_occupancy=bool(params.get("sign_occupancy", False)),
            sdf_supersample=int(params.get("sdf_supersample", 1)),
            sdf_supersample_mode=str(params.get("sdf_supersample_mode", "mean")),
            sign_occupancy_mode=str(params.get("sign_occupancy_mode", "center")),
            watertight_voxelize=bool(params.get("watertight_voxelize", False)),
            wt_voxel_scale=int(params.get("wt_voxel_scale", 1)),
            sdf_downsample=int(params.get("sdf_downsample", 1)),
            sdf_downsample_mode=str(params.get("sdf_downsample_mode", "minabs")),
        )
        warns = pre_warns + warns

        if "repair" not in meta:
            meta["repair"] = {}
        meta["repair"]["o3d_clean"] = bool(params.get("o3d_clean", False))
        meta["repair"]["o3d_subdivide"] = int(params.get("o3d_subdivide", 0))
        meta["repair"]["meshfix"] = bool(params.get("meshfix", False) and mesh_backend == "trimesh")
        meta["repair"]["watertight_voxelize"] = bool(params.get("watertight_voxelize", False))
        meta["repair"]["wt_voxel_scale"] = int(params.get("wt_voxel_scale", 1))
        meta["repair"]["min_component_faces"] = int(params.get("min_component_faces", 0))
        meta["repair"]["keep_largest_component"] = bool(params.get("keep_largest_component", False))

        meta["source_relpath"] = str(rel).replace(os.sep, "/")
        meta["source_abspath"] = str(stl_path)
        meta["mesh_backend"] = str(mesh_backend)

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
    parser.add_argument(
        "--mesh_backend",
        choices=["open3d", "trimesh"],
        default="open3d",
        help="Mesh loading backend. open3d avoids trimesh usage.",
    )

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
    parser.add_argument(
        "--o3d_clean",
        action="store_true",
        help="Apply Open3D mesh cleanup before SDF (remove degenerate/duplicated/non-manifold).",
    )
    parser.add_argument(
        "--o3d_subdivide",
        type=int,
        default=0,
        help="Open3D midpoint subdivision iterations (0 disables).",
    )
    parser.add_argument(
        "--meshfix",
        action="store_true",
        help="Apply pymeshfix to enforce watertightness (may alter geometry).",
    )
    parser.add_argument(
        "--watertight_voxelize",
        action="store_true",
        help="Build a watertight proxy mesh by voxel occupancy + marching cubes (topology-robust).",
    )
    parser.add_argument(
        "--wt_voxel_scale",
        type=int,
        default=1,
        help="Voxel scale for watertight proxy (>=1). Larger preserves thin parts but slower.",
    )
    parser.add_argument(
        "--min_component_faces",
        type=int,
        default=0,
        help="Drop disconnected components with fewer faces (0 disables).",
    )
    parser.add_argument(
        "--keep_largest_component",
        action="store_true",
        help="Keep only the largest connected component.",
    )
    parser.add_argument(
        "--sign_occupancy",
        action="store_true",
        help="Use Open3D occupancy to decide sign (slower but robust).",
    )
    parser.add_argument(
        "--sign_occupancy_mode",
        choices=["center", "any", "majority"],
        default="center",
        help="Occupancy decision from supersampled points: center/any/majority.",
    )
    parser.add_argument(
        "--sdf_supersample",
        type=int,
        default=1,
        help="Sub-voxel samples per axis for SDF (>=1). Increases compute but keeps memory stable.",
    )
    parser.add_argument(
        "--sdf_supersample_mode",
        choices=["mean", "minabs"],
        default="mean",
        help="Reduction for supersampled SDF: mean (smooth) or minabs (conservative).",
    )
    parser.add_argument(
        "--sdf_downsample",
        type=int,
        default=1,
        help="Compute SDF at higher res (factor) then downsample to target res.",
    )
    parser.add_argument(
        "--sdf_downsample_mode",
        choices=["minabs"],
        default="minabs",
        help="Downsample reduction mode.",
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
            f"dtype={args.dtype}, distance_method={args.distance_method}, "
            f"o3d_clean={bool(args.o3d_clean)}, o3d_subdivide={int(args.o3d_subdivide)}, "
            f"meshfix={bool(args.meshfix)}, watertight_voxelize={bool(args.watertight_voxelize)}, "
            f"wt_voxel_scale={int(args.wt_voxel_scale)}, min_component_faces={int(args.min_component_faces)}, "
            f"keep_largest_component={bool(args.keep_largest_component)}, "
            f"sign_occupancy={bool(args.sign_occupancy)}, "
            f"sign_occupancy_mode={args.sign_occupancy_mode}, mesh_backend={args.mesh_backend}, "
            f"sdf_supersample={int(args.sdf_supersample)}, sdf_supersample_mode={args.sdf_supersample_mode}, "
            f"sdf_downsample={int(args.sdf_downsample)}, sdf_downsample_mode={args.sdf_downsample_mode}"
        )

    params = {
        "res": int(args.res),
        "padding": float(args.padding),
        "normalize_to_cube": bool(args.normalize_to_cube),
        "signed_policy": str(args.signed_policy),
        "trunc": None if args.trunc is None else float(args.trunc),
        "out_dtype": out_dtype,
        "overwrite": bool(args.overwrite),
        "mesh_backend": str(args.mesh_backend),
        "distance_method": str(args.distance_method),
        "morph_close_iters": int(args.morph_close_iters),
        "o3d_clean": bool(args.o3d_clean),
        "o3d_subdivide": int(args.o3d_subdivide),
        "meshfix": bool(args.meshfix),
        "watertight_voxelize": bool(args.watertight_voxelize),
        "wt_voxel_scale": int(max(1, args.wt_voxel_scale)),
        "min_component_faces": int(args.min_component_faces),
        "keep_largest_component": bool(args.keep_largest_component),
        "sign_occupancy": bool(args.sign_occupancy),
        "sign_occupancy_mode": str(args.sign_occupancy_mode),
        "repair": bool(args.repair),
        "merge_digits": None if args.merge_digits is None else int(args.merge_digits),
        "sdf_supersample": int(max(1, args.sdf_supersample)),
        "sdf_supersample_mode": str(args.sdf_supersample_mode),
        "sdf_downsample": int(max(1, args.sdf_downsample)),
        "sdf_downsample_mode": str(args.sdf_downsample_mode),
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
    
