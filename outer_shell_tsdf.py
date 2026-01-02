#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive STL outer-shell extractor with Open3D.
- Load each STL
- Render depth from multiple external viewpoints (offscreen)
- Fuse depths via TSDF (SDF-based) to keep only the visible outer shell
- Extract mesh, clean & simplify (compression), save as STL to output root
"""

import argparse
import copy
import math
import os
from pathlib import Path
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import open3d as o3d


def _proc_worker(kwargs):
    """子プロセス側で1ファイル処理。例外は文字列にして返す。"""
    try:
        process_one_stl(**kwargs)
        return (str(kwargs["in_path"]), True, "")
    except Exception as e:
        return (str(kwargs["in_path"]), False, f"{e}")
    
    
def _fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    ga = (3.0 - math.sqrt(5.0)) * math.pi * 2.0
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z * z)
    phi = i * ga
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1) * radius

def _look_at(cam_pos: np.ndarray, target: np.ndarray, up=np.array([0,0,1.0])):
    f = (target - cam_pos); f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, up);    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    cam2world = np.eye(4, dtype=np.float64)
    cam2world[:3,0] = r; cam2world[:3,1] = u; cam2world[:3,2] = f; cam2world[:3,3] = cam_pos
    world2cam = np.linalg.inv(cam2world)
    return world2cam

def render_depths_from_views_ray(mesh_legacy, views=64, img_size=768,
                                 margin_scale=1.4, z_near=0.05, z_far=10_000.0):
    aabb = mesh_legacy.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    diameter = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    if diameter == 0:
        raise ValueError("Degenerate mesh (zero diameter).")

    radius = (diameter * 0.5) * margin_scale
    cam_positions = _fibonacci_sphere(views, radius=radius) + center

    fx = fy = img_size / 2.0 / math.tan(math.radians(30.0))
    cx = (img_size - 1) / 2.0
    cy = (img_size - 1) / 2.0

    # ===== ここが差し替え部分 =====
    try:
        m_legacy = mesh_legacy.clone()
    except AttributeError:
        m_legacy = copy.deepcopy(mesh_legacy)

    if hasattr(m_legacy, "remove_duplicated_vertices"):
        m_legacy.remove_duplicated_vertices()
    if hasattr(m_legacy, "remove_duplicated_triangles"):
        m_legacy.remove_duplicated_triangles()
    if hasattr(m_legacy, "remove_degenerate_triangles"):
        m_legacy.remove_degenerate_triangles()
    if hasattr(m_legacy, "remove_non_manifold_edges"):
        m_legacy.remove_non_manifold_edges()

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(m_legacy)
    # dtype を Float32 に統一（RaycastingScene が安定）
    if "positions" in mesh_t.vertex:
        mesh_t.vertex["positions"] = mesh_t.vertex["positions"].to(o3d.core.Dtype.Float32)
    if "normals" in mesh_t.vertex:
        mesh_t.vertex["normals"] = mesh_t.vertex["normals"].to(o3d.core.Dtype.Float32)

    if "indices" in mesh_t.triangle:
        if mesh_t.triangle["indices"].dtype != o3d.core.Dtype.Int32:
            mesh_t.triangle["indices"] = mesh_t.triangle["indices"].to(o3d.core.Dtype.Int32)
    # ===== ここまで =====

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    depths_legacy, extrinsics = [], []
    z_far_eff = max(z_far, diameter * 4.0)

    for p in cam_positions:
        extr = _look_at(p, center)
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(
                [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=o3d.core.Dtype.Float32
            ),
            extrinsic_matrix=o3d.core.Tensor(extr, dtype=o3d.core.Dtype.Float32),
            width_px=int(img_size),
            height_px=int(img_size),
        )
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy().reshape(img_size, img_size).astype(np.float32)
        # t_hit: Raycasting のレイ長 [mm]
        t = t_hit.astype(np.float32)
        invalid = ~np.isfinite(t) | (t < z_near) | (t > z_far_eff)
        t[invalid] = 0.0
        depths_legacy.append(o3d.geometry.Image(t))
        extrinsics.append(extr.copy())

    intrinsic_leg = o3d.camera.PinholeCameraIntrinsic(img_size, img_size, fx, fy, cx, cy)
    return depths_legacy, extrinsics, intrinsic_leg


# -----------------------------
# TSDF fusion (SDF-based union)
# -----------------------------
def fuse_tsdf_from_depths(depths, extrinsics, intrinsic,
                          voxel_len_mm, sdf_trunc_mm):
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_len_mm),    # ← mm
        sdf_trunc=float(sdf_trunc_mm),       # ← mm
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    color = o3d.geometry.Image(np.zeros((intrinsic.height, intrinsic.width,3), np.uint8))
    for depth, ext in zip(depths, extrinsics):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,       # ← mmをそのまま使う
            depth_trunc=1e9,       # 前段で 0 にしてるので大きめでOK
        )
        tsdf.integrate(rgbd, intrinsic, np.asarray(ext))
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def _sanitize_legacy_mesh(m: o3d.geometry.TriangleMesh, area_eps: float = 1e-12) -> o3d.geometry.TriangleMesh:
    # 1) NaN/Inf 頂点を除去し、面を再構成
    V = np.asarray(m.vertices)      # (Nv, 3) float64
    F = np.asarray(m.triangles)     # (Nf, 3) int
    finite = np.isfinite(V).all(axis=1)
    if not finite.all():
        idx_map = -np.ones(len(V), dtype=np.int64)
        idx_map[finite] = np.arange(finite.sum(), dtype=np.int64)
        keep_face = finite[F].all(axis=1)
        F = F[keep_face]
        F = idx_map[F]
        V = V[finite]
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(F)
        )

    # 2) 退化・重複・非多様体・未参照頂点の除去
    if hasattr(m, "remove_degenerate_triangles"): m.remove_degenerate_triangles()
    if hasattr(m, "remove_duplicated_triangles"): m.remove_duplicated_triangles()
    if hasattr(m, "remove_duplicated_vertices"):  m.remove_duplicated_vertices()
    if hasattr(m, "remove_non_manifold_edges"):   m.remove_non_manifold_edges()
    if hasattr(m, "remove_unreferenced_vertices"): m.remove_unreferenced_vertices()

    # 面積ゼロ近傍の三角形を落とす
    if len(m.triangles) > 0:
        # m.compute_triangle_normals()
        # areas = np.asarray(m.get_triangle_areas())
        # keep = areas > area_eps
        # if not keep.all():
        #     F = np.asarray(m.triangles)[keep]
        #     V = np.asarray(m.vertices)
        #     m = o3d.geometry.TriangleMesh(
        #         o3d.utility.Vector3dVector(V),
        #         o3d.utility.Vector3iVector(F)
        #     )
        #     if hasattr(m, "remove_unreferenced_vertices"):
        #         m.remove_unreferenced_vertices()
        # 置換後（自前で三角形面積を計算）
        V = np.asarray(m.vertices)                 # (Nv, 3) float
        F = np.asarray(m.triangles, dtype=np.int64)  # (Nf, 3) int
        if F.size > 0:
            tris = V[F]                            # (Nf, 3, 3)
            e1 = tris[:, 1] - tris[:, 0]
            e2 = tris[:, 2] - tris[:, 0]
            areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)  # mm^2

            # 動的なしきい値（超小面積を除去）。固定値より安全。
            aabb = m.get_axis_aligned_bounding_box()
            dia = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()) + 1e-12  # mm
            rel_eps = max(area_eps, (dia * 1e-6) ** 2)  # 物体スケールの1e-6倍の長さ → 面積しきい値
            keep = areas > rel_eps

            if not np.all(keep):
                F = F[keep]
                m = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(V),
                    o3d.utility.Vector3iVector(F.astype(np.int32, copy=False))
                )
                if hasattr(m, "remove_unreferenced_vertices"):
                    m.remove_unreferenced_vertices()

    # 法線を再計算（STLは面法線を書き出す実装もある）
    if len(m.vertices) > 0:
        m.compute_vertex_normals()
        m.compute_triangle_normals()
    return m

def _safe_write_stl(path: Path, m: o3d.geometry.TriangleMesh) -> None:
    if m.is_empty() or len(m.triangles) == 0:
        raise RuntimeError("Empty mesh after sanitization.")
    ok = o3d.io.write_triangle_mesh(
        str(path),
        m,
        write_ascii=False,          # バイナリSTLのみ
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write STL: {path}")
    

# -----------------------------
# Mesh cleanup & compression
# -----------------------------
def clean_and_compress(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int = 100_000,
    clustering_voxel: float = None,
    smoothing_iters: int = 5,
) -> o3d.geometry.TriangleMesh:
    """
    Basic cleanup and size reduction:
      - remove degenerate / duplicated / non-manifold
      - quadric decimation
      - optional vertex clustering
      - light Laplacian smoothing
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if target_triangles is not None and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    if clustering_voxel is not None and clustering_voxel > 0:
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=clustering_voxel,
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    if smoothing_iters and smoothing_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iters)
        mesh.compute_vertex_normals()

    return mesh


# -----------------------------
# Per-file processing
# -----------------------------
def process_one_stl(
    in_path: Path,
    out_root: Path,
    views: int,
    img_size: int,
    margin_scale: float,
    z_near: float,
    z_far: float,
    voxel_len_ratio: float,
    sdf_trunc_ratio: float,
    target_tris: int,
    clustering_voxel_ratio: float,
) -> None:
    print(f"[INFO] Processing: {in_path}")

    mesh = o3d.io.read_triangle_mesh(str(in_path))
    if mesh.is_empty():
        print(f"[WARN] Empty mesh: {in_path}")
        return

    # Use object scale to set TSDF resolution & clustering voxel
    aabb = mesh.get_axis_aligned_bounding_box()
    diameter = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    # Voxel length proportional to size (e.g., diameter / 512)
    voxel_len = max(diameter * voxel_len_ratio, 1e-5)
    sdf_trunc = max(voxel_len * (sdf_trunc_ratio), voxel_len * 3.0)

    clustering_voxel = diameter * clustering_voxel_ratio if clustering_voxel_ratio > 0 else None

    # 1) Render depths from multi-views
    depths, extrinsics, intrinsic = render_depths_from_views_ray(
        mesh, views=views, img_size=img_size, margin_scale=margin_scale,
        z_near=z_near, z_far=max(z_far, diameter * 4.0)
    )

    # 2) TSDF fuse → outer shell mesh
    shell = fuse_tsdf_from_depths(depths, extrinsics, intrinsic, voxel_len, sdf_trunc)

    # 3) Clean & compress
    shell = clean_and_compress(
        shell,
        target_triangles=target_tris,
        clustering_voxel=clustering_voxel,
        smoothing_iters=3,
    )
    shell = _sanitize_legacy_mesh(shell)

    # 4) Save (mirror subdir layout)
    rel = in_path.with_suffix(".stl").relative_to(in_path.anchor if in_path.is_absolute() else Path("."))
    # Keep path under input root:
    try:
        rel = in_path.relative_to(common_prefix(in_path, out_root)).with_suffix(".stl")
    except Exception:
        rel = Path(in_path.name).with_suffix(".stl")

    out_path = out_root / rel
    print(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(
        str(out_path),
        shell,
        write_ascii=False,        # ← ASCIIは使わない
        compressed=False,         # ← STLには効かないので False 固定でOK
        write_vertex_normals=True,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write STL: {out_path}")
    print(f"[OK] Saved outer shell → {out_path}")


def common_prefix(a: Path, b: Path) -> Path:
    """Longest common prefix directory (for relative pathing)."""
    ap = a.resolve().parts
    bp = b.resolve().parts
    i = 0
    for x, y in zip(ap, bp):
        if x != y:
            break
        i += 1
    return Path(*ap[:i]) if i else a.parent


# -----------------------------
# Driver
# -----------------------------
def find_stls(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.stl") if p.is_file()]


def main():
    ap = argparse.ArgumentParser(description="Extract outer-shell STL via multi-view TSDF fusion (Open3D).")
    # 既存の引数定義の下あたりに
    default_jobs = max(1, (os.cpu_count() or 1) - 1)  # コア数-1を既定
    ap.add_argument("--jobs", type=int, default=default_jobs,
                    help=f"Number of parallel workers (default: cores-1 ≈ {default_jobs})")
    ap.add_argument("input_root", type=str, help="Folder A (recursively search for .stl)")
    ap.add_argument("output_root", type=str, help="Folder B (mirrors relative paths)")
    ap.add_argument("--views", type=int, default=128, help="Number of viewpoints on sphere")
    ap.add_argument("--img", type=int, default=1024, help="Offscreen render image size (square, px)")
    ap.add_argument("--margin", type=float, default=1.8, help="Camera radius = 0.5*diameter*margin (in mm units)")
    ap.add_argument("--znear", type=float, default=0.05, help="Near clip in mm")
    ap.add_argument("--zfar", type=float, default=10_000.0, help="Far clip in mm (will be maxed with 4x diameter)")
    ap.add_argument("--voxel_ratio", type=float, default=1.0/1000.0,
                help="TSDF voxel length = diameter * ratio (mm). e.g., 200 mm dia -> 0.2 mm voxels")
    ap.add_argument("--sdf_trunc_ratio", type=float, default=3.0,
                help="SDF truncation = voxel_len * ratio")
    ap.add_argument("--target_tris", type=int, default=50_000,
                help="Quadric decimation target triangles")
    ap.add_argument("--cluster_ratio", type=float, default=1.0/2000.0,
                help="Vertex clustering voxel = diameter * ratio (mm). 0 to disable")
    args = ap.parse_args()

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stls = find_stls(in_root)
    if not stls:
        print("[WARN] No .stl files found.")
        return

    # for stl in stls:
    #     try:
    #         process_one_stl(
    #             stl, out_root,
    #             views=args.views,
    #             img_size=args.img,
    #             margin_scale=args.margin,
    #             z_near=args.znear,
    #             z_far=args.zfar,
    #             voxel_len_ratio=args.voxel_ratio,
    #             sdf_trunc_ratio=args.sdf_trunc_ratio,
    #             target_tris=args.target_tris,
    #             clustering_voxel_ratio=args.cluster_ratio,
    #         )
    #     except Exception as e:
    #         print(f"[ERROR] {stl}: {e}")
    
    
    # 並列（ProcessPoolExecutor）
    jobs = max(1, min(args.jobs, len(stls)))
    print(f"[INFO] Parallel workers = {jobs} (CPU cores = {os.cpu_count()})")

    # 過度なスレッド競合を避ける（NumPy/MKL/OMP）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tasks = []
    for stl in stls:
        tasks.append(dict(
            in_path=stl,
            out_root=out_root,
            views=args.views,
            img_size=args.img,
            margin_scale=args.margin,
            z_near=args.znear,
            z_far=args.zfar,
            voxel_len_ratio=args.voxel_ratio,
            sdf_trunc_ratio=args.sdf_trunc_ratio,
            target_tris=args.target_tris,
            clustering_voxel_ratio=args.cluster_ratio,
        ))

    ok_cnt = 0
    err_cnt = 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_proc_worker, kw) for kw in tasks]
        for fut in as_completed(futs):
            path, ok, msg = fut.result()
            if ok:
                ok_cnt += 1
            else:
                err_cnt += 1
                print(f"[ERROR] {path}: {msg}")

    print(f"[INFO] Done. success={ok_cnt}, error={err_cnt}")


if __name__ == "__main__":
    main()
