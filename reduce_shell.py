#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel STL simplifier with Open3D (no rendering / no TSDF).
- Recursively find .stl files under input_root
- For each STL:
    * load mesh
    * sanitize/cleanup
    * simplify (quadric decimation, optional vertex clustering, optional smoothing)
    * save to output_root while mirroring relative paths
"""

import argparse
import os
from pathlib import Path
from typing import List

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import open3d as o3d


# -----------------------------
# Mesh utilities
# -----------------------------
def _sanitize_legacy_mesh(m: o3d.geometry.TriangleMesh, area_eps: float = 1e-12) -> o3d.geometry.TriangleMesh:
    """数値異常・退化・重複・非多様体・未参照の除去＋面積しきい値で三角形除去＋法線再計算"""
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)

    # 1) NaN/Inf 頂点を除去し、面を再構成
    finite = np.isfinite(V).all(axis=1)
    if not finite.all():
        idx_map = -np.ones(len(V), dtype=np.int64)
        idx_map[finite] = np.arange(finite.sum(), dtype=np.int64)
        keep_face = finite[F].all(axis=1)
        F = idx_map[F[keep_face]]
        V = V[finite]
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(F.astype(np.int32, copy=False)),
        )

    # 2) 基本クリーンアップ
    if hasattr(m, "remove_degenerate_triangles"): m.remove_degenerate_triangles()
    if hasattr(m, "remove_duplicated_triangles"): m.remove_duplicated_triangles()
    if hasattr(m, "remove_duplicated_vertices"):  m.remove_duplicated_vertices()
    if hasattr(m, "remove_non_manifold_edges"):   m.remove_non_manifold_edges()
    if hasattr(m, "remove_unreferenced_vertices"): m.remove_unreferenced_vertices()

    # 3) 超小面積三角形を除去（スケール相対の閾値）
    if len(m.triangles) > 0:
        V = np.asarray(m.vertices)
        F = np.asarray(m.triangles, dtype=np.int64)
        if F.size > 0:
            tris = V[F]
            e1 = tris[:, 1] - tris[:, 0]
            e2 = tris[:, 2] - tris[:, 0]
            areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

            aabb = m.get_axis_aligned_bounding_box()
            dia = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()) + 1e-12
            rel_eps = max(area_eps, (dia * 1e-6) ** 2)
            keep = areas > rel_eps
            if not np.all(keep):
                F = F[keep]
                m = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(V),
                    o3d.utility.Vector3iVector(F.astype(np.int32, copy=False)),
                )
                if hasattr(m, "remove_unreferenced_vertices"):
                    m.remove_unreferenced_vertices()

    if len(m.vertices) > 0:
        m.compute_vertex_normals()
        m.compute_triangle_normals()
    return m


def clean_and_compress(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int = 100_000,
    clustering_voxel: float | None = None,
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
            voxel_size=float(clustering_voxel),
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    if smoothing_iters and smoothing_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(smoothing_iters))
        mesh.compute_vertex_normals()

    return mesh


def _safe_write_stl(path: Path, m: o3d.geometry.TriangleMesh) -> None:
    if m.is_empty() or len(m.triangles) == 0:
        raise RuntimeError("Empty mesh after sanitization.")
    ok = o3d.io.write_triangle_mesh(
        str(path),
        m,
        write_ascii=False,          # バイナリSTL
        compressed=False,           # STLでは無効
        write_vertex_normals=True,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write STL: {path}")


# -----------------------------
# Per-file processing (simplify only)
# -----------------------------
def process_one_stl(
    in_path: Path,
    in_root: Path,
    out_root: Path,
    target_tris: int,
    clustering_voxel_ratio: float,
    smoothing_iters: int,
) -> None:
    print(f"[INFO] Processing: {in_path}")
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    if mesh.is_empty():
        print(f"[WARN] Empty mesh: {in_path}")
        return

    # 0) 入力をまずサニタイズ（頂点異常や退化対策）
    mesh = _sanitize_legacy_mesh(mesh)

    # 1) 物体スケールに基づきクラスタリング体素サイズを決定
    aabb = mesh.get_axis_aligned_bounding_box()
    diameter = float(np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()))
    clustering_voxel = diameter * float(clustering_voxel_ratio) if clustering_voxel_ratio > 0 else None

    # 2) 単純化
    mesh = clean_and_compress(
        mesh,
        target_triangles=int(target_tris) if target_tris and target_tris > 0 else None,
        clustering_voxel=clustering_voxel,
        smoothing_iters=int(smoothing_iters) if smoothing_iters and smoothing_iters > 0 else 0,
    )

    # 3) 最終サニタイズ（単純化後の微小三角形・未参照頂点などの除去）
    mesh = _sanitize_legacy_mesh(mesh)

    # 4) 保存（input_root からの相対を維持）
    try:
        rel = in_path.resolve().relative_to(in_root.resolve()).with_suffix(".stl")
    except Exception:
        rel = Path(in_path.name).with_suffix(".stl")

    out_path = (out_root / rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_stl(out_path, mesh)
    print(f"[OK] Saved simplified → {out_path}")


# -----------------------------
# Parallel driver
# -----------------------------
def find_stls(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.stl") if p.is_file()]


def _proc_worker(kwargs):
    """子プロセス側で1ファイル処理。例外は文字列で返す。"""
    try:
        process_one_stl(**kwargs)
        return (str(kwargs["in_path"]), True, "")
    except Exception as e:
        return (str(kwargs["in_path"]), False, f"{e}")


def main():
    ap = argparse.ArgumentParser(description="Simplify STL meshes in parallel (Open3D).")
    default_jobs = max(1, (os.cpu_count() or 1) - 1)  # コア数-1を既定
    ap.add_argument("--jobs", type=int, default=default_jobs,
                    help=f"Number of parallel workers (default: cores-1 ≈ {default_jobs})")
    ap.add_argument("input_root", type=str, help="Folder A (recursively search for .stl)")
    ap.add_argument("output_root", type=str, help="Folder B (mirrors relative paths)")
    ap.add_argument("--target_tris", type=int, default=20_000,
                    help="Quadric decimation target triangles (<=0 to disable)")
    ap.add_argument("--cluster_ratio", type=float, default=1.0/2000.0,
                    help="Vertex clustering voxel = diameter * ratio (0 to disable)")
    ap.add_argument("--smooth", type=int, default=3,
                    help="Laplacian smoothing iterations (0 to disable)")
    args = ap.parse_args()

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stls = find_stls(in_root)
    if not stls:
        print("[WARN] No .stl files found.")
        return

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
            in_root=in_root,
            out_root=out_root,
            target_tris=args.target_tris,
            clustering_voxel_ratio=args.cluster_ratio,
            smoothing_iters=args.smooth,
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
