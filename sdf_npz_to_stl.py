#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python sdf_npz_to_stl.py --npz xxx.npz --out yyy.stl --restore_size --mc_auto --mc_methods lorensen --to_original_scale

python sdf_npz_to_stl.py --root_dir /mnt/ssd2tb/new_format_sdf --dst_dir /mnt/ssd2tb/new_format_sdf_stl --mc_auto --mc_methods lorensen --to_original_scale
"""


import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def parse_level_offsets(text: Optional[str], signed_used: bool) -> List[float]:
    """
    Parse comma-separated offsets (in voxel_size units).
    """
    if text is None:
        if signed_used:
            offsets = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        else:
            offsets = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    else:
        parts = [p.strip() for p in text.split(",")]
        offsets = [float(p) for p in parts if p]
    if 0.0 not in offsets:
        offsets.insert(0, 0.0)
    return offsets


def parse_methods(text: Optional[str], default: List[str]) -> List[str]:
    """
    Parse comma-separated marching cubes methods.
    """
    if text is None:
        methods = list(default)
    else:
        parts = [p.strip() for p in text.split(",")]
        methods = [p for p in parts if p]
    seen = set()
    uniq = []
    for m in methods:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def count_mesh_components(mesh: trimesh.Trimesh) -> int:
    """
    Count connected components from face adjacency.
    """
    try:
        from trimesh.graph import connected_components
        n_faces = int(len(mesh.faces))
        if n_faces == 0:
            return 0
        comps = connected_components(mesh.face_adjacency, nodes=np.arange(n_faces))
        return int(len(comps))
    except Exception:
        return 1


def filter_components_by_faces(
    mesh: trimesh.Trimesh,
    min_faces: int,
) -> trimesh.Trimesh:
    """
    Remove small connected components by face count.
    """
    if min_faces <= 0:
        return mesh
    try:
        from trimesh.graph import connected_components
        n_faces = int(len(mesh.faces))
        if n_faces == 0:
            return mesh
        comps = connected_components(mesh.face_adjacency, nodes=np.arange(n_faces))
        keep_faces = np.zeros((n_faces,), dtype=bool)
        for comp in comps:
            if len(comp) >= min_faces:
                keep_faces[np.asarray(comp, dtype=np.int64)] = True
        if keep_faces.all():
            return mesh
        new_mesh = mesh.copy()
        new_mesh.update_faces(keep_faces)
        new_mesh.remove_unreferenced_vertices()
        return new_mesh
    except Exception:
        return mesh


def iter_npz_files(root_dir: Path):
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".npz":
            yield p


def replace_dirname(path: Path, src: str = "sdf", dst: str = "sdf_stl") -> Path:
    parts = [dst if p == src else p for p in path.parts]
    return Path(*parts)


def convert_one_npz(npz_path: Path, out_path: Path, args: argparse.Namespace) -> Dict:
    try:
        sdf, meta = load_npz(npz_path)
        bbox_min = np.asarray(meta["bbox_min"], dtype=np.float64)
        voxel_size = float(meta["voxel_size"])
        signed_used = bool(meta.get("signed_used", False))

        try:
            from skimage import measure
            from skimage.transform import resize
        except Exception as e:
            raise RuntimeError("scikit-image が必要です．pip install scikit-image") from e

        up = int(max(1, args.mc_upsample))
        if up > 1:
            r = sdf.shape[0]
            target_shape = (r * up, r * up, r * up)
            sdf = resize(
                sdf,
                target_shape,
                order=1,
                mode="edge",
                anti_aliasing=False,
                preserve_range=True,
            ).astype(sdf.dtype, copy=False)
            voxel_size = voxel_size / float(up)

        if args.level is None:
            base_level = 0.0 if signed_used else 0.5 * voxel_size
        else:
            base_level = float(args.level)

        use_auto = bool(args.mc_auto)
        if args.mc_auto:
            methods = parse_methods(args.mc_methods, default=["lewiner", "lorensen"])
        else:
            methods = parse_methods(None, default=[args.method])

        if use_auto:
            offsets = parse_level_offsets(args.level_offsets, signed_used=signed_used)
            levels = [base_level + o * voxel_size for o in offsets]
            best_level = None
            best_method = None
            best_verts = None
            best_faces = None
            best_watertight = False
            best_comp = 0
            best_score = None
            for method in methods:
                for lvl in levels:
                    try:
                        verts, faces, normals, values = measure.marching_cubes(
                            volume=sdf,
                            level=float(lvl),
                            spacing=(voxel_size, voxel_size, voxel_size),
                            method=method,
                            step_size=int(args.step_size),
                            allow_degenerate=False,
                        )
                    except Exception:
                        continue
                    if verts.shape[0] == 0 or faces.shape[0] == 0:
                        continue
                    mesh_tmp = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    watertight = bool(mesh_tmp.is_watertight)
                    n_comp = count_mesh_components(mesh_tmp)
                    score = (0 if watertight else 1, n_comp, -int(len(faces)))
                    keep = False
                    if best_level is None or score < best_score:
                        if best_verts is not None:
                            del best_verts, best_faces
                        best_level = float(lvl)
                        best_method = method
                        best_verts = verts
                        best_faces = faces
                        best_watertight = watertight
                        best_comp = n_comp
                        best_score = score
                        keep = True
                    del mesh_tmp, normals, values
                    if not keep:
                        del verts, faces
            if best_level is None:
                raise RuntimeError(
                    "Auto selection failed: all candidates produced empty meshes. "
                    "Try explicit --level or adjust --level_offsets."
                )
            level = best_level
            verts = best_verts
            faces = best_faces
            method = best_method if best_method is not None else args.method
        else:
            level = float(base_level)
            method = args.method
            verts, faces, normals, values = measure.marching_cubes(
                volume=sdf,
                level=level,
                spacing=(voxel_size, voxel_size, voxel_size),
                method=method,
                step_size=int(args.step_size),
                allow_degenerate=False,
            )
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                raise RuntimeError(
                    f"Empty mesh from marching cubes (level={level}). "
                    f"Try --level {0.5*voxel_size:.6g} or smaller/larger."
                )

        verts_world = verts.astype(np.float64, copy=False) + bbox_min[None, :]

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
        mesh = filter_components_by_faces(mesh, min_faces=int(args.min_component_faces))
        mesh.export(str(out_path))

        return {
            "status": "ok",
            "npz": str(npz_path),
            "out": str(out_path),
            "level": float(level),
            "method": str(method),
            "signed_used": bool(signed_used),
            "faces": int(len(mesh.faces)),
        }
    except Exception as e:
        return {"status": "fail", "npz": str(npz_path), "out": str(out_path), "msg": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="Convert voxel SDF(.npz) to STL via marching cubes.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", type=str, help="Input npz file.")
    group.add_argument("--root_dir", type=str, help="Root dir to search npz files recursively.")
    ap.add_argument("--out", type=str, help="Output stl path (single file mode).")
    ap.add_argument("--dst_dir", type=str, help="Output root dir (batch mode).")

    ap.add_argument(
        "--level",
        type=float,
        default=None,
        help="Isosurface level for marching cubes. Default: 0.0 if signed_used else 0.5*voxel_size. "
             "If --mc_auto is set, this is treated as the base level.",
    )
    ap.add_argument(
        "--level_offsets",
        type=str,
        default=None,
        help="Comma-separated offsets in voxel_size units (e.g., -0.2,-0.1,0,0.1).",
    )
    ap.add_argument(
        "--mc_auto",
        action="store_true",
        help="Auto-select level and method by trying multiple candidates.",
    )
    ap.add_argument(
        "--mc_methods",
        type=str,
        default=None,
        help="Comma-separated methods to try when --mc_auto is set (default: lewiner,lorensen).",
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
    ap.add_argument(
        "--mc_upsample",
        type=int,
        default=1,
        help="Upsample SDF grid by integer factor before marching cubes (trilinear).",
    )
    ap.add_argument(
        "--min_component_faces",
        type=int,
        default=0,
        help="Remove connected components with fewer faces (0 disables).",
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
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1).")

    args = ap.parse_args()

    if args.npz:
        if args.out is None:
            raise ValueError("--out is required when --npz is used.")
        npz_path = Path(args.npz).expanduser().resolve()
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r = convert_one_npz(npz_path, out_path, args)
        if r["status"] == "fail":
            raise RuntimeError(r["msg"])
        print(f"OK: {npz_path} -> {out_path}")
        print(
            f"  level={r['level']} method={r['method']} signed_used(meta)={r['signed_used']} faces={r['faces']}"
        )
        return

    if args.root_dir is None or args.dst_dir is None:
        raise ValueError("--root_dir and --dst_dir are required for batch mode.")
    root_dir = Path(args.root_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    npz_list = list(iter_npz_files(root_dir))
    if not args.quiet:
        print(f"Found {len(npz_list)} NPZ files under: {root_dir}")
        print(f"workers={args.workers}")

    n_ok = n_skip = n_fail = 0

    if args.workers <= 1:
        for i, p in enumerate(npz_list, start=1):
            rel = p.relative_to(root_dir)
            rel_out = replace_dirname(rel, src="sdf", dst="sdf_stl")
            out_path = (dst_dir / rel_out).with_suffix(".stl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and (not args.overwrite):
                n_skip += 1
                if not args.quiet:
                    print(f"[{i}/{len(npz_list)}] SKIP: {rel} (exists)")
                continue
            r = convert_one_npz(p, out_path, args)
            if r["status"] == "ok":
                n_ok += 1
                if not args.quiet:
                    print(f"[{i}/{len(npz_list)}] OK: {rel}")
            else:
                n_fail += 1
                print(f"[{i}/{len(npz_list)}] FAIL: {rel}\n{r['msg']}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = []
            for p in npz_list:
                rel = p.relative_to(root_dir)
                rel_out = replace_dirname(rel, src="sdf", dst="stl")
                out_path = (dst_dir / rel_out).with_suffix(".stl")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and (not args.overwrite):
                    n_skip += 1
                    if not args.quiet:
                        print(f"[skip] {rel}")
                    continue
                futs.append(ex.submit(convert_one_npz, p, out_path, args))

            for i, fut in enumerate(as_completed(futs), start=1):
                r = fut.result()
                rel = Path(r["npz"]).relative_to(root_dir)
                if r["status"] == "ok":
                    n_ok += 1
                    if not args.quiet:
                        print(f"[{i}/{len(futs)}] OK: {rel}")
                else:
                    n_fail += 1
                    print(f"[{i}/{len(futs)}] FAIL: {rel}\n{r['msg']}")

    print(f"Done. ok={n_ok}, skip={n_skip}, fail={n_fail}, dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
