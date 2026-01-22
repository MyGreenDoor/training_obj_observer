import json
import numpy as np
import trimesh

def load_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    sdf = z["sdf"]
    meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    return sdf, meta

def normalize_mesh_if_needed(mesh: trimesh.Trimesh, meta: dict) -> trimesh.Trimesh:
    if meta.get("normalize_to_cube", False) and "transform" in meta:
        c = np.array(meta["transform"]["center"], dtype=np.float64)
        s = float(meta["transform"]["scale"])
        v = (mesh.vertices.astype(np.float64) - c) * s
        return trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    return mesh

def trilinear_sample(sdf: np.ndarray, xyz: np.ndarray, bbox_min: np.ndarray, voxel_size: float) -> np.ndarray:
    # xyz: (N,3) in world coords of the SDF grid
    idx = (xyz - bbox_min[None, :]) / voxel_size  # float index
    i0 = np.floor(idx).astype(np.int64)
    d = idx - i0

    r = sdf.shape[0]
    # clamp to valid range for i0+1
    i0 = np.clip(i0, 0, r - 2)
    i1 = i0 + 1

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    c000 = sdf[x0, y0, z0]
    c001 = sdf[x0, y0, z1]
    c010 = sdf[x0, y1, z0]
    c011 = sdf[x0, y1, z1]
    c100 = sdf[x1, y0, z0]
    c101 = sdf[x1, y0, z1]
    c110 = sdf[x1, y1, z0]
    c111 = sdf[x1, y1, z1]

    c00 = c000 * (1 - dz) + c001 * dz
    c01 = c010 * (1 - dz) + c011 * dz
    c10 = c100 * (1 - dz) + c101 * dz
    c11 = c110 * (1 - dz) + c111 * dz

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    c = c0 * (1 - dx) + c1 * dx
    return c

def check_sdf_vs_mesh(npz_path: str, stl_path: str, n_surf: int = 5000, n_rand: int = 20000):
    sdf, meta = load_npz(npz_path)
    bbox_min = np.array(meta["bbox_min"], dtype=np.float64)
    bbox_max = np.array(meta["bbox_max"], dtype=np.float64)
    voxel_size = float(meta["voxel_size"])

    mesh = trimesh.load(stl_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    mesh = normalize_mesh_if_needed(mesh, meta)

    # 1) surface points should have |SDF| small
    pts_surf, _ = trimesh.sample.sample_surface(mesh, n_surf)
    s_surf = trilinear_sample(sdf, pts_surf, bbox_min, voxel_size)
    print(f"[surface] |sdf| mean={np.mean(np.abs(s_surf)):.6g}  p95={np.percentile(np.abs(s_surf),95):.6g}  max={np.max(np.abs(s_surf)):.6g}")

    # 2) sign agreement with mesh.contains (watertight前提)
    rng = np.random.default_rng(0)
    pts = rng.uniform(bbox_min, bbox_max, size=(n_rand, 3))
    s = trilinear_sample(sdf, pts, bbox_min, voxel_size)

    inside_mesh = mesh.contains(pts)  # True=inside
    inside_sdf = (s < 0)  # inside_negative の想定

    agree = np.mean(inside_mesh == inside_sdf)
    print(f"[sign ] agreement(mesh.contains vs sdf<0) = {agree:.4f}")

# 例
check_sdf_vs_mesh("/mnt/ssd2tb/new_format_sdf/auto_generated/0002/00020000/stl/0002000000011.npz", "/mnt/ssd2tb/new_format/auto_generated/0002/00020000/stl/0002000000011.stl")
