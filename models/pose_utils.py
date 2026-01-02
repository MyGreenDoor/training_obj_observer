import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def _weighted_mean_and_cov(points: torch.Tensor,
                           weights: Optional[torch.Tensor],
                           eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    points:  (B,N,3)
    weights: (B,N) or None
    return:
      mu: (B,3)
      C:  (B,3,3)  （重み付き共分散の “分母はΣw” 版、N-1 ではない）
    """
    B, N, _ = points.shape
    if weights is None:
        w = torch.full((B, N), 1.0 / max(N, 1), device=points.device, dtype=points.dtype)
    else:
        w = weights.clamp_min(0)
        s = w.sum(dim=1, keepdim=True).clamp_min(eps)   # (B,1)
        w = w / s

    mu = (w.unsqueeze(-1) * points).sum(dim=1)          # (B,3)
    xc = points - mu.unsqueeze(1)                       # (B,N,3)
    C  = xc.transpose(1,2) @ (w.unsqueeze(-1) * xc)     # (B,3,3)
    return mu, C

def _pca_eig(C: torch.Tensor):
    """
    batched eigendecomposition
    C: (B,3,3)
    return:
      evals_desc: (B,3)  降順
      evecs_desc: (B,3,3) 列が主成分
    """
    evals, evecs = torch.linalg.eigh(C)                 # 昇順
    idx = torch.argsort(evals, dim=-1, descending=True) # (B,3)
    # 並べ替え
    B = C.shape[0]
    batch_idx = torch.arange(B, device=C.device).unsqueeze(-1)
    evals_desc = evals[batch_idx, idx]
    evecs_desc = evecs.gather(dim=-1, index=idx.unsqueeze(1).expand(B,3,3))
    # 右手系に揃える（det>0）
    dets = torch.det(evecs_desc)
    flip_mask = dets < 0
    if flip_mask.any():
        evecs_desc[flip_mask, :, 2] *= -1
    return evals_desc, evecs_desc

def estimate_pose_pca6d(
    obs_points_cam: torch.Tensor,        # (B,N,3)  観測3D（カメラ座標）
    mesh_verts_obj: torch.Tensor,        # (B,M,3)  モデル3D（物体座標）
    weights: Optional[torch.Tensor] = None  # (B,N) or None
) -> dict:
    """
    どのバッチでも 6DoF を返す（R: (B,3,3), t: (B,3)）。
    - 非対称物体に対して主成分軸を一致（符号は4候補から最適化）、det>0に補正。
    - weights=None でも動作（等重み）。
    - 数値退化/全ゼロ重みなどのケースは (R=I, t=0) にフォールバック。
    """
    B, N, _ = obs_points_cam.shape
    device, dtype = obs_points_cam.device, obs_points_cam.dtype

    # --- 観測側の重み付き PCA ---
    mu_obs, C_obs = _weighted_mean_and_cov(obs_points_cam, weights)
    evals_o, V_obs = _pca_eig(C_obs)                    # 列: v1,v2,v3

    # --- モデル側の等重み PCA（※本番は物体ごとに事前計算しても良い）---
    mu_mdl, C_mdl = _weighted_mean_and_cov(mesh_verts_obj, None)
    evals_m, V_mdl = _pca_eig(C_mdl)

    # --- 符号不定性の解消：4通りの符号反転で tr(R) を最大化 ---
    S_list = torch.stack([
        torch.diag(torch.tensor([ 1., 1., 1.], device=device, dtype=dtype)),
        torch.diag(torch.tensor([ 1.,-1.,-1.], device=device, dtype=dtype)),
        torch.diag(torch.tensor([-1., 1.,-1.], device=device, dtype=dtype)),
        torch.diag(torch.tensor([-1.,-1., 1.], device=device, dtype=dtype)),
    ], dim=0)  # (4,3,3)

    # R_candidates[b,k] = V_obs[b] @ S_list[k] @ V_mdl[b]^T
    # batchedでスコア（trace）比較
    # 先に A = V_obs, B = V_mdl
    A = V_obs                                    # (B,3,3)
    BmT = V_mdl.transpose(1,2)                   # (B,3,3)

    # (B,4,3,3): A @ S @ B^T
    R_cands = torch.einsum('bij,kjk,bkl->bkil', A, S_list, BmT)
    traces  = torch.einsum('bkii->bk', R_cands)  # (B,4) trace

    best_idx = traces.argmax(dim=1)              # (B,)
    # gather best R
    R = R_cands[torch.arange(B, device=device), best_idx]   # (B,3,3)

    # 安全のため det>0 に補正（稀に -1 が選ばれる可能性を潰す）
    dets = torch.det(R)
    bad = dets < 0
    if bad.any():
        R[bad, :, 2] *= -1

    # --- 並進 t = mu_obs - R @ mu_mdl ---
    t = mu_obs - torch.einsum('bij,bj->bi', R, mu_mdl)      # (B,3)

    # --- フォールバック（退化/NaN など）---
    # 評価値が極端に小さい（点が線状/面状/ゼロ重みなど） or 非有限なら I,0
    invalid = (
        ~torch.isfinite(R).all(dim=(1,2))
        | ~torch.isfinite(t).all(dim=1)
        | (evals_o[:, 0] <= 1e-12)  # 第一主成分がほぼゼロ
        | (evals_m[:, 0] <= 1e-12)
    )
    if invalid.any():
        R_fallback = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
        t_fallback = torch.zeros(B, 3, device=device, dtype=dtype)
        R = torch.where(invalid.view(B,1,1), R_fallback, R)
        t = torch.where(invalid.view(B,1),   t_fallback, t)

    return {"R": R, "t": t}
