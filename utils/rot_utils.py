

import torch
from typing import Literal, Optional, Tuple


def _axis_angle_to_R(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    axis_angle: (B,3)  ロドリゲスベクトル r = theta * axis
    return: (B,3,3)
    """
    B = axis_angle.shape[0]
    theta = torch.linalg.norm(axis_angle, dim=1, keepdim=True)  # (B,1)
    small = theta < 1e-12
    # unit axis (safe)
    axis = torch.where(
        small,
        torch.tensor([1.0,0.0,0.0], device=axis_angle.device, dtype=axis_angle.dtype).expand(B,3),
        axis_angle / theta
    )  # (B,3)

    x, y, z = axis[:,0], axis[:,1], axis[:,2]
    zeros = torch.zeros(B, device=axis.device, dtype=axis.dtype)
    K = torch.stack([
        zeros, -z,    y,
        z,    zeros, -x,
       -y,     x,   zeros
    ], dim=-1).reshape(B,3,3)

    ct = torch.cos(theta).view(B,1,1)
    st = torch.sin(theta).view(B,1,1)
    I  = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(B,3,3)

    R = I + st*K + (1.0-ct) * (K @ K)
    # for very small angles, fall back to identity
    R = torch.where(small.view(B,1,1), I, R)
    return R

def _quat_to_R(q: torch.Tensor, order: Literal["wxyz","xyzw"]="wxyz") -> torch.Tensor:
    """
    q: (B,4) quaternion; normalized inside. orderを指定。
    return: (B,3,3)
    """
    q = q / (q.norm(dim=1, keepdim=True).clamp(min=1e-12))
    if order == "wxyz":
        w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    else:  # "xyzw"
        x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        1 - 2*(yy+zz),  2*(xy - wz),    2*(xz + wy),
        2*(xy + wz),    1 - 2*(xx+zz),  2*(yz - wx),
        2*(xz - wy),    2*(yz + wx),    1 - 2*(xx+yy),
    ], dim=-1).reshape(-1,3,3)
    return R

def make_T_from_pose(
    rot: torch.Tensor,          # (B,4) quaternion もしくは (B,3) axis-angle
    pos: torch.Tensor,          # (B,3) or (B,4)  （4のとき先頭3成分を使用）
    quat_order: Literal["wxyz","xyzw"]="xyzw",
) -> torch.Tensor:
    """
    obj_in_left を生成（物体→左カメラ）。右手座標、+Z前方の想定。
    return: (B,4,4)
    """
    B = rot.shape[0]
    device, dtype = rot.device, rot.dtype

    # 位置
    if pos.shape[-1] == 4:
        t = pos[..., :3]
    else:
        t = pos
    t = t.view(B,3)

    # 回転
    if rot.shape[-1] == 4:
        R = _quat_to_R(rot.view(B,4), order=quat_order)        # (B,3,3)
    elif rot.shape[-1] == 3:
        R = _axis_angle_to_R(rot.view(B,3))                    # (B,3,3)
    else:
        raise ValueError("rot must be (B,4) quaternion or (B,3) axis-angle")

    # 同次行列
    T = torch.zeros(B,4,4, device=device, dtype=dtype)
    T[:, :3, :3] = R
    T[:, :3,  3] = t
    T[:,  3,  3] = 1.0
    return T


@torch.jit.script
def compose_T_from_Rt(
    R: torch.Tensor,      # (B,K,3,3)
    t: torch.Tensor,      # (B,K,3)
    valid: torch.Tensor   # (B,K) bool  （無いなら size(0)==0 の空テンソルを渡す運用でもOK）
) -> torch.Tensor:
    B = R.size(0)
    K = R.size(1)
    T = torch.zeros((B, K, 4, 4), dtype=R.dtype, device=R.device)  # 実体確保
    T[..., :3, :3] = R
    T[..., :3,  3] = t
    T[...,  3,  3] = 1.0

    # valid が渡されていれば、無効は単位行列に置換（任意）
    if valid.numel() != 0:
        eye44 = torch.eye(4, dtype=R.dtype, device=R.device).view(1,1,4,4)
        m = valid.to(T.dtype).view(B, K, 1, 1)
        T = T * m + eye44 * (1.0 - m)
    return T


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build rotation matrices from axes and angles with safe normalization."""
    axis = axis.reshape(-1, 3)
    angle = angle.reshape(-1)
    axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype).view(1, 3)
    axis_unit = torch.where(axis_norm > eps, axis / axis_norm, default_axis)
    axis_angle = axis_unit * angle.unsqueeze(1)
    R = _axis_angle_to_R(axis_angle)
    return R.reshape(*axis.shape[:-1], 3, 3)


def _proj_perp(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Project vectors v onto the plane orthogonal to n."""
    return v - (v * n).sum(dim=-1, keepdim=True) * n


def canonicalize_pose_gspose_torch(
    R: torch.Tensor,
    t: torch.Tensor,
    axis: torch.Tensor,
    order: torch.Tensor,
    eps: float = 1e-8,
    boundary_bias: float = 1e-12,
    ensure_objx_toward_pos_camx: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Canonicalize object-to-camera rotations under rotational symmetry.

    Args:
        R: (...,3,3) rotation matrices (object -> camera).
        t: (...,3) translation vectors in camera frame.
        axis: (...,3) symmetry axes in object frame.
        order: (...) symmetry orders (0=continuous, 1=no symmetry, 2/4/... finite).
        eps: Numerical stability epsilon.
        boundary_bias: Bias for half-open rounding in finite symmetry.
        ensure_objx_toward_pos_camx: Apply the +X_obj toward +X_cam tie-break.

    Returns:
        Tuple of (R_can, delta) with R_can shape (...,3,3) and delta shape (...,).
    """
    orig_shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    tf = t.reshape(-1, 3)
    af = axis.reshape(-1, 3)
    of = order.reshape(-1).to(Rf.dtype)

    axis_norm = torch.linalg.norm(af, dim=1, keepdim=True)
    axis_valid = axis_norm.squeeze(1) > eps
    af_unit = af / axis_norm.clamp_min(eps)

    is_finite = torch.isfinite(of)
    is_cont = (~is_finite) | (of == 0.0)
    is_noop = is_finite & (of == 1.0)
    is_finite_sym = is_finite & (of >= 2.0)
    sym_active = axis_valid & (is_cont | is_finite_sym) & (~is_noop)

    c_obj = -torch.einsum("bij,bj->bi", Rf.transpose(1, 2), tf)
    p = _proj_perp(c_obj, af_unit)
    p_norm = torch.linalg.norm(p, dim=1)

    x = torch.tensor([1.0, 0.0, 0.0], device=Rf.device, dtype=Rf.dtype).view(1, 3)
    y = torch.tensor([0.0, 1.0, 0.0], device=Rf.device, dtype=Rf.dtype).view(1, 3)
    z = torch.tensor([0.0, 0.0, 1.0], device=Rf.device, dtype=Rf.dtype).view(1, 3)
    ux = _proj_perp(x, af_unit)
    uz = _proj_perp(z, af_unit)
    uy = _proj_perp(y, af_unit)
    nx = torch.linalg.norm(ux, dim=1)
    nz = torch.linalg.norm(uz, dim=1)
    ny = torch.linalg.norm(uy, dim=1)

    use_x = nx > eps
    use_z = (~use_x) & (nz > eps)
    use_y = (~use_x) & (~use_z) & (ny > eps)
    valid_basis = use_x | use_z | use_y

    U = torch.zeros_like(af_unit)
    U = torch.where(use_x.unsqueeze(1), ux, U)
    U = torch.where(use_z.unsqueeze(1), uz, U)
    U = torch.where(use_y.unsqueeze(1), uy, U)
    U = U / torch.linalg.norm(U, dim=1, keepdim=True).clamp_min(eps)
    V = torch.cross(af_unit, U, dim=1)
    V = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(eps)

    phi = torch.atan2((p * V).sum(dim=1), (p * U).sum(dim=1))
    delta_base = torch.zeros_like(phi)
    if is_cont.any():
        delta_base = torch.where(is_cont, phi, delta_base)
    if is_finite_sym.any():
        step = (2.0 * torch.pi) / of.clamp_min(2.0)
        k = torch.floor(phi / step + 0.5 - boundary_bias)
        delta_base = torch.where(is_finite_sym, k * step, delta_base)

    delta_base = torch.where(sym_active & (p_norm > eps) & valid_basis, delta_base, torch.zeros_like(delta_base))

    if ensure_objx_toward_pos_camx:
        even_mask = is_cont | (is_finite_sym & torch.isclose(torch.remainder(of, 2.0), torch.zeros_like(of)))
        R_axis_base = _axis_angle_to_matrix(af_unit, delta_base, eps=eps)
        R_new_base = torch.einsum("bij,bjk->bik", Rf, R_axis_base)
        x_cam = R_new_base[:, :, 0]
        flip = even_mask & (x_cam[:, 0] < 0.0)
        delta = torch.where(flip, delta_base + torch.pi, delta_base)
    else:
        delta = delta_base

    delta = torch.where(sym_active, delta, torch.zeros_like(delta))
    R_axis = _axis_angle_to_matrix(af_unit, delta, eps=eps)
    R_can = torch.einsum("bij,bjk->bik", Rf, R_axis)
    R_can = torch.where(sym_active.view(-1, 1, 1), R_can, Rf)

    return R_can.reshape(*orig_shape, 3, 3), delta.reshape(*orig_shape)


def align_pose_by_symmetry_min_rotation(
    R_src: torch.Tensor,
    R_dst: torch.Tensor,
    axis: torch.Tensor,
    order: torch.Tensor,
    eps: float = 1e-6,
    boundary_bias: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align R_dst by symmetry to minimize rotation from R_src.

    Args:
        R_src: (...,3,3) source rotations.
        R_dst: (...,3,3) destination rotations.
        axis: (...,3) symmetry axes.
        order: (...) symmetry orders.
        eps: Numerical stability epsilon.
        boundary_bias: Bias for finite symmetry rounding.

    Returns:
        Tuple of (R_dst_aligned, delta) with shapes (...,3,3) and (...,).
    """
    orig_shape = R_src.shape[:-2]
    Rs = R_src.reshape(-1, 3, 3)
    Rd = R_dst.reshape(-1, 3, 3)
    af = axis.reshape(-1, 3)
    of = order.reshape(-1).to(Rs.dtype)

    axis_norm = torch.linalg.norm(af, dim=1, keepdim=True)
    axis_valid = axis_norm.squeeze(1) > eps
    af_unit = af / axis_norm.clamp_min(eps)

    is_finite = torch.isfinite(of)
    is_cont = (~is_finite) | (of == 0.0)
    is_noop = is_finite & (of == 1.0)
    is_finite_sym = is_finite & (of >= 2.0)
    sym_active = axis_valid & (is_cont | is_finite_sym) & (~is_noop)

    M = torch.einsum("bij,bjk->bik", Rs.transpose(1, 2), Rd)
    trM = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    aMa = torch.einsum("bi,bij,bj->b", af_unit, M, af_unit)

    ax, ay, az = af_unit[:, 0], af_unit[:, 1], af_unit[:, 2]
    K = torch.stack(
        [
            torch.zeros_like(ax), -az, ay,
            az, torch.zeros_like(ax), -ax,
            -ay, ax, torch.zeros_like(ax),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    trMK = (M * K.transpose(1, 2)).sum(dim=(1, 2))

    B = trM - aMa
    C = trMK
    phi = torch.atan2(C, B)

    delta = torch.zeros_like(phi)
    if is_cont.any():
        delta = torch.where(is_cont, phi, delta)
    if is_finite_sym.any():
        step = (2.0 * torch.pi) / of.clamp_min(2.0)
        k = torch.floor(phi / step + 0.5 - boundary_bias)
        delta = torch.where(is_finite_sym, k * step, delta)

    delta = torch.where(sym_active, delta, torch.zeros_like(delta))
    R_axis = _axis_angle_to_matrix(af_unit, delta, eps=eps)
    R_aligned = torch.einsum("bij,bjk->bik", Rd, R_axis)
    R_aligned = torch.where(sym_active.view(-1, 1, 1), R_aligned, Rd)
    return R_aligned.reshape(*orig_shape, 3, 3), delta.reshape(*orig_shape)


def apply_symmetry_rotation_map(
    rot_map: torch.Tensor,
    axis: torch.Tensor,
    delta: torch.Tensor,
    inst_id_map: torch.Tensor,
    fg_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply per-instance symmetry rotations to a rotation map.

    Args:
        rot_map: (B,3,3,H,W) rotation map.
        axis: (B,K,3) symmetry axes.
        delta: (B,K) symmetry deltas.
        inst_id_map: (B,H,W) instance IDs (0-based).
        fg_mask: Optional (B,1,H,W) foreground mask.
        eps: Numerical stability epsilon.

    Returns:
        rot_map aligned to symmetry as (B,3,3,H,W).
    """
    B, _, _, H, W = rot_map.shape
    if axis.numel() == 0 or delta.numel() == 0 or inst_id_map.numel() == 0:
        return rot_map
    K = axis.shape[1]
    if K == 0:
        return rot_map
    idx = inst_id_map.clamp(0, K - 1).long()
    idx_flat = idx.reshape(B, -1)
    axis_map = axis.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)).view(B, H, W, 3)
    delta_map = delta.gather(1, idx_flat).view(B, H, W)
    if fg_mask is not None:
        valid = fg_mask.squeeze(1) > 0
        delta_map = torch.where(valid, delta_map, torch.zeros_like(delta_map))

    R_axis = _axis_angle_to_matrix(axis_map, delta_map, eps=eps).reshape(B, H, W, 3, 3)
    rot_hw = rot_map.permute(0, 3, 4, 1, 2)
    rot_aligned = torch.einsum("bhwij,bhwjk->bhwik", rot_hw, R_axis)
    return rot_aligned.permute(0, 3, 4, 1, 2)


@torch.jit.script
def _rotvec_to_rotmat_map(rvec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # rvec: (B,3,H,W) → (B,3,3,H,W)
    B, _, H, W = rvec.shape
    rx, ry, rz = rvec[:, 0], rvec[:, 1], rvec[:, 2]
    theta = torch.sqrt(torch.clamp(rx*rx + ry*ry + rz*rz, min=0.0))  # ← clamp で負ゼロ防止
    inv_theta = 1.0 / torch.clamp(theta, min=eps)                    # ← eps ガード

    ux = torch.where(theta > eps, rx * inv_theta, torch.zeros_like(rx))
    uy = torch.where(theta > eps, ry * inv_theta, torch.zeros_like(ry))
    uz = torch.where(theta > eps, rz * inv_theta, torch.zeros_like(rz))

    c = torch.cos(theta); s = torch.sin(theta); one_c = 1.0 - c
    # ...（ロドリゲスはそのまま）...
    R = torch.stack([
        torch.stack([c + ux*ux*one_c,     ux*uy*one_c - uz*s, ux*uz*one_c + uy*s], dim=1),
        torch.stack([uy*ux*one_c + uz*s,  c + uy*uy*one_c,    uy*uz*one_c - ux*s], dim=1),
        torch.stack([uz*ux*one_c - uy*s,  uz*uy*one_c + ux*s, c + uz*uz*one_c],    dim=1),
    ], dim=1)                # (B,3,3,H,W)
    return R

@torch.jit.script
def se3_from_delta_map(delta6: torch.Tensor):
    """
    delta6: (B,6,H,W) = [dωx,dωy,dωz, dtx,dty,dtz]
    returns:
      R_delta: (B,3,3,H,W)
      t_delta: (B,3,H,W)
    """
    R_delta = _rotvec_to_rotmat_map(delta6[:, 0:3])   # (B,3,3,H,W)
    t_delta = delta6[:, 3:6]                          # (B,3,H,W)
    return R_delta, t_delta


@torch.no_grad()
def scale_K(K_1x: torch.Tensor, down: int) -> torch.Tensor:
    # K_1x: (B,3,3), down: 4 など
    K = K_1x.clone()
    K[:, 0, 0] /= down  # fx
    K[:, 1, 1] /= down  # fy
    K[:, 0, 2] /= down  # cx
    K[:, 1, 2] /= down  # cy
    return K


def apply_delta_to_pointmap_projective(
    X_map: torch.Tensor,        # (B,3,H,W)  旧（レンダ）point map in camera coords
    R_delta: torch.Tensor,      # (B,3,3,H,W) per-pixel ΔR （列ベクトル系の合成前提）
    t_delta: torch.Tensor,      # (B,3,H,W)  per-pixel Δt
    K_14: torch.Tensor,         # (B,3,3)    1/4解像度用の内部パラメータ
    z_soft: bool = True,        # True: 奥行き優先をsoftに
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Δpose を適用した結果を (u',v') に再投影し、前向きスプラットで新しい point_map を構成。
    画素外/裏面は無視、穴は元の X_map で埋める。
    """
    device = X_map.device
    dtype  = X_map.dtype
    B, _, H, W = X_map.shape

    # 1) per-pixel SE3: X' = RΔ * X + tΔ
    # einsum('bijhw,bjhw->bihw'): (B,3,3,H,W) × (B,3,H,W) → (B,3,H,W)
    Xp = torch.einsum('bijhw,bjhw->bihw', R_delta.to(dtype), X_map) + t_delta.to(dtype)  # (B,3,H,W)

    # 2) project to pixels (u', v') with K (H/4,W/4 専用)
    fx, fy = K_14[:, 0, 0], K_14[:, 1, 1]
    cx, cy = K_14[:, 0, 2], K_14[:, 1, 2]
    X, Y, Z = Xp[:, 0], Xp[:, 1], Xp[:, 2] + eps

    # valid depth (>0)
    valid = (Z > eps)

    # u' = fx*X/Z + cx, v' = fy*Y/Z + cy
    # ここで fx, cx はバッチごとにスカラーなのでブロードキャスト
    u_p = fx.view(B, 1, 1) * (X / Z) + cx.view(B, 1, 1)  # (B,H,W)
    v_p = fy.view(B, 1, 1) * (Y / Z) + cy.view(B, 1, 1)  # (B,H,W)

    # 3) 前向きスプラット（双一次）
    # 出力バッファと重み
    out = torch.zeros_like(X_map)                         # (B,3,H,W)
    wts = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    # Z に基づく重み（近いほど重く）: softmax はコスト高なので 1/Z か exp(-αZ) 程度でOK
    if z_soft:
        wz = (1.0 / Z.clamp_min(1e-3))                   # 近いほど大
    else:
        wz = torch.ones_like(Z)

    # 元の整数格子（ソース）
    uu = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W).to(dtype)
    vv = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W).to(dtype)

    # マスク（画面内 & 正の Z）
    inb = valid & (u_p >= 0.0) & (u_p <= (W - 1)) & (v_p >= 0.0) & (v_p <= (H - 1))

    if inb.any():
        up = u_p[inb];  vp = v_p[inb]
        xp0 = up.floor().clamp_(0, W - 1)
        yp0 = vp.floor().clamp_(0, H - 1)
        xp1 = (xp0 + 1).clamp_(0, W - 1)
        yp1 = (yp0 + 1).clamp_(0, H - 1)

        wx = (up - xp0)   # ∈[0,1)
        wy = (vp - yp0)
        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy

        # 重みに深度重みを掛ける
        wz_sel = wz[inb].to(dtype)
        w00 *= wz_sel; w01 *= wz_sel; w10 *= wz_sel; w11 *= wz_sel

        # バッチ/位置の線形インデックス
        b_idx, y_src, x_src = torch.nonzero(inb, as_tuple=True)
        # ソースの値（B,3,H,W）→ inb で抽出 → (N,3)
        vals = Xp.permute(0, 2, 3, 1)[inb]  # (N,3)

        def _scatter_add(weight, x_ind, y_ind, wval):
            lin = (b_idx * H + y_ind.long()) * W + x_ind.long()
            # flatten
            weight_flat = wts.view(B * H * W, 1)
            out_flat    = out.view(B *  H * W, 3)
            weight_flat.index_add_(0, lin, wval.unsqueeze(1).to(dtype))
            out_flat.index_add_(0, lin, (vals * wval.unsqueeze(1).to(dtype)))

        _scatter_add(wts, xp0, yp0, w00)
        _scatter_add(wts, xp0, yp1, w01)
        _scatter_add(wts, xp1, yp0, w10)
        _scatter_add(wts, xp1, yp1, w11)

    # 4) 正規化して穴を埋める（重み0は元の値を残す）
    out = torch.where(
        (wts > 0.0),
        out / wts.clamp_min(1e-6),
        X_map
    )
    return out


@torch.jit.script
def compose_rotmap(R_delta: torch.Tensor, R_cur: torch.Tensor) -> torch.Tensor:
    # R_new = R_delta @ R_cur
    # どちらも (B,3,3,H,W)
    return torch.einsum('bijhw,bjkhw->bikhw', R_delta, R_cur)


@torch.jit.script
def update_translation_map(
    t_cur: torch.Tensor,          # (B,3,H,W)
    d_map: torch.Tensor,          # (B,3,H,W) : (dx,dy,dz)
    weight: float = 100.0,
    depth_transform: str = "exp",
    detach_depth_for_xy: bool = True,
    dz_clip: float = 4.0,
    eps: float = 1e-6
) -> torch.Tensor:
    tx, ty, tz = t_cur[:, 0], t_cur[:, 1], t_cur[:, 2]
    tz_safe = torch.clamp(tz, min=eps)

    dx = d_map[:, 0]
    dy = d_map[:, 1]
    dz = d_map[:, 2]

    # 1) depth update
    if depth_transform == "exp":
        dz = torch.clamp(dz, min=-dz_clip, max=dz_clip)  # 安定化
        vz = tz / torch.exp(dz)
    else:
        # 線形なら軽く制限しておく
        dz = torch.clamp(dz, min=-0.95, max=0.95)
        vz = tz * (1.0 + dz)

    vz = torch.clamp(vz, min=eps)  # 最終安全弁

    # 2) normalized plane offsets
    # ray_x = (tx/tz) + dx/weight
    # ray_y = (ty/tz) + dy/weight
    ray_x = torch.addcdiv(dx / weight, tx, tz_safe)
    ray_y = torch.addcdiv(dy / weight, ty, tz_safe)

    # 3) scale back by new depth
    scale = vz.detach() if detach_depth_for_xy else vz
    vx = scale * ray_x
    vy = scale * ray_y

    return torch.stack((vx, vy, vz), dim=1)


@torch.jit.script
def update_pose_maps(
    R_cur: torch.Tensor,          # (B,3,3,H,W) 現在の回転マップ
    t_cur: torch.Tensor,          # (B,3,H,W)   現在の並進マップ
    R_delta: torch.Tensor,        # (B,3,3,H,W) 増分回転（回転行列）
    d_map: torch.Tensor,          # (B,3,H,W)   増分並進パラメタ（dx,dy,dz）
    weight: float = 100.0,
    depth_transform: str = "exp",
    detach_depth_for_xy: bool = False,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    R_new = compose_rotmap(R_delta, R_cur)
    t_new = update_translation_map(
        t_cur, d_map, weight, depth_transform, detach_depth_for_xy, eps
    )
    return R_new, t_new


@torch.jit.script
def update_pose_maps_se3(
    R_cur: torch.Tensor,          # (B,3,3,H,W) current rotation
    t_cur: torch.Tensor,          # (B,3,H,W) current translation in camera frame
    R_delta: torch.Tensor,        # (B,3,3,H,W) left-multiplied rotation delta
    t_delta: torch.Tensor,        # (B,3,H,W) translation delta in camera frame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update pose maps by SE(3) left-multiplication in camera frame."""
    R_new = compose_rotmap(R_delta, R_cur)
    t_new = torch.einsum("bijhw,bjhw->bihw", R_delta, t_cur) + t_delta
    return R_new, t_new


def splat_per_instance(point_map_rend, inst_id_map, Rk, tk, K14, tau=0.02, eps=1e-6):
    B, _, H, W = point_map_rend.shape
    device = point_map_rend.device
    # 出力バッファ
    num = torch.zeros(B, 3, H, W, device=device, dtype=point_map_rend.dtype)
    den = torch.zeros(B, 1, H, W, device=device, dtype=point_map_rend.dtype)

    # 物体ごと（K が小さい前提; 大きければマスクでまとめて処理）
    K = Rk.size(1)
    for k in range(K):
        # 1) マスク抽出（その物体のピクセルだけ取り出し）
        Mk = (inst_id_map[:,0] == k)                      # (B,H,W) bool
        if not Mk.any(): 
            continue

        # (y,x) の座標（flatten）
        b_idx, y_idx, x_idx = torch.where(Mk)
        # その画素の点群 X
        X = point_map_rend[b_idx, :, y_idx, x_idx]        # (N,3)
        # 物体ポーズ
        R = Rk[b_idx, k]                                  # (N,3,3)
        t = tk[b_idx, k]                                  # (N,3)

        # 2) 変換
        Xp = (R @ X.unsqueeze(-1)).squeeze(-1) + t        # (N,3)
        Xp_x, Xp_y, Xp_z = Xp[:,0], Xp[:,1], Xp[:,2]
        valid = (Xp_z > 1e-6)
        if not valid.any(): 
            continue
        b_idx, y_idx, x_idx, Xp_x, Xp_y, Xp_z = b_idx[valid], y_idx[valid], x_idx[valid], Xp_x[valid], Xp_y[valid], Xp_z[valid]

        # 3) 投影（1/4解像度のK）
        fx = K14[b_idx, 0, 0]; fy = K14[b_idx, 1, 1]
        cx = K14[b_idx, 0, 2]; cy = K14[b_idx, 1, 2]
        uf = fx * (Xp_x / Xp_z) + cx
        vf = fy * (Xp_y / Xp_z) + cy

        # 画面内のみ
        in_img = (uf >= 0) & (uf <= (W-1)) & (vf >= 0) & (vf <= (H-1))
        if not in_img.any(): 
            continue
        b_idx = b_idx[in_img]; y_idx = y_idx[in_img]; x_idx = x_idx[in_img]
        uf = uf[in_img]; vf = vf[in_img]; Xp = Xp[valid][in_img]; z = Xp[:,2]

        # 4) 4近傍 bilinear とソフトZ重み
        x0 = torch.floor(uf).to(torch.long); x1 = (x0 + 1).clamp_max(W-1)
        y0 = torch.floor(vf).to(torch.long); y1 = (y0 + 1).clamp_max(H-1)
        wx = (uf - x0.float()); wy = (vf - y0.float())
        nbrs = [
            (y0, x0, (1-wx)*(1-wy)),
            (y0, x1, (wx)*(1-wy)),
            (y1, x0, (1-wx)*(wy)),
            (y1, x1, (wx)*(wy)),
        ]
        logwz = -(z / tau)               # ソフトZ: exp(-Z/τ)

        for yy, xx, wb in nbrs:
            w = (wb * torch.exp(logwz)).unsqueeze(1)      # (N,1)
            # index_add のためにフラット化
            lin = (b_idx*H + yy) * W + xx                 # (N,)
            # 3chと1chを分けて scatter_add
            num_flat = num.view(B, 3, H*W)
            den_flat = den.view(B, 1, H*W)
            num_flat[b_idx, :, lin] += (w * Xp).transpose(0,1)  # (3,N)
            den_flat[b_idx, :, lin] += w.transpose(0,1)         # (1,N)

    out = num / den.clamp_min(eps)  # 穴は別途埋め戻し推奨
    return out



def depth_to_pointmap_from_K(depth_14: torch.Tensor,  # (B,1,H/4,W/4) [m]
                             K_14: torch.Tensor,      # (B,3,3)       (1/4解像度のK)
                             eps: float = 1e-6) -> torch.Tensor:
    """
    射影モデル: X = Z * ( (u - cx)/fx, (v - cy)/fy, 1 )
    戻り: (B,3,H/4,W/4)  [m]
    """
    B, _, H4, W4 = depth_14.shape
    device = depth_14.device
    dtype  = depth_14.dtype

    # ピクセル座標（中心合わせ）
    ys = torch.arange(H4, device=device, dtype=dtype).view(1, H4, 1).expand(B, H4, W4)
    xs = torch.arange(W4, device=device, dtype=dtype).view(1, 1, W4).expand(B, H4, W4)

    fx = K_14[:, 0, 0].view(B, 1, 1)
    fy = K_14[:, 1, 1].view(B, 1, 1)
    cx = K_14[:, 0, 2].view(B, 1, 1)
    cy = K_14[:, 1, 2].view(B, 1, 1)

    Z = depth_14[:, 0].clamp_min(eps)                   # (B,H4,W4)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    return torch.stack([X, Y, Z], dim=1)



@torch.jit.script
def _skew_batch(r: torch.Tensor) -> torch.Tensor:
    # r: (N,3) -> K: (N,3,3)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    K = torch.stack([
        zero, -rz,  ry,
         rz,  zero,-rx,
        -ry,  rx,  zero
    ], dim=-1).view(-1, 3, 3)
    return K


@torch.jit.script
def so3_exp_batch(r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    r: (N,3)  -> R: (N,3,3)
    Rodrigues (微小角安定版). 64bitで計算して最後に戻す。
    """
    orig_dtype = r.dtype
    r64 = r.to(torch.float64)                   # 安定化
    K = _skew_batch(r64)                        # (N,3,3)
    theta = torch.linalg.norm(r64, dim=-1)      # (N,)

    # 係数（小角と通常をブレンド）。最終的に (N,1,1) に整形すること！
    theta2 = theta * theta
    small = (theta < 1e-4)

    A_small = 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0            # 近似: sin(x)/x
    B_small = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0           # 近似: (1-cos)/x^2

    A_full  = torch.sin(theta) / torch.clamp(theta, min=eps)
    B_full  = (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=eps)

    A = torch.where(small, A_small, A_full).view(-1, 1, 1)               # (N,1,1)
    B = torch.where(small, B_small, B_full).view(-1, 1, 1)               # (N,1,1)

    I = torch.eye(3, dtype=torch.float64, device=r.device).expand(K.shape)
    R = I + A * K + B * (K @ K)
    return R.to(orig_dtype)


@torch.jit.script
def so3_exp_vec(r_vec: torch.Tensor) -> torch.Tensor:
    """
    r_vec: (B,K,3)  -> R: (B,K,3,3)
    """
    B, K = r_vec.shape[0], r_vec.shape[1]
    r_flat = r_vec.reshape(-1, 3)                       # (N,3) N=B*K
    R_flat = so3_exp_batch(r_flat)                      # (N,3,3)
    return R_flat.reshape(B, K, 3, 3)


def so3_log_batch(R: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1.0 + 1e-6, 3.0 - 1e-6)
    cos_t = (tr - 1.0) * 0.5
    theta = torch.acos(cos_t)

    A = (R - R.transpose(-1, -2))
    vee = torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], dim=-1)

    sin_t = torch.sin(theta).clamp_min(eps)
    scale = theta / (2.0 * sin_t)

    small = (theta < 1e-3)
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)  # 小角では log ≈ 0.5*vee(R-R^T)
    return scale.unsqueeze(-1) * vee


def so3_log_map(rot_map: torch.Tensor) -> torch.Tensor:
    # rot_map: (B,3,3,H,W) -> r_map: (B,3,H,W)
    B, _, _, H, W = rot_map.shape
    Rf = rot_map.permute(0, 3, 4, 1, 2).reshape(-1, 3, 3).to(torch.float64)
    rv = so3_log_batch(Rf)  # (-1,3)
    rv = rv.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
    return rv.to(rot_map.dtype)


# ===== Extra: average auxiliary maps with the same weights =====
@torch.jit.script
def _avg_optional_map(m: torch.Tensor, wn: torch.Tensor) -> torch.Tensor:
    # m: (B,C,H,W) -> (B,K,C), weighted average by wn
    C = m.size(1)
    mk = (m.unsqueeze(1) * wn)  # (B,K,C,H,W)
    mk = mk.sum(dim=(3,4))      # (B,K,C)
    return mk


@torch.jit.script
def _gather_peak_map(
    m: torch.Tensor,
    b_ix: torch.Tensor,
    lin: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    # m: (B,C,H,W) -> (B,K,C) at (y_pk,x_pk)
    C = m.size(1)
    m_flat = m.permute(0, 2, 3, 1).reshape(m.size(0), H * W, C)  # (B,HW,C)
    return m_flat[b_ix, lin]



# def pose_from_maps_auto(
#     rot_map: torch.Tensor,      # (B,3,3,H,W)
#     pos_map: torch.Tensor,      # (B,3,H,W)
#     Wk_1_4: torch.Tensor,       # (B,K,1,H,W)
#     wfg: torch.Tensor,          # (B,1,H,W)
#     peaks_yx: torch.Tensor = None,  # (B,K,2) or None, yx
#     min_px: int = 10,
#     min_wsum: float = 1e-6,
#     tau_peak: float = 0.0,          # 例: 0.2（peak信頼ゲート）
#     pos_logvar: Optional[torch.Tensor] = None,        # (B,Cp,H,W) 例: Cp=1 or 3
#     rot_logvar_theta: Optional[torch.Tensor] = None,  # (B,Cr,H,W) 例: Cr=1
# ):
#     """
#     Always pick R, t (and logvars if provided) at peak.

#     return:
#       R_hat: (B,K,3,3)
#       t_hat: (B,K,3)
#       valid: (B,K)  <- 面積/総和の有効判定 + ピーク信頼ゲート
#       pos_logvar_k: (B,K,Cp) or None
#       rot_logvar_k: (B,K,Cr) or None
#     """
#     B = rot_map.size(0)
#     H, W = pos_map.shape[-2:]
#     device, dtype = rot_map.device, rot_map.dtype

#     # K==0 なら即返し
#     if Wk_1_4.numel() == 0:
#         zR = rot_map.new_zeros(B, 0, 3, 3)
#         zt = pos_map.new_zeros(B, 0, 3)
#         zv = torch.zeros(B, 0, dtype=torch.bool, device=device)
#         return zR, zt, zv, None, None

#     B_, K = Wk_1_4.shape[:2]
#     assert B_ == B, "Batch size mismatch."

#     # --- 重み前処理 ---
#     Wk = Wk_1_4.clamp_min(0).to(torch.float32)    # (B,K,1,H,W)
#     wfg = wfg.clamp_min(0).to(torch.float32)      # (B,1,H,W)
#     w = Wk * wfg.unsqueeze(1)                     # (B,K,1,H,W)

#     # 面積・総和（valid 判定用のみ）
#     px = (w > 0).sum(dim=(2,3,4))                 # (B,K)
#     wsum = w.sum(dim=(2,3,4))                     # (B,K)

#     # --- peak 取得 ---
#     if (peaks_yx is None) or (peaks_yx.numel() == 0):
#         w2d = w_peak.squeeze(2)                        # (B,K,H,W)
#         wflat = w2d.reshape(B, K, -1)             # (B,K,HW)
#         idx = wflat.argmax(dim=-1)                # (B,K)
#         y_pk = (idx // W).clamp_(0, H-1)
#         x_pk = (idx %  W).clamp_(0, W-1)
#     else:
#         y_pk = peaks_yx[..., 0].to(torch.long).clamp_(0, H-1)
#         x_pk = peaks_yx[..., 1].to(torch.long).clamp_(0, W-1)

#     # --- peak 上の前景信頼 ---
#     b_ix = torch.arange(B, device=device).view(B,1).expand(B,K)  # (B,K)
#     lin = (y_pk * W + x_pk).clamp_(0, H*W - 1)                   # (B,K)
#     wfg_flat = wfg[:, 0].reshape(B, H*W)                         # (B,HW)
#     wfg_at_peak = wfg_flat[b_ix, lin]                            # (B,K)

#     # --- peak で R, t を gather（常に peak 由来） ---
#     # rot_map: (B,3,3,H,W) -> (B,HW,3,3)
#     R_hw = rot_map.permute(0,3,4,1,2).reshape(B, H*W, 3, 3)
#     # pos_map: (B,3,H,W) -> (B,HW,3)
#     t_hw = pos_map.permute(0,2,3,1).reshape(B, H*W, 3)

#     R_hat = R_hw[b_ix, lin].to(dtype)            # (B,K,3,3)
#     t_hat = t_hw[b_ix, lin].to(dtype)            # (B,K,3)

#     # --- logvar も peak で gather ---
#     def _gather_peak_map(m: torch.Tensor):
#         # m: (B,C,H,W) -> (B,K,C)
#         C = m.size(1)
#         m_flat = m.permute(0,2,3,1).reshape(B, H*W, C)   # (B,HW,C)
#         return m_flat[b_ix, lin].to(dtype)               # (B,K,C)

#     pos_logvar_k = _gather_peak_map(pos_logvar) if (pos_logvar is not None) else None
#     rot_logvar_k = _gather_peak_map(rot_logvar_theta) if (rot_logvar_theta is not None) else None

#     # --- valid 判定（R/t は常に peak だが、品質フラグは返す） ---
#     valid_area = (px >= min_px) & (wsum >= min_wsum)     # (B,K)
#     valid_peak = (wfg_at_peak > tau_peak)                # (B,K)
#     valid = valid_area & valid_peak                      # (B,K)

#     return R_hat, t_hat, valid, pos_logvar_k, rot_logvar_k


@torch.jit.script
def pose_from_maps_auto(
    rot_map: torch.Tensor,      # (B,3,3,H,W)
    pos_map: torch.Tensor,      # (B,3,H,W)
    Wk_1_4: torch.Tensor,       # (B,K,1,H,W)
    wfg: torch.Tensor,          # (B,1,H,W)
    peaks_yx: Optional[torch.Tensor] = None,  # (B,K,2) or None
    min_px: int = 10,
    min_wsum: float = 1e-6,
    tau_peak: float = 0.0,          # 例: 0.2 など（peak信頼の閾値）
    pos_logvar: Optional[torch.Tensor] = None,        # (B,Cp,H,W) 例: Cp=1(Zのみ) or 3(XYZ)
    rot_logvar_theta: Optional[torch.Tensor] = None,  # (B,Cr,H,W) 例: Cr=1（角度分散など）
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    return:
      R_hat: (B,K,3,3)
      t_hat: (B,K,3)
      valid: (B,K)  <- 面積/総和の有効判定 + ピーク信頼ゲート
      pos_logvar_k: (B,K,Cp) or None
      rot_logvar_k: (B,K,Cr) or None
    """
    B, K = Wk_1_4.shape[:2]
    device, dtype = rot_map.device, rot_map.dtype
    H, W = pos_map.shape[-2:]

    # K==0 なら即返し
    if K == 0:
        zR = rot_map.new_zeros(B, 0, 3, 3)
        zt = pos_map.new_zeros(B, 0, 3)
        zv = torch.zeros(B, 0, dtype=torch.bool, device=device)
        return zR, zt, zv, None, None

    # --- 重みの前処理 ---
    Wk = Wk_1_4.clamp_min(0).to(torch.float32)           # (B,K,1,H,W)
    wfg = wfg.clamp_min(0).to(torch.float32)             # (B,1,H,W)
    w = (Wk * wfg.unsqueeze(1))                          # (B,K,1,H,W)

    # Validity check uses base weights only.
    px = (w > 0).sum(dim=(2,3,4))                        # (B,K)
    wsum = w.sum(dim=(2,3,4))                            # (B,K)
    valid = (px >= min_px) & (wsum >= min_wsum)          # (B,K)

    w_pos = w
    w_rot = w
    w_peak = w
    if pos_logvar is not None:
        pos_logvar_t = torch.jit._unwrap_optional(pos_logvar)
        pos_lv = pos_logvar_t.mean(dim=1, keepdim=True)
        pos_conf = torch.exp(-0.5 * pos_lv)
        w_pos = w_pos * pos_conf.unsqueeze(1)
        w_peak = w_peak * pos_conf.unsqueeze(1)
    if rot_logvar_theta is not None:
        rot_logvar_t = torch.jit._unwrap_optional(rot_logvar_theta)
        rot_conf = torch.exp(-0.5 * rot_logvar_t)
        w_rot = w_rot * rot_conf.unsqueeze(1)
        w_peak = w_peak * rot_conf.unsqueeze(1)

    denom_pos = w_pos.sum(dim=(2,3,4)).view(B, K, 1, 1, 1).clamp_min(min_wsum)
    wn_pos = w_pos / denom_pos
    denom_rot = w_rot.sum(dim=(2,3,4)).view(B, K, 1, 1, 1).clamp_min(min_wsum)
    wn_rot = w_rot / denom_rot

    # --- t: weighted average ---
    t_hat = (pos_map.to(torch.float32).unsqueeze(1) * wn_pos).sum(dim=(3,4))  # (B,K,3)

    # --- R: Lie average (log -> avg -> exp) ---
    r_map = so3_log_map(rot_map.to(torch.float32))                # (B,3,H,W)
    r_hat = (r_map.unsqueeze(1) * wn_rot).sum(dim=(3,4))          # (B,K,3)
    R_hat = so3_exp_vec(r_hat).to(dtype)                          # (B,K,3,3)
    t_hat = t_hat.to(dtype)
    pos_logvar_k = torch.jit.annotate(Optional[torch.Tensor], None)
    rot_logvar_k = torch.jit.annotate(Optional[torch.Tensor], None)
    if pos_logvar is not None:
        pos_logvar_t = torch.jit._unwrap_optional(pos_logvar)
        pos_logvar_k = _avg_optional_map(pos_logvar_t, wn_pos)         # (B,K,Cp)
    if rot_logvar_theta is not None:
        rot_logvar_t = torch.jit._unwrap_optional(rot_logvar_theta)
        rot_logvar_k = _avg_optional_map(rot_logvar_t, wn_rot)         # (B,K,Cr)

    # --- Fallback if needed (peak selection) ---
    use_pk = ~valid
    need_pk = bool(use_pk.any().item())

    if need_pk:
        # peaks が無ければ「最大重み画素」で代用
        if (peaks_yx is None) or (peaks_yx.numel() == 0):
            w2d = w_peak.squeeze(2)                               # (B,K,H,W)
            wflat = w2d.reshape(B, K, -1)                    # (B,K,HW)
            idx = wflat.argmax(dim=-1)                       # (B,K)
            y_pk = (idx // W).clamp_(0, H-1)
            x_pk = (idx %  W).clamp_(0, W-1)
        else:
            y_pk = peaks_yx[..., 0].to(torch.long).clamp_(0, H-1)
            x_pk = peaks_yx[..., 1].to(torch.long).clamp_(0, W-1)

        # ピーク位置の前景信頼（wfg）による gate
        wfg_pk = wfg[:, 0]                                   # (B,H,W)
        b_ix = torch.arange(B, device=device).view(B,1).expand(B,K)
        lin = (y_pk * W + x_pk).clamp_(0, H*W - 1)
        wfg_flat = wfg_pk.reshape(B, H*W)
        wfg_at_peak = wfg_flat[b_ix, lin]                    # (B,K)
        valid = (valid | use_pk) & (wfg_at_peak > tau_peak)

        # ピークから R,t を抽出（全件計算→ where で混ぜる）
        R_hw = rot_map.permute(0,3,4,1,2).reshape(B, H*W, 3, 3)  # (B,HW,3,3)
        t_hw = pos_map.permute(0,2,3,1).reshape(B, H*W, 3)       # (B,HW,3)
        R_pk = R_hw[b_ix, lin]                                   # (B,K,3,3)
        t_pk = t_hw[b_ix, lin]                                   # (B,K,3)

        mR = use_pk.unsqueeze(-1).unsqueeze(-1)
        mT = use_pk.unsqueeze(-1)
        R_hat = torch.where(mR, R_pk.to(R_hat.dtype), R_hat)
        t_hat = torch.where(mT, t_pk.to(t_hat.dtype), t_hat)

        # ▼ 追加：logvar 系もピーク値でフォールバックして整合
        if pos_logvar is not None and pos_logvar_k is not None:
            pos_logvar_t = torch.jit._unwrap_optional(pos_logvar)
            pos_logvar_pk = _gather_peak_map(pos_logvar_t, b_ix, lin, H, W).to(dtype)       # (B,K,Cp)
            m = use_pk.unsqueeze(-1)                              # (B,K,1)
            pos_logvar_k = torch.where(m, pos_logvar_pk, pos_logvar_k)
        if rot_logvar_theta is not None and rot_logvar_k is not None:
            rot_logvar_t = torch.jit._unwrap_optional(rot_logvar_theta)
            rot_logvar_pk = _gather_peak_map(rot_logvar_t, b_ix, lin, H, W).to(dtype)       # (B,K,Cr)
            m = use_pk.unsqueeze(-1)
            rot_logvar_k = torch.where(m, rot_logvar_pk, rot_logvar_k)

    return R_hat, t_hat, valid, pos_logvar_k, rot_logvar_k
