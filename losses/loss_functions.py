
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head import nearest_so3_batched, pos_mu_to_pointmap
from models.sscflow2 import scale_intrinsics_pair_for_feature
from utils import rot_utils

# ----- 1) Center heatmap: focal (logit版, f32固定) -----
def focal_loss_logits(logits: torch.Tensor, target: torch.Tensor,
                      alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    z = logits.to(torch.float32)
    t = target.to(torch.float32).clamp(0.0, 1.0)

    # 安定形
    sp_pos = F.softplus( z)   # = log(1+e^z)
    sp_neg = F.softplus(-z)   # = log(1+e^-z)

    log_p  = -sp_neg          # log σ(z)
    log_np = -sp_pos          # log (1-σ(z))

    p_pow  = torch.exp(-alpha * sp_neg)  # σ(z)^alpha
    np_pow = torch.exp(-alpha * sp_pos)  # (1-σ(z))^alpha

    pos_w = t
    neg_w = (1.0 - t).pow(beta)

    pos = -(np_pow * log_p)  * pos_w
    neg = -(p_pow  * log_np) * neg_w

    denom = pos_w.sum().clamp(min=1.0)
    return (pos.sum() + neg.sum()) / denom


# ----- 2) Charbonnier -----
@torch.jit.script
def charbonnier1(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def pos_loss_hetero(mu_pos: torch.Tensor,   # (B,K, 3) [mm]
                       lv_pos: torch.Tensor,   # (B,K, 3) [mm]
                       pos_gt: torch.Tensor,   # (B,K, 3) [mm]
                       valid: torch.Tensor     # (B,K) [mm]
                ) -> torch.Tensor:
    rho = charbonnier1(mu_pos - pos_gt)                 # (B,K,3)
    inv = torch.exp(-lv_pos).clamp_min(1e-8)       # (B,K,3)
    loss = inv * rho + lv_pos                  # (B,K,3)
    if valid.dim() == 2:
        valid = valid.unsqueeze(-1)                            # (.., .., 1)
    return (loss * valid).sum() / (valid.sum().clamp_min(1.0)  * mu_pos.size(-1))


@torch.jit.script
def pos_loss_hetero_map(
    mu_pos_map: torch.Tensor,           # (B, C, H, W)
    lv_pos_map: typing.Optional[torch.Tensor], # (B, C, H, W)  (log-variance) or None
    pos_gt_map: torch.Tensor,           # (B, C, H, W)
    valid_map: torch.Tensor             # (B, 1, H, W)  (0/1 or float mask)
) -> torch.Tensor:
    """
    Heteroscedastic position loss with lv.

    if lv_pos_map is not None:
        loss = sum valid * ( exp(-lv) * rho + lv ) / max(1.0, sum(valid))
    else:
        loss = sum valid * rho / max(1.0, sum(valid))

    where rho = charbonnier(mu_pos - pos_gt)
    """
    # robust error per-pixel per-channel
    rho = charbonnier1(mu_pos_map - pos_gt_map)  # (B,C,H,W)

    # ensure valid is float and broadcastable; keep denominator counting spatial valid only
    valid_f = valid_map.float()                  # (B,1,H,W)

    if lv_pos_map is None:
        # 単純なCharbonnier位置loss
        weighted = rho * valid_f
        denom = valid_f.sum().clamp_min(1.0)
        return weighted.sum() / denom

    # ----- ここからヘテロスケ版 -----

    lv_pos_map = lv_pos_map.clamp(-10.0, 10.0)
    # inverse-variance weight (B,C,H,W)
    inv = torch.exp(-lv_pos_map).clamp_min(1e-8)  # (B,C,H,W)

    # per-pixel per-channel hetero loss
    loss = inv * rho + lv_pos_map                 # (B,C,H,W)

    weighted = loss * valid_f

    # normalize by number of valid spatial locations (not multiplied by channels)
    denom = valid_f.sum().clamp_min(1.0)  * mu_pos_map.size(1)

    return weighted.sum() / denom


def _pos_mu_gt_from_t_map(
    t_map: torch.Tensor,  # (B,3,H/4,W/4)
    K_left_1x: torch.Tensor,
    downsample: int = 4,
    eps: float = 1e-6,
    use_logz: bool = False,
) -> torch.Tensor:
    B, _, H4, W4 = t_map.shape
    device = t_map.device
    dtype = t_map.dtype

    X = t_map[:, 0:1]
    Y = t_map[:, 1:2]
    Z = t_map[:, 2:3].clamp_min(eps)

    fx = K_left_1x[:, 0, 0].view(B, 1, 1, 1)
    fy = K_left_1x[:, 1, 1].view(B, 1, 1, 1)
    cx = K_left_1x[:, 0, 2].view(B, 1, 1, 1)
    cy = K_left_1x[:, 1, 2].view(B, 1, 1, 1)

    u_c = fx * (X / Z) + cx
    v_c = fy * (Y / Z) + cy

    u = (torch.arange(W4, device=device, dtype=dtype) + 0.5) * float(downsample)
    v = (torch.arange(H4, device=device, dtype=dtype) + 0.5) * float(downsample)
    u = u.view(1, 1, 1, W4).expand(B, 1, H4, W4)
    v = v.view(1, 1, H4, 1).expand(B, 1, H4, W4)

    dx = u_c - u
    dy = v_c - v
    z = t_map[:, 2:3]
    if use_logz:
        z = torch.log(Z)
    return torch.cat([dx, dy, z], dim=1)


# ----- 4) geodesic θ (f32) -----
@torch.jit.script
def _geodesic_angle_so3(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    # R_pred, R_gt: (...,3,3)
    # θ = atan2(s, t),  t = (tr(R_rel) - 1)/2,  s = ||vee(R_rel - R_rel^T)||/2
    R_rel = (R_pred.transpose(-2, -1) @ R_gt).to(torch.float32)

    # t ≈ cos θ（丸め誤差に強くするため軽くクランプ）
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    t = 0.5 * (tr - 1.0)
    t = torch.clamp(t, -1.0, 1.0)

    # s ≈ sin θ の非負スカラー（負値にならないように保護）
    rx = R_rel[..., 2, 1] - R_rel[..., 1, 2]
    ry = R_rel[..., 0, 2] - R_rel[..., 2, 0]
    rz = R_rel[..., 1, 0] - R_rel[..., 0, 1]
    s2 = (rx * rx + ry * ry + rz * rz) * 0.25
    s = torch.sqrt(torch.clamp(s2, min=0.0) + eps)

    theta = torch.atan2(s, t)  # ∈ [0, π]
    return theta


@torch.jit.script
def rotation_loss_hetero(R_pred: torch.Tensor,  # (B,K,3,3)
                         R_gt: torch.Tensor,    # (B,K,3,3)
                         lv_raw: torch.Tensor,  # (B,K,1)  predicted log-variance for angle
                         valid: torch.Tensor     # (B,K)
                         ) -> torch.Tensor:
    """
    ヘテロスケ回転ロス（測地角 θ に対するガウスNLL）
      L = 0.5 * (θ^2 * exp(-logσ^2) + logσ^2)

    引数:
      R_pred: (B,K,3,3) 予測回転
      R_gt  : (B,K,3,3) GT回転
      lv_raw: (B,K,1)   予測ログ分散 log σ^2（未加工）

    返値:
      スカラー損失（バッチ・K平均）
    """
    # 1) 測地角（float32 安定計算）
    theta = _geodesic_angle_so3(R_pred, R_gt)          # (B,K) float32
    theta = theta                           # (B,K)

    # 2) 予測ログ分散の安定化
    #    極端な過小/過大分散は学習を壊すので範囲制限（例: [-10, 4]）
    lv = lv_raw.squeeze(-1).to(torch.float32)          # (B,K)
    lv = torch.clamp(lv, min=-10.0, max=4.0)

    # 3) ヘテロスケ NLL
    inv_var = torch.exp(-lv)                           # exp(-logσ^2) = 1/σ^2
    loss = 0.5 * (theta * inv_var + lv)              # (B,K)

    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


@torch.jit.script
def rotation_loss_hetero_map(
    R_pred_map: torch.Tensor,            # (B,3,3,H,W)
    R_gt_map: torch.Tensor,              # (B,3,3,H,W)
    lv_raw_map: typing.Optional[torch.Tensor],  # (B,1,H,W) or None, predicted log-variance: log σ^2
    valid_map: torch.Tensor,             # (B,1,H,W)  0/1 or weight
    use_atan2: bool = True               # True: 安定版の角度推定, False: acos 版
) -> torch.Tensor:
    """
    Map版ヘテロスケ回転ロス（各ピクセルでの回転行列）

    lv_raw_map is not None:
        Gaussian NLL を仮定:
          L = 0.5 * (θ^2 * exp(-logσ^2) + logσ^2)

    lv_raw_map is None:
        σ^2 = 1 を仮定した等方ガウス:
          L = 0.5 * θ^2
    """
    # 1) 型整備 & 形状 (B,3,3,H,W) -> (B,H,W,3,3)
    Rp = R_pred_map.to(torch.float32).permute(0, 3, 4, 1, 2)  # (B,H,W,3,3)
    Rg = R_gt_map  .to(torch.float32).permute(0, 3, 4, 1, 2)  # (B,H,W,3,3)

    # 2) 相対回転
    R_rel = Rp.transpose(-2, -1).matmul(Rg)                   # (B,H,W,3,3)

    # 3) 角度 θ の計算
    if use_atan2:
        tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]       # (B,H,W)
        t  = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        vx = R_rel[..., 2, 1] - R_rel[..., 1, 2]
        vy = R_rel[..., 0, 2] - R_rel[..., 2, 0]
        vz = R_rel[..., 1, 0] - R_rel[..., 0, 1]
        s  = 0.5 * torch.sqrt(vx*vx + vy*vy + vz*vz + 1e-12)
        theta = torch.atan2(s, t)                                           # (B,H,W)
    else:
        tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_val = (tr - 1.0) * 0.5
        cos_clamped = torch.clamp(cos_val, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_clamped)                                     # (B,H,W)
    theta = theta                                              # (B,H,W)

    # 4) valid マップ（重み）整形
    valid = valid_map.float().squeeze(1)                                    # (B,H,W)
    denom = valid.sum().clamp_min(1.0)

    # 5) lv が無い場合: 0.5 * θ^2 を weighted average
    if lv_raw_map is None:
        loss_map = 0.5 * theta                                             # (B,H,W)
        weighted = loss_map * valid
        return weighted.sum() / denom

    # ----- ここからヘテロスケ版 -----

    # lv 整形・クリップ
    lv = lv_raw_map.squeeze(1).to(torch.float32)                            # (B,H,W)
    lv = torch.clamp(lv, min=-10.0, max=4.0)  # σ^2 ∈ [e^-10, e^4]

    # 6) ヘテロスケ NLL（Gaussian）
    inv_var = torch.exp(-lv)                                                # (B,H,W)
    loss_map = 0.5 * (theta* inv_var + lv)                                # (B,H,W)

    # 7) valid マスク & 平均（weight平均）
    weighted = loss_map * valid
    return weighted.sum() / denom

# ----- 7) Classification -----
def classification_loss(logits: torch.Tensor, target: torch.Tensor,
                        use_focal: bool = False, gamma: float = 2.0,
                        alpha: float = 0.25, ignore_index: int = -1) -> torch.Tensor:
    z = logits.to(torch.float32)
    if not use_focal:
        return F.cross_entropy(z, target.long(), ignore_index=ignore_index)

    B, C, H, W = z.shape
    valid = (target >= 0) & (target < C)
    if valid.sum() == 0:
        return torch.zeros((), dtype=z.dtype, device=z.device)

    t = torch.where(valid, target, torch.zeros_like(target)).long()
    logp  = F.log_softmax(z, dim=1)
    logpt = logp.gather(1, t.unsqueeze(1)).squeeze(1)
    pt    = logpt.exp().clamp(1e-6, 1.0)

    foc = (1.0 - pt).pow(gamma)
    if alpha is not None:
        foc = alpha * foc

    m = valid.float()
    loss = -(foc * logpt * m).sum() / m.sum().clamp_min(1.0)
    return loss


# ----- 8) Dice (with optional mask) -----
def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits.to(torch.float32))
    t = target.to(torch.float32).clamp(0.0, 1.0)

    if mask is not None:
        m = mask.to(torch.float32)
        while m.dim() < p.dim():
            m = m.unsqueeze(1)
        p = p * m
        t = t * m

    inter = (p * t).sum(dim=(1,2,3))
    denom = (p + t).sum(dim=(1,2,3)).clamp_min(eps)
    dice  = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


# ----- 9) weight map 正規化 -----
def make_weight_fg_h4(mask_1_4: torch.Tensor, weight_map_1_4: torch.Tensor,
                      clip_max: float = 5.0, eps: float = 1e-6) -> torch.Tensor:
    w = (weight_map_1_4.to(torch.float32) * mask_1_4.to(torch.float32)).clamp_min(0.0)
    sum_w = w.sum(dim=(1,2,3), keepdim=True).clamp_min(eps)
    sum_m = mask_1_4.to(torch.float32).sum(dim=(1,2,3), keepdim=True).clamp_min(1.0)
    w = (w * (sum_m / sum_w)).clamp(0.0, clip_max)
    return w.detach()


# --- ヘルパ: H/4格子の(y,x)→元解像度の(u,v)[px]
@torch.jit.script
def _peaks_to_uv_full(idxs_yx: torch.Tensor, down: int, W: int, H: int) -> torch.Tensor:
    u = (idxs_yx[..., 1].to(torch.float32) + 0.5) * float(down)
    v = (idxs_yx[..., 0].to(torch.float32) + 0.5) * float(down)
    u = u.clamp(0.0, float(W-1)); v = v.clamp(0.0, float(H-1))
    return torch.stack([u, v], dim=-1)  # (B,K,2)

@torch.jit.script
def _xy_from_uvZ(K: torch.Tensor, uv: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    # K: (B,3,3), uv:(B,K,2), Z:(B,K)
    fx = K[:,0,0].unsqueeze(1); fy = K[:,1,1].unsqueeze(1)
    s  = K[:,0,1].unsqueeze(1)
    cx = K[:,0,2].unsqueeze(1); cy = K[:,1,2].unsqueeze(1)
    u = uv[...,0]; v = uv[...,1]
    Zu = (u - cx) * Z / fx
    Zv = (v - cy) * Z / fy
    X = Zu - (s / fx) * Zv
    Y = Zv
    return torch.stack([X, Y], dim=-1)  # (B,K,2)

@torch.jit.script
def _gather_map_at_peaks(map_: torch.Tensor, idxs_yx: torch.Tensor) -> torch.Tensor:
    # map_: (B,C,H,W), idxs_yx:(B,K,2[y,x]) -> (B,K,C)
    B, C, H, W = map_.shape
    y = idxs_yx[..., 0].clamp(0, H-1).long()
    x = idxs_yx[..., 1].clamp(0, W-1).long()
    b = torch.arange(B, device=map_.device).view(B,1).expand_as(y)
    val = map_[b, :, y, x]           # → (B, K, C)
    return val 


def _build_inst_id_map(
    weight_map_inst: torch.Tensor,
    mask_1_4: typing.Optional[torch.Tensor],
) -> torch.Tensor:
    """Build 0-based instance ID map from weight maps."""
    inst_id = weight_map_inst.squeeze(2).argmax(dim=1)
    if mask_1_4 is not None:
        fg = mask_1_4.squeeze(1) > 0
        inst_id = torch.where(fg, inst_id, torch.zeros_like(inst_id))
    return inst_id


def pick_representatives_mask_only(inst_mask, valid_map, pos_map, rot_map=None, n_erode=0):
    """
    inst_mask: (B,K,H,W)  0/1
    valid_map: (B,1,H,W)  0/1（可視・信頼）
    pos_map:   (B,3,H,W)
    rot_map:   (B,3,3,H,W) or None
    """
    B, K, H, W = inst_mask.shape
    device = inst_mask.device

    # 簡易 inside スコア（n_erode 回の収縮で“内側回数”を積算）
    with torch.no_grad():
        kernel = torch.ones(1,1,3,3, device=device)
        inside = torch.zeros(B, K, H, W, device=device)
        erode = inst_mask.float()
        for _ in range(n_erode):
            erode = (F.conv2d(erode.view(B*K,1,H,W), kernel, padding=1) == 9).view(B,K,H,W)
            inside += erode.float()

    mask_ok = inst_mask.bool() & (valid_map>0).bool()             # (B,K,H,W)
    very_neg = torch.finfo(inside.dtype).min/4
    score = torch.where(mask_ok, inside, very_neg)                 # 内側ほど高スコア

    # 各インスタンスで argmax → 代表画素
    score_flat = score.view(B, K, -1)
    idx_flat = torch.argmax(score_flat, dim=-1)                    # (B,K)
    y = (idx_flat // W).clamp(0, H-1)
    x = (idx_flat %  W).clamp(0, W-1)
    b = torch.arange(B, device=device).view(B,1).expand(B,K)

    pos_at = pos_map[b, :, y, x]                                   # (B,K,3)
    rot_at = None
    if rot_map is not None:
        rot_at = rot_map[b, :, :, y, x]                            # (B,K,3,3)
    idx_yx = torch.stack([y, x], dim=-1)                           # (B,K,2)
    return idx_yx, pos_at, rot_at
     

def _make_T_from_RZuvK(rot_map, mu_z_map, peaks_yx, K_left_1x, H_full, W_full, down=4):
    # rot_map:(B,3,3,H/4,W/4), mu_z_map:(B,1,H/4,W/4)
    B,_,_,Hq,Wq = rot_map.shape
    # R をピークで取得
    R_hw = rot_map.permute(0,3,4,1,2).reshape(B, Hq*Wq, 3, 3)
    idx = (peaks_yx[...,0]*Wq + peaks_yx[...,1]).clamp(0, Hq*Wq-1)  # (B,K)
    b = torch.arange(B, device=rot_map.device).view(B,1).expand_as(idx)
    R = R_hw[b, idx]  # (B,K,3,3)

    # Z を取得 & uv→XY
    Z = _gather_map_at_peaks(mu_z_map, peaks_yx).squeeze(-1)  # (B,K)
    uv = _peaks_to_uv_full(peaks_yx, down, W_full, H_full)    # (B,K,2)
    XY = _xy_from_uvZ(K_left_1x, uv, Z)                       # (B,K,2)
    t  = torch.cat([XY, Z.unsqueeze(-1)], dim=-1)             # (B,K,3)
    return R, t

def _transform_points(R, t, P):  # R:(B,K,3,3) t:(B,K,3) P:(B,K,N,3 or 1,N,3)
    if P.dim() == 4:
        B,K,N,_ = P.shape
        X = (R @ P.transpose(-1,-2)).transpose(-1,-2) + t.unsqueeze(-2)  # (B,K,N,3)
    else:  # P:(B,N,3) を K にブロードキャスト
        B,N,_ = P.shape
        X = (R @ P.unsqueeze(1).transpose(-1,-2)).transpose(-1,-2) + t.unsqueeze(-2)  # (B,K,N,3)
    return X


@torch.jit.script
def _project_R_safe(M: torch.Tensor, thr: float = 1e-4) -> torch.Tensor:
    # M: (B,K,3,3) f32
    B, K = M.shape[0], M.shape[1]
    F = (M**2).sum(dim=(2,3))                # (B,K)
    U,S,Vh = torch.linalg.svd(M, full_matrices=False)
    UV = U @ Vh
    det = torch.det(UV)
    s = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    D = torch.stack((torch.ones_like(s), torch.ones_like(s), s), dim=-1).diag_embed()
    R = U @ D @ Vh
    I = torch.eye(3, device=M.device, dtype=M.dtype).view(1,1,3,3).expand(B,K,3,3)
    return torch.where((F < thr*thr).view(B,K,1,1), I, R)


def _scale_K(K_1x: torch.Tensor, down:int=4) -> torch.Tensor:
    # K: (B,3,3). fx,fy,cx,cy を 1/down
    K14 = K_1x.clone()
    K14[:, 0, 0] /= down
    K14[:, 1, 1] /= down
    K14[:, 0, 2] /= down
    K14[:, 1, 2] /= down
    return K14


# 共有: R/t → ADD(-S)
def adds_core_from_Rt(
    R_pred: torch.Tensor, t_pred: torch.Tensor,   # (B,K,3,3), (B,K,3)
    R_gt: torch.Tensor,   t_gt: torch.Tensor,     # (B,K,3,3), (B,K,3)
    model_points: torch.Tensor,                   # (B,K,N,3) or (B,1,N,3)
    diameters: torch.Tensor,
    use_symmetric: bool = True,
    max_points: int = 4096,
    valid_mask: torch.Tensor = None,              # (B,K) or None
    rot_only: bool = True,
) -> torch.Tensor:
    device = R_pred.device
    P = model_points if model_points.dim()==4 else model_points.unsqueeze(1)
    N = P.size(2)
    if N > max_points:
        idx = torch.arange(N, device=device)[:max_points]
        P = P[:, :, idx, :]
    
    if rot_only:
        t_pred = t_gt

    def _xfm(R, t, Q):
        return (R @ Q.transpose(-1,-2)).transpose(-1,-2) + t.unsqueeze(2)


    Xp = _xfm(R_pred.to(torch.float32), t_pred.to(torch.float32), P.to(torch.float32))
    Xg = _xfm(R_gt.to(torch.float32),   t_gt.to(torch.float32),   P.to(torch.float32))

    if use_symmetric:
        D  = torch.cdist(Xp, Xg, p=2)
        d1 = D.min(dim=3).values.mean(dim=2)
        d2 = D.min(dim=2).values.mean(dim=2)
        d  = torch.minimum(d1, d2)                # (B,K)
    else:
        d  = (Xp - Xg).norm(dim=-1).mean(dim=-1)  # (B,K)
    d /= (diameters / 100.0) # normalize
    if valid_mask is not None:
        v = valid_mask.to(d.dtype)
        return (d * d * v).sum() / v.sum().clamp_min(1.0)
    return d.mean()


def loss_pose_sequence(
    R_pred_list: torch.Tensor,   # (B,3,3,H,W)
    t_pred_list: torch.Tensor,   # (B,3,H,W)
    R_gt_map: torch.Tensor,     # (B,3,3,H,W)
    t_gt_map: torch.Tensor,     # (B,3,H,W),
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    wfg: torch.Tensor,
    weight_map_inst: torch.Tensor,
    peak_yx: torch.Tensor,
    pos_logvar_list: typing.Optional[typing.List[torch.Tensor]] = None,
    rot_logvar_theta_list: typing.Optional[typing.List[torch.Tensor]] = None,
    w_rot: float = 1.0, 
    w_pos: float = 1.0,
    w_adds_per_t: float = 1.0,
    model_points: typing.Optional[torch.Tensor] = None,    # (B,K,N,3) or (B,1,N,3)
    diameters: typing.Optional[torch.Tensor] = None,
    gamma: float = 0.8, use_charbonnier: bool = True,
    valid_mask: typing.Optional[torch.Tensor] = None,      # (B,K)
    adds_use_symmetric: bool = True,
    adds_max_points: int = 4096,
    eps_small: float = 1e-6,
    rot_only: bool = True,
    symmetry_axes: typing.Optional[torch.Tensor] = None,
    symmetry_orders: typing.Optional[torch.Tensor] = None,
    inst_id_map: typing.Optional[torch.Tensor] = None,
) -> typing.Dict[str, torch.Tensor]:
    total = R_gt_map.new_tensor(0.0)
    sum_w = R_gt_map.new_tensor(0.0)
    logs: typing.Dict[str, torch.Tensor] = {}
    if symmetry_axes is not None and symmetry_orders is not None and inst_id_map is None:
        inst_id_map = _build_inst_id_map(weight_map_inst, t_gt_map[:, -1:])
    for t, (R_pred_map, t_pred_map) in enumerate(zip(R_pred_list, t_pred_list)):
        wt = (gamma ** t)

        # ----- rotation loss (pixel-weighted) -----
        rot_lv = None
        if rot_logvar_theta_list is not None and t < len(rot_logvar_theta_list):
            rot_lv = rot_logvar_theta_list[t]
        R_map_use = R_pred_map
        R_t, t_t, _, _, _ = rot_utils.pose_from_maps_auto(
            rot_map=R_pred_map,
            pos_map=t_pred_map,
            Wk_1_4=weight_map_inst,
            wfg=wfg,
            peaks_yx=peak_yx,
        )
        R_t_use = R_t
        if symmetry_axes is not None and symmetry_orders is not None and inst_id_map is not None:
            R_t_use, delta = rot_utils.canonicalize_pose_gspose_torch(
                R_t,
                t_t,
                symmetry_axes,
                symmetry_orders,
            )
            R_map_use = rot_utils.apply_symmetry_rotation_map(
                R_map_use,
                symmetry_axes,
                delta,
                inst_id_map,
                fg_mask=t_gt_map[:, -1:],
            )
        L_rot_t = rotation_loss_hetero_map(R_map_use, R_gt_map, rot_lv, t_gt_map[:, -1:] > 0)

        # ----- translation loss (pixel-weighted) -----
        
        pos_scale = t_pred_map.new_tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
        pos_lv = None
        if pos_logvar_list is not None and t < len(pos_logvar_list):
            log_scale = - 2.0 * torch.log(pos_scale)
            pos_lv = pos_logvar_list[t] -log_scale
        L_pos_t = pos_loss_hetero_map(t_pred_map / pos_scale, pos_lv, t_gt_map / pos_scale, t_gt_map[:, -1:] > 0)
        
        # ----- ADD/ADD-S（任意; 安全化付き）-----
        with torch.no_grad():
            L_adds_t = adds_core_from_Rt(
                    R_pred=R_t_use, t_pred=t_t,
                    R_gt=R_gt,     t_gt=t_gt,
                    model_points=model_points,
                    diameters=diameters,
                    use_symmetric=adds_use_symmetric,
                    max_points=adds_max_points,
                    valid_mask=valid_mask,      # (B,K)
                    rot_only=rot_only,
                )
            
            L_add_t = adds_core_from_Rt(
                    R_pred=R_t_use, t_pred=t_t,
                    R_gt=R_gt,     t_gt=t_gt,
                    model_points=model_points,
                    diameters=diameters,
                    use_symmetric=False,
                    max_points=adds_max_points,
                    valid_mask=valid_mask,      # (B,K)
                    rot_only=rot_only,
                )

        Lt = w_rot * L_rot_t + w_pos * L_pos_t #+ w_adds_per_t * L_adds_t + w_adds_per_t * L_add_t * 0.2
        total = total + wt * Lt
        sum_w = sum_w + wt

        logs[f"L_rot_seq_t{t}"]  = L_rot_t.detach()
        logs[f"L_pos_seq_t{t}"]  = L_pos_t.detach()
        if L_adds_t.requires_grad or float(L_adds_t) != 0.0:
            logs[f"L_adds_seq_t{t}"] = L_adds_t.detach()
        if L_add_t.requires_grad or float(L_add_t) != 0.0:
            logs[f"L_add_seq_t{t}"] = L_add_t.detach()

    total = total / sum_w.clamp_min(1e-6)
    logs["L_seqs_total"] = total.detach()
    if len(R_pred_list) > 0:
        logs["L_rot_seqs"] = torch.stack([logs[k] for k in logs if k.startswith("L_rot_seq_t")]).mean()
        logs["L_pos_seqs"] = torch.stack([logs[k] for k in logs if k.startswith("L_pos_seq_t")]).mean()

    return {"loss": total, "logs": logs}


def gather_gt_at_peaks(
    gt_pos_14_m: torch.Tensor,     # (B,3,H,W)  ← gt_iter0['pos_1_4']（m）
    gt_R_14: torch.Tensor,         # (B,3,3,H,W)← gt_iter0['R_1_4']
    yx: torch.Tensor,              # (B,K,2) int64  ← pred['instances']['yx']
) -> typing.Tuple[torch.Tensor, torch.Tensor]:                  # → (B,K,4,4)
    B, _, H, W = gt_pos_14_m.shape
    K = yx.shape[1]
    device = gt_pos_14_m.device

    b_ix = torch.arange(B, device=device).view(B,1).expand(B,K)
    ys = yx[..., 0].clamp(0, H-1)
    xs = yx[..., 1].clamp(0, W-1)

    # t をピーク座標で取得
    tx = gt_pos_14_m[b_ix, 0, ys, xs]
    ty = gt_pos_14_m[b_ix, 1, ys, xs]
    tz = gt_pos_14_m[b_ix, 2, ys, xs]
    t  = torch.stack([tx, ty, tz], dim=-1)  # (B,K,3)

    # R をピーク座標で取得
    R00 = gt_R_14[b_ix, 0, 0, ys, xs]; R01 = gt_R_14[b_ix, 0, 1, ys, xs]; R02 = gt_R_14[b_ix, 0, 2, ys, xs]
    R10 = gt_R_14[b_ix, 1, 0, ys, xs]; R11 = gt_R_14[b_ix, 1, 1, ys, xs]; R12 = gt_R_14[b_ix, 1, 2, ys, xs]
    R20 = gt_R_14[b_ix, 2, 0, ys, xs]; R21 = gt_R_14[b_ix, 2, 1, ys, xs]; R22 = gt_R_14[b_ix, 2, 2, ys, xs]
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)  # (B,K,3,3)
    return R, t


def loss_step_iter0(out, gt, 
                    w_ctr=1.0, w_mask=1.0, w_pos=1.0, w_rot=1.0, w_cls=0.5,
                    w_adds: float = 1.0, w_rot_update=1.0, w_pos_update=1.0,
                    w_adds_update: float = 1.0,
                    update_gamma=0.8,
                    use_adds_symmetric: bool = True,    # ← True: ADD-S
                    adds_max_points: int = 4096,        # ← 安全上限
                    # w_mask_bootstrap=0.2
                    ):
    device = out["mask_logits"].device
    # 1) CenterNet focal
    L_ctr = focal_loss_logits(out["center_logits"], gt["weight_map"])

    # 2) Mask（BCE+Dice）＋ 入力マスク bootstrap
    dice_loss = dice_loss_with_logits(out["mask_logits"], gt["mask_1_4"])
    L_mask = dice_loss

    K_left_14 = _scale_K(gt["K_left_1x"])
    assert out["pos_mu"].shape[1] == 3
    assert out["pos_logvar"].shape[1] == 3
    pos_map_pred = pos_mu_to_pointmap(out["pos_mu"], gt["K_left_1x"], downsample=4)
    wfg = torch.ones_like(gt["mask_1_4"])
    R_pred, t_pred, valid, pos_log_var, rot_log_theta = rot_utils.pose_from_maps_auto(
        rot_map=out["rot_mat"], pos_map=pos_map_pred,
        Wk_1_4=gt["weight_map_inst"], wfg=wfg,
        peaks_yx=out["instances"]["gt_yx"],
        pos_logvar=out["pos_logvar"],
        rot_logvar_theta=out["rot_logvar_theta"]
    ) # (B,K,3,3), (B,K,3), (B,K)

    sym_axes = gt.get("symmetry_axes", None)
    sym_orders = gt.get("symmetry_orders", None)
    rot_map_use = out["rot_mat"]
    R_pred_use = R_pred
    inst_id_map = None
    if sym_axes is not None and sym_orders is not None:
        inst_id_map = _build_inst_id_map(gt["weight_map_inst"], gt.get("mask_1_4", None))
        R_pred_use, delta = rot_utils.canonicalize_pose_gspose_torch(
            R_pred,
            t_pred,
            sym_axes,
            sym_orders,
        )
        rot_map_use = rot_utils.apply_symmetry_rotation_map(
            rot_map_use,
            sym_axes,
            delta,
            inst_id_map,
            fg_mask=gt.get("mask_1_4", None),
        )

    idx_yx, t_gt, R_gt = pick_representatives_mask_only(
        inst_mask=gt["pos_1_4"][:, -1:] > 0,
        valid_map=gt["pos_1_4"][:, -1:] > 0,
        pos_map=gt["pos_1_4"],
        rot_map=gt["R_1_4"],
        n_erode=1,
    ) # (B,K,3,3), (B,K,3), (B,K)

    # 3) Position
    L_pos = pos_loss_hetero(
        t_pred, 
        pos_log_var, 
        t_gt, 
        valid
        ) # 1mm, 1mm, 1mm
    # position map
    gt_pos_mu_map = _pos_mu_gt_from_t_map(gt["pos_1_4"], gt["K_left_1x"], downsample=4)
    valid_mask = gt["pos_1_4"][:, -1:] > 0
    pos_scale = out["pos_mu"].new_tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
    log_scale = torch.log(pos_scale)
    L_pos_map = pos_loss_hetero_map(
        out["pos_mu"] / pos_scale,
        out["pos_logvar"] - 2.0 * log_scale,
        gt_pos_mu_map / pos_scale,
        valid_mask,
    ) # 1mm, 1mm, 1mm

    # 4) Rotation
    L_rot = rotation_loss_hetero(
        R_pred_use, R_gt, rot_log_theta, valid
    )
    L_rot_map = rotation_loss_hetero_map(rot_map_use, gt["R_1_4"], out["rot_logvar_theta"], gt["pos_1_4"][:, -1:] > 0)
    
        # ★ 追加: iter0 ADD/ADD-S
    L_adds = torch.tensor(0.0, device=device)
    can_adds = all(k in gt for k in ["peaks_yx", "K_left_1x", "model_points"])
    if can_adds:
        with torch.no_grad():
            L_adds = adds_core_from_Rt(
                R_pred = R_pred_use,
                t_pred= t_pred,          # ここは「Zのみ」のデザインでもOK（μ_zを使う）
                R_gt     = R_gt,
                t_gt   = t_gt,          # (X,Y,Z)。Zしか無いなら X,Y を GT から計算して入れる
                model_points = gt["model_points"],     # (B,K,N,3) or (B,N,3)
                diameters = gt['diameters'],
                use_symmetric= use_adds_symmetric,
                max_points   = adds_max_points,
                valid_mask   = valid,
                rot_only=True,
            )
    can_add = all(k in gt for k in ["peaks_yx", "K_left_1x", "model_points"])
    if can_add:
        with torch.no_grad():
            L_add = adds_core_from_Rt(
                R_pred = R_pred_use,
                t_pred= t_pred,          # ここは「Zのみ」のデザインでもOK（μ_zを使う）
                R_gt     = R_gt,
                t_gt   = t_gt,          # (X,Y,Z)。Zしか無いなら X,Y を GT から計算して入れる
                model_points = gt["model_points"],     # (B,K,N,3) or (B,N,3)
                diameters = gt['diameters'],
                use_symmetric= False,
                max_points   = adds_max_points,
                valid_mask   = valid,
                rot_only=True,
            )


    # 5) Classification（任意）
    if "cls_target" in gt:
        L_cls = classification_loss(out["cls_logits"], gt["cls_target"], use_focal=False)
    else:
        L_cls = torch.tensor(0.0, device=device)
    maps_out = loss_pose_sequence(
        R_pred_list= out["rot_maps"],   # [(B,K,3,3), ...]
        t_pred_list= out["pos_maps"],   # [(B,K,3), ...]
        R_gt_map     = gt["R_1_4"],
        t_gt_map   = gt["pos_1_4"],
        R_gt=R_gt,
        t_gt=t_gt,
        pos_logvar_list=out.get("pos_logvar_maps", None),
        rot_logvar_theta_list=out.get("rot_logvar_theta_maps", None),
        wfg = wfg,
        weight_map_inst=gt["weight_map_inst"],
        peak_yx=out["instances"]["gt_yx"],
        model_points = gt["model_points"],     # (B,K,N,3) or (B,N,3)
        diameters = gt['diameters'],
        valid_mask   = valid,
        w_rot=w_rot_update,
        w_pos=w_pos_update, 
        w_adds_per_t  = w_adds_update,
        gamma=update_gamma,
        adds_use_symmetric = use_adds_symmetric,
        adds_max_points = adds_max_points,
        rot_only=True,
        symmetry_axes=sym_axes,
        symmetry_orders=sym_orders,
        inst_id_map=inst_id_map,
    )
    L_maps   = maps_out["loss"]            # Tensor
    map_logs = maps_out.get("logs", {})    # dict
    # 合成
    L = w_ctr*L_ctr + w_mask*L_mask# + w_pos*L_pos + w_rot*L_rot
    L = L + w_cls*L_cls# + w_adds*L_adds + w_adds * L_add
    L = L + w_pos*L_pos_map + w_rot*L_rot_map + L_maps
    logs = {
        "L": L.detach(),
        "L_ctr": L_ctr.detach(),
        "L_mask": L_mask.detach(),
        "L_pos": L_pos.detach(),
        "L_pos_map": L_pos_map.detach(),
        "L_rot": L_rot.detach(),
        "L_rot_map": L_rot_map.detach(),
        "L_cls": L_cls.detach(),
        "L_adds": L_adds.detach(),        
        "L_add": L_add.detach(),        
    }
    if out["pos_mu"].shape[1] == 3 and valid_mask.any():
        dx = out["pos_mu"][:, 0:1][valid_mask]
        dy = out["pos_mu"][:, 1:2][valid_mask]
        # logs.update(
        #     {
        #         "pos_dx_mean": dx.mean().detach(),
        #         "pos_dx_std": dx.std().detach(),
        #         "pos_dx_min": dx.min().detach(),
        #         "pos_dx_max": dx.max().detach(),
        #         "pos_dy_mean": dy.mean().detach(),
        #         "pos_dy_std": dy.std().detach(),
        #         "pos_dy_min": dy.min().detach(),
        #         "pos_dy_max": dy.max().detach(),
        #         "pos_logvar_min": out["pos_logvar"].min().detach(),
        #         "pos_logvar_max": out["pos_logvar"].max().detach(),
        #     }
        # )
    logs.update({k: (v.detach() if torch.is_tensor(v) else v) for k, v in map_logs.items()})
    return L, logs
