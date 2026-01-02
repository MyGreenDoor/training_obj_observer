
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

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
def charbonnier1(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)

@torch.jit.script
def depthZ_loss_hetero(mu_z: torch.Tensor,   # (B,1,H/4,W/4) [m]
                       lv_z: torch.Tensor,   # (B,1,H/4,W/4) log σ^2
                       z_gt: torch.Tensor,   # (B,1,H/4,W/4) [m]
                       weight: torch.Tensor) -> torch.Tensor:  # (B,1,H/4,W/4)
    # f32安定
    m  = mu_z.to(torch.float32)
    lv = lv_z.to(torch.float32).clamp(-8.0, 4.0)
    zg = z_gt.to(torch.float32)

    rho = charbonnier1(m - zg)                 # (B,1,H/4,W/4)
    inv = torch.exp(-lv).clamp_min(1e-8)       # (B,1,H/4,W/4)
    loss_map = inv * rho + lv                  # (B,1,H/4,W/4)

    w = weight.to(torch.float32).clamp_min(0.0)
    denom = w.sum().clamp_min(1.0)
    return (loss_map * w).sum() / denom


# ----- 4) geodesic θ (f32) -----
def geodesic_dense_safe(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, _, _, H, W = R_pred.shape
    Rt   = R_pred.transpose(1, 2).contiguous()
    R_rel = torch.einsum('bijhw,bjkhw->bikhw', Rt.to(torch.float32), R_gt.to(torch.float32))
    tr   = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos  = 0.5 * (tr - 1.0)
    cos  = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)  # (B,H,W) f32
    return theta


# ----- 5) Rotation (hetero) : TorchScript OK 版（autocastなし） -----
@torch.jit.script
def rotation_loss_hetero(R_pred: torch.Tensor,  # (B,3,3,H,W)
                         R_gt: torch.Tensor,    # (B,3,3,H,W)
                         lv_raw: torch.Tensor,  # (B,1,H,W)
                         weight: torch.Tensor = None  # (B,1,H,W) or None
                         ) -> torch.Tensor:
    # geodesic は f32 返す前提
    theta = geodesic_dense_safe(R_pred, R_gt)    # (B,H,W)
    lv    = lv_raw.to(torch.float32).clamp(-8.0, 4.0)
    inv   = torch.exp(-lv).squeeze(1).clamp_min(1e-8)

    loss_map = inv * theta + lv.squeeze(1)       # (B,H,W)

    if weight is not None:
        w = weight.to(loss_map.dtype).squeeze(1).clamp_min(0.0)
        denom = w.sum().clamp_min(1.0)
        return (loss_map * w).sum() / denom
    else:
        return loss_map.mean()


# ----- 6) Rotation (hetero + symmetry min) ※非scriptでOK -----
def rotation_loss_hetero_symmin(R_pred: torch.Tensor, R_gt: torch.Tensor,
                                logvar_theta: torch.Tensor, S: torch.Tensor,
                                weight: torch.Tensor = None) -> torch.Tensor:
    R_gtS   = torch.einsum('bijhw,kjl->bkilhw', R_gt, S)
    R_predK = R_pred.unsqueeze(1).expand_as(R_gtS)
    thetaK  = geodesic_dense_safe(R_predK.flatten(0,1), R_gtS.flatten(0,1)) \
                .view(R_pred.size(0), S.size(0), R_pred.size(-2), R_pred.size(-1))
    theta   = thetaK.min(dim=1).values  # (B,H,W)

    lv = torch.clamp(logvar_theta.to(torch.float32), -8.0, 4.0)
    inv = torch.exp(-lv).squeeze(1).clamp_min(1e-8)
    loss_map = inv * theta + lv.squeeze(1)

    if weight is not None:
        w = weight.to(loss_map.dtype).squeeze(1).clamp_min(0.0)
        denom = w.sum().clamp_min(1.0)
        return (loss_map * w).sum() / denom
    else:
        return loss_map.mean()


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


def add_s_loss_iter0(
    rot_map_pred: torch.Tensor,      # (B,3,3,H/4,W/4)
    mu_z_map_pred: torch.Tensor,     # (B,1,H/4,W/4) [m]
    gt_R_map: torch.Tensor,          # (B,3,3,H/4,W/4)
    gt_pos_map: torch.Tensor,        # (B,3,H/4,W/4)  (X,Y,Z)[m]
    peaks_yx: torch.Tensor,          # (B,K,2)  H/4格子の[y,x]
    K_left_1x: torch.Tensor,         # (B,3,3)
    model_points: torch.Tensor,      # (B,K,N,3) or (B,N,3)  [m]
    use_symmetric: bool = True,      # True: ADD-S, False: ADD
    max_points: int = 4096,
    valid_mask: torch.Tensor = None, # (B,K) 1/0
    eps: float = 1e-6,
    rot_only: bool = True
) -> torch.Tensor:
    device = rot_map_pred.device
    B, K = peaks_yx.shape[:2]
    if K == 0:
        return torch.zeros((), device=device, dtype=rot_map_pred.dtype)

    # --- Pred pose at peaks ---
    R_pred, t_pred = _make_T_from_RZuvK(
        rot_map_pred, mu_z_map_pred, peaks_yx, K_left_1x,
        H_full=gt_pos_map.size(-2) * 4, W_full=gt_pos_map.size(-1) * 4, down=4
    )  # R_pred:(B,K,3,3)  t_pred:(B,K,3)
    t_pred = t_pred
    # --- GT pose at peaks ---
    R_gt_hw = gt_R_map.permute(0, 3, 4, 1, 2).reshape(
        B, gt_R_map.size(-2)*gt_R_map.size(-1), 3, 3
    )
    lin = (peaks_yx[..., 0] * gt_R_map.size(-1) + peaks_yx[..., 1]).clamp(0, R_gt_hw.size(1)-1)
    bix = torch.arange(B, device=device).view(B, 1).expand_as(lin)
    R_gt = R_gt_hw[bix, lin]                      # (B,K,3,3)
    t_gt = _gather_map_at_peaks(gt_pos_map, peaks_yx) # (B,K,3)
    if rot_only:
        t_pred = t_gt

    # --- 点群（B,K,N,3）へ揃える ---
    if model_points.dim() == 4:     # (B,K,N,3)
        P = model_points
        N = P.size(2)
    else:                           # (B,N,3) → (B,1,N,3)
        P = model_points.unsqueeze(1)
        N = P.size(2)

    if N > max_points:
        idx_pts = torch.randperm(N, device=device)[:max_points]
        P = P[:, :, idx_pts, :]
        N = P.size(2)

    # --- 変換（f32で計算） ---
    Xp = _transform_points(R_pred.to(torch.float32), t_pred.to(torch.float32), P.to(torch.float32))  # (B,K,N,3)
    Xg = _transform_points(R_gt.to(torch.float32),   t_gt.to(torch.float32),   P.to(torch.float32))  # (B,K,N,3)

    if use_symmetric:
        D = torch.cdist(Xp, Xg, p=2)                        # (B,K,N,N)
        d_pred_to_gt = D.min(dim=3).values.mean(dim=2)      # (B,K)  min over N(gt), mean over N(pred)
        d_gt_to_pred = D.min(dim=2).values.mean(dim=2)      # (B,K)  ← ここを dim=2 に修正！
        d = torch.minimum(d_pred_to_gt, d_gt_to_pred)       # (B,K)
    else:
        d = (Xp - Xg).norm(dim=-1).mean(dim=-1)             # (B,K)

    if valid_mask is not None:
        w = valid_mask.to(d.dtype)
        return (d * w).sum() / w.sum().clamp_min(1.0)
    else:
        return d.mean()


def loss_step_iter0(out, gt,
                    w_ctr=1.0, w_mask=1.0, w_pos=1.0, w_rot=1.0, w_cls=0.5,
                    w_adds: float = 1.0,                # ← 追加
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

    # 前景重み（pos/rot 用）
    weight_fg = make_weight_fg_h4(gt["mask_1_4"], gt["weight_map"])

    # 3) Position（ヘテロスケ）
    L_pos = depthZ_loss_hetero(
        out["pos_mu"], out["pos_logvar"], gt["pos_1_4"][:, -1:], # z_only
        weight=weight_fg
    )

    # 4) Rotation（ヘテロスケ, 対称なら min-over-group）
    if "symR" in gt and gt["symR"] is not None:
        L_rot = rotation_loss_hetero_symmin(
            out["rot_mat"], gt["R_1_4"], out["rot_logvar_theta"], S=gt["symR"], weight=weight_fg
        )
    else:
        L_rot = rotation_loss_hetero(
            out["rot_mat"], gt["R_1_4"], out["rot_logvar_theta"], weight=weight_fg
        )
    
        # ★ 追加: iter0 ADD/ADD-S
    L_adds = torch.tensor(0.0, device=device)
    can_adds = all(k in gt for k in ["peaks_yx", "K_left_1x", "model_points"])
    if can_adds:
        L_adds = add_s_loss_iter0(
            rot_map_pred = out["rot_mat"],
            mu_z_map_pred= out["pos_mu"],          # ここは「Zのみ」のデザインでもOK（μ_zを使う）
            gt_R_map     = gt["R_1_4"],
            gt_pos_map   = gt["pos_1_4"],          # (X,Y,Z)。Zしか無いなら X,Y を GT から計算して入れる
            peaks_yx     = gt["peaks_yx"],         # (B,K,2) on H/4
            K_left_1x    = gt["K_left_1x"],        # (B,3,3)
            model_points = gt["model_points"],     # (B,K,N,3) or (B,N,3)
            use_symmetric= use_adds_symmetric,
            max_points   = adds_max_points,
            valid_mask   = gt.get("valid_inst", None)  # (B,K) 任意
        )

    # 5) Classification（任意）
    if "cls_target" in gt:
        L_cls = classification_loss(out["cls_logits"], gt["cls_target"], use_focal=False)
    else:
        L_cls = torch.tensor(0.0, device=device)

    # 合成
    L = w_ctr*L_ctr + w_mask*L_mask + w_pos*L_pos + w_rot*L_rot + w_cls*L_cls + w_adds*L_adds
    logs = {
        "L": L.detach(),
        "L_ctr": L_ctr.detach(),
        "L_mask": L_mask.detach(),
        "L_pos": L_pos.detach(),
        "L_rot": L_rot.detach(),
        "L_cls": L_cls.detach(),
        "L_adds": L_adds.detach(),
    }
    return L, logs