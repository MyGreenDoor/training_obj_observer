
import typing

import torch
import torch.nn.functional as F



@torch.no_grad()
def mean_abs_flow_over_valid(
    flow: torch.Tensor,       # (B,2,H,W) or (B,K,2,H,W)
    invalid_value: float = 400.0,
):
    """
    flow の有効画素だけを対象に，|flow| の平均ベクトルを求める．

    - invalid な画素は flow_x==invalid_value かつ flow_y==invalid_value とみなす．
    - abs は成分ごとにとる：mean(|fx|)，mean(|fy|)
    - 返り値：
        flow: (B,2,H,W)   → (B,2)
        flow: (B,K,2,H,W) → (B,K,2)
    """
    if flow.dim() == 4:
        # (B,2,H,W)
        B, C, H, W = flow.shape
        assert C == 2
        flow_x = flow[:, 0]                  # (B,H,W)
        flow_y = flow[:, 1]                  # (B,H,W)

        invalid = (flow_x == invalid_value) & (flow_y == invalid_value)
        valid = ~invalid                     # (B,H,W)

        valid_count = valid.sum(dim=(1, 2))  # (B,)

        # abs してから，有効画素のみ集計
        fx_abs = flow_x.abs()
        fy_abs = flow_y.abs()

        fx_abs_masked = fx_abs.clone()
        fy_abs_masked = fy_abs.clone()
        fx_abs_masked[~valid] = 0.0
        fy_abs_masked[~valid] = 0.0

        sum_fx = fx_abs_masked.reshape(B, -1).sum(dim=-1)  # (B,)
        sum_fy = fy_abs_masked.reshape(B, -1).sum(dim=-1)  # (B,)

        denom = valid_count.clamp_min(1).to(flow.dtype)    # (B,)
        mean_fx = sum_fx / denom
        mean_fy = sum_fy / denom

        mean_flow = torch.stack([mean_fx, mean_fy], dim=-1)  # (B,2)

        # 有効画素ゼロのバッチは 0 にしておく（必要なら invalid_value に変更してもよい）
        zero_mask = (valid_count == 0)                       # (B,)
        mean_flow[zero_mask] = 0.0

        return mean_flow

    elif flow.dim() == 5:
        # (B,K,2,H,W)
        B, K, C, H, W = flow.shape
        assert C == 2
        flow_x = flow[:, :, 0]              # (B,K,H,W)
        flow_y = flow[:, :, 1]              # (B,K,H,W)

        invalid = (flow_x == invalid_value) & (flow_y == invalid_value)
        valid = ~invalid                    # (B,K,H,W)

        valid_count = valid.sum(dim=(2, 3)) # (B,K)

        fx_abs = flow_x.abs()
        fy_abs = flow_y.abs()

        fx_abs_masked = fx_abs.clone()
        fy_abs_masked = fy_abs.clone()
        fx_abs_masked[~valid] = 0.0
        fy_abs_masked[~valid] = 0.0

        sum_fx = fx_abs_masked.reshape(B, K, -1).sum(dim=-1)  # (B,K)
        sum_fy = fy_abs_masked.reshape(B, K, -1).sum(dim=-1)  # (B,K)

        denom = valid_count.clamp_min(1).to(flow.dtype)       # (B,K)
        mean_fx = sum_fx / denom
        mean_fy = sum_fy / denom

        mean_flow = torch.stack([mean_fx, mean_fy], dim=-1)   # (B,K,2)

        zero_mask = (valid_count == 0)                        # (B,K)
        mean_flow[zero_mask] = 0.0

        return mean_flow

    else:
        raise ValueError("flow must be (B,2,H,W) or (B,K,2,H,W)")
    
    
@torch.no_grad()
def warp_depth_with_forward_flow(
    depth_src: torch.Tensor,          # (B,1,H,W) or (B,H,W)
    flow: torch.Tensor,               # (B,2,H,W) source→target
    invalid_flow_value: float = 400.0,
    invalid_depth_value: float = 0.0,
):
    """
    Forward warp depth with Z-buffer behavior (nearest landing).
    - If multiple source pixels land on the same target pixel, keep the smallest depth.
    - Invalid flow (sentinel) and invalid depth are ignored (do not write).
    """
    if depth_src.ndim == 3:
        depth_src = depth_src.unsqueeze(1)
    if depth_src.ndim != 4 or depth_src.shape[1] != 1:
        raise ValueError(f"depth_src must be (B,1,H,W) or (B,H,W). got {tuple(depth_src.shape)}")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B,2,H,W). got {tuple(flow.shape)}")

    B, _, H, W = depth_src.shape
    device, dtype = depth_src.device, depth_src.dtype

    # meshgrid (fixed src pixel coords)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)
    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)

    fx = flow[:, 0:1]  # (B,1,H,W)
    fy = flow[:, 1:2]

    # invalid flow sentinel
    invalid_flow = (fx == invalid_flow_value) | (fy == invalid_flow_value)

    # compute target coords
    u_dst = xx + fx
    v_dst = yy + fy

    # nearest landing
    u_i = torch.round(u_dst).to(torch.long)  # (B,1,H,W)
    v_i = torch.round(v_dst).to(torch.long)

    in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)

    # invalid depth: you said 0 is never a valid distance
    invalid_depth = (depth_src == invalid_depth_value) | (depth_src <= 0)

    valid = (~invalid_flow) & (~invalid_depth) & in_bounds  # (B,1,H,W)

    # flatten for scatter
    idx = (v_i * W + u_i).view(B, -1)          # (B, H*W)
    z   = depth_src.view(B, -1)               # (B, H*W)

    # z-buffer: initialize with +inf, then amin-reduce
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    out = torch.full((B, H * W), inf, device=device, dtype=dtype)

    valid_flat = valid.view(B, -1)
    # mask-out invalid writers: send them to index 0 with src=+inf (doesn't affect amin)
    idx = idx.clone()
    z = z.clone()
    idx[~valid_flat] = 0
    z[~valid_flat] = inf

    # requires PyTorch scatter_reduce support
    out.scatter_reduce_(dim=1, index=idx, src=z, reduce="amin", include_self=True)

    depth_tgt = out.view(B, 1, H, W)
    depth_tgt = torch.where(torch.isfinite(depth_tgt), depth_tgt,
                            torch.as_tensor(invalid_depth_value, device=device, dtype=dtype))
    return depth_tgt


@torch.no_grad()
def gt_flow_from_pointmap_and_poses(
    point_map: torch.Tensor,          # (B,3,H,W) or (B,K,3,H,W)
    K: torch.Tensor,                  # (B,3,3) or (3,3)
    R_src: torch.Tensor,              # (B,3,3) or (B,K,3,3)   object→camera（source）
    t_src_mm: torch.Tensor,           # (B,3)   or (B,K,3)     [mm]
    R_dst: torch.Tensor,              # (B,3,3) or (B,K,3,3)   object→camera（target）
    t_dst_mm: torch.Tensor,           # (B,3)   or (B,K,3)     [mm]
    point_frame: str = "camera",      # "camera"（X_cam_src）or "object"（X_obj）
    point_unit_m: bool = False,       # point_map が [m] なら True，[mm] なら False
    invalid_value: float = 400.0,
    z_eps: float = 1e-6,
    collapse_k_by_depth: bool = True,
):
    """
    Returns:
      - point_map: (B,3,H,W)        -> flow: (B,1,2,H,W)  (collapse=Trueなら (B,2,H,W))
      - point_map: (B,K,3,H,W)      -> flow: (B,K,2,H,W)  (collapse=Trueなら (B,2,H,W))

    Spec (per your answers):
      - flow = (u_dst - u_src, v_dst - v_src), u_src/v_src は meshgrid 固定
      - invalid: Z_src <= z_eps（≒欠損0もここで弾く）, 追加で Z_dst <= z_eps も invalid
      - invalid flow は invalid_value (=400) を埋める
      - collapse: Z_src の最小（手前）を選ぶ。Z_src invalid は +inf 扱い。
    """
    if point_frame not in ("camera", "object"):
        raise ValueError(f"point_frame must be 'camera' or 'object', got {point_frame}")

    # ---- normalize point_map to (B,K,3,H,W)
    if point_map.ndim == 4:
        point_map = point_map.unsqueeze(1)
    elif point_map.ndim != 5:
        raise ValueError(f"point_map must be 4D or 5D, got shape {tuple(point_map.shape)}")

    B, Kobj, C, H, W = point_map.shape
    if C != 3:
        raise ValueError(f"point_map channel must be 3, got {C}")

    device = point_map.device
    dtype = point_map.dtype

    # ---- mm固定（point_map が m のときは mm に寄せる）
    if point_unit_m:
        point_map = point_map * 1000.0

    # ---- intrinsics K: (3,3) or (B,3,3) -> (B,3,3)
    if K.ndim == 2 and K.shape == (3, 3):
        K_b = K.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    elif K.ndim == 3 and K.shape == (B, 3, 3):
        K_b = K.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"K must be (3,3) or (B,3,3). got {tuple(K.shape)}")

    def _ensure_pose(R: torch.Tensor, t: torch.Tensor, name: str) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        R = R.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)

        if Kobj == 1:
            if R.shape == (B, 3, 3):
                R = R.unsqueeze(1)
            elif R.shape != (B, 1, 3, 3):
                raise ValueError(f"{name} R must be (B,3,3) or (B,1,3,3) when K=1. got {tuple(R.shape)}")
            if t.shape == (B, 3):
                t = t.unsqueeze(1)
            elif t.shape != (B, 1, 3):
                raise ValueError(f"{name} t must be (B,3) or (B,1,3) when K=1. got {tuple(t.shape)}")
        else:
            if R.shape != (B, Kobj, 3, 3):
                raise ValueError(f"{name} R must be (B,K,3,3) when K={Kobj}. got {tuple(R.shape)}")
            if t.shape != (B, Kobj, 3):
                raise ValueError(f"{name} t must be (B,K,3) when K={Kobj}. got {tuple(t.shape)}")

        return R, t

    R_src_bk, t_src_bk = _ensure_pose(R_src, t_src_mm, "src")
    R_dst_bk, t_dst_bk = _ensure_pose(R_dst, t_dst_mm, "dst")

    # ---- meshgrid fixed source pixels (u,v)
    v = torch.arange(H, device=device, dtype=dtype)
    u = torch.arange(W, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H,W)
    uu = uu.reshape(1, 1, H, W)  # reshape を推奨
    vv = vv.reshape(1, 1, H, W)

    def _R_times_X(R_bk33: torch.Tensor, X_bk3hw: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bkij,bkjhw->bkihw", R_bk33, X_bk3hw)

    # ---- prepare X_cam_src and X_obj
    if point_frame == "object":
        X_obj = point_map
        X_cam_src = _R_times_X(R_src_bk, X_obj) + t_src_bk.view(B, Kobj, 3, 1, 1)
    else:
        X_cam_src = point_map
        X_obj = _R_times_X(R_src_bk.transpose(-1, -2), (X_cam_src - t_src_bk.view(B, Kobj, 3, 1, 1)))

    # ---- to destination camera
    X_cam_dst = _R_times_X(R_dst_bk, X_obj) + t_dst_bk.view(B, Kobj, 3, 1, 1)

    # ---- invalid masks (Z_src/Z_dst)
    Z_src = X_cam_src[:, :, 2, :, :]
    Z_dst = X_cam_dst[:, :, 2, :, :]
    invalid = (Z_src <= z_eps) | (Z_dst <= z_eps) | (~torch.isfinite(Z_src)) | (~torch.isfinite(Z_dst))

    # ---- project dst
    p = torch.einsum("bij,bkjhw->bkihw", K_b, X_cam_dst)  # (B,K,3,H,W)
    denom = p[:, :, 2, :, :]

    # （安全）非有限も invalid にする
    invalid = invalid | (~torch.isfinite(denom))

    denom = denom.clamp_min(z_eps)
    u_dst = p[:, :, 0, :, :] / denom
    v_dst = p[:, :, 1, :, :] / denom

    # ★追加：dst が画像外なら invalid
    in_bounds = (u_dst >= 0) & (u_dst <= (W - 1)) & (v_dst >= 0) & (v_dst <= (H - 1))
    invalid = invalid | (~in_bounds) | (~torch.isfinite(u_dst)) | (~torch.isfinite(v_dst))

    # ---- flow
    flow = torch.stack((u_dst - uu, v_dst - vv), dim=2)  # (B,K,2,H,W)
    inv_val = torch.as_tensor(invalid_value, device=device, dtype=dtype)
    flow = flow.masked_fill(invalid.unsqueeze(2), inv_val)

    # ---- optional collapse by nearest Z_src (valid なところだけ Z_src で argmin)
    if collapse_k_by_depth:
        Z_for_argmin = Z_src.clone()
        Z_for_argmin = Z_for_argmin.masked_fill(invalid, float("inf"))  # ★invalid 全部 inf
        idx = torch.argmin(Z_for_argmin, dim=1)  # (B,H,W)

        idx_g = idx[:, None, None, :, :].expand(-1, 1, 2, -1, -1)  # (B,1,2,H,W)
        flow = torch.gather(flow, dim=1, index=idx_g).squeeze(1)    # (B,2,H,W)

        # （任意）全 K が invalid の画素は明示的に invalid_value に
        any_valid = torch.isfinite(Z_for_argmin).any(dim=1)  # (B,H,W) ただし inf 以外があるか
        flow = flow.masked_fill((~any_valid).unsqueeze(1), inv_val)

    return flow


def _hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    h,s,v ∈ [0,1]，shape は同じ（B,1,H,W でも可）．→ (B,3,H,W) RGB∈[0,1]
    """
    i = torch.floor(h * 6).to(torch.int64)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i_mod = i % 6
    r = torch.where(i_mod==0, v, torch.where(i_mod==1, q, torch.where(i_mod==2, p, torch.where(i_mod==3, p, torch.where(i_mod==4, t, v)))))
    g = torch.where(i_mod==0, t, torch.where(i_mod==1, v, torch.where(i_mod==2, v, torch.where(i_mod==3, q, torch.where(i_mod==4, p, p)))))
    b = torch.where(i_mod==0, p, torch.where(i_mod==1, p, torch.where(i_mod==2, t, torch.where(i_mod==3, v, torch.where(i_mod==4, v, q)))))

    return torch.cat([r, g, b], dim=1)

@torch.no_grad()
def visualize_flow_torch(flow: torch.Tensor,
                         invalid_value: float = 400.0,
                         max_mag: float = None) -> torch.Tensor:
    """
    flow: (B,2,H,W)．無効画素は invalid_value で埋められている前提．
    return: (B,3,H,W) ∈ [0,1]
    """
    B, _, H, W = flow.shape
    fx, fy = flow[:, 0:1], flow[:, 1:2]
    # 無効マスク
    invalid = (fx.abs() >= invalid_value*0.99) | (fy.abs() >= invalid_value*0.99) | torch.isnan(fx) | torch.isnan(fy)
    fx = torch.where(invalid, torch.zeros_like(fx), fx)
    fy = torch.where(invalid, torch.zeros_like(fy), fy)

    mag = torch.sqrt(fx*fx + fy*fy)  # (B,1,H,W)
    ang = torch.atan2(fy, fx)        # [-pi, pi]
    hue = (ang / (2*torch.pi) + 0.5).clamp(0, 1)  # [0,1] に正規化
    sat = torch.ones_like(hue)

    if max_mag is None:
        # 有効画素の 95 パーセンタイルでクリップしてコントラストを確保
        flat = mag[~invalid].flatten()
        if flat.numel() == 0:
            vmax = torch.tensor(1.0, device=flow.device, dtype=flow.dtype)
        else:
            vmax = torch.quantile(flat, 0.95)
        vmax = torch.clamp(vmax, min=1e-6)
    else:
        vmax = torch.tensor(max_mag, device=flow.device, dtype=flow.dtype)

    val = (mag / vmax).clamp(0, 1)
    rgb = _hsv_to_rgb_torch(hue, sat, val)

    # 無効画素は黒に
    rgb = torch.where(invalid.expand_as(rgb), torch.zeros_like(rgb), rgb)
    return rgb


@torch.no_grad()
def visualize_flow_pair_same_scale(
    flow_pred: torch.Tensor,          # (B,2,H,W)
    flow_gt:   torch.Tensor,          # (B,2,H,W)
    invalid_value: float = 400.0,
    vmax: float = None,               # 固定レンジにしたい場合に指定．None なら合算95%分位
    quantile: float = 0.95,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred と GT を同一レンジで HSV→RGB 化する．
    戻り値: (rgb_pred, rgb_gt, vmax_used)．rgb は (B,3,H,W)∈[0,1]．
    """
    def _prep(flow: torch.Tensor):
        fx, fy = flow[:, 0:1], flow[:, 1:2]
        invalid = (fx.abs() >= invalid_value*0.99) | (fy.abs() >= invalid_value*0.99) | torch.isnan(fx) | torch.isnan(fy)
        fx = torch.where(invalid, torch.zeros_like(fx), fx)
        fy = torch.where(invalid, torch.zeros_like(fy), fy)
        mag = torch.sqrt(fx*fx + fy*fy)                 # (B,1,H,W)
        ang = torch.atan2(fy, fx)                       # (B,1,H,W), [-pi,pi]
        return invalid, mag, ang

    inv_p, mag_p, ang_p = _prep(flow_pred)
    inv_g, mag_g, ang_g = _prep(flow_gt)

    if vmax is None:
        flat = torch.cat([mag_p[~inv_p], mag_g[~inv_g]], dim=0)
        if flat.numel() == 0:
            vmax_used = torch.tensor(1.0, device=flow_pred.device, dtype=flow_pred.dtype)
        else:
            vmax_used = torch.quantile(flat, quantile).clamp_min(1e-6)
    else:
        vmax_used = torch.tensor(vmax, device=flow_pred.device, dtype=flow_pred.dtype)

    def _hsv_to_rgb(h, s, v):
        # 既存の _hsv_to_rgb_torch を使うなら置き換え可
        return _hsv_to_rgb_torch(h, s, v)

    def _to_rgb(mag, ang, invalid):
        hue = (ang / (2*torch.pi) + 0.5).clamp(0, 1)    # 方向→色相
        sat = torch.ones_like(hue)
        val = (mag / vmax_used).clamp(0, 1)             # 大きさ→明度
        rgb = _hsv_to_rgb(hue, sat, val)
        return torch.where(invalid.expand_as(rgb), torch.zeros_like(rgb), rgb)

    rgb_pred = _to_rgb(mag_p, ang_p, inv_p)
    rgb_gt   = _to_rgb(mag_g, ang_g, inv_g)
    return rgb_pred, rgb_gt, vmax_used