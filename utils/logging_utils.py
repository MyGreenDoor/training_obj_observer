
from contextlib import contextmanager
import time
import typing

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

@torch.no_grad()
def visualize_mono_torch(
    pred: torch.Tensor,              # (B,1,H,W)
    pred_mask: torch.Tensor,         # (B,1,H,W) in {0,1} / bool
    gt: torch.Tensor,                # (B,1,H,W)
    gt_mask: torch.Tensor,           # (B,1,H,W) in {0,1} / bool
    min_val: typing.Optional[float] = None,  # 共有レンジの明示的min（なければ有効画素から算出）
    max_val: typing.Optional[float] = None,  # 共有レンジの明示的max
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    pred/gt を同一カラーレンジで JET 風に可視化する。
    - レンジは pred_mask ∪ gt_mask の有効画素のみに基づいて決定（各画像ごと）。
    - 無効画素は黒。
    返り値: (pred_color, gt_color) いずれも (B,3,H,W) uint8
    """
    assert pred.ndim == 4 and pred.size(1) == 1
    assert gt.shape == pred.shape
    assert pred_mask.shape == pred.shape and gt_mask.shape == pred.shape

    dev = pred.device
    B, _, H, W = pred.shape

    x_pred = pred.to(torch.float32)
    x_gt   = gt.to(torch.float32)

    m_pred = (pred_mask > 0).to(torch.float32)
    m_gt   = (gt_mask   > 0).to(torch.float32)
    m_union = torch.clamp(m_pred + m_gt, max=1.0)  # (B,1,H,W)

    # --- 共有レンジの算出（各画像ごと） ---
    if min_val is None or max_val is None:
        # 無効画素を除いた min/max を計算（無効は +inf/-inf に置換）
        x_pred_m = torch.where(m_pred.bool(), x_pred, torch.tensor(float('inf'),  device=dev, dtype=torch.float32))
        x_gt_m   = torch.where(m_gt.bool(),   x_gt,   torch.tensor(float('inf'),  device=dev, dtype=torch.float32))
        per_img_min = torch.minimum(x_pred_m.amin(dim=(-2,-1), keepdim=True),
                                    x_gt_m.amin(dim=(-2,-1), keepdim=True))
        # 有効画素が皆無のときのフォールバック（inf→0）
        per_img_min = torch.where(torch.isinf(per_img_min), torch.zeros_like(per_img_min), per_img_min)

        x_pred_M = torch.where(m_pred.bool(), x_pred, torch.tensor(-float('inf'), device=dev, dtype=torch.float32))
        x_gt_M   = torch.where(m_gt.bool(),   x_gt,   torch.tensor(-float('inf'), device=dev, dtype=torch.float32))
        per_img_max = torch.maximum(x_pred_M.amax(dim=(-2,-1), keepdim=True),
                                    x_gt_M.amax(dim=(-2,-1), keepdim=True))
        per_img_max = torch.where(torch.isinf(per_img_max), torch.ones_like(per_img_max), per_img_max)
    else:
        # 明示レンジが与えられた場合はそれを各画像に適用
        per_img_min = torch.full((B,1,1,1), float(min_val), device=dev, dtype=torch.float32)
        per_img_max = torch.full((B,1,1,1), float(max_val), device=dev, dtype=torch.float32)

    # 正規化
    denom = (per_img_max - per_img_min).clamp(min=1e-12)
    v_pred = ((x_pred - per_img_min) / denom).clamp(0.0, 1.0)
    v_gt   = ((x_gt   - per_img_min) / denom).clamp(0.0, 1.0)

    # --- JET-like colormap（三角波近似）---
    def jet_like(v: torch.Tensor) -> torch.Tensor:  # v in [0,1], (B,1,H,W) -> (B,3,H,W)
        r = (1.5 - (4.0 * v - 3.0).abs()).clamp(0.0, 1.0)
        g = (1.5 - (4.0 * v - 2.0).abs()).clamp(0.0, 1.0)
        b = (1.5 - (4.0 * v - 1.0).abs()).clamp(0.0, 1.0)
        return torch.cat([r, g, b], dim=1)

    color_pred = jet_like(v_pred)
    color_gt   = jet_like(v_gt)

    # 無効画素は黒（各自の mask を使用）
    m_pred3 = m_pred.expand(B, 3, H, W)
    m_gt3   = m_gt.expand(B, 3, H, W)
    color_pred = torch.where(m_pred3.bool(), color_pred, torch.zeros_like(color_pred))
    color_gt   = torch.where(m_gt3.bool(),   color_gt,   torch.zeros_like(color_gt))

    # 0-255 uint8
    color_pred = (color_pred * 255.0).round().to(torch.uint8)
    color_gt   = (color_gt   * 255.0).round().to(torch.uint8)
    return color_pred, color_gt


def colorize_disparity(disp, max_disp=64.0):
    """
    Simple gray -> 3ch visualization normalized to [0,1].
    disp: (B,1,H,W)
    returns: (B,3,H,W)
    """
    disp = disp.clamp(0.0, max_disp) / max(1e-6, float(max_disp))
    return disp.repeat(1, 3, 1, 1)

def save_image_grid(tensor, path, nrow=4, normalize=True, scale_each=True):
    import torchvision.utils as vutils
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=normalize, scale_each=scale_each)
    from PIL import Image
    img = (grid.permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype("uint8")
    Image.fromarray(img).save(path)
    
    
def project_points_no_dist(
    pts_obj: torch.Tensor,      # (B,N,3)
    K: torch.Tensor,            # (B,3,3)
    T_cam_obj: torch.Tensor     # (B,4,4)  物体→カメラ（右手+Z前方）
) -> torch.Tensor:
    B, N, _ = pts_obj.shape
    R = T_cam_obj[:, :3, :3]                           # (B,3,3)
    t = T_cam_obj[:, :3, 3].unsqueeze(1)               # (B,1,3)
    X_cam = pts_obj @ R.transpose(1, 2) + t            # (B,N,3)
    X, Y, Z = X_cam.unbind(-1)
    Z = Z.clamp(min=1e-12)
    x = X / Z
    y = Y / Z
    fx = K[:, 0, 0].unsqueeze(1); fy = K[:, 1, 1].unsqueeze(1)
    s  = K[:, 0, 1].unsqueeze(1); cx = K[:, 0, 2].unsqueeze(1); cy = K[:, 1, 2].unsqueeze(1)
    u = fx * x + s * y + cx
    v = fy * y + cy
    return torch.stack([u, v], dim=-1)                 # (B,N,2)

def make_axes_points_batch(axis_len: float, B: int, device=None, dtype=None) -> torch.Tensor:
    """各バッチで同一の [origin, Xend, Yend, Zend] を返す (B,4,3)"""
    base = torch.tensor([[0.0,       0.0,       0.0],
                         [axis_len,  0.0,       0.0],
                         [0.0,       axis_len,  0.0],
                         [0.0,       0.0,       axis_len]], device=device, dtype=dtype)  # (4,3)
    return base.unsqueeze(0).expand(B, -1, -1).contiguous()

# -------------------------
# 2) 描画ユーティリティ (純PyTorch)
# -------------------------
def _draw_disk_(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor, r: int, color: torch.Tensor):
    """
    img: (3,H,W) [0..1]、inplaceで上書き
    x,y: (M,) float座標（画素中心基準）。半径rのディスクを塗る（最近傍）
    color: (3,) [0..1]
    """
    H, W = img.shape[-2], img.shape[-1]
    if r <= 0:
        xi = x.round().long().clamp(0, W-1)
        yi = y.round().long().clamp(0, H-1)
        img[:, yi, xi] = color[:, None]
        return

    rr = torch.arange(-r, r+1, device=img.device)
    dx, dy = torch.meshgrid(rr, rr, indexing='ij')
    mask = (dx*dx + dy*dy) <= r*r
    dyv = dy[mask].reshape(-1)  # (K,)
    dxv = dx[mask].reshape(-1)  # (K,)

    xi = (x.round().long().unsqueeze(1) + dxv).clamp(0, W-1)
    yi = (y.round().long().unsqueeze(1) + dyv).clamp(0, H-1)
    # ブロードキャストしてまとめて塗る
    img[:, yi, xi] = color.view(3,1,1).expand(-1, yi.shape[0], yi.shape[1])

def _draw_line_(img: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, color: torch.Tensor, thickness: int = 2):
    """
    直線をサンプルしてディスク塗りで太さを出す簡易実装（アンチエイリアスなし）
    img: (3,H,W)  color:(3,)
    p0,p1: (2,) 画素座標 (u,v)
    """
    H, W = img.shape[-2], img.shape[-1]
    x0, y0 = p0[0].item(), p0[1].item()
    x1, y1 = p1[0].item(), p1[1].item()
    steps = int(max(abs(x1-x0), abs(y1-y0))) + 1
    if steps <= 1:
        _draw_disk_(img, torch.tensor([x0], device=img.device), torch.tensor([y0], device=img.device), thickness//2, color)
        return
    t = torch.linspace(0, 1, steps, device=img.device)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    _draw_disk_(img, xs, ys, max(0, thickness//2), color)

def _draw_arrowhead_(img: torch.Tensor, base: torch.Tensor, tip: torch.Tensor,
                     color: torch.Tensor, thickness: int = 2, tip_length: float = 0.1):
    """
    簡易矢印ヘッド：先端から逆向きに2本の小線（V字）を引く
    base, tip: (2,)
    tip_length: 全体の線分長に対する割合
    """
    v = tip - base
    L = torch.linalg.norm(v)
    if L.item() < 1e-6:
        return
    v_dir = v / L
    # 垂直ベクトル（2D）
    perp = torch.tensor([ -v_dir[1], v_dir[0] ], device=img.device)
    head_len = L * tip_length
    spread = head_len * 0.5  # 開き具合
    p_left  = tip - v_dir * head_len + perp * spread
    p_right = tip - v_dir * head_len - perp * spread
    _draw_line_(img, tip, p_left,  color, thickness)
    _draw_line_(img, tip, p_right, color, thickness)

# -------------------------
# 3) 画像へ座標軸を描画（バッチ）
# -------------------------
@torch.no_grad()
def draw_axes_on_images(
    imgs: torch.Tensor,         # (B,3,H,W) [0..1] (RGB想定)
    K: torch.Tensor,            # (B,3,3)
    T_cam_obj: torch.Tensor,    # (B,4,4)
    axis_len: float = 0.05,
    thickness: int = 2,
    tip_length: float = 0.12
) -> torch.Tensor:
    """
    X=赤, Y=緑, Z=青 を imgs に直接描画して返す（値域はclamp(0,1)）
    """
    assert imgs.dim() == 4 and imgs.shape[1] == 3, "imgs should be (B,3,H,W)"
    B, _, H, W = imgs.shape
    device = imgs.device
    dtype  = imgs.dtype

    # 4点を物体座標で作り、投影
    pts_obj = make_axes_points_batch(axis_len, B, device=device, dtype=dtype)  # (B,4,3)
    uv = project_points_no_dist(pts_obj, K, T_cam_obj)                         # (B,4,2)

    # 色（RGB, [0..1]）
    col_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    col_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    col_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    out = imgs.clone()
    for b in range(B):
        o = uv[b, 0]; px = uv[b, 1]; py = uv[b, 2]; pz = uv[b, 3]

        # 画像境界外に大きく外れたらスキップ（任意）
        def _inside(p):
            return (-W <= p[0] <= 2*W) and (-H <= p[1] <= 2*H)

        if _inside(o) and _inside(px):
            _draw_line_(out[b], o, px, col_x, thickness)
            _draw_arrowhead_(out[b], o, px, col_x, thickness, tip_length)

        if _inside(o) and _inside(py):
            _draw_line_(out[b], o, py, col_y, thickness)
            _draw_arrowhead_(out[b], o, py, col_y, thickness, tip_length)

        if _inside(o) and _inside(pz):
            _draw_line_(out[b], o, pz, col_z, thickness)
            _draw_arrowhead_(out[b], o, pz, col_z, thickness, tip_length)

    return out.clamp(0.0, 1.0)


@torch.no_grad()
def draw_axes_on_images_bk(
    imgs: torch.Tensor,         # (B,3,H,W) [0..1] RGB
    K: torch.Tensor,            # (B,3,3) or (B,K,3,3)
    T_cam_obj: torch.Tensor,    # (B,K,4,4)
    axis_len: float = 0.05,
    thickness: int = 2,
    tip_length: float = 0.12,
    valid: torch.Tensor = None  # (B,K) bool, optional
) -> torch.Tensor:
    """
    各バッチ画像に、全インスタンス(K個)の座標軸(X=赤, Y=緑, Z=青)を重ね描きして返す。
    返り値: (B,3,H,W) [0..1]
    """
    assert imgs.dim() == 4 and imgs.shape[1] == 3, "imgs must be (B,3,H,W)"
    assert T_cam_obj.dim() == 4 and T_cam_obj.shape[-2:] == (4,4), "T_cam_obj must be (B,K,4,4)"

    B, _, H, W = imgs.shape
    Knum = T_cam_obj.shape[1]
    device, dtype = imgs.device, imgs.dtype

    # valid が無ければ全True
    if valid is None:
        valid = torch.ones((B, Knum), dtype=torch.bool, device=device)

    # K を (B,K,3,3) に揃える
    if K.dim() == 3:
        K_bk = K.unsqueeze(1).expand(B, Knum, 3, 3).contiguous()
    elif K.dim() == 4:
        assert K.shape[:2] == (B, Knum), "K must be (B,3,3) or (B,K,3,3)"
        K_bk = K
    else:
        raise ValueError("K must be (B,3,3) or (B,K,3,3)")

    # 4点（原点＋各軸先端）を BK バッチで投影
    BK = B * Knum
    pts_obj = make_axes_points_batch(axis_len, BK, device=device, dtype=dtype)        # (BK,4,3)
    uv = project_points_no_dist(
        pts_obj,
        K_bk.view(BK, 3, 3),
        T_cam_obj.view(BK, 4, 4)
    ).view(B, Knum, 4, 2)  # (B,K,4,2)

    # 軸の色
    col_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    col_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    col_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    out = imgs.clone()

    # 1枚のキャンバスに K 本の軸を重ね描き
    for b in range(B):
        canvas = out[b]  # (3,H,W)

        def _inside(p):
            x, y = p[0].item(), p[1].item()
            # ちょっと余裕を持って範囲チェック
            return (-W <= x <= 2*W) and (-H <= y <= 2*H)

        for k in range(Knum):
            if not bool(valid[b, k]):
                continue
            o, px, py, pz = uv[b, k, 0], uv[b, k, 1], uv[b, k, 2], uv[b, k, 3]

            if _inside(o) and _inside(px):
                _draw_line_(canvas, o, px, col_x, thickness)
                _draw_arrowhead_(canvas, o, px, col_x, thickness, tip_length)

            if _inside(o) and _inside(py):
                _draw_line_(canvas, o, py, col_y, thickness)
                _draw_arrowhead_(canvas, o, py, col_y, thickness, tip_length)

            if _inside(o) and _inside(pz):
                _draw_line_(canvas, o, pz, col_z, thickness)
                _draw_arrowhead_(canvas, o, pz, col_z, thickness, tip_length)

    return out.clamp(0.0, 1.0)


@torch.no_grad()
def _ensure_size(x, H, W, mode="bilinear"):
    # x: (B,*,h,w) -> (B,*,H,W)
    if x.shape[-2:] != (H, W):
        align = (mode == "bilinear")
        x = F.interpolate(x, size=(H, W), mode=mode, align_corners=False if align else None)
    return x

@torch.no_grad()
def _clip01(x):
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def _blend(img, overlay, alpha):
    # img, overlay: (B,3,H,W), alpha: (B,1,H,W) or scalar
    if isinstance(alpha, torch.Tensor):
        while alpha.dim() < overlay.dim():
            alpha = alpha.unsqueeze(1)  # -> (B,1,H,W)
        out = img * (1 - alpha) + overlay * alpha
    else:
        out = img * (1 - alpha) + overlay * alpha
    return _clip01(out)

@torch.no_grad()
def _colormap_jet_like(v):
    # v in [0,1], shape: (B,1,H,W) -> (B,3,H,W)
    r = (1.5 - (4.0 * v - 3.0).abs()).clamp(0.0, 1.0)
    g = (1.5 - (4.0 * v - 2.0).abs()).clamp(0.0, 1.0)
    b = (1.5 - (4.0 * v - 1.0).abs()).clamp(0.0, 1.0)
    return torch.cat([r, g, b], dim=1)

# ----------------------------------------------------------
# 1) Mask overlay（GT=Green, Pred=Red, Overlap=Yellow）
# ----------------------------------------------------------
@torch.no_grad()
def make_mask_overlay_grid(
    imgs,               # (B,3,H,W) [0..1]
    mask_logits_pred,   # (B,1,hp,wp) 予測ロジット
    mask_gt,            # (B,1,hg,wg)  GT (0/1) ※リサイズしない
    nrow=4,
    alpha=0.45
):
    B, _, H, W = imgs.shape
    dev = imgs.device

    # --- まず GT 解像度で可視化を作る ---
    Hg, Wg = mask_gt.shape[-2], mask_gt.shape[-1]

    # Pred を GT 解像度にだけ合わせる
    pred_p_g = torch.sigmoid(F.interpolate(mask_logits_pred, size=(Hg, Wg), mode="bilinear", align_corners=False)).clamp(0.0, 1.0)  # (B,1,Hg,Wg)
    gt_g     = mask_gt.to(torch.float32).clamp(0.0, 1.0)  # (B,1,Hg,Wg)（非リサイズ）

    # GT解像度で色付け
    red_g   = torch.zeros((B,3,Hg,Wg), device=dev);   red_g[:, 0:1] = pred_p_g
    green_g = torch.zeros((B,3,Hg,Wg), device=dev); green_g[:, 1:2] = gt_g
    both_g  = _clip01(red_g + green_g)
    a_map_g = torch.max(pred_p_g, gt_g)  # 透明度

    # 画像解像へ拡大
    red   = F.interpolate(red_g,   size=(H,W), mode="bilinear", align_corners=False)
    green = F.interpolate(green_g, size=(H,W), mode="bilinear", align_corners=False)
    both  = F.interpolate(both_g,  size=(H,W), mode="bilinear", align_corners=False)
    a_map = F.interpolate(a_map_g, size=(H,W), mode="bilinear", align_corners=False)

    over_pred = _blend(imgs, red,   alpha * F.interpolate(pred_p_g, size=(H,W), mode="bilinear", align_corners=False))
    over_gt   = _blend(imgs, green, alpha * F.interpolate(gt_g,     size=(H,W), mode="nearest"))
    over_both = _blend(imgs, both,  alpha * a_map)

    grid = vutils.make_grid(
        torch.cat([imgs, over_gt, over_pred, over_both], dim=0),
        nrow=nrow, normalize=False
    )
    return grid

# ----------------------------------------------------------
# 2) Center heat overlay（Pred だけ GT 解像度にリサイズ）
#   - Pred を GT の解像度へ
#   - GT/Pred をその解像度で色付け → 最後に画像解像へ拡大してブレンド
# ----------------------------------------------------------
@torch.no_grad()
def make_center_overlay_grid(
    imgs,                # (B,3,H,W) [0..1]
    ctr_logits_pred,     # (B,1,hp,wp) 予測ロジット
    ctr_gt,              # (B,1,hg,wg)  GT ∈[0,1] ※リサイズしない
    nrow=4,
    alpha=0.60,
    vmin=None,
    vmax=None
):
    B, _, H, W = imgs.shape

    Hg, Wg = ctr_gt.shape[-2], ctr_gt.shape[-1]
    pred_p_g = torch.sigmoid(F.interpolate(ctr_logits_pred, size=(Hg, Wg), mode="bilinear", align_corners=False)).clamp(0.0, 1.0)  # (B,1,Hg,Wg)
    gt_g     = ctr_gt.to(torch.float32).clamp(0.0, 1.0)  # 非リサイズ

    # 共有レンジ（必要なら）
    if vmin is not None or vmax is not None:
        vmin = 0.0 if vmin is None else float(vmin)
        vmax = 1.0 if vmax is None else float(vmax)
        denom = max(vmax - vmin, 1e-6)
        pred_v_g = ((pred_p_g - vmin) / denom).clamp(0.0, 1.0)
        gt_v_g   = ((gt_g     - vmin) / denom).clamp(0.0, 1.0)
    else:
        pred_v_g, gt_v_g = pred_p_g, gt_g

    # GT解像度で色
    pred_color_g = _colormap_jet_like(pred_v_g)  # (B,3,Hg,Wg)
    gt_color_g   = _colormap_jet_like(gt_v_g)

    # 画像解像へ拡大
    pred_color = F.interpolate(pred_color_g, size=(H,W), mode="bilinear", align_corners=False)
    gt_color   = F.interpolate(gt_color_g,   size=(H,W), mode="bilinear", align_corners=False)
    pred_a     = F.interpolate(pred_v_g,     size=(H,W), mode="bilinear", align_corners=False)  # 透明度
    gt_a       = F.interpolate(gt_v_g,       size=(H,W), mode="bilinear", align_corners=False)

    over_pred = _blend(imgs, pred_color, alpha * pred_a)
    over_gt   = _blend(imgs, gt_color,   alpha * gt_a)

    grid = vutils.make_grid(
        torch.cat([imgs, over_gt, over_pred], dim=0),
        nrow=nrow, normalize=False
    )
    return grid


@contextmanager
def section_timer(name: str, writer=None, step: int = None, use_cuda: bool = True, print_fn=print):
    """
    name: 区間名（ログ名）
    writer: TensorBoard writer（任意）
    step: そのときの global_step / epoch（任意）
    use_cuda: CUDAイベントで測るか（デフォルト True）
    """
    if use_cuda and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  # milliseconds (float)
    else:
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000.0

    msg = f"[timing] {name}: {ms:.3f} ms"
    if print_fn is not None:
        print_fn(msg)
    if writer is not None and step is not None:
        writer.add_scalar(f"timing/{name}", ms, step)