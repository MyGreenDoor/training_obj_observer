from typing import Tuple, Dict, Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_intrinsics_pair_for_feature(
    K_pair_1x: torch.Tensor,
    downsample: int = 4,
) -> torch.Tensor:
    """
    Scale intrinsics to match a 1/downsample feature grid.
    """
    B = K_pair_1x.shape[0]
    s = float(downsample)

    S = torch.zeros(B, 2, 3, 3, device=K_pair_1x.device, dtype=K_pair_1x.dtype)
    S[:, :, 0, 0] = 1.0 / s
    S[:, :, 1, 1] = 1.0 / s
    S[:, :, 2, 2] = 1.0

    return S @ K_pair_1x


def disparity_to_pointmap_from_Kpair(
    disp_1_4: torch.Tensor,
    K_pair_1x: torch.Tensor,
    baseline_mm: Union[float, torch.Tensor],
    downsample: int = 4,
    eps: float = 1e-1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert disparity to a point map in the left camera frame.
    """
    assert disp_1_4.dim() == 4 and disp_1_4.size(1) == 1, "disp shape must be (B,1,H/4,W/4)"
    B, _, Hq, Wq = disp_1_4.shape
    device = disp_1_4.device
    dtype = disp_1_4.dtype

    if isinstance(baseline_mm, (int, float)):
        b = torch.full((B, 1, 1, 1), float(baseline_mm), device=device, dtype=dtype)
    else:
        b = baseline_mm.view(B, 1, 1, 1).to(device=device, dtype=dtype)

    K_pair_14 = scale_intrinsics_pair_for_feature(K_pair_1x, downsample=downsample)

    fx_l = K_pair_14[:, 0, 0, 0].view(B, 1, 1, 1).to(dtype)
    fy_l = K_pair_14[:, 0, 1, 1].view(B, 1, 1, 1).to(dtype)
    cx_l = K_pair_14[:, 0, 0, 2].view(B, 1, 1, 1).to(dtype)
    cy_l = K_pair_14[:, 0, 1, 2].view(B, 1, 1, 1).to(dtype)

    disparity_shift = (K_pair_14[:, 1, 0, 2] - K_pair_14[:, 0, 0, 2]).view(B, 1, 1, 1).to(dtype)
    d_eff = torch.clamp(disp_1_4 + disparity_shift, min=eps)

    Z = fx_l * b / d_eff

    u = torch.arange(Wq, device=device, dtype=dtype).view(1, 1, 1, Wq).expand(B, 1, Hq, Wq) + 0.5
    v = torch.arange(Hq, device=device, dtype=dtype).view(1, 1, Hq, 1).expand(B, 1, Hq, Wq) + 0.5

    X = (u - cx_l) / fx_l * Z
    Y = (v - cy_l) / fy_l * Z

    pts = torch.cat([X, Y, Z], dim=1)
    return pts, K_pair_14


def disparity_to_pointmap_from_Kpair_with_logvar(
    disp_1_4: torch.Tensor,
    disp_log_var_1_4: torch.Tensor,
    K_pair_1x: torch.Tensor,
    baseline_mm: Union[float, torch.Tensor],
    downsample: int = 4,
    eps: float = 1e-1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert disparity + log-variance to a point map and depth log-variance.
    """
    assert disp_1_4.dim() == 4 and disp_1_4.size(1) == 1
    assert disp_log_var_1_4.shape == disp_1_4.shape, "disp_log_var must match disp shape"
    B, _, Hq, Wq = disp_1_4.shape
    device = disp_1_4.device
    dtype = disp_1_4.dtype

    if torch.jit.is_scripting():
        b = baseline_mm.view(B, 1, 1, 1).to(device=device, dtype=dtype)  # type: ignore[attr-defined]
    else:
        if isinstance(baseline_mm, (int, float)):
            b = torch.full((B, 1, 1, 1), float(baseline_mm), device=device, dtype=dtype)
        else:
            b = baseline_mm.view(B, 1, 1, 1).to(device=device, dtype=dtype)

    K_pair_14 = scale_intrinsics_pair_for_feature(K_pair_1x, downsample=downsample)

    fx_l = K_pair_14[:, 0, 0, 0].view(B, 1, 1, 1).to(dtype)
    fy_l = K_pair_14[:, 0, 1, 1].view(B, 1, 1, 1).to(dtype)
    cx_l = K_pair_14[:, 0, 0, 2].view(B, 1, 1, 1).to(dtype)
    cy_l = K_pair_14[:, 0, 1, 2].view(B, 1, 1, 1).to(dtype)

    disparity_shift = (K_pair_14[:, 1, 0, 2] - K_pair_14[:, 0, 0, 2]).view(B, 1, 1, 1).to(dtype)

    d_eff = torch.clamp(disp_1_4 + disparity_shift, min=eps)

    Z = fx_l * b / d_eff

    u = torch.arange(Wq, device=device, dtype=dtype).view(1, 1, 1, Wq).expand(B, 1, Hq, Wq) + 0.5
    v = torch.arange(Hq, device=device, dtype=dtype).view(1, 1, Hq, 1).expand(B, 1, Hq, Wq) + 0.5

    X = (u - cx_l) / fx_l * Z
    Y = (v - cy_l) / fy_l * Z
    pts = torch.cat([X, Y, Z], dim=1)

    absZ = torch.clamp(Z.abs(), min=eps)
    absd = torch.clamp(d_eff.abs(), min=eps)
    depth_log_var = disp_log_var_1_4 + 2.0 * (torch.log(absZ) - torch.log(absd))

    return pts, K_pair_14, depth_log_var


def disparity_to_pointmap_from_Kpair_with_conf(
    disp_1_4: torch.Tensor,
    disp_log_var_1_4: torch.Tensor,
    K_pair_1x: torch.Tensor,
    baseline_mm: Union[float, torch.Tensor],
    downsample: int = 4,
    eps: float = 1e-1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert disparity + log-variance to a point map and confidence.
    """
    pts, K_pair_14, depth_log_var = disparity_to_pointmap_from_Kpair_with_logvar(
        disp_1_4,
        disp_log_var_1_4,
        K_pair_1x,
        baseline_mm,
        downsample=downsample,
        eps=eps,
    )
    point_map_conf = torch.exp(-0.5 * depth_log_var).clamp(0.0, 1.0)
    return pts, K_pair_14, point_map_conf


def make_gn(groups: int = 16) -> Callable[[int], nn.Module]:
    def _gn(c: int) -> nn.Module:
        g = min(groups, c)
        while c % g != 0 and g > 1:
            g //= 2
        return nn.GroupNorm(g, c)

    return _gn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: Callable[[int], nn.Module], stride: int = 1, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=pad, dilation=dilation, bias=True)
        self.bn1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=pad, dilation=dilation, bias=True)
        self.bn2 = norm(out_ch)
        self.down = nn.Identity() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True), norm(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(self.down(x) + y)


def _two_blocks(in_ch: int, out_ch: int, norm: Callable[[int], nn.Module], first_stride: int) -> nn.Sequential:
    return nn.Sequential(
        ResidualBlock(in_ch, out_ch, norm, stride=first_stride),
        ResidualBlock(out_ch, out_ch, norm, stride=1),
    )


class RAFTStyleFeatureEncoder14_128(nn.Module):
    def __init__(self, norm_layer: Callable[[int], nn.Module], strides: Tuple[int, int, int] = (2, 1, 2), l2_normalize: bool = True):
        super().__init__()
        s_stem, s_l1, s_l2 = strides
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=s_stem, padding=3, bias=True), norm_layer(64), nn.ReLU(True))
        self.l1 = _two_blocks(64, 64, norm_layer, first_stride=s_l1)
        self.l2 = _two_blocks(64, 96, norm_layer, first_stride=s_l2)
        self.l3 = _two_blocks(96, 128, norm_layer, first_stride=1)
        self.out = nn.Conv2d(128, 128, 1, bias=True)
        assert s_stem * s_l1 * s_l2 == 4
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.out(x)
        return x


class ASPPLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: Tuple[int, int, int, int] = (1, 6, 12, 18), norm: Callable[[int], nn.Module] = None):
        super().__init__()
        self.br = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=r, dilation=r, groups=in_ch, bias=False),
                (nn.Identity() if norm is None else norm(in_ch)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch // 2, 1, bias=False),
                (nn.Identity() if norm is None else norm(in_ch // 2)),
                nn.ReLU(inplace=True),
            ) for r in rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch // 2 * len(rates), out_ch, 1, bias=False),
            (nn.Identity() if norm is None else norm(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [b(x) for b in self.br]
        return self.fuse(torch.cat(xs, dim=1))


def _make_divisible(v: float, divisor: int = 8, min_ch: int = 16) -> int:
    """Round channels to be compatible with normalization layers."""
    if v <= 0:
        return min_ch
    new_v = max(min_ch, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ContextUNet14(nn.Module):
    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        use_aspp: bool = True,
        l2_normalize: bool = False,
        out_ch: int = 128,
        width_ratios: Tuple[float, float, float] = (0.5, 0.75, 1.0),
        divisor: int = 8,
        min_ch: int = 16,
    ):
        super().__init__()
        gn = norm_layer

        r1, r2, r3 = width_ratios
        c1 = _make_divisible(out_ch * r1, divisor, min_ch)
        c2 = _make_divisible(out_ch * r2, divisor, min_ch)
        c3 = _make_divisible(out_ch * r3, divisor, min_ch)

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3, bias=True), gn(c1), nn.ReLU(True),
            ResidualBlock(c1, c1, gn, stride=1),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, gn, stride=2),
            ResidualBlock(c2, c2, gn, stride=1),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, gn, stride=2),
            ResidualBlock(c3, c3, gn, stride=1, dilation=2),
        )

        self.aspp = ASPPLite(c3, c3, rates=(1, 6, 12, 18), norm=gn) if use_aspp else nn.Identity()

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c2, c3, 3, padding=1, bias=False), gn(c3), nn.ReLU(True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False), gn(c3), nn.ReLU(True),
        )

        self.head = nn.Conv2d(c3, out_ch, 1, bias=True)
        self.l2_normalize = l2_normalize
        self.out_ch = out_ch
        self._widths = (c1, c2, c3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.aspp(e3)

        up = self.up3(b)
        if up.shape[-2:] != e2.shape[-2:]:
            up = F.interpolate(up, size=e2.shape[-2:], mode="bilinear", align_corners=False)

        d3 = self.dec3(torch.cat([up, e2], dim=1))
        y = self.head(d3)

        if self.l2_normalize:
            y = F.normalize(y, p=2, dim=1, eps=1e-12)
        return y


def full_rowwise_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    Compute full row-wise correlation.
    """
    return torch.einsum("bchi,bchj->bjhi", left, right)


def build_corr_pyramid(full_corr: torch.Tensor, num_levels: int = 4) -> List[torch.Tensor]:
    """
    Build a correlation pyramid by pooling along the right image axis.
    """
    B, Wr, H, Wl = full_corr.shape
    pyr = [full_corr]
    cur = full_corr
    for _ in range(1, num_levels):
        cur = F.avg_pool1d(
            cur.permute(0, 2, 3, 1).reshape(B * H * Wl, 1, cur.size(1)),
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )
        cur = cur.view(B, H, Wl, -1).permute(0, 3, 1, 2)
        pyr.append(cur)
    return pyr


def sample_corr_pyramid_bilinear(pyr: List[torch.Tensor], disp_1_4: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """
    Sample 1D correlation neighborhoods from the pyramid with bilinear interpolation.
    """
    B, _, H, Wl = pyr[0].shape
    L = len(pyr)
    K = 2 * radius + 1
    neighs = []
    base = torch.arange(Wl, device=disp_1_4.device, dtype=disp_1_4.dtype).view(1, 1, 1, Wl)
    offs = torch.arange(-radius, radius + 1, device=disp_1_4.device, dtype=disp_1_4.dtype).view(1, K, 1, 1)
    for l, corr in enumerate(pyr):
        Wr_l = corr.size(1)
        idx_c = (base - disp_1_4) / (2 ** l)
        idx = (idx_c + offs).clamp(0, Wr_l - 1)
        idx0 = torch.floor(idx).clamp(0, Wr_l - 1)
        idx1 = (idx0 + 1).clamp(0, Wr_l - 1)
        w1 = (idx - idx0)
        w0 = 1.0 - w1

        corr_exp = corr.unsqueeze(1).expand(B, K, Wr_l, H, Wl)
        v0 = torch.gather(corr_exp, 2, idx0.long().unsqueeze(2)).squeeze(2)
        v1 = torch.gather(corr_exp, 2, idx1.long().unsqueeze(2)).squeeze(2)
        neighs.append(w0 * v0 + w1 * v1)

    return torch.cat(neighs, dim=1)


def _bilinear_gather_2d(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Bilinear gather for 2D sampling on the (right, height) axes.
    """
    B, Cw, H, Wl = t.shape
    x0 = torch.floor(x).clamp(0, Cw - 1)
    x1 = (x0 + 1).clamp(0, Cw - 1)
    y0 = torch.floor(y).clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    wx1 = x - x0
    wx0 = 1.0 - wx1
    wy1 = y - y0
    wy0 = 1.0 - wy1

    t_exp = t.view(B, Cw, 1, 1, H, Wl)

    x0i = x0.long().unsqueeze(1)
    x1i = x1.long().unsqueeze(1)

    v00x = torch.gather(t_exp, 1, x0i).squeeze(1)
    v10x = torch.gather(t_exp, 1, x1i).squeeze(1)
    vx = wx0 * v00x + wx1 * v10x

    y0i = y0.long().unsqueeze(1)
    y1i = y1.long().unsqueeze(1)

    vx_exp = vx.unsqueeze(1)
    v00 = torch.gather(vx_exp, 4, y0i).squeeze(1)
    v11 = torch.gather(vx_exp, 4, y1i).squeeze(1)

    vy = wy0 * v00 + wy1 * v11
    return vy


def sample_corr_pyramid_bilinear_2d(
    pyr: List[torch.Tensor],
    disp_1_4: torch.Tensor,
    radius_w: int = 4,
    radius_h: int = 1,
) -> torch.Tensor:
    B, _, H, Wl = pyr[0].shape
    off_w = torch.arange(-radius_w, radius_w + 1, device=disp_1_4.device, dtype=disp_1_4.dtype)
    off_h = torch.arange(-radius_h, radius_h + 1, device=disp_1_4.device, dtype=disp_1_4.dtype)
    Kw = off_w.numel()
    Kh = off_h.numel()

    base_i = torch.arange(Wl, device=disp_1_4.device, dtype=disp_1_4.dtype).view(1, 1, 1, Wl)
    base_v = torch.arange(H, device=disp_1_4.device, dtype=disp_1_4.dtype).view(1, 1, H, 1)

    outs = []
    for l, corr in enumerate(pyr):
        Wr_l = corr.size(1)
        j_center = (base_i - disp_1_4) / (2 ** l)

        j = j_center.unsqueeze(1).unsqueeze(1) + off_w.view(1, 1, Kw, 1, 1)
        j = j.expand(B, Kh, Kw, H, Wl).clamp(0, Wr_l - 1)
        v = base_v.unsqueeze(1).unsqueeze(2) + off_h.view(1, Kh, 1, 1, 1)
        v = v.expand(B, Kh, Kw, H, Wl).clamp(0, H - 1)

        nb = _bilinear_gather_2d(corr, j, v)
        outs.append(nb.view(B, Kh * Kw, H, Wl))

    return torch.cat(outs, dim=1)


class MotionEncoder1D(nn.Module):
    def __init__(self, k_channels: int, out_ch: int = 128, norm: Callable[[int], nn.Module] = make_gn(16)):
        super().__init__()
        in_ch = k_channels + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False), norm(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), norm(128), nn.ReLU(True),
            nn.Conv2d(128, out_ch, 1, bias=True),
        )

    def forward(self, corr_nb: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([corr_nb, disp], dim=1))


class ConvGRUCell(nn.Module):
    def __init__(self, hidden_ch: int = 128, input_ch: int = 256, norm: Callable[[int], nn.Module] = make_gn(16)):
        super().__init__()
        self.conv_z = nn.Conv2d(hidden_ch + input_ch, hidden_ch, 3, padding=1)
        self.conv_r = nn.Conv2d(hidden_ch + input_ch, hidden_ch, 3, padding=1)
        self.conv_h = nn.Conv2d(hidden_ch + input_ch, hidden_ch, 3, padding=1)
        self.norm = norm(hidden_ch)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv_z(hx))
        r = torch.sigmoid(self.conv_r(hx))
        q = torch.tanh(self.conv_h(torch.cat([r * h, x], dim=1)))
        h_new = (1 - z) * h + z * q
        return self.norm(h_new)


class UpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_ch: int = 128,
        context_ch: int = 128,
        k_per_level: int = 9,
        levels: int = 4,
        norm: Callable[[int], nn.Module] = None,
        min_log_var: float = -8.0,
        max_log_var: float = 4.0,
        init_log_var: float = 0.0,
    ):
        super().__init__()
        if norm is None:
            norm = lambda c: nn.GroupNorm(16, c)

        self.k_total = k_per_level * levels
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

        self.motion_enc = MotionEncoder1D(k_channels=self.k_total, out_ch=hidden_ch)
        self.gru = ConvGRUCell(hidden_ch=hidden_ch, input_ch=hidden_ch + context_ch, norm=norm)

        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden_ch, 1, 3, padding=1),
        )

        self.logvar_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden_ch, 1, 3, padding=1),
        )

        nn.init.zeros_(self.logvar_head[-1].weight)
        nn.init.constant_(self.logvar_head[-1].bias, float(init_log_var))

    def forward(
        self,
        h: torch.Tensor,
        context0: torch.Tensor,
        corr_neigh_all_levels: torch.Tensor,
        disp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mot = self.motion_enc(corr_neigh_all_levels, disp)
        inp = torch.cat([mot, context0], dim=1)
        h = self.gru(h, inp)
        delta = self.delta_head(h)
        log_var = self.logvar_head(h)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        return h, delta, log_var
