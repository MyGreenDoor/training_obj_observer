# raftlike_pose_.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ----------------- small utils -----------------
def conv3x3(ci: int, co: int, s: int = 1, g: int = 1, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, groups=g, bias=bias)


@torch.jit.script
def l2norm(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = torch.linalg.norm(f, dim=1, keepdim=True)
    n = torch.clamp(n, min=eps)
    return f / n


class ResBlock(nn.Module):
    def __init__(self, c: int, groups: int = 16):
        super().__init__()
        self.c1 = conv3x3(c, c)
        self.n1 = nn.GroupNorm(groups, c)
        self.c2 = conv3x3(c, c)
        self.n2 = nn.GroupNorm(groups, c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(x + y)


class Down(nn.Module):
    def __init__(self, c: int, groups: int = 16):
        super().__init__()
        self.m = nn.Sequential(
            conv3x3(c, c, s=2),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            ResBlock(c, groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class Up(nn.Module):
    def __init__(self, c: int, groups: int = 16):
        super().__init__()
        self.conv = conv3x3(c, c)
        self.norm = nn.GroupNorm(groups, c)
        self.act = nn.SiLU(inplace=True)
        self.rb = ResBlock(c, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act(self.norm(self.conv(x)))
        x = self.rb(x)
        return x


# ----------------- Feature / Context encoders (UNet-ish, output @32x32) -----------------
class FeatureEncoderUNet32(nn.Module):
    """
    Input:  (B,3,64,64)
    Output: (B,C,32,32) L2-normalized
    """
    def __init__(self, c_in: int = 3, base_ch: int = 128, groups: int = 16):
        super().__init__()
        C = base_ch
        self.stem = nn.Sequential(
            conv3x3(c_in, C),
            nn.GroupNorm(groups, C),
            nn.SiLU(inplace=True),
            ResBlock(C, groups),
        )  # 64
        self.d1 = Down(C, groups)  # 32
        self.d2 = Down(C, groups)  # 16
        self.d3 = Down(C, groups)  # 8

        self.u2 = Up(C, groups)    # 16
        self.u1 = Up(C, groups)    # 32
        self.fuse32 = ResBlock(C, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s64 = self.stem(x)   # 64
        s32 = self.d1(s64)   # 32
        s16 = self.d2(s32)   # 16
        s8  = self.d3(s16)   # 8

        y16 = self.u2(s8) + s16
        y32 = self.u1(y16) + s32
        y32 = self.fuse32(y32)
        return l2norm(y32)


class ContextEncoderUNet32(nn.Module):
    """
    Input:  (B,Cctx,64,64)
    Output: h32,c32: (B,128,32,32)
    """
    def __init__(self, c_in: int, base_ch: int = 128, groups: int = 16, h_ch: int = 128, ctx_ch: int = 128):
        super().__init__()
        C = base_ch
        self.stem = nn.Sequential(
            conv3x3(c_in, C),
            nn.GroupNorm(groups, C),
            nn.SiLU(inplace=True),
            ResBlock(C, groups),
        )  # 64
        self.d1 = Down(C, groups)  # 32
        self.d2 = Down(C, groups)  # 16
        self.d3 = Down(C, groups)  # 8

        self.u2 = Up(C, groups)    # 16
        self.u1 = Up(C, groups)    # 32
        self.fuse32 = ResBlock(C, groups)

        self.proj_h = nn.Sequential(conv3x3(C, h_ch), nn.Tanh())
        self.proj_c = nn.Sequential(conv3x3(C, ctx_ch), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s64 = self.stem(x)
        s32 = self.d1(s64)
        s16 = self.d2(s32)
        s8  = self.d3(s16)

        y16 = self.u2(s8) + s16
        y32 = self.u1(y16) + s32
        y32 = self.fuse32(y32)

        h32 = self.proj_h(y32)
        c32 = self.proj_c(y32)
        return h32, c32


# ----------------- Corr pyramid (fixed 3 levels: 32,16,8) -----------------
@torch.jit.script
def build_allpairs_corr_5d(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """
    f1,f2: (B,C,H,W) -> corr: (B,H,W,H,W)
    """
    B, C, H, W = f1.shape
    a = f1.reshape(B, C, H * W).transpose(1, 2)  # (B,HW,C)
    b = f2.reshape(B, C, H * W)                  # (B,C,HW)
    corr = torch.matmul(a, b).reshape(B, H, W, H, W)
    return corr


@torch.jit.script
def corr_pool_last2(corr: torch.Tensor) -> torch.Tensor:
    """
    corr: (B,H1,W1,H2,W2) -> pooled on (H2,W2) by 2: (B,H1,W1,H2/2,W2/2)
    """
    B, H1, W1, H2, W2 = corr.shape
    vol = corr.reshape(B * H1 * W1, 1, H2, W2)
    vol = F.avg_pool2d(vol, kernel_size=2, stride=2)
    H2n = vol.shape[-2]
    W2n = vol.shape[-1]
    return vol.reshape(B, H1, W1, H2n, W2n)


@torch.jit.script
def build_corr_pyr3(corr0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    corr0: (B,32,32,32,32)
    return: (corr32, corr16, corr8) each (B,32,32,*,*)
    """
    corr1 = corr_pool_last2(corr0)   # -> (B,32,32,16,16)
    corr2 = corr_pool_last2(corr1)   # -> (B,32,32,8,8)
    return corr0, corr1, corr2


@torch.jit.script
def sample_corr_local(corr: torch.Tensor, coords: torch.Tensor, radius: int) -> torch.Tensor:
    """
    corr:   (B,H1,W1,h2,w2)
    coords: (B,2,H1,W1) pixel coords on (h2,w2)
    return: (B,K,H1,W1), K=(2r+1)^2
    """
    B, H1, W1, h2, w2 = corr.shape
    device = corr.device
    dtype = coords.dtype

    rng = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(rng, rng, indexing="ij")
    dx = dx.reshape(1, -1, 1, 1)  # (1,K,1,1)
    dy = dy.reshape(1, -1, 1, 1)
    K = dx.numel()

    x0 = coords[:, 0].unsqueeze(1)  # (B,1,H1,W1)
    y0 = coords[:, 1].unsqueeze(1)

    x = x0 + dx  # (B,K,H1,W1)
    y = y0 + dy

    N = B * H1 * W1
    x = x.permute(0, 2, 3, 1).reshape(N, K)  # (N,K)
    y = y.permute(0, 2, 3, 1).reshape(N, K)

    w2m1 = (w2 - 1) if w2 > 1 else 1
    h2m1 = (h2 - 1) if h2 > 1 else 1
    gx = 2.0 * x / float(w2m1) - 1.0
    gy = 2.0 * y / float(h2m1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (N,K,1,2)

    vol = corr.reshape(N, 1, h2, w2)
    samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (N,1,K,1)
    samp = samp[:, 0, :, 0]  # (N,K)
    samp = samp.reshape(B, H1, W1, K).permute(0, 3, 1, 2).contiguous()
    return samp


@torch.jit.script
def corr_lookup_pyr3(
    corr32: torch.Tensor, corr16: torch.Tensor, corr8: torch.Tensor,
    coords32: torch.Tensor, radius: int
) -> torch.Tensor:
    """
    coords32: (B,2,32,32) on level32 grid.
    For level16/8, coords are scaled by /2, /4.
    return: concat([K32, K16, K8]) -> (B, 3*K, 32,32)
    """
    # level32
    c0 = sample_corr_local(corr32, coords32, radius)
    # level16
    coords16 = coords32 * 0.5
    c1 = sample_corr_local(corr16, coords16, radius)
    # level8
    coords8 = coords32 * 0.25
    c2 = sample_corr_local(corr8, coords8, radius)
    return torch.cat((c0, c1, c2), dim=1)


# ----------------- Motion / GRU / Heads -----------------
class MotionEncoder(nn.Module):
    def __init__(self, corr_ch: int, out_ch: int = 128, groups: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(corr_ch + 2, out_ch),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            ResBlock(out_ch, groups),
        )

    def forward(self, corr_feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((corr_feat, flow), dim=1))


class ConvGRUCell(nn.Module):
    def __init__(self, c_x: int = 128, c_h: int = 128):
        super().__init__()
        self.z = conv3x3(c_x + c_h, c_h)
        self.r = conv3x3(c_x + c_h, c_h)
        self.h = conv3x3(c_x + c_h, c_h)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.z(torch.cat((x, h), dim=1)))
        r = torch.sigmoid(self.r(torch.cat((x, h), dim=1)))
        h_t = torch.tanh(self.h(torch.cat((x, r * h), dim=1)))
        return (1.0 - z) * h + z * h_t


class RAFTUp(nn.Module):
    """
    Learned convex upsampling (scale=2) like RAFT/RAFT-Stereo.
    """
    def __init__(self, hidden_ch: int = 128, up_factor: int = 2, groups: int = 16):
        super().__init__()
        self.s = int(up_factor)
        self.mask_head = nn.Sequential(
            conv3x3(hidden_ch, hidden_ch),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, 9 * (self.s * self.s), kernel_size=1, bias=True),
        )

    def forward(self, flow: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, _, H, W = flow.shape
        mask = self.mask_head(h)  # (B,9*s*s,H,W)
        mask = mask.view(B, 1, 9, self.s, self.s, H, W)
        mask = torch.softmax(mask, dim=2)

        unfold = F.unfold(flow, kernel_size=3, padding=1)  # (B,2*9,H*W)
        unfold = unfold.view(B, 2, 9, H, W).unsqueeze(3).unsqueeze(3)  # (B,2,9,1,1,H,W)

        up = (mask * unfold).sum(dim=2)  # (B,2,s,s,H,W)
        up = up.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, 2, H * self.s, W * self.s)
        return up * float(self.s)


# ----------------- Core: fixed output flows64 (B,T,2,64,64) -----------------
class RAFTCore32(nn.Module):
    """
    - feature @32x32
    - corr pyramid over target dims: 32/16/8 (built once)
    - iterative update at 32x32
    - output flows64 stacked: (B,T,2,64,64)
    """
    def __init__(
        self,
        ctx_in_ch: int,
        base_ch: int = 64,
        groups: int = 16,
        iters: int = 6,
        radius: int = 4,
    ):
        super().__init__()
        assert iters > 0
        self.iters = int(iters)
        self.radius = int(radius)

        self.feat1 = FeatureEncoderUNet32(3, base_ch, groups)
        self.feat2 = FeatureEncoderUNet32(3, base_ch, groups)
        self.ctx_enc = ContextEncoderUNet32(ctx_in_ch, base_ch, groups, h_ch=base_ch, ctx_ch=base_ch)

        K = (2 * self.radius + 1) * (2 * self.radius + 1)
        corr_ch = 3 * K  # 32/16/8
        self.motion = MotionEncoder(corr_ch, base_ch, groups)
        self.ctx_fuse = nn.Conv2d(base_ch + base_ch, base_ch, kernel_size=1, bias=False)
        self.gru = ConvGRUCell(base_ch, base_ch)
        self.flow_head = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

        self.up32to64 = RAFTUp(base_ch, up_factor=2, groups=groups)

    def forward(
        self,
        point_map_cur: torch.Tensor,   # (B,3,64,64)
        point_map_rend: torch.Tensor,  # (B,3,64,64)
        ctx_in: torch.Tensor,          # (B,ctx_in_ch,64,64)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = point_map_cur.size(0)
        device = point_map_cur.device
        dtype = point_map_cur.dtype

        # features @32
        f1_32 = self.feat1(point_map_cur)    # (B,C,32,32)
        f2_32 = self.feat2(point_map_rend)   # (B,C,32,32)

        # all-pairs corr and pyramid (once)
        corr0 = build_allpairs_corr_5d(f1_32, f2_32)     # (B,32,32,32,32)
        corr32, corr16, corr8 = build_corr_pyr3(corr0)   # fixed tuple

        # context -> initial hidden and context @32
        h32, c32 = self.ctx_enc(ctx_in)  # (B,128,32,32) each

        # base grid @32
        yy, xx = torch.meshgrid(
            torch.arange(0, 32, device=device, dtype=dtype),
            torch.arange(0, 32, device=device, dtype=dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,32,32)

        flow32 = torch.zeros((B, 2, 32, 32), device=device, dtype=dtype)
        flows_list = torch.jit.annotate(list[torch.Tensor], [])
        for i in range(self.iters):
            coords32 = base + flow32  # (B,2,32,32)
            corr_feat = corr_lookup_pyr3(corr32, corr16, corr8, coords32, self.radius)  # (B,3K,32,32)

            m = self.motion(corr_feat, flow32)                 # (B,128,32,32)
            x = self.ctx_fuse(torch.cat((m, c32), dim=1))      # (B,128,32,32)
            h32 = self.gru(x, h32)                              # (B,128,32,32)
            flow32 = flow32 + self.flow_head(h32)               # (B,2,32,32)

            flow64_i = self.up32to64(flow32, h32)  # (B,2,64,64)
            flows_list.append(flow64_i)
        flows64 = torch.stack(flows_list, dim=1)  # (B,T,2,64,64)
        return flows64, flow32


# ----------------- Delta regressor (64->4->64 UNet) -----------------
class DeltaRegressorUNet64(nn.Module):
    """
    Input:  concat(point_map_cur(3), flow64(2)) -> (B,5,64,64)
    Output: delta_map (B,6,64,64) with tanh on rot
    """
    def __init__(
        self,
        base_ch: int = 128,
        groups: int = 16,
        out_scale_rot: float = 0.15,
        out_scale_trans: float = 0.005,
        in_ch: int = 5,
    ):
        super().__init__()
        self.out_scale_rot = float(out_scale_rot)
        self.out_scale_trans = float(out_scale_trans)
        C = base_ch

        self.stem = nn.Sequential(
            conv3x3(in_ch, C),
            nn.GroupNorm(groups, C),
            nn.SiLU(inplace=True),
            ResBlock(C, groups),
        )  # 64
        self.d1 = Down(C, groups)  # 32
        self.d2 = Down(C, groups)  # 16
        self.d3 = Down(C, groups)  # 8
        self.d4 = Down(C, groups)  # 4

        self.bottleneck = nn.Sequential(
            ResBlock(C, groups),
            conv3x3(C, C),
            nn.GroupNorm(groups, C),
            nn.SiLU(inplace=True),
            ResBlock(C, groups),
        )

        self.u3 = Up(C, groups)  # 8
        self.f3 = ResBlock(C, groups)
        self.u2 = Up(C, groups)  # 16
        self.f2 = ResBlock(C, groups)
        self.u1 = Up(C, groups)  # 32
        self.f1 = ResBlock(C, groups)
        self.u0 = Up(C, groups)  # 64
        self.f0 = ResBlock(C, groups)

        self.head = nn.Sequential(
            conv3x3(C, C),
            nn.GroupNorm(groups, C),
            nn.SiLU(inplace=True),
            nn.Conv2d(C, 6, kernel_size=1, bias=True),
        )

    def forward(self, point_map_cur: torch.Tensor, flow64: torch.Tensor, pos_map: torch.Tensor, rot_map: torch.Tensor) -> torch.Tensor:
        x = torch.cat((point_map_cur, flow64, pos_map, rot_map), dim=1)  # (B,5,64,64)

        s0 = self.stem(x)  # 64
        s1 = self.d1(s0)   # 32
        s2 = self.d2(s1)   # 16
        s3 = self.d3(s2)   # 8
        s4 = self.d4(s3)   # 4

        b = self.bottleneck(s4)

        y = self.u3(b)
        y = self.f3(y + s3)
        y = self.u2(y)
        y = self.f2(y + s2)
        y = self.u1(y)
        y = self.f1(y + s1)
        y = self.u0(y)
        y = self.f0(y + s0)

        out = self.head(y)  # (B,6,64,64)
        rvec = torch.tanh(out[:, :3]) * self.out_scale_rot
        tvec = out[:, 3:] * self.out_scale_trans
        return torch.cat((rvec, tvec), dim=1)


# ----------------- Wrapper: fixed output (delta_map, flows64) -----------------
class DeltaPoseRegressorRAFTLike(nn.Module):
    """
    forward() inputs are fixed:
      - point_map_cur : (B,3,64,64)
      - point_map_rend: (B,3,64,64)
      - ctx_in        : (B,ctx_ch,64,64)  # already concatenated outside

    outputs are fixed:
      - delta_map: (B,6,64,64)
      - flows64 : (B,T,2,64,64)
    """
    def __init__(
        self,
        ctx_in_ch: int,
        base_ch: int = 64,
        groups: int = 16,
        iters: int = 4,
        radius: int = 4,
        out_scale_rot: float = 0.5,
        out_scale_trans: float = 0.01,
    ):
        super().__init__()
        self.core = RAFTCore32(
            ctx_in_ch=ctx_in_ch,
            base_ch=base_ch,
            groups=groups,
            iters=iters,
            radius=radius,
        )
        self.delta = DeltaRegressorUNet64(
            base_ch=base_ch,
            groups=groups,
            out_scale_rot=out_scale_rot,
            out_scale_trans=out_scale_trans,
            in_ch=11,
        )

    def forward(
        self,
        point_map_cur: torch.Tensor,
        point_map_rend: torch.Tensor,
        pos_map: torch.Tensor,
        rot_map: torch.Tensor,
        ctx_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flows64, _flow32_last = self.core(point_map_cur, point_map_rend, ctx_in)  # (B,T,2,64,64)
        flow64_last = flows64[:, -1]                                              # (B,2,64,64)
        delta_map = self.delta(point_map_cur, flow64_last, pos_map, rot_map)                        # (B,6,64,64)
        return delta_map, flows64


# ----------------- quick script check -----------------
def _script_check() -> None:
    B = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 例: ctx_in = concat([point_map_cur(3), current_pos_map(3), mask(1)]) => ctx_in_ch=7
    ctx_in_ch = 7

    model = DeltaPoseRegressorRAFTLike(
        ctx_in_ch=ctx_in_ch,
        base_ch=128,
        groups=16,
        iters=6,
        radius=4,
        out_scale_rot=0.5,
        out_scale_trans=0.01,
    ).to(device=device, dtype=dtype)

    point_map_cur = torch.randn(B, 3, 64, 64, device=device, dtype=dtype)
    point_map_rend = torch.randn(B, 3, 64, 64, device=device, dtype=dtype)
    ctx_in = torch.randn(B, ctx_in_ch, 64, 64, device=device, dtype=dtype)

    # eager
    delta_map, flows64 = model(point_map_cur, point_map_rend, ctx_in)
    assert delta_map.shape == (B, 6, 64, 64)
    assert flows64.shape == (B, 6, 2, 64, 64)

    # script
    sm = torch.jit.script(model)
    d2, f2 = sm(point_map_cur, point_map_rend, ctx_in)
    assert d2.shape == (B, 6, 64, 64)
    assert f2.shape == (B, 6, 2, 64, 64)


if __name__ == "__main__":
    _script_check()
