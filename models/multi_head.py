"""Heads and utility functions for multi-task instance pose estimation."""

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rot_utils


def conv3x3(in_ch: int, out_ch: int, s: int = 1) -> nn.Conv2d:
    """3x3 convolution with optional stride and padding.

    Args:
        in_ch: Input channel size.
        out_ch: Output channel size.
        s: Stride to apply.

    Returns:
        nn.Conv2d: Convolution layer without bias.
    """
    return nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)


class ConvBlock(nn.Module):
    """Two-layer convolutional block with normalization and SiLU."""

    def __init__(self, in_ch, out_ch, norm_layer):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(in_ch, out_ch), norm_layer(out_ch), nn.SiLU(True),
            conv3x3(out_ch, out_ch), norm_layer(out_ch), nn.SiLU(True),
        )

    def forward(self, x): return self.net(x)


class Bottleneck(nn.Module):
    """Residual bottleneck supporting downsampling."""

    def __init__(self, in_ch, mid_ch, out_ch, norm_layer, s=1):
        super().__init__()
        self.down = (s!=1) or (in_ch!=out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), norm_layer(mid_ch), nn.SiLU(True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=s, padding=1, bias=False), norm_layer(mid_ch), nn.SiLU(True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False), norm_layer(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False), norm_layer(out_ch)
        ) if self.down else nn.Identity()
        self.act = nn.SiLU(True)

    def forward(self, x):
        y = self.conv(x) + self.skip(x)
        return self.act(y)


@torch.jit.script
def peaks_to_uv_1_4(idxs_yx: torch.Tensor,  # (B,K,2) int64, yx on H/4 grid
                     down: int,
                     W: int, H: int) -> torch.Tensor:
    """Convert peak indices on quarter resolution to full-res pixel centers.

    Args:
        idxs_yx: Peak indices on H/4 grid ``(B, K, 2)`` as ``(y, x)``.
        down: Downsample factor between feature map and image.
        W: Full-resolution width.
        H: Full-resolution height.

    Returns:
        torch.Tensor: Pixel coordinates ``(B, K, 2)`` ordered as ``(u, v)``.
    """
    # 画素中心系に合わせて +0.5 を入れて、元解像度に戻す
    # u = x + 0.5, v = y + 0.5
    u = (idxs_yx[..., 1].to(torch.float32) + 0.5) * float(down)
    v = (idxs_yx[..., 0].to(torch.float32) + 0.5) * float(down)
    # clamp（安全）
    u = u.clamp(0.0, float(W-1))
    v = v.clamp(0.0, float(H-1))
    return torch.stack([u, v], dim=-1)  # (B,K,2)


@torch.jit.script
def rotvec_to_rotmat_map(rvec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert Rodrigues rotation vectors to rotation matrices per pixel.

    Args:
        rvec: Rotation vectors shaped ``(B, 3, H, W)``.
        eps: Small clamp to avoid division by zero.

    Returns:
        torch.Tensor: Rotation matrices shaped ``(B, 3, 3, H, W)``.
    """
    # rvec: (B,3,H,W) → R: (B,3,3,H,W)
    B, _, H, W = rvec.shape
    rx, ry, rz = rvec[:, 0:1], rvec[:, 1:2], rvec[:, 2:3]     # (B,1,H,W)

    theta = torch.sqrt(rx*rx + ry*ry + rz*rz).clamp_min(0.0)  # (B,1,H,W)
    inv_theta = 1.0 / torch.clamp(theta, min=eps)

    ux = torch.where(theta > eps, rx * inv_theta, torch.zeros_like(rx))
    uy = torch.where(theta > eps, ry * inv_theta, torch.zeros_like(ry))
    uz = torch.where(theta > eps, rz * inv_theta, torch.zeros_like(rz))

    c = torch.cos(theta)
    s = torch.sin(theta)
    one_c = 1.0 - c

    r00 = c + ux*ux*one_c
    r01 = ux*uy*one_c - uz*s
    r02 = ux*uz*one_c + uy*s

    r10 = uy*ux*one_c + uz*s
    r11 = c + uy*uy*one_c
    r12 = uy*uz*one_c - ux*s

    r20 = uz*ux*one_c - uy*s
    r21 = uz*uy*one_c + ux*s
    r22 = c + uz*uz*one_c

    # まず各行を (B,3,H,W) に cat、そのあと rows を stack → (B,3,3,H,W)
    row0 = torch.cat([r00, r01, r02], dim=1)  # (B,3,H,W)
    row1 = torch.cat([r10, r11, r12], dim=1)  # (B,3,H,W)
    row2 = torch.cat([r20, r21, r22], dim=1)  # (B,3,H,W)

    R = torch.stack([row0, row1, row2], dim=1)  # (B,3,3,H,W)
    return R.to(rvec.dtype)


class ASPPLite(nn.Module):
    """Lightweight ASPP with dilation rates 1, 2, 4, 8 and global context."""

    def __init__(self, ch, norm_layer):
        super().__init__()
        br = []
        for r in (1, 2, 4, 8):
            br.append(nn.Sequential(
                nn.Conv2d(ch, ch//2, 3, padding=r, dilation=r, bias=False),
                norm_layer(ch//2), nn.SiLU(True)
            ))
        self.branches = nn.ModuleList(br)
        # global context branch
        self.gc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//2, 1, bias=False),
            nn.SiLU(True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d( (len(br)+1)*(ch//2), ch, 1, bias=False),
            norm_layer(ch), nn.SiLU(True)
        )
        
    def forward(self, x):
        ys = [b(x) for b in self.branches]
        g  = self.gc(x)
        H, W = x.size(-2), x.size(-1)      # ← 明示
        g  = F.interpolate(g, size=(H, W), mode='nearest')
        y  = torch.cat(ys + [g], dim=1)
        return self.proj(y)



class ConvBlock(nn.Module):
    """Single convolution + normalization + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, norm_layer, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = norm_layer(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MLP(nn.Module):
    """Two-layer feed-forward block used in transformer heads."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding2D(nn.Module):
    """Dynamic 2D sine-cosine positional encoding (ViT compatible)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.dim = dim

    def forward(self, H: int, W: int, device=None, dtype=None):
        """Create positional encodings for a given spatial size.

        Args:
            H: Height of the grid.
            W: Width of the grid.
            device: Optional device for the output.
            dtype: Optional dtype for the output.

        Returns:
            torch.Tensor: Positional encodings shaped ``(1, H * W, dim)``.
        """
        d = self.dim // 2
        d2 = d // 2

        y = torch.linspace(0, 1, steps=H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)

        omega = torch.arange(d2, device=device, dtype=dtype) / d2
        omega = 1.0 / (10000 ** omega)  # (d2,)

        pos_y = yy[..., None] * omega  # (H,W,d2)
        pos_x = xx[..., None] * omega  # (H,W,d2)

        pe = torch.cat([pos_y.sin(), pos_y.cos(), pos_x.sin(), pos_x.cos()], dim=-1)  # (H,W,d)
        # pad (if dim is not exactly 2*d)
        if pe.shape[-1] < self.dim:
            pad = self.dim - pe.shape[-1]
            pe = F.pad(pe, (0, pad), mode="constant", value=0.0)
        # -> (1, H*W, dim)
        pe = pe.view(1, H * W, self.dim)
        return pe


class CrossAttnBlock(nn.Module):
    """Cross-attention block with residual connection and MLP."""

    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim_q)
        self.ln_kv = nn.LayerNorm(dim_kv)
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True, kdim=dim_kv, vdim=dim_kv, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.mlp = MLP(dim_q, mlp_ratio, drop)
        self.ln2 = nn.LayerNorm(dim_q)

    def forward(self, q, kv):
        """Apply cross attention followed by feed-forward network.

        Args:
            q: Query tensor shaped ``(B, Lq, Dq)``.
            kv: Key/value tensor shaped ``(B, Lkv, Dkv)``.

        Returns:
            torch.Tensor: Updated queries.
        """
        # q: (B, Lq, Dq), kv: (B, Lkv, Dkv)
        h = self.ln_q(q)
        kvn = self.ln_kv(kv)
        out, _ = self.attn(h, kvn, kvn, need_weights=False)
        q = q + self.drop(out)
        q = q + self.mlp(self.ln2(q))
        return q


class SelfAttnBlock(nn.Module):
    """Self-attention block with residual MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        """Apply self attention and MLP.

        Args:
            x: Token tensor shaped ``(B, L, D)``.

        Returns:
            torch.Tensor: Updated tokens.
        """
        # x: (B, L, D)
        h = self.ln1(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(out)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerMultiTaskHead(nn.Module):
    """Perceiver-style transformer head for mask/pose/class predictions."""
    def __init__(
        self,
        norm_layer,
        num_classes: int,
        ctx_ch: int = 128,
        hidden_ch: int = 128,
        use_pixelshuffle: bool = True,          # 互換のため残す（未使用）
        rot_repr: str = "r6d",
        out_pos_scale: float = 1.0,
        *,
        d_model: int = 192,
        n_heads: int = 6,
        n_latents: int = 256,
        depth: int = 3,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.rot_repr = rot_repr.lower()
        self.out_pos_scale = out_pos_scale
        self.use_pixelshuffle = use_pixelshuffle

        in_ch = ctx_ch + hidden_ch + 1 + 3  # ctx + hidden + mask + point_map

        # トークン埋め込み
        self.stem = ConvBlock(in_ch, d_model, norm_layer, k=3, s=1, p=1)

        # 位置埋め込み（動的生成）
        self.posenc = PositionalEncoding2D(d_model)

        # 潜在ベクトル（学習可能）
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        # Encoder（cross->self）を depth 回
        enc_blocks = []
        for _ in range(depth):
            enc_blocks.append(nn.ModuleDict({
                "cross": CrossAttnBlock(d_model, d_model, n_heads, mlp_ratio, drop),
                "self":  SelfAttnBlock(d_model, n_heads, mlp_ratio, drop),
            }))
        self.encoder = nn.ModuleList(enc_blocks)

        # デコーダ（tokensへ戻す）
        self.decoder = CrossAttnBlock(d_model, d_model, n_heads, mlp_ratio, drop)

        # トークン特徴を 128ch に集約
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model, 224, 3, padding=1, bias=False),
            norm_layer(224),
            nn.SiLU(inplace=True),
        )

        # --- heads ---
        rot_ch = 6 if self.rot_repr == "r6d" else 3
        self.neck_geo = self.make_neck(224, 160, norm_layer)   # 幾何側: mask/ctr/pos/rot
        self.neck_sem = self.make_neck(224, 128, norm_layer)   # 语義側: cls

        # heads
        self.head_mask = nn.Conv2d(160, 1, 3, padding=1)
        self.head_ctr  = nn.Conv2d(160, 1, 3, padding=1)
        self.head_posz = nn.Conv2d(160, 6, 3, padding=1)
        self.head_rot  = nn.Conv2d(160, rot_ch + 1, 3, padding=1)
        self.head_cls  = nn.Conv2d(128, num_classes, 1)
        self._init_head(self.head_ctr)
        self._init_head(self.head_mask)

    def _init_head(self, m: nn.Module, bias_neg: float = -2.0, std: float = 1e-3) -> None:
        for l in m.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=std)
                if l.bias is not None:
                    nn.init.constant_(l.bias, bias_neg)
    
    
    def make_neck(self, in_ch: int, mid_ch: int, norm_layer: typing.Callable[[int], nn.Module]) -> nn.Module:
        """Create lightweight neck with depthwise and pointwise convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # depthwise
            norm_layer(in_ch), nn.SiLU(True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), norm_layer(mid_ch), nn.SiLU(True),
        )

    def forward(
        self,
        ctx_14: torch.Tensor,       # (B,Cc,H4,W4)
        hidden_14: torch.Tensor,    # (B,Ch,H4,W4)
        mask_14: torch.Tensor,      # (B,1,H4,W4)
        point_map_14: torch.Tensor  # (B,3,H4,W4)
    ) -> typing.Dict[str, torch.Tensor]:
        """Run transformer head to produce geometry and semantic predictions.

        Args:
            ctx_14: Context features ``(B, Cc, H/4, W/4)``.
            hidden_14: Hidden state features ``(B, Ch, H/4, W/4)``.
            mask_14: Foreground mask logits ``(B, 1, H/4, W/4)``.
            point_map_14: Point map ``(B, 3, H/4, W/4)``.

        Returns:
            Dict[str, torch.Tensor]: Predicted maps including mask, center, position, rotation, and class logits.
        """

        B, _, H4, W4 = ctx_14.shape
        x_in = torch.cat((ctx_14, hidden_14, mask_14, point_map_14), dim=1)  # (B,in_ch,H4,W4)
        x_tok = self.stem(x_in)                                              # (B,D,H4,W4)

        # flatten tokens: (B, N, D)
        N = H4 * W4
        tokens = x_tok.flatten(2).transpose(1, 2)                            # (B,N,D)

        # 2D pos enc
        pe = self.posenc(H4, W4, device=tokens.device, dtype=tokens.dtype)   # (1,N,D)
        tok_pos = tokens + pe                                                # (B,N,D)

        # prepare latents
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)                    # (B,L,D)

        # encoder blocks
        for blk in self.encoder:
            lat = blk["cross"](lat, tok_pos)  # lat <- attend to tokens
            lat = blk["self"](lat)            # lat self-attn

        # decode to tokens (queries=tokens+pos, kv=latents)
        tok_dec = self.decoder(tok_pos, lat)                                 # (B,N,D)

        # reshape back to (B,D,H4,W4)
        feat = tok_dec.transpose(1, 2).reshape(B, -1, H4, W4)                # (B,D,H4,W4)
        x = self.fuse(feat)                                                  # (B,128,H4,W4)

        # --- heads ---        
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        mask_logits = self.head_mask(x_geo)
        ctr_logits = self.head_ctr(x_geo)
        out_posz    = self.head_posz(x_geo)
        out_rot    = self.head_rot(x_geo)
        cls_logits  = self.head_cls(x_sem)

        mu_raw = out_posz[:, 0:3]
        lv_raw = out_posz[:, 3:6]
        mu_z = mu_raw[:, 2:3] * 1000.0 * self.out_pos_scale
        mu_pos = torch.cat([mu_raw[:, 0:2], mu_z], dim=1)
        lv_pos = lv_raw.clamp(-8.0, 4.0)
        
        if self.rot_repr == "r6d":
            r6d          = out_rot[:, :6]
            # 期待: r6d_to_rotmat: (B,6,H,W) -> (B,3,3,H,W)
            rot_R        = r6d_to_rotmat(r6d)
            logvar_theta = out_rot[:, 6:7].clamp(-10.0, 5.0)
        else:  # "rotvec"
            rvec         = out_rot[:, :3]             # (B,3,H4,W4)
            # 期待: rotvec_to_rotmat_map: (B,3,H,W) -> (B,3,3,H,W)
            rot_R        = rotvec_to_rotmat_map(rvec)
            logvar_theta = out_rot[:, 3:4].clamp(-10.0, 5.0)

        return {
            "mask_logits":        mask_logits,
            "center_logits":      ctr_logits,
            "pos_mu":             mu_pos,  # [dx, dy, Z(mm)]
            "pos_logvar":         lv_pos,
            "rot_mat":            rot_R,
            "rot_logvar_theta":   logvar_theta,
            "cls_logits":         cls_logits,
        }
        
        
class LiteFPNMultiTaskHead(nn.Module):
    """Lightweight FPN-style multi-task head for mask/pose/class outputs."""

    def __init__(self, 
                 norm_layer, 
                 num_classes: int,
                 ctx_ch: int = 128, 
                 hidden_ch: int = 128,
                 use_pixelshuffle: bool = True,
                 rot_repr: str = "r6d",
                 out_pos_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.use_pixelshuffle = use_pixelshuffle
        self.rot_repr = rot_repr.lower()
        
        in_ch = ctx_ch + hidden_ch + 1 + 3  # ctx + hidden + mask + point_map
        c4, c8 = 160, 192

        self.enc4  = ConvBlock(in_ch, c4, norm_layer)
        self.down4 = Bottleneck(c4, c4 // 2, c8, norm_layer, s=2)  # H/8
        self.enc8  = Bottleneck(c8, c8 // 2, c8, norm_layer, s=1)
        self.bridge = ASPPLite(c8, norm_layer)

        if use_pixelshuffle:
            self.up = nn.Sequential(
                nn.Conv2d(c8, c8 * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2), nn.SiLU(True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv3x3(c8, c8), norm_layer(c8), nn.SiLU(True)
            )
        self.fuse4 = ConvBlock(c8 + c4, 224, norm_layer)
        # 各ヘッド前に軽いneck（DW+PW or 1x1→3x3→1x1）

        rot_ch = 6 if self.rot_repr == "r6d" else 3
        self.neck_geo = self.make_neck(224, 160, norm_layer)   # 幾何
        self.neck_sem = self.make_neck(224, 128, norm_layer)   # 分類

        # heads
        self.head_mask = nn.Conv2d(160, 1, 3, padding=1)
        self.head_ctr  = nn.Conv2d(160, 1, 3, padding=1)
        self.head_posz = nn.Conv2d(160, 6, 3, padding=1)
        self.head_rot  = nn.Conv2d(160, rot_ch + 1, 3, padding=1)
        self.head_cls  = nn.Conv2d(128, num_classes, 1)
        self.out_pos_scale = out_pos_scale

        self._init_head(self.head_ctr)
        self._init_head(self.head_mask)

    
    def make_neck(self, in_ch: int, mid_ch: int, norm_layer: typing.Callable[[int], nn.Module]) -> nn.Module:
        """Create lightweight convolutional neck for head branches."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # depthwise
            norm_layer(in_ch), nn.SiLU(True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), norm_layer(mid_ch), nn.SiLU(True),
        )
            
    def _init_head(self, m: nn.Module, bias_neg: float = -2.0, std: float = 1e-3) -> None:
        for l in m.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=std)
                if l.bias is not None:
                    nn.init.constant_(l.bias, bias_neg)

    def forward(self, 
                ctx_14: torch.Tensor, 
                hidden_14: torch.Tensor, 
                mask_14: torch.Tensor, 
                point_map_14: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        """Run FPN-style head on 1/4 resolution features.

        Args:
            ctx_14: Context features ``(B, Cc, H/4, W/4)``.
            hidden_14: Hidden state features ``(B, Ch, H/4, W/4)``.
            mask_14: Foreground mask logits ``(B, 1, H/4, W/4)``.
            point_map_14: Point map ``(B, 3, H/4, W/4)``.

        Returns:
            Dict[str, torch.Tensor]: Predicted logits and pose outputs.
        """
        # 入力はすべて H/4 解像で、mask:1ch, point_map:3ch と固定
        x4_in = torch.cat((ctx_14, hidden_14, mask_14, point_map_14), dim=1)
        x4 = self.enc4(x4_in)
        x8 = self.enc8(self.down4(x4))
        x8 = self.bridge(x8)
        x4u = self.up(x8)
        x   = self.fuse4(torch.cat((x4u, x4), dim=1))
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        mask_logits = self.head_mask(x_geo)
        ctr_logits = self.head_ctr(x_geo)
        out_posz    = self.head_posz(x_geo)
        out_rot    = self.head_rot(x_geo)
        cls_logits  = self.head_cls(x_sem)

        mu_raw = out_posz[:, 0:3]
        lv_raw = out_posz[:, 3:6]
        mu_z = mu_raw[:, 2:3] * 1000.0 * self.out_pos_scale
        mu_pos = torch.cat([mu_raw[:, 0:2], mu_z], dim=1)
        lv_pos = lv_raw.clamp(-8.0, 4.0)
        if self.rot_repr == "r6d":
            r6d          = out_rot[:, :6]
            rot_R        = r6d_to_rotmat(r6d)                   # (B,3,3,H,W)
            logvar_theta = out_rot[:, 6:7].clamp(-10.0, 5.0)
        else:  # "rotvec"
            rvec         = out_rot[:, :3]                       # (B,3,H,W)
            rot_R        = rotvec_to_rotmat_map(rvec)           # (B,3,3,H,W)
            logvar_theta = out_rot[:, 3:4].clamp(-10.0, 5.0)

        return {
            "mask_logits":  mask_logits,              # ← ロジットで返す
            "center_logits": ctr_logits,              # ← ロジットで返す
            "pos_mu":       mu_pos,                   # (B,3,H/4,W/4) [dx, dy, Z(mm)]
            "pos_logvar":   lv_pos,
            "rot_mat":      rot_R,                    # これは行列
            "rot_logvar_theta": logvar_theta,
            "cls_logits":   cls_logits,               # ← ロジットで返す
        }


@torch.jit.script
def r6d_to_rotmat(r6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices.

    Args:
        r6d: Rotation representation shaped ``(B, 6, H, W)``.
        eps: Clamp for numerical stability.

    Returns:
        torch.Tensor: Rotation matrices ``(B, 3, 3, H, W)``.
    """
    # r6d: (B,6,H,W) -> R: (B,3,3,H,W)
    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]
    n1 = torch.linalg.norm(a1, dim=1, keepdim=True).clamp_min(eps)
    b1 = a1 / n1
    # 直交化
    proj = (b1 * a2).sum(dim=1, keepdim=True) * b1
    v2 = a2 - proj
    n2 = torch.linalg.norm(v2, dim=1, keepdim=True).clamp_min(eps)
    b2 = v2 / n2
    b3 = torch.cross(b1, b2, dim=1)
    R = torch.stack([b2, b3, b1], dim=2)
    return R


# @torch.jit.script
# def _nearest_so3_batched(M: torch.Tensor) -> torch.Tensor:
#     # M: (B,K,3,3) -> R: (B,K,3,3) with det=+1
#     B, K = M.shape[0], M.shape[1]
#     M2 = M.reshape(B*K, 3, 3)

#     # ★ half/bfloat16でもOKなように、線形代数はfloat32で実施
#     M2_f32 = M2.to(torch.float32)

#     U, S, Vh = torch.linalg.svd(M2_f32, full_matrices=False)
#     R32 = U @ Vh  # (BK,3,3)

#     # 反射補正（det>0にする）もfloat32で
#     detR = torch.det(R32)                       # (BK,)
#     neg = (detR < 0.0).view(-1, 1, 1)          # (BK,1,1)

#     U_fix = U.clone()
#     # det<0 のとき U の最後の列を反転
#     U_fix[:, :, 2] = torch.where(neg.squeeze(-1), -U_fix[:, :, 2], U_fix[:, :, 2])
#     R32 = U_fix @ Vh                             # (BK,3,3), det=+1

#     # 元のdtypeに戻す
#     R = R32.reshape(B, K, 3, 3).to(M.dtype)
#     return R


@torch.jit.script
def _sign_det_from_cols(A: torch.Tensor) -> torch.Tensor:
    """Compute sign of determinant from column vectors.

    Args:
        A: Matrices shaped ``(N, 3, 3)``.

    Returns:
        torch.Tensor: Sign of determinant per matrix.
    """
    a0 = A[:, :, 0]                            # (N,3)
    a1 = A[:, :, 1]                            # (N,3)
    a2 = A[:, :, 2]                            # (N,3)
    c01 = torch.cross(a0, a1, dim=1)           # (N,3)
    vol = (c01 * a2).sum(dim=1)                # (N,)
    # sign: >=0 → +1,  <0 → -1  （0 は +1 扱い）
    s = (vol >= 0).to(A.dtype) * 2.0 - 1.0     # (N,)
    return s

@torch.jit.script
def nearest_so3_batched(M: torch.Tensor) -> torch.Tensor:
    """Project matrices onto SO(3) using SVD in a batched manner."""
    M2 = M.reshape(-1, 3, 3).to(torch.float32)               # flatten して f32
    U, S, Vh = torch.linalg.svd(M2, full_matrices=False)     # (N,3,3)
    UV = U @ Vh                                              # (N,3,3)

    # det(UV) の符号で反射補正（0 は +1 扱い）
    det_uv = torch.linalg.det(UV)                            # (N,)
    ones   = torch.ones_like(det_uv)
    sign   = torch.where(det_uv < 0.0, -ones, ones)          # (N,)

    # D = diag(1, 1, sign)
    D = torch.diag_embed(torch.stack([ones, ones, sign], dim=-1))  # (N,3,3)

    R32 = U @ D @ Vh                                          # (N,3,3) in f32
    return R32.to(M.dtype).view_as(M) 


@torch.jit.script
def _gaussian_disk_batched(H: int, W: int, idxs: torch.Tensor,
                           sigma: float, radius: int,
                           dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate batched Gaussian disks around peak indices."""
    B, K = idxs.shape[0], idxs.shape[1]
    y = torch.arange(0, H, device=device, dtype=dtype).view(1,1,H,1)
    x = torch.arange(0, W, device=device, dtype=dtype).view(1,1,1,W)
    ys = idxs[..., 0].to(dtype).view(B, K, 1, 1)
    xs = idxs[..., 1].to(dtype).view(B, K, 1, 1)
    dy2 = (y - ys) ** 2
    dx2 = (x - xs) ** 2
    g = torch.exp(-(dy2 + dx2) / (2.0 * (sigma * sigma)))  # (B,K,H,W)
    if radius > 0:
        within = (dy2 + dx2) <= float(radius * radius)
        g = g * within
    return g.unsqueeze(2)  # (B,K,1,H,W)


@torch.jit.script
def xy_from_uvZ(K: torch.Tensor,   # (B,3,3)
                uv: torch.Tensor,  # (B,K,2) [px]
                Z: torch.Tensor) -> torch.Tensor:  # (B,K) [m]
    """Lift pixel coordinates with depth to XY in camera frame."""
    # X = (u - cx)*Z/fx - (s/fx)*(v - cy)*Z/fy  （一般形; 実務では s≈0が多い）
    fx = K[:, 0, 0].unsqueeze(1)   # (B,1)
    fy = K[:, 1, 1].unsqueeze(1)
    s  = K[:, 0, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = uv[..., 0]  # (B,K)
    v = uv[..., 1]
    Zu = (u - cx) * Z / fx
    Zv = (v - cy) * Z / fy
    X = Zu - (s / fx) * Zv
    Y = Zv
    return torch.stack([X, Y], dim=-1)  # (B,K,2)


@torch.jit.script
def compose_t_from_Z_uvK(mu_z_map: torch.Tensor,      # (B,1,H/4,W/4)
                         peaks_yx: torch.Tensor,      # (B,K,2) on H/4 grid
                         K_left_1x: torch.Tensor,     # (B,3,3)
                         H: int, W: int, down: int) -> torch.Tensor:
    """Back-project peak points and depths into camera translation vectors."""
    # 1) ピクセル座標へ
    uv = peaks_to_uv_1_4(peaks_yx, down, W, H)  # (B,K,2) @ 1/1解像
    # 2) Z をサンプル（nearestでOK）
    By, Kk = peaks_yx.shape[0], peaks_yx.shape[1]
    Hq = mu_z_map.size(-2); Wq = mu_z_map.size(-1)
    y = peaks_yx[..., 0].clamp(0, Hq-1).long()
    x = peaks_yx[..., 1].clamp(0, Wq-1).long()
    z_samples = mu_z_map.squeeze(1)  # (B,H/4,W/4)
    z_at = z_samples[torch.arange(By).unsqueeze(1), y, x]  # (B,K) [m]
    # 3) X,Y を復元
    xy = xy_from_uvZ(K_left_1x, uv, z_at)  # (B,K,2)
    t = torch.cat([xy, z_at.unsqueeze(-1)], dim=-1)  # (B,K,3) [m]
    return t


def pos_mu_to_pointmap(
    pos_mu: torch.Tensor,       # (B,1 or 3,H/4,W/4)
    K_left_1x: torch.Tensor,    # (B,3,3) full-res intrinsics
    downsample: int = 4,
) -> torch.Tensor:
    """Convert pos_mu map to XYZ point map at 1/4 resolution.

    For 1-channel pos_mu, treat it as Z and use scaled intrinsics.
    For 3-channel pos_mu, interpret as (dx_px, dy_px, Z_mm).
    """
    if pos_mu.size(1) == 1:
        K14 = K_left_1x.clone()
        K14[:, 0, 0] /= downsample
        K14[:, 1, 1] /= downsample
        K14[:, 0, 2] /= downsample
        K14[:, 1, 2] /= downsample
        return rot_utils.depth_to_pointmap_from_K(pos_mu, K14)

    if pos_mu.size(1) != 3:
        raise ValueError(f"pos_mu must have 1 or 3 channels, got {pos_mu.size(1)}")

    B, _, H4, W4 = pos_mu.shape
    device = pos_mu.device
    dtype = pos_mu.dtype

    dx = pos_mu[:, 0:1]
    dy = pos_mu[:, 1:2]
    z = pos_mu[:, 2:3]

    u = (torch.arange(W4, device=device, dtype=dtype) + 0.5) * float(downsample)
    v = (torch.arange(H4, device=device, dtype=dtype) + 0.5) * float(downsample)
    u = u.view(1, 1, 1, W4).expand(B, 1, H4, W4)
    v = v.view(1, 1, H4, 1).expand(B, 1, H4, W4)

    u_c = u + dx
    v_c = v + dy

    fx = K_left_1x[:, 0, 0].view(B, 1, 1, 1)
    fy = K_left_1x[:, 1, 1].view(B, 1, 1, 1)
    cx = K_left_1x[:, 0, 2].view(B, 1, 1, 1)
    cy = K_left_1x[:, 1, 2].view(B, 1, 1, 1)

    X = (u_c - cx) / fx * z
    Y = (v_c - cy) / fy * z
    return torch.cat([X, Y, z], dim=1)


@torch.jit.script
def _greedy_match_yx(pred_yx: torch.Tensor, gt_yx: torch.Tensor) -> torch.Tensor:
    """Greedily match predicted peaks to ground truth peaks.

    Args:
        pred_yx: Predicted peak coordinates ``(B, K, 2)``.
        gt_yx: Ground-truth peak coordinates ``(B, K, 2)``.

    Returns:
        torch.Tensor: Permutation indices ``(B, K)`` aligning preds to GT order.
    """
    B, K, _ = pred_yx.shape
    perm = torch.zeros((B, K), dtype=torch.long, device=pred_yx.device)
    big = torch.tensor(1e9, dtype=pred_yx.dtype, device=pred_yx.device)

    for b in range(B):
        # 距離行列 (K,K)
        # d_ij = ||pred_i - gt_j||_2
        p = pred_yx[b]  # (K,2)
        g = gt_yx[b]    # (K,2)
        # (K,1,2) - (1,K,2) -> (K,K,2) -> (K,K)
        diff = p.unsqueeze(1) - g.unsqueeze(0)
        D = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)

        used_p = torch.zeros(K, dtype=torch.bool, device=pred_yx.device)
        for j in range(K):  # GTの順番で割り当て
            # 未使用の pred 行だけを対象に最小を取る
            Dj = D[:, j].clone()
            Dj = torch.where(used_p, big, Dj)   # 既に使用済み行は∞に
            i = int(torch.argmin(Dj).item())
            perm[b, j] = i
            used_p[i] = True

    return perm


def extract_instances_from_head_vec(
    center_logits: torch.Tensor,                 # (B,1,H/4,W/4) logits
    mask_logits: typing.Optional[torch.Tensor],  # (B,1,H/4,W/4) or None
    posz_mu: torch.Tensor,                       # (B,1 or 3, H/4, W/4)  Zのみ or (X,Y,Z)[m]
    rot_mat: torch.Tensor,                       # (B,3,3,H/4,W/4)
    cls_logits: typing.Optional[torch.Tensor],   # (B,C,H/4,W/4) or None
    k_left_1x: torch.Tensor,                     # (B,3,3) 1x解像度のK
    use_gt_peaks: bool,
    gt_center_1_4: typing.Optional[torch.Tensor],# (B,1,H/4,W/4) prob/heat or None
    gt_Wk: typing.Optional[torch.Tensor],        # (B,K,1,H/4,W/4) or None
    topk: int,
    nms_radius: int,
    center_thresh: float,
    use_uncertainty: bool,
    pos_logvar: typing.Optional[torch.Tensor],   # (B,1 or 3,H/4,W/4) or None（Z or XYZ）
    rot_logvar_theta: typing.Optional[torch.Tensor],  # (B,1,H/4,W/4) or None
    gauss_sigma: float,
    gauss_radius: int,
    class_from_logits: bool
) -> typing.Dict[str, torch.Tensor]:
    """Extract per-instance pose and classification from network heads.

    Args:
        center_logits: Center heatmap logits ``(B, 1, H/4, W/4)``.
        mask_logits: Optional mask logits ``(B, 1, H/4, W/4)``.
        posz_mu: Depth or position map.
        rot_mat: Rotation matrices ``(B, 3, 3, H/4, W/4)``.
        cls_logits: Optional class logits ``(B, C, H/4, W/4)``.
        k_left_1x: Camera intrinsics at full resolution.
        use_gt_peaks: Whether to use GT peaks for extraction.
        gt_center_1_4: Optional GT center map ``(B, 1, H/4, W/4)``.
        gt_Wk: Optional GT weights ``(B, K, 1, H/4, W/4)``.
        topk: Max number of instances to return.
        nms_radius: NMS radius on the heatmap.
        center_thresh: Threshold for peaks when not using GT.
        use_uncertainty: Whether uncertainty maps are provided.
        pos_logvar: Optional log-variance for position map.
        rot_logvar_theta: Optional log-variance for rotation.
        gauss_sigma: Gaussian sigma for peak weighting.
        gauss_radius: Radius for Gaussian disk.
        class_from_logits: If ``True``, apply softmax; otherwise sigmoid for classes.

    Returns:
        Dict[str, torch.Tensor]: Instance predictions and auxiliary weights.
    """

    B, _, H, W = center_logits.shape
    device = center_logits.device
    dtype  = center_logits.dtype

    # --- base prob/weight ---
    ctr_p = torch.sigmoid(center_logits)                         # (B,1,H,W)
    if mask_logits is not None:
        # msk_p = torch.sigmoid(mask_logits)
        msk_p = torch.ones_like(ctr_p)
    else:
        msk_p = torch.ones_like(ctr_p)
    base_w = ctr_p * msk_p                                       # (B,1,H,W)

    # GTピーク座標（教師Wkがある時に返す用）
    idxs_gt = torch.empty((B, 0, 2), dtype=torch.long, device=device)

    # =========================
    # 1) インスタンスのピーク/重み Wk を決める
    # =========================
    if use_gt_peaks and (gt_Wk is not None) and (gt_Wk.numel() > 0):
        Wk_in = gt_Wk.to(device=device, dtype=dtype)             # (B,K,1,H,W)
        Bk, Kg, _, Hk, Wk_ = Wk_in.shape
        if not (Bk == B and Hk == H and Wk_ == W):
            # 形が合わなければ空に
            Kg = 0
            Wk_in = torch.zeros((B, 0, 1, H, W), dtype=dtype, device=device)

        # GTのargmax座標
        flat_gt = Wk_in.reshape(B, Kg, H*W)
        argmx_gt = flat_gt.argmax(dim=-1)                         # (B,Kg)
        ys_gt = argmx_gt // W
        xs_gt = argmx_gt %  W
        idxs_gt = torch.stack((ys_gt, xs_gt), dim=-1)             # (B,Kg,2)

        # 予測側からも K=Kg 個のピーク（NMS）を拾って、GT順に並べ替え
        pool = F.max_pool2d(ctr_p, kernel_size=2*nms_radius+1, stride=1, padding=nms_radius)
        keep = (ctr_p == pool)
        scores_nms = ctr_p.masked_fill(~keep, float("-inf")).reshape(B, H*W)
        prob_flat  = (base_w).reshape(B, H*W)

        score1, topk_idx1 = torch.topk(scores_nms, k=max(Kg, 1), dim=1, largest=True)
        need_fb = (score1 <= -1e30).any(dim=1, keepdim=True)
        score2, topk_idx2 = torch.topk(prob_flat,  k=max(Kg, 1), dim=1, largest=True)
        topk_idx = torch.where(need_fb, topk_idx2, topk_idx1)

        ys_pred = topk_idx // W
        xs_pred = topk_idx %  W
        idxs_pred = torch.stack((ys_pred, xs_pred), dim=-1)       # (B,Kg,2)

        # Greedy で GT順へ
        perm = _greedy_match_yx(idxs_pred.to(torch.float32), idxs_gt.to(torch.float32))  # (B,Kg)
        b_ix = torch.arange(B, device=device).view(B,1).expand(B, Kg)
        idxs = idxs_pred[b_ix, perm]                                                     # (B,Kg,2)

        # そのピーク周りにガウスで Wk 生成（base_w でゲート）
        gk = _gaussian_disk_batched(H, W, idxs, gauss_sigma, gauss_radius, dtype, device) # (B,Kg,1,H,W)
        Wk = (base_w.unsqueeze(1) * gk)                                                   # (B,Kg,1,H,W)
        Wk_sum = Wk.sum(dim=(3,4), keepdim=True).clamp_min(1e-6)
        Wk = Wk / Wk_sum
        valid = (Wk_sum.squeeze(2).squeeze(2).squeeze(2) > 0)                             # (B,Kg)

        score = prob_flat[b_ix, (idxs[...,0]*W + idxs[...,1])]                            # (B,Kg)

    else:
        # 予測 or GT center heat から NMS → ガウスWk
        prob = gt_center_1_4.to(device=device, dtype=dtype) if (use_gt_peaks and (gt_center_1_4 is not None)) else ctr_p
        pool = F.max_pool2d(prob, kernel_size=2*nms_radius+1, stride=1, padding=nms_radius)
        keep = (prob == pool) & (prob >= center_thresh)

        scores_flat = prob.masked_fill(~keep, 0.0).reshape(B, H*W)
        Ksel = min(topk, H*W)
        score, topk_idx = torch.topk(scores_flat, k=Ksel, dim=1)

        ys = topk_idx // W
        xs = topk_idx %  W
        idxs = torch.stack((ys, xs), dim=-1)                                              # (B,Ksel,2)
        valid = (score > 0)

        gk = _gaussian_disk_batched(H, W, idxs, gauss_sigma, gauss_radius, dtype, device) # (B,Ksel,1,H,W)
        Wk = (base_w.unsqueeze(1) * gk)                                                   # (B,Ksel,1,H,W)
        Wk_sum = Wk.sum(dim=(3,4), keepdim=True).clamp_min(1e-6)
        Wk = (Wk / Wk_sum) * valid.view(B, -1, 1, 1, 1)

    K = Wk.shape[1]

    # =========================
    # 2) pos_mu → (X,Y,Z) point map
    # =========================
    pos_map_14 = pos_mu_to_pointmap(posz_mu, k_left_1x, downsample=4)

    # 前景重み（pose_from_maps_auto の wfg）
    wfg = msk_p                                                 # (B,1,H/4,W/4)

    # =========================
    # 3) R,t &（あれば）logvar 集約
    # =========================
    peaks_for_pose = idxs  # ピーク/最大重み画素のfallbackにも使える
    R_hat, t_hat, inst_valid, pos_lv_k, rot_lv_k = rot_utils.pose_from_maps_auto(
        rot_map=rot_mat, pos_map=pos_map_14,
        Wk_1_4=Wk, wfg=wfg,
        peaks_yx=peaks_for_pose,
        min_px=10, min_wsum=1e-6, tau_peak=0.0,
        pos_logvar=pos_logvar, rot_logvar_theta=rot_logvar_theta
    )  # (B,K,3,3), (B,K,3), (B,K), (B,K,Cp or None), (B,K,Cr or None)

    # =========================
    # 4) クラス投票
    # =========================
    if cls_logits is not None:
        cls_prob = F.softmax(cls_logits, dim=1) if class_from_logits else torch.sigmoid(cls_logits)
        cls_prob_k = (cls_prob.unsqueeze(1) * Wk).sum(dim=(3,4))    # (B,K,C)
        class_score, class_id = torch.max(cls_prob_k, dim=-1)       # (B,K)
    else:
        class_id    = torch.full((B, K), -1, dtype=torch.long, device=device)
        class_score = torch.zeros((B, K), dtype=dtype, device=device)

    area_px = (Wk > 0).sum(dim=(3,4)).squeeze(2).to(torch.int64)    # (B,K)

    out: typing.Dict[str, torch.Tensor] = {
        "gt_yx": idxs_gt.to(torch.long),     # (B,Kg,2) or empty
        "yx": idxs.to(torch.long),           # (B,K,2)
        "score": score,                      # (B,K)
        "R": R_hat,                          # (B,K,3,3)
        "t": t_hat,                          # (B,K,3)  [m]
        "class_id": class_id,                # (B,K)
        "class_score": class_score,          # (B,K)
        "area_px": area_px,                  # (B,K)
        "weight_map": Wk,                    # (B,K,1,H/4,W/4)
        "valid": inst_valid,                 # (B,K)
    }
    # 不確かさを返せるなら追加（下流のヘテロロスで使用）
    if pos_lv_k is not None:
        out["pos_logvar_k"] = pos_lv_k       # (B,K,1 or 3)
    if rot_lv_k is not None:
        out["rot_logvar_k"] = rot_lv_k       # (B,K,1)
    return out
