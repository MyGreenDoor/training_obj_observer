"""Pose updater modules and utilities designed for TorchScript."""

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(c_in, c_out, s=1): 
    """3x3 convolution without bias."""
    return nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=False)

class DSBlock(nn.Module):
    """Down: Conv→GN→SiLU×2（2回目で stride=2）"""
    def __init__(self, c_in, c_out, groups=16):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(c_in,  c_out), nn.GroupNorm(groups, c_out), nn.SiLU(True),
            conv3x3(c_out, c_out, s=2), nn.GroupNorm(groups, c_out), nn.SiLU(True),
        )
    def forward(self, x): return self.net(x)

class USBlock(nn.Module):
    """Up: upsample→Conv→GN→SiLU、skip 結合後に Conv→GN→SiLU"""
    def __init__(self, c_in, c_out, groups=16):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(c_in, c_out), nn.GroupNorm(groups, c_out), nn.SiLU(True),
        )
        self.post = nn.Sequential(
            conv3x3(c_out*2, c_out), nn.GroupNorm(groups, c_out), nn.SiLU(True),
        )
    def forward(self, x, skip):
        y = self.pre(x)
        y = torch.cat([y, skip], dim=1)
        return self.post(y)

class DeltaPoseRegressorMapHourglass4L(nn.Module):
    """
    入力: concat(point_map_cur, point_map_rend, context[, hidden]) を想定
    出力: (B,6,H/4,W/4) = (dω, dt)
    64x64 → 32 → 16 → 8 → 4 → 8 → 16 → 32 → 64
    """
    def __init__(self, c_in: int, base_ch: int = 128, groups: int = 16, 
                 out_scale_rot=0.1, out_scale_trans=0.01,
                 use_head_bn: bool = False,
                 use_tanh_gate_rot=True,  # ← 回転のみ tanh
                 ):
        super().__init__()
        self.out_scale_rot = out_scale_rot
        self.out_scale_trans = out_scale_trans

        # Encoder (stem + 4 downs)
        self.stem = nn.Sequential(
            conv3x3(c_in, base_ch), nn.GroupNorm(groups, base_ch), nn.SiLU(True),
            conv3x3(base_ch, base_ch), nn.GroupNorm(groups, base_ch), nn.SiLU(True),
        )                                     # 64
        self.down1 = DSBlock(base_ch, base_ch, groups)   # 32
        self.down2 = DSBlock(base_ch, base_ch, groups)   # 16
        self.down3 = DSBlock(base_ch, base_ch, groups)   # 8
        self.down4 = DSBlock(base_ch, base_ch, groups)   # 4

        # Bottleneck: 最小構成（ASPP なし）
        self.bottleneck = nn.Sequential(
            conv3x3(base_ch, base_ch), nn.GroupNorm(groups, base_ch), nn.SiLU(True)
        )

        # Decoder (4 ups)
        self.up3 = USBlock(base_ch, base_ch, groups)     # 4→8,  skip: s3
        self.up2 = USBlock(base_ch, base_ch, groups)     # 8→16, skip: s2
        self.up1 = USBlock(base_ch, base_ch, groups)     # 16→32,skip: s1
        self.up0 = USBlock(base_ch, base_ch, groups)     # 32→64,skip: s0

        # Head
        head = [conv3x3(base_ch, base_ch), nn.GroupNorm(groups, base_ch), nn.SiLU(True),
                nn.Conv2d(base_ch, 6, 1, bias=True)]
        if use_head_bn:  # 好みで
            head.insert(1, nn.BatchNorm2d(base_ch))
        self.head = nn.Sequential(*head)
        self.use_tanh_gate_rot = use_tanh_gate_rot

    def forward(self, point_map_cur, point_map_rend, context, mask=None, extra_feats=None):
        diff = point_map_cur - point_map_rend
        x_in = [point_map_cur, point_map_rend, diff, context, mask]
        if extra_feats is not None:
            x_in.append(extra_feats)
        x = torch.cat(x_in, dim=1)
        # if mask is not None:
        #     x = x * (mask > 0)

        # Enc
        s0 = self.stem(x)     # 64
        s1 = self.down1(s0)   # 32
        s2 = self.down2(s1)   # 16
        s3 = self.down3(s2)   # 8
        s4 = self.down4(s3)   # 4

        y  = self.bottleneck(s4)

        # Dec (+ skip)
        y = self.up3(y, s3)   # 8
        y = self.up2(y, s2)   # 16
        y = self.up1(y, s1)   # 32
        y = self.up0(y, s0)   # 64

        out = self.head(y)  # (B,6,H/4,W/4)
        rvec = out[:, 0:3]
        tvec = out[:, 3:6]

        # 回転のみ tanh で飽和
        if self.use_tanh_gate_rot:
            rvec = torch.tanh(rvec)

        # 追加の安全策：指数写像前にノルムをクリップ（小角制約）
        # ノルムが大きいときのみ縮めるので勾配の潰れが少ない
        # with torch.no_grad():
        #     norm = rvec.norm(dim=1, keepdim=True)  # (B,1,H/4,W/4)
        #     scale = torch.clamp(self.rvec_max_rad / (norm + 1e-8), max=1.0)
        # rvec = rvec * scale

        rvec = rvec * self.out_scale_rot
        tvec = tvec * self.out_scale_trans  # 並進は線形スケールのみ

        return torch.cat([rvec, tvec], dim=1)
    

class DeltaPoseRegressor(DeltaPoseRegressorMapHourglass4L):
    def __init__(self, c_in, base_ch=128, groups=16,
                 out_scale_rot=0.5, out_scale_trans=0.01,
                 use_gate=False, use_posenc=False):
        super().__init__(c_in=c_in, base_ch=base_ch, groups=groups, out_scale_rot=out_scale_rot, out_scale_trans=out_scale_trans)
        self.use_gate = use_gate
        if use_posenc:
            raise NotImplementedError()
        self.use_pose_enc=use_posenc

        # 追加の小ヘッド（ゲート）
        if use_gate:
            self.gate_head = nn.Sequential(
                conv3x3(base_ch, base_ch), nn.GroupNorm(groups, base_ch), nn.SiLU(True),
                nn.Conv2d(base_ch, 1, 1, bias=True)
            )
            nn.init.zeros_(self.gate_head[-1].weight); nn.init.zeros_(self.gate_head[-1].bias)

        # 出力ゼロ初期化
        # nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, point_map_cur, point_map_rend, current_pos_map, R_cur_log=None,
                context=None, mask=None, extra_feats=None, pose_enc=None,
                add_feats: list = None):
        # 追加特徴の組み立て（ΔZ比, 正規化平面差などは呼び出し側で作って add_feats へ）
        diff = point_map_cur - point_map_rend
        x_in = [point_map_cur, point_map_rend, diff, current_pos_map]
        if R_cur_log is not None: x_in.append(R_cur_log)
        if context is not None:   x_in.append(context)
        if mask is not None:      x_in.append(mask)
        if extra_feats is not None: x_in.append(extra_feats)
        if add_feats is not None:
            x_in += add_feats
        x = torch.cat(x_in, dim=1)

        # Hourglass 本体（親の実装）
        s0 = self.stem(x); s1 = self.down1(s0); s2 = self.down2(s1); s3 = self.down3(s2); s4 = self.down4(s3)
        y  = self.bottleneck(s4)
        y = self.up3(y, s3); y = self.up2(y, s2); y = self.up1(y, s1); y = self.up0(y, s0)

        out = self.head(y)  # (B,6,H,W)
        d_omega_raw, dt = out[:, :3], out[:, 3:] * self.out_scale_trans

        d_omega_raw = torch.tanh(d_omega_raw) * self.out_scale_rot

        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(y))  # (B,1,H,W)
            d_omega_raw = gate * d_omega_raw
            dt = gate * dt

        return torch.cat([d_omega_raw, dt], dim=1)


def conv3x3(ci, co, s=1, g=1): return nn.Conv2d(ci, co, 3, stride=s, padding=1, groups=g, bias=False)

class ResBlock(nn.Module):
    def __init__(self, c, g=16):
        super().__init__()
        self.f = nn.Sequential(
            conv3x3(c,c), nn.GroupNorm(g,c), nn.SiLU(True),
            conv3x3(c,c), nn.GroupNorm(g,c)
        )
    def forward(self, x): return F.silu(x + self.f(x), inplace=True)

class Up(nn.Module):
    def __init__(self, ci, co, g=16):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(conv3x3(ci, co), nn.GroupNorm(g, co), nn.SiLU(True))
    def forward(self, x): return self.conv(self.up(x))

# --- 幾何バイアスつき局所相関 ---
class LocalCorrWithGeom(nn.Module):
    """
    入力：
      phi_cur, phi_rnd: (B,C,h,w)  画素特徴（cur/rendは別Enc）
      P_cur,   P_rnd  : (B,3,h,w)  3D点マップ（同スケール）
    出力：
      du, dv, conf: 各 (B,1,h,w)
    仕組み：
      1) unfoldベースの局所相関 vol_corr: (B,D,h,w)
      2) 幾何バイアス vol_geom: (B,D,h,w) を加点 → logits = vol_corr + γ·vol_geom
      3) softmax(logits) から Δu，Δv の期待値，conf を計算
    """
    def __init__(self, c_in: int, c_mid: int = 64, radius: int = 3, use_geom_bias: bool = True):
        super().__init__()
        self.q = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.k = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.c_mid = c_mid
        self.R = radius
        self.use_geom = use_geom_bias
        D = (2*radius+1)**2
        self.vol_head = nn.Conv2d(D, D, 1, bias=True)  # 相関側の前処理
        # 幾何バイアスの係数（学習可能スカラー）
        if use_geom_bias:
            self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, phi_cur, phi_rnd, P_cur, P_rnd):
        B, _, H, W = phi_cur.shape
        R, D, C = self.R, (2*self.R+1)**2, self.c_mid

        # 相関体積
        q = F.normalize(self.q(phi_cur), dim=1)                 # (B,C,H,W)
        k = F.normalize(self.k(phi_rnd), dim=1)                 # (B,C,H,W)
        patches = F.unfold(k, kernel_size=2*R+1, padding=R)     # (B,C*D,HW)
        patches = patches.view(B, C, D, H*W)                    # (B,C,D,HW)
        qf = q.view(B, C, H*W)                                  # (B,C,HW)
        corr = (qf.unsqueeze(2) * patches).sum(dim=1)           # (B,D,HW)
        vol_corr = self.vol_head(corr.view(B, D, H, W))         # (B,D,H,W)

        # 幾何バイアス体積（Δz，‖Δp‖，⟨n_cur,n_rnd⟩*任意*）
        if self.use_geom:
            # P_rnd の局所パッチ抽出
            Pr = F.unfold(P_rnd, kernel_size=2*R+1, padding=R)  # (B, 3*D, HW)
            Pr = Pr.view(B, 3, D, H*W)                          # (B,3,D,HW)
            Pc = P_cur.view(B, 3, 1, H*W).expand_as(Pr)         # (B,3,D,HW)
            dP = Pc - Pr                                        # (B,3,D,HW)
            # 代表的な3つの幾何量
            dz   = dP[:, 2].view(B, D, H, W)                    # (B,D,H,W)
            dist = (dP**2).sum(dim=1).clamp_min(1e-12).sqrt().view(B, D, H, W)
            # 法線は無ければスキップ（あるなら別引数で渡して同様にunfold）
            # 単純で強いバイアス（小さい距離，高いZ一致を好む）
            vol_geom = (-dist) + (-0.5 * dz.abs())
            logits = vol_corr + self.gamma * vol_geom
        else:
            logits = vol_corr

        # softmax → 期待値
        w = torch.softmax(logits, dim=1)                        # (B,D,H,W)
        dy = torch.arange(-R, R+1, device=phi_cur.device, dtype=phi_cur.dtype)
        dx = torch.arange(-R, R+1, device=phi_cur.device, dtype=phi_cur.dtype)
        dY, dX = torch.meshgrid(dy, dx, indexing='ij')
        offx = dX.reshape(1, -1, 1, 1); offy = dY.reshape(1, -1, 1, 1)
        du = (w * offx).sum(dim=1, keepdim=True)                # (B,1,H,W)
        dv = (w * offy).sum(dim=1, keepdim=True)                # (B,1,H,W)
        conf = w.max(dim=1, keepdim=True).values                # (B,1,H,W)
        return du, dv, conf


def _enc64_cur(enc, x):
    s64 = enc["e0"](x); s32 = enc["d1"](s64); s16 = enc["d2"](s32); s8 = enc["d3"](s16)
    y16 = enc["f2"](enc["up2"](s8)  + s16)
    y32 = enc["f1"](enc["up1"](y16) + s32)
    y64 = enc["f0"](enc["up0"](y32) + s64)
    return s8, s16, y64

def _enc_down(enc, x):
    s64 = enc["e0"](x); s32 = enc["d1"](s64); s16 = enc["d2"](s32); s8 = enc["d3"](s16)
    return s8, s16


class DeltaPoseRegressor_CF(nn.Module):
    """
    cur／rend を別Encに固定．coarse→fine 相関（8→16→64）＋ rend warp．
    出力：(B,6,H/4,W/4)（回転のみ tanh でスケール，並進は線形スケール）
    前提：point_map_* は H/4 解像度（例：64×64），単位[m]．
    """
    def __init__(self, c_in: int,
                 base_ch: int = 128, groups: int = 16,
                 out_scale_rot: float = 0.5, out_scale_trans: float = 0.01,
                 corr_cmid_c: int = 64, corr_cmid_f: int = 48,
                 radius_c: int = 3, radius_f: int = 2,
                 use_geom_bias: bool = True,
                 use_gate=False, use_posenc=False):
        super().__init__()
        self.out_scale_rot = out_scale_rot
        self.out_scale_trans = out_scale_trans
        self.use_tanh_gate_rot = True

        c = base_ch

        # --- Encoders（別Enc，重み共有なし） ---
        # 64→32→16→8（受容野確保），各スケールで特徴と点群を用意
        def enc_branch_cur(c, groups):
            # cur側：64→32→16→8，さらにskipで64へ復元（y64_cを使う）
            return nn.ModuleDict({
                "e0": nn.Sequential(conv3x3(3, c), nn.GroupNorm(groups,c), nn.SiLU(True)),         # 64
                "d1": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 32
                "d2": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 16
                "d3": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 8
                "up2": Up(c,c,groups), "up1": Up(c,c,groups), "up0": Up(c,c,groups),
                "f2": ResBlock(c,groups), "f1": ResBlock(c,groups), "f0": ResBlock(c,groups)
            })

        def enc_branch_rend_down(c, groups):
            # rend側：down専用（y64_rを作らない）
            return nn.ModuleDict({
                "e0": nn.Sequential(conv3x3(3, c), nn.GroupNorm(groups,c), nn.SiLU(True)),         # 64
                "d1": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 32
                "d2": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 16
                "d3": nn.Sequential(conv3x3(c,c, s=2), nn.GroupNorm(groups,c), nn.SiLU(True), ResBlock(c,groups)),  # 8
            })

        self.enc_cur  = enc_branch_cur(base_ch, groups)
        self.enc_rend = enc_branch_rend_down(base_ch, groups)

        # --- 相関（coarse: 8×8，fine: 16×16 → 64×64） ---
        self.corr_coarse = LocalCorrWithGeom(c_in=c, c_mid=corr_cmid_c, radius=radius_c, use_geom_bias=use_geom_bias)
        self.corr_fine   = LocalCorrWithGeom(c_in=c, c_mid=corr_cmid_f, radius=radius_f, use_geom_bias=use_geom_bias)

        # 相関特徴 [du,dv,conf] を 64×64 で投影
        self.corr_proj = nn.Sequential(
            nn.Conv2d(3, c//4, 1, bias=False), nn.GroupNorm(groups, c//4), nn.SiLU(True)
        )

        # --- 最終 Decoder → Head（出力は (B,6,64,64) を維持） ---
        stem_in_extra = c//4  # 相関特徴の分
        self.dec_tail = nn.Sequential(
            ResBlock(c + stem_in_extra, groups),
            conv3x3(c + stem_in_extra, c), nn.GroupNorm(groups,c), nn.SiLU(True),
            ResBlock(c, groups)
        )
        self.head = nn.Sequential(
            conv3x3(c, c), nn.GroupNorm(groups,c), nn.SiLU(True),
            nn.Conv2d(c, 6, 1, bias=True)
        )

    def _enc64(self, enc, x):  # 64→32→16→8 → 64 の復元特徴
        s64 = enc["e0"](x)
        s32 = enc["d1"](s64)
        s16 = enc["d2"](s32)
        s8  = enc["d3"](s16)
        y16 = enc["f2"](enc["up2"](s8)  + s16)
        y32 = enc["f1"](enc["up1"](y16) + s32)
        y64 = enc["f0"](enc["up0"](y32) + s64)
        return s8, s16, y64  # 8×8, 16×16, 64×64 特徴

    def forward(self, point_map_cur, point_map_rend,
                R_cur_log=None, context=None, mask=None, extra_feats=None, pose_enc=None, add_feats=None):
        B, _, H, W = point_map_cur.shape  # 64×64 を想定

        # --- 別Enc ---
        s8_c, s16_c, y64_c = _enc64_cur(self.enc_cur,  point_map_cur)
        s8_r, s16_r        = _enc_down   (self.enc_rend, point_map_rend)

        # --- 幾何も同スケールに揃える（点群は入力そのものをスケールダウン） ---
        P8_c  = F.interpolate(point_map_cur, size=(H//8, W//8), mode='bilinear', align_corners=False)
        P8_r  = F.interpolate(point_map_rend, size=(H//8, W//8), mode='bilinear', align_corners=False)
        P16_c = F.interpolate(point_map_cur, size=(H//4, W//4), mode='bilinear', align_corners=False)
        P16_r = F.interpolate(point_map_rend, size=(H//4, W//4), mode='bilinear', align_corners=False)

        # --- Coarse 相関（8×8）→ Δu8,Δv8 を 16×16 に拡大して rend 特徴・点群を warp ---
        du8, dv8, conf8 = self.corr_coarse(s8_c, s8_r, P8_c, P8_r)  # (B,1,8,8)
        # 正規化座標に変換して warp
        def shift_grid(du, dv, h, w):
            # du,dv: (B,1,h,w)
            B = du.shape[0]
            gx = torch.linspace(-1, 1, w, device=du.device, dtype=du.dtype)
            gy = torch.linspace(-1, 1, h, device=du.device, dtype=du.dtype)
            yy, xx = torch.meshgrid(gy, gx, indexing='ij')  # yy: (h,w), xx: (h,w)

            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, h, w, 2)  # (B,h,w,2)

            # ピクセル→正規化
            du_n = 2.0 * du / max(w - 1, 1)
            dv_n = 2.0 * dv / max(h - 1, 1)

            # ★ permute不要：すでに (B,h,w,2)
            offset = torch.stack([du_n.squeeze(1), dv_n.squeeze(1)], dim=-1)  # (B,h,w,2)
            return grid + offset
        grid8 = shift_grid(du8, dv8, P8_r.shape[-2], P8_r.shape[-1])
        s8_r_w  = F.grid_sample(s8_r,  grid8, mode='bilinear', padding_mode='border', align_corners=True)
        P8_r_w  = F.grid_sample(P8_r,  grid8, mode='bilinear', padding_mode='border', align_corners=True)

        # --- Fine 相関（16×16） ---
        # 粗Δを 16×16 に拡大して初期合わせをよくする（オプショナル）
        du16_init = F.interpolate(du8, scale_factor=2, mode='nearest')
        dv16_init = F.interpolate(dv8, scale_factor=2, mode='nearest')
        # 16×16の相関
        du16, dv16, conf16 = self.corr_fine(s16_c, s16_r, P16_c, P16_r)
        # 累積Δ（粗＋微）
        du16_sum = du16 + du16_init
        dv16_sum = dv16 + dv16_init

        # --- 64×64 に拡大して相関特徴を作る（最終は map 出力なので64で使う） ---
        du64   = F.interpolate(du16_sum, size=(H, W), mode='bilinear', align_corners=False)
        dv64   = F.interpolate(dv16_sum, size=(H, W), mode='bilinear', align_corners=False)
        # conf の融合（解像度を合わせてから max）
        conf16_from8 = F.interpolate(conf8, size=conf16.shape[-2:], mode='nearest')  # 8→16
        conf16 = torch.maximum(conf16_from8.to(conf16.dtype), conf16)                # (B,1,16,16)

        # 64×64へアップサンプル（信頼度は nearest 推奨）
        conf64 = F.interpolate(conf16, size=(H, W), mode='nearest')
        corr_feat = self.corr_proj(torch.cat([du64, dv64, conf64], dim=1))  # (B,c/4,64,64)

        # --- 従来のスタックに corr_feat を追加して最終回帰 ---
        diff = point_map_cur - point_map_rend
        x_in = [point_map_cur, point_map_rend, diff]
        if R_cur_log is not None: x_in.append(R_cur_log)
        if context   is not None: x_in.append(context)
        if mask      is not None: x_in.append(mask)
        if extra_feats is not None: x_in.append(extra_feats)
        if add_feats is not None: x_in += add_feats
        x_in.append(corr_feat)
        x = torch.cat([t for t in x_in if t is not None], dim=1)

        z = torch.cat([y64_c, corr_feat], dim=1)            # cur側の64特徴＋相関特徴
        z = self.dec_tail(z)
        out = self.head(z)                                   # (B,6,64,64)

        rvec = torch.tanh(out[:, :3]) * self.out_scale_rot  # 回転のみtanh
        tvec = out[:, 3:] * self.out_scale_trans            # 並進は線形スケール（上限なし）
        return torch.cat([rvec, tvec], dim=1)
    

def _gn(ch: int, groups: int = 8):
    g = min(groups, max(1, ch // 8))
    return nn.GroupNorm(g, ch)

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=_gn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch), nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=_gn):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=False)
        self.block = _ConvBlock(in_ch, out_ch, norm)
    def forward(self, x): return self.block(self.pool(x))

class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm=_gn, use_deconv=False):
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = _ConvBlock(in_ch + skip_ch, out_ch, norm)
    def forward(self, x, skip):
        x = self.up(x)
        # 形がズレるケースを吸収（安全側; 基本は同じはず）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class DeltaPoseRegressorUNet(nn.Module):
    """
    入力: cur/rend の point_map（mask で有効画素ゲート）、context、(任意)extra
    出力: (B,6,H/4,W/4) = [d_rvec(3), d_t(3)]
    """
    def __init__(
        self,
        context_ch: int = 128,
        extra_ch: int = 0,              # 例: hidden を入れるなら 128
        base_ch: int = 64,
        num_down: int = 3,              # H/4 -> H/16 程度（安定 & 速い）
        use_deconv: bool = False,
        norm_layer= nn.BatchNorm2d,
    ):
        super().__init__()
        in_ch = 3 + 3 + 3 + 1 + context_ch + (extra_ch if extra_ch > 0 else 0)
        # enc
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8]
        self.enc0 = _ConvBlock(in_ch, chs[0], norm_layer)
        self.down1 = _Down(chs[0], chs[1], norm_layer)
        self.down2 = _Down(chs[1], chs[2], norm_layer) if num_down >= 2 else None
        self.down3 = _Down(chs[2], chs[3], norm_layer) if num_down >= 3 else None

        # bottleneck
        bot_in = chs[min(num_down, 3)]
        self.bottleneck = _ConvBlock(bot_in, bot_in, norm_layer)

        # dec（skip と対になるチャネル）
        self.up3 = _Up(bot_in, chs[2], chs[2], norm_layer, use_deconv) if num_down >= 3 else None
        self.up2 = _Up(chs[2] if num_down>=3 else bot_in, chs[1], chs[1], norm_layer, use_deconv) if num_down >= 2 else None
        self.up1 = _Up(chs[1] if num_down>=2 else bot_in, chs[0], chs[0], norm_layer, use_deconv)

        # head: 6ch（ゼロ初期化で Δ=0 スタート）
        self.head = nn.Conv2d(chs[0], 6, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        point_map_cur: torch.Tensor,      # (B,3,H,W)
        point_map_rend: torch.Tensor,     # (B,3,H,W)
        context: torch.Tensor,            # (B,Cc,H,W)
        mask: torch.Tensor,               # (B,1,H,W)
        extra_feats: typing.Optional[torch.Tensor] = None  # (B,Ce,H,W)
    ) -> torch.Tensor:                    # -> (B,6,H,W)
        B, _, H, W = point_map_cur.shape

        # --- mask で有効画素をゲート（NaN/Inf も 0 に）
        m = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        cur  = torch.nan_to_num(point_map_cur,  nan=0.0, posinf=0.0, neginf=0.0)  * m
        rend = torch.nan_to_num(point_map_rend, nan=0.0, posinf=0.0, neginf=0.0)  * m
        diff = (cur - rend)

        feats = [cur, rend, diff, m, context]
        if extra_feats is not None:
            feats.append(extra_feats)
        x_in = torch.cat(feats, dim=1)

        # enc
        e0 = self.enc0(x_in)          # H/4
        e1 = self.down1(e0)           # H/8
        x  = e1
        if self.down2 is not None:    # H/16
            e2 = self.down2(e1)
            x  = e2
        if self.down3 is not None:    # H/32
            e3 = self.down3(e2)
            x  = e3

        # bottleneck
        x = self.bottleneck(x)

        # dec + skip
        if self.up3 is not None:
            x = self.up3(x, e2)
        if self.up2 is not None:
            x = self.up2(x, e1)
        x = self.up1(x, e0)

        out = self.head(x)            # (B,6,H/4,W/4)
        # ここではスケール拘束を掛けず、生の Δ を返す（学習で自動調整）
        return out
