"""Panoptic stereo model with disparity + multi-task heads."""

from typing import Dict, Callable, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stereo_disparity import (
    ContextUNet14,
    UpdateBlock,
    build_corr_pyramid,
    sample_corr_pyramid_bilinear,
    sample_corr_pyramid_bilinear_2d,
    disparity_to_pointmap_from_Kpair_with_conf,
    make_gn,
)
from models.multi_head import (
    ASPPLite,
    Bottleneck,
    ConvBlock,
    conv3x3,
    r6d_to_rotmat,
    rotvec_to_rotmat_map,
)


class AffEmbHead(nn.Module):
    """Affinity and embedding heads for instance segmentation."""

    def __init__(self, in_ch: int, emb_dim: int, norm_layer: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
            norm_layer(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
            norm_layer(in_ch),
            nn.ReLU(inplace=True),
        )
        self.aff = nn.Conv2d(in_ch, 4, kernel_size=1)
        self.emb = nn.Conv2d(in_ch, emb_dim, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(feat)
        aff_logits = self.aff(x)
        emb = self.emb(x)
        emb = F.normalize(emb, dim=1)
        return {"aff_logits": aff_logits, "emb": emb}


class LiteFPNMultiTaskHeadWithAffEmb(nn.Module):
    """LiteFPN-style multi-task head with affinity/embedding outputs."""

    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        num_classes: int,
        ctx_ch: int = 128,
        hidden_ch: int = 128,
        use_pixelshuffle: bool = True,
        rot_repr: str = "r6d",
        out_pos_scale: float = 1.0,
        emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.use_pixelshuffle = use_pixelshuffle
        self.rot_repr = rot_repr.lower()

        in_ch = ctx_ch + hidden_ch + 1 + 3
        c4, c8 = 160, 192

        self.enc4 = ConvBlock(in_ch, c4, norm_layer)
        self.down4 = Bottleneck(c4, c4 // 2, c8, norm_layer, s=2)
        self.enc8 = Bottleneck(c8, c8 // 2, c8, norm_layer, s=1)
        self.bridge = ASPPLite(c8, norm_layer)

        if use_pixelshuffle:
            self.up = nn.Sequential(
                nn.Conv2d(c8, c8 * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.SiLU(True),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                conv3x3(c8, c8),
                norm_layer(c8),
                nn.SiLU(True),
            )
        self.fuse4 = ConvBlock(c8 + c4, 224, norm_layer)

        rot_ch = 6 if self.rot_repr == "r6d" else 3
        self.neck_geo = self.make_neck(224, 160, norm_layer)
        self.neck_sem = self.make_neck(224, 128, norm_layer)
        self.neck_inst = self.make_neck(224, 128, norm_layer)

        self.head_mask = nn.Conv2d(160, 1, 3, padding=1)
        self.head_ctr = nn.Conv2d(160, 1, 3, padding=1)
        self.head_posz = nn.Conv2d(160, 6, 3, padding=1)
        self.head_rot = nn.Conv2d(160, rot_ch + 1, 3, padding=1)
        self.head_cls = nn.Conv2d(128, num_classes, 1)
        self.affemb_head = AffEmbHead(128, emb_dim, norm_layer)
        self.out_pos_scale = out_pos_scale

        self._init_head(self.head_ctr)
        self._init_head(self.head_mask)

    def make_neck(self, in_ch: int, mid_ch: int, norm_layer: Callable[[int], nn.Module]) -> nn.Module:
        """Create lightweight convolutional neck for head branches."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            norm_layer(in_ch),
            nn.SiLU(True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            norm_layer(mid_ch),
            nn.SiLU(True),
        )

    def _init_head(self, m: nn.Module, bias_neg: float = -2.0, std: float = 1e-3) -> None:
        for l in m.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=std)
                if l.bias is not None:
                    nn.init.constant_(l.bias, bias_neg)

    def forward(
        self,
        ctx_14: torch.Tensor,
        hidden_14: torch.Tensor,
        mask_14: torch.Tensor,
        point_map_14: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run FPN-style head on 1/4 resolution features."""
        x4_in = torch.cat((ctx_14, hidden_14, mask_14, point_map_14), dim=1)
        x4 = self.enc4(x4_in)
        x8 = self.enc8(self.down4(x4))
        x8 = self.bridge(x8)
        x4u = self.up(x8)
        x = self.fuse4(torch.cat((x4u, x4), dim=1))
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        x_inst = self.neck_inst(x)

        mask_logits = self.head_mask(x_geo)
        ctr_logits = self.head_ctr(x_geo)
        out_posz = self.head_posz(x_geo)
        out_rot = self.head_rot(x_geo)
        cls_logits = self.head_cls(x_sem)

        mu_raw = out_posz[:, 0:3]
        lv_raw = out_posz[:, 3:6]
        log_z = mu_raw[:, 2:3] * self.out_pos_scale
        mu_pos = torch.cat([mu_raw[:, 0:2], log_z], dim=1)
        lv_pos = lv_raw.clamp(-8.0, 4.0)
        if self.rot_repr == "r6d":
            r6d = out_rot[:, :6]
            rot_R = r6d_to_rotmat(r6d)
            logvar_theta = out_rot[:, 6:7].clamp(-10.0, 5.0)
        else:
            rvec = out_rot[:, :3]
            rot_R = rotvec_to_rotmat_map(rvec)
            logvar_theta = out_rot[:, 3:4].clamp(-10.0, 5.0)

        affemb_out = self.affemb_head(x_inst)

        return {
            "mask_logits": mask_logits,
            "center_logits": ctr_logits,
            "pos_mu": mu_pos,
            "pos_logvar": lv_pos,
            "rot_mat": rot_R,
            "rot_logvar_theta": logvar_theta,
            "cls_logits": cls_logits,
            **affemb_out,
        }


class LiteFPNMultiTaskHeadNoHiddenWithAffEmb(nn.Module):
    """LiteFPN-style multi-task head without hidden input, with aff/emb outputs."""

    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        num_classes: int,
        ctx_ch: int = 128,
        use_pixelshuffle: bool = True,
        rot_repr: str = "r6d",
        out_pos_scale: float = 1.0,
        emb_dim: int = 8,
        head_c4: int = 160,
        head_c8: int = 192,
        head_fuse_ch: int = 224,
        head_geo_ch: int = 160,
        head_sem_ch: int = 128,
        head_inst_ch: int = 128,
    ) -> None:
        super().__init__()
        self.use_pixelshuffle = use_pixelshuffle
        self.rot_repr = rot_repr.lower()

        in_ch = ctx_ch + 1 + 3
        c4, c8 = int(head_c4), int(head_c8)
        fuse_ch = int(head_fuse_ch)
        geo_ch = int(head_geo_ch)
        sem_ch = int(head_sem_ch)
        inst_ch = int(head_inst_ch)

        self.enc4 = ConvBlock(in_ch, c4, norm_layer)
        self.down4 = Bottleneck(c4, c4 // 2, c8, norm_layer, s=2)
        self.enc8 = Bottleneck(c8, c8 // 2, c8, norm_layer, s=1)
        self.bridge = ASPPLite(c8, norm_layer)

        if use_pixelshuffle:
            self.up = nn.Sequential(
                nn.Conv2d(c8, c8 * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.SiLU(True),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                conv3x3(c8, c8),
                norm_layer(c8),
                nn.SiLU(True),
            )
        self.fuse4 = ConvBlock(c8 + c4, fuse_ch, norm_layer)

        rot_ch = 6 if self.rot_repr == "r6d" else 3
        self.neck_geo = self.make_neck(fuse_ch, geo_ch, norm_layer)
        self.neck_sem = self.make_neck(fuse_ch, sem_ch, norm_layer)
        self.neck_inst = self.make_neck(fuse_ch, inst_ch, norm_layer)

        self.head_posz = nn.Conv2d(geo_ch, 6, 3, padding=1)
        self.head_rot = nn.Conv2d(geo_ch, rot_ch + 1, 3, padding=1)
        self.head_cls = nn.Conv2d(sem_ch, num_classes, 1)
        self.affemb_head = AffEmbHead(inst_ch, emb_dim, norm_layer)
        self.out_pos_scale = out_pos_scale

    def make_neck(self, in_ch: int, mid_ch: int, norm_layer: Callable[[int], nn.Module]) -> nn.Module:
        """Create lightweight convolutional neck for head branches."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            norm_layer(in_ch),
            nn.SiLU(True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            norm_layer(mid_ch),
            nn.SiLU(True),
        )

    def _init_head(self, m: nn.Module, bias_neg: float = -2.0, std: float = 1e-3) -> None:
        for l in m.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=std)
                if l.bias is not None:
                    nn.init.constant_(l.bias, bias_neg)

    def forward(
        self,
        ctx_1x: torch.Tensor,
        mask_1x: torch.Tensor,
        point_map_1x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run FPN-style head on full resolution features."""
        x4_in = torch.cat((ctx_1x, mask_1x, point_map_1x), dim=1)
        x4 = self.enc4(x4_in)
        x8 = self.enc8(self.down4(x4))
        x8 = self.bridge(x8)
        x4u = self.up(x8)
        x = self.fuse4(torch.cat((x4u, x4), dim=1))
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        x_inst = self.neck_inst(x)

        out_posz = self.head_posz(x_geo)
        out_rot = self.head_rot(x_geo)
        cls_logits = self.head_cls(x_sem)

        mu_raw = out_posz[:, 0:3]
        lv_raw = out_posz[:, 3:6]
        log_z = mu_raw[:, 2:3] * self.out_pos_scale
        mu_pos = torch.cat([mu_raw[:, 0:2], log_z], dim=1)
        lv_pos = lv_raw.clamp(-8.0, 4.0)
        if self.rot_repr == "r6d":
            r6d = out_rot[:, :6]
            rot_R = r6d_to_rotmat(r6d)
            logvar_theta = out_rot[:, 6:7].clamp(-10.0, 5.0)
        else:
            rvec = out_rot[:, :3]
            rot_R = rotvec_to_rotmat_map(rvec)
            logvar_theta = out_rot[:, 3:4].clamp(-10.0, 5.0)

        affemb_out = self.affemb_head(x_inst)

        return {
            "pos_mu": mu_pos,
            "pos_logvar": lv_pos,
            "rot_mat": rot_R,
            "rot_logvar_theta": logvar_theta,
            "cls_logits": cls_logits,
            **affemb_out,
        }


class DummyMultiTaskHead(nn.Module):
    """Dummy multi-task head for memory profiling."""

    def __init__(self, num_classes: int, emb_dim: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)

    def forward(
        self,
        ctx_1x: torch.Tensor,
        mask_1x: torch.Tensor,
        point_map_1x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = ctx_1x.shape
        device = ctx_1x.device
        dtype = ctx_1x.dtype
        rot = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3, 1, 1).expand(B, 3, 3, H, W)
        return {
            "mask_logits": torch.zeros((B, 1, H, W), device=device, dtype=dtype),
            "center_logits": torch.zeros((B, 1, H, W), device=device, dtype=dtype),
            "pos_mu": torch.zeros((B, 3, H, W), device=device, dtype=dtype),
            "pos_logvar": torch.zeros((B, 3, H, W), device=device, dtype=dtype),
            "rot_mat": rot,
            "rot_logvar_theta": torch.zeros((B, 1, H, W), device=device, dtype=dtype),
            "cls_logits": torch.zeros((B, self.num_classes, H, W), device=device, dtype=dtype),
            "aff_logits": torch.zeros((B, 4, H, W), device=device, dtype=dtype),
            "emb": torch.zeros((B, self.emb_dim, H, W), device=device, dtype=dtype),
        }


class ContextUNet14Up(nn.Module):
    """ContextUNet14 variant with skip from enc1 and 1x output."""

    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        out_ch_1_4: int,
        out_ch_1x: int,
        use_aspp: bool = True,
        l2_normalize: bool = False,
        width_ratios: Tuple[float, float, float] = (0.5, 0.75, 1.0),
        divisor: int = 8,
        min_ch: int = 16,
    ) -> None:
        super().__init__()
        gn = norm_layer

        r1, r2, r3 = width_ratios
        c1 = max(min_ch, int(out_ch_1_4 * r1) // divisor * divisor)
        c2 = max(min_ch, int(out_ch_1_4 * r2) // divisor * divisor)
        c3 = max(min_ch, int(out_ch_1_4 * r3) // divisor * divisor)

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3, bias=True),
            gn(c1),
            nn.ReLU(True),
            Bottleneck(c1, c1 // 2, c1, gn, s=1),
        )
        self.enc2 = nn.Sequential(
            Bottleneck(c1, c1 // 2, c2, gn, s=2),
            Bottleneck(c2, c2 // 2, c2, gn, s=1),
        )
        self.enc3 = nn.Sequential(
            Bottleneck(c2, c2 // 2, c3, gn, s=2),
            Bottleneck(c3, c3 // 2, c3, gn, s=1),
        )

        self.aspp = ASPPLite(c3, gn) if use_aspp else nn.Identity()

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c3 + c2, c3, gn)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(c3 + c1, c2, gn)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head_1_4 = nn.Conv2d(c3, out_ch_1_4, 1, bias=True)
        self.head_1x = nn.Conv2d(c2, out_ch_1x, 1, bias=True)

        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.aspp(e3)

        up3 = self.up3(b)
        if up3.shape[-2:] != e2.shape[-2:]:
            up3 = F.interpolate(up3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([up3, e2], dim=1))

        up2 = self.up2(d3)
        if up2.shape[-2:] != e1.shape[-2:]:
            up2 = F.interpolate(up2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e1], dim=1))

        up1 = self.up1(d2)
        if up1.shape[-2:] != x.shape[-2:]:
            up1 = F.interpolate(up1, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_1_4 = self.head_1_4(d3)
        out_1x = self.head_1x(up1)

        if self.l2_normalize:
            out_1_4 = F.normalize(out_1_4, p=2, dim=1, eps=1e-12)
            out_1x = F.normalize(out_1x, p=2, dim=1, eps=1e-12)
        return out_1_4, out_1x


class ConvexUpsampler(nn.Module):
    """RAFT-style convex upsampling for arbitrary channels."""

    def __init__(self, hidden_ch: int, up_factor: int = 4, groups: int = 16) -> None:
        super().__init__()
        self.s = int(up_factor)
        self.mask_head = nn.Sequential(
            conv3x3(hidden_ch, hidden_ch),
            nn.GroupNorm(groups, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, 9 * (self.s * self.s), kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        B, C, H, W = x.shape
        mask = self.mask_head(h)
        mask = mask.view(B, 1, 9, self.s, self.s, H, W)
        mask = torch.softmax(mask, dim=2)

        unfold = F.unfold(x, kernel_size=3, padding=1)
        unfold = unfold.view(B, C, 9, H, W).unsqueeze(3).unsqueeze(3)
        up = (mask * unfold).sum(dim=2)
        up = up.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H * self.s, W * self.s)
        return up * float(scale_factor)


class PanopticStereoMultiHead(nn.Module):
    """Stereo disparity with LiteFPN multi-task head and instance heads."""

    def __init__(
        self,
        levels: int = 4,
        norm_layer: Callable[[int], nn.Module] = make_gn(16),
        l2_normalize_feature: bool = True,
        use_ctx_aspp: bool = True,
        lookup_mode: str = "1d",
        radius_w: int = 4,
        radius_h: int = 0,
        hidden_ch: int = 96,
        context_ch: int = 96,
        num_classes: int = 1,
        rot_repr: str = "r6d",
        emb_dim: int = 16,
        use_dummy_head: bool = False,
        head_c4: int = 160,
        head_c8: int = 192,
        head_fuse_ch: int = 224,
        head_geo_ch: int = 160,
        head_sem_ch: int = 128,
        head_inst_ch: int = 128,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.lookup_mode = lookup_mode.lower()
        self.hidden_ch = int(hidden_ch)
        self.context_ch = int(context_ch)

        if self.lookup_mode == "1d":
            self.radius_w = radius_w
            self.radius_h = 0
        elif self.lookup_mode == "2d":
            self.radius_w = radius_w
            self.radius_h = radius_w
        elif self.lookup_mode in ("a2d", "anisotropic2d", "anisotropic_2d"):
            self.radius_w = radius_w
            self.radius_h = radius_h
        else:
            raise ValueError(f"Unknown lookup_mode={lookup_mode}")

        if self.radius_h == 0:
            k_per_level = 2 * self.radius_w + 1
        else:
            k_per_level = (2 * self.radius_w + 1) * (2 * self.radius_h + 1)

        self.feature = ContextUNet14(
            norm_layer,
            use_aspp=use_ctx_aspp,
            l2_normalize=l2_normalize_feature,
            out_ch=64,
        )
        self.context = ContextUNet14Up(
            norm_layer=norm_layer,
            out_ch_1_4=context_ch + hidden_ch,
            out_ch_1x=context_ch,
            use_aspp=use_ctx_aspp,
        )
        self.update = UpdateBlock(
            hidden_ch=hidden_ch,
            context_ch=context_ch,
            k_per_level=k_per_level,
            levels=levels,
            norm=norm_layer,
        )
        if use_dummy_head:
            self.pose_head = DummyMultiTaskHead(num_classes=num_classes, emb_dim=emb_dim)
        else:
            self.pose_head = LiteFPNMultiTaskHeadNoHiddenWithAffEmb(
                norm_layer=norm_layer,
                num_classes=num_classes,
                ctx_ch=context_ch,
                use_pixelshuffle=True,
                rot_repr=rot_repr,
                emb_dim=emb_dim,
                head_c4=head_c4,
                head_c8=head_c8,
                head_fuse_ch=head_fuse_ch,
                head_geo_ch=head_geo_ch,
                head_sem_ch=head_sem_ch,
                head_inst_ch=head_inst_ch,
            )
        self.upsampler = ConvexUpsampler(hidden_ch=context_ch, up_factor=4, groups=16)

    def extract(self, stereo: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert stereo.size(1) == 6
        B = stereo.size(0)
        left, right = stereo[:, :3], stereo[:, 3:]
        both = torch.cat([left, right], dim=0)
        feats = self.feature(both)
        featL, featR = feats[:B], feats[B:]
        ctx_1_4, ctx_1x = self.context(left)
        hidden0 = torch.tanh(ctx_1_4[:, : self.hidden_ch])
        context0 = F.relu(ctx_1_4[:, self.hidden_ch :], inplace=False)
        return {
            "featL_1_4": featL,
            "featR_1_4": featR,
            "hidden0": hidden0,
            "context0": context0,
            "context_1x": ctx_1x,
        }

    def forward(
        self,
        stereo: torch.Tensor,
        K_pair_1x: torch.Tensor,
        baseline_mm: Union[float, torch.Tensor],
        iters: int = 8,
        disp_init: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        stuff = self.extract(stereo)
        featL, featR = stuff["featL_1_4"], stuff["featR_1_4"]
        hidden, context0 = stuff["hidden0"], stuff["context0"]
        B, _, H4, W4 = featL.shape

        corr_full = torch.einsum("bchi,bchj->bjhi", featL, featR)
        corr_pyr = build_corr_pyramid(corr_full, num_levels=self.levels)

        sc_disp = torch.zeros(B, 1, H4, W4, device=featL.device, dtype=featL.dtype) if disp_init is None else disp_init
        disp_preds: List[torch.Tensor] = []
        disp_log_var_preds: List[torch.Tensor] = []

        for _ in range(iters):
            if self.lookup_mode == "1d":
                corr_nb = sample_corr_pyramid_bilinear(corr_pyr, disp_1_4=sc_disp, radius=self.radius_w)
            else:
                corr_nb = sample_corr_pyramid_bilinear_2d(
                    corr_pyr, disp_1_4=sc_disp, radius_w=self.radius_w, radius_h=self.radius_h
                )
            hidden, delta, disp_log_var = self.update(hidden, context0, corr_nb, sc_disp)
            disp = sc_disp + delta
            disp_preds.append(disp)
            disp_log_var_preds.append(disp_log_var)
            sc_disp = disp.detach()

        point_map_cur, K_pair_14, point_map_conf = disparity_to_pointmap_from_Kpair_with_conf(
            disp_preds[-1],
            disp_log_var_preds[-1],
            K_pair_1x=K_pair_1x,
            baseline_mm=baseline_mm,
            downsample=4,
            eps=1e-6,
        )
        point_map_cur = point_map_cur * 0.001
        depth_1_4 = point_map_cur[:, 2:3]
        disp_1x = self.upsampler(disp_preds[-1], context0, scale_factor=4.0)
        disp_log_var_1x = self.upsampler(disp_log_var_preds[-1], context0, scale_factor=1.0)
        point_map_1x, _, point_map_conf_1x = disparity_to_pointmap_from_Kpair_with_conf(
            disp_1x,
            disp_log_var_1x,
            K_pair_1x=K_pair_1x,
            baseline_mm=baseline_mm,
            downsample=1,
            eps=1e-6,
        )
        point_map_1x = point_map_1x * 0.001
        point_map_1x_norm = normalize_point_map(point_map_1x)

        head_out = self.pose_head(
            ctx_1x=stuff["context_1x"],
            mask_1x=point_map_conf_1x,
            point_map_1x=point_map_1x_norm,
        )
        pos_mu_norm = head_out["pos_mu"]
        pos_logvar_norm = head_out["pos_logvar"]
        pos_mu = denormalize_pos_mu(pos_mu_norm)
        pos_logvar = denormalize_pos_logvar(pos_logvar_norm, pos_mu_norm)

        out: Dict[str, torch.Tensor] = {
            **stuff,
            "disp_preds": disp_preds,
            "disp_log_var_preds": disp_log_var_preds,
            "depth_1_4": depth_1_4,
            "left_mask_1_4": point_map_conf,
            "point_map_conf": point_map_conf,
            "disp_1x": disp_1x,
            "disp_log_var_1x": disp_log_var_1x,
            "point_map_1x": point_map_1x,
            "point_map_conf_1x": point_map_conf_1x,
            "sem_logits": head_out["cls_logits"],
            "pos_mu": pos_mu,
            "pos_logvar": pos_logvar,
            "pos_mu_norm": pos_mu_norm,
            "pos_logvar_norm": pos_logvar_norm,
        }
        out.update({k: v for k, v in head_out.items() if k not in ("pos_mu", "pos_logvar")})
        return out


def normalize_point_map(point_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize XYZ point map to (X/Z, Y/Z, logZ)."""
    z = point_map[:, 2:3].clamp_min(eps)
    xz = point_map[:, 0:1] / z
    yz = point_map[:, 1:2] / z
    logz = torch.log(z)
    return torch.cat([xz, yz, logz], dim=1)


def denormalize_point_map(point_map_norm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Recover XYZ point map from (X/Z, Y/Z, logZ)."""
    z = torch.exp(point_map_norm[:, 2:3]).clamp_min(eps)
    x = point_map_norm[:, 0:1] * z
    y = point_map_norm[:, 1:2] * z
    return torch.cat([x, y, z], dim=1)


def normalize_pos_mu(pos_mu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize (dx, dy, Z) into (dx, dy, logZ)."""
    z = pos_mu[:, 2:3].clamp_min(eps)
    logz = torch.log(z)
    return torch.cat([pos_mu[:, 0:1], pos_mu[:, 1:2], logz], dim=1)


def denormalize_pos_mu(pos_mu_norm: torch.Tensor) -> torch.Tensor:
    """Recover (dx, dy, Z) from (dx, dy, logZ)."""
    z = torch.exp(pos_mu_norm[:, 2:3])
    return torch.cat([pos_mu_norm[:, 0:1], pos_mu_norm[:, 1:2], z], dim=1)


def denormalize_pos_logvar(pos_logvar_norm: torch.Tensor, pos_mu_norm: torch.Tensor) -> torch.Tensor:
    """Approximate log-variance for (dx, dy, Z) from (dx, dy, logZ)."""
    logz = pos_mu_norm[:, 2:3]
    lv_z = pos_logvar_norm[:, 2:3] + 2.0 * logz
    return torch.cat([pos_logvar_norm[:, 0:1], pos_logvar_norm[:, 1:2], lv_z], dim=1)
