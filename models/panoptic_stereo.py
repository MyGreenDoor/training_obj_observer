"""Panoptic stereo model with disparity + multi-task heads."""
import math
from typing import Dict, Callable, List, Union, Tuple, Optional, Sequence

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
from models.sdf_pose_net import SDFPoseDeltaNet, SDFVolumeDecoder
from utils import rot_utils
from utils.normalization_utils import normalize_point_map, denormalize_pos_mu, denormalize_pos_logvar


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
    """LiteFPN-style multi-task head with U-Net pyramid and aff/emb outputs."""

    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        num_classes: int,
        ctx_ch: int = 128,
        use_pixelshuffle: bool = True,
        rot_repr: str = "r6d",
        out_pos_scale: float = 1.0,
        emb_dim: int = 8,
        head_base_ch: int = 128,
        head_ch_scale: float = 1.35,
        head_downsample: int = 4,
    ) -> None:
        super().__init__()
        self.use_pixelshuffle = use_pixelshuffle
        self.rot_repr = rot_repr.lower()
        self.head_downsample = int(head_downsample)
        if self.head_downsample < 1:
            raise ValueError("head_downsample must be >= 1")

        in_ch = ctx_ch + 1 + 3
        base_ch = int(head_base_ch)
        ch_scale = float(head_ch_scale)
        fuse_ch = base_ch * 2
        geo_ch = base_ch
        sem_ch = base_ch
        inst_ch = base_ch

        def scaled_ch(level: int) -> int:
            """Compute stage channels from base and scale."""
            return max(8, int(round(base_ch * (ch_scale ** level))))

        c1 = scaled_ch(0)
        c2 = scaled_ch(1)
        c3 = scaled_ch(2)
        c4 = scaled_ch(3)
        c5 = scaled_ch(4)
        c6 = scaled_ch(5)

        self.enc1 = ConvBlock(in_ch, c1, norm_layer)
        self.down1 = Bottleneck(c1, c1 // 2, c2, norm_layer, s=2)
        self.down2 = Bottleneck(c2, c2 // 2, c3, norm_layer, s=2)
        self.down3 = Bottleneck(c3, c3 // 2, c4, norm_layer, s=2)
        self.down4 = Bottleneck(c4, c4 // 2, c5, norm_layer, s=2)
        self.down5 = Bottleneck(c5, c5 // 2, c6, norm_layer, s=2)

        self.bridge = ASPPLite(c6, norm_layer)

        self.up5 = self.make_up(c6, c5, norm_layer)
        self.dec5 = ConvBlock(c5 + c5, c5, norm_layer)
        self.up4 = self.make_up(c5, c4, norm_layer)
        self.dec4 = ConvBlock(c4 + c4, c4, norm_layer)
        self.up3 = self.make_up(c4, c3, norm_layer)
        self.dec3 = ConvBlock(c3 + c3, c3, norm_layer)
        self.up2 = self.make_up(c3, c2, norm_layer)
        self.dec2 = ConvBlock(c2 + c2, c2, norm_layer)
        self.up1 = self.make_up(c2, c1, norm_layer)
        self.dec1 = ConvBlock(c1 + c1, fuse_ch, norm_layer)

        self.detail_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(in_ch, fuse_ch, 1, bias=False),
            norm_layer(fuse_ch),
            nn.SiLU(True),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 1, bias=True),
            nn.Sigmoid(),
        )

        rot_ch = 6 if self.rot_repr == "r6d" else 3
        self.neck_geo = self.make_neck(fuse_ch, geo_ch, norm_layer)
        self.neck_sem = self.make_neck(fuse_ch, sem_ch, norm_layer)
        self.neck_inst = self.make_neck(fuse_ch, inst_ch, norm_layer)

        self.head_posz_pre = ConvBlock(geo_ch, geo_ch, norm_layer)
        self.head_posz = nn.Conv2d(geo_ch, 6, 3, padding=1)
        self.head_rot_pre = ConvBlock(geo_ch, geo_ch, norm_layer)
        self.head_rot = nn.Conv2d(geo_ch, rot_ch + 1, 3, padding=1)
        self.head_cls_pre = ConvBlock(sem_ch, sem_ch, norm_layer)
        self.head_cls = nn.Conv2d(sem_ch, num_classes, 1)
        self.affemb_head = AffEmbHead(inst_ch, emb_dim, norm_layer)
        self.out_pos_scale = out_pos_scale

    def make_up(self, in_ch: int, out_ch: int, norm_layer: Callable[[int], nn.Module]) -> nn.Module:
        """Create a 2x upsampling block with optional pixel shuffle."""
        if self.use_pixelshuffle:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.SiLU(True),
            )
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv3x3(in_ch, out_ch),
            norm_layer(out_ch),
            nn.SiLU(True),
        )

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
        out_hw = ctx_1x.shape[-2:]
        if self.head_downsample > 1:
            low_hw = (max(1, out_hw[0] // self.head_downsample), max(1, out_hw[1] // self.head_downsample))
            ctx_1x = F.interpolate(ctx_1x, size=low_hw, mode="bilinear", align_corners=False)
            mask_1x = F.interpolate(mask_1x, size=low_hw, mode="bilinear", align_corners=False)
            point_map_1x = F.interpolate(point_map_1x, size=low_hw, mode="bilinear", align_corners=False)

        x4_in = torch.cat((ctx_1x, mask_1x, point_map_1x), dim=1)
        detail = x4_in - self.detail_pool(x4_in)
        e1 = self.enc1(x4_in)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        e6 = self.down5(e5)

        b = self.bridge(e6)

        u5 = self.up5(b)
        if u5.shape[-2:] != e5.shape[-2:]:
            u5 = F.interpolate(u5, size=e5.shape[-2:], mode="bilinear", align_corners=False)
        d5 = self.dec5(torch.cat((u5, e5), dim=1))

        u4 = self.up4(d5)
        if u4.shape[-2:] != e4.shape[-2:]:
            u4 = F.interpolate(u4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat((u4, e4), dim=1))

        u3 = self.up3(d4)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat((u3, e3), dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((u2, e2), dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((u1, e1), dim=1))
        detail = self.detail_proj(detail)
        x = x + detail * self.detail_gate(x)
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        x_inst = self.neck_inst(x)

        out_posz = self.head_posz(self.head_posz_pre(x_geo))
        out_rot = self.head_rot(self.head_rot_pre(x_geo))
        cls_logits = self.head_cls(self.head_cls_pre(x_sem))
        affemb_out = self.affemb_head(x_inst)
        if self.head_downsample > 1:
            out_posz = F.interpolate(out_posz, size=out_hw, mode="bilinear", align_corners=False)
            out_rot = F.interpolate(out_rot, size=out_hw, mode="bilinear", align_corners=False)
            cls_logits = F.interpolate(cls_logits, size=out_hw, mode="bilinear", align_corners=False)
            aff_logits = F.interpolate(affemb_out["aff_logits"], size=out_hw, mode="bilinear", align_corners=False)
            emb = F.interpolate(affemb_out["emb"], size=out_hw, mode="bilinear", align_corners=False)
            emb = F.normalize(emb, dim=1, eps=1e-6)
            affemb_out = {"aff_logits": aff_logits, "emb": emb}

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

        return {
            "pos_mu": mu_pos,
            "pos_logvar": lv_pos,
            "rot_mat": rot_R,
            "rot_logvar_theta": logvar_theta,
            "cls_logits": cls_logits,
            **affemb_out,
        }


class LiteFPNMultiTaskHeadNoRotWithAffEmbLatent(nn.Module):
    """LiteFPN-style head for translation plus latent map (no rotation outputs)."""

    def __init__(
        self,
        norm_layer: Callable[[int], nn.Module],
        num_classes: int,
        ctx_ch: int = 128,
        use_pixelshuffle: bool = True,
        emb_dim: int = 8,
        latent_dim: int = 16,
        latent_l2_norm: bool = True,
        head_base_ch: int = 128,
        head_ch_scale: float = 1.35,
        head_downsample: int = 4,
        out_pos_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_pixelshuffle = use_pixelshuffle
        self.head_downsample = int(head_downsample)
        self.latent_l2_norm = bool(latent_l2_norm)
        if self.head_downsample < 1:
            raise ValueError("head_downsample must be >= 1")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        in_ch = ctx_ch + 1 + 3
        base_ch = int(head_base_ch)
        ch_scale = float(head_ch_scale)
        fuse_ch = base_ch * 2
        sem_ch = base_ch
        inst_ch = base_ch

        def scaled_ch(level: int) -> int:
            return max(8, int(round(base_ch * (ch_scale ** level))))

        c1 = scaled_ch(0)
        c2 = scaled_ch(1)
        c3 = scaled_ch(2)
        c4 = scaled_ch(3)
        c5 = scaled_ch(4)
        c6 = scaled_ch(5)

        self.enc1 = ConvBlock(in_ch, c1, norm_layer)
        self.down1 = Bottleneck(c1, c1 // 2, c2, norm_layer, s=2)
        self.down2 = Bottleneck(c2, c2 // 2, c3, norm_layer, s=2)
        self.down3 = Bottleneck(c3, c3 // 2, c4, norm_layer, s=2)
        self.down4 = Bottleneck(c4, c4 // 2, c5, norm_layer, s=2)
        self.down5 = Bottleneck(c5, c5 // 2, c6, norm_layer, s=2)

        self.bridge = ASPPLite(c6, norm_layer)

        self.up5 = self.make_up(c6, c5, norm_layer)
        self.dec5 = ConvBlock(c5 + c5, c5, norm_layer)
        self.up4 = self.make_up(c5, c4, norm_layer)
        self.dec4 = ConvBlock(c4 + c4, c4, norm_layer)
        self.up3 = self.make_up(c4, c3, norm_layer)
        self.dec3 = ConvBlock(c3 + c3, c3, norm_layer)
        self.up2 = self.make_up(c3, c2, norm_layer)
        self.dec2 = ConvBlock(c2 + c2, c2, norm_layer)
        self.up1 = self.make_up(c2, c1, norm_layer)
        self.dec1 = ConvBlock(c1 + c1, fuse_ch, norm_layer)

        self.detail_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(in_ch, fuse_ch, 1, bias=False),
            norm_layer(fuse_ch),
            nn.SiLU(True),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 1, bias=True),
            nn.Sigmoid(),
        )

        self.neck_geo = self.make_neck(fuse_ch, base_ch, norm_layer)
        self.neck_sem = self.make_neck(fuse_ch, sem_ch, norm_layer)
        self.neck_inst = self.make_neck(fuse_ch, inst_ch, norm_layer)

        self.head_posz_pre = ConvBlock(base_ch, base_ch, norm_layer)
        self.head_posz = nn.Conv2d(base_ch, 6, 3, padding=1)
        self.head_cls_pre = ConvBlock(sem_ch, sem_ch, norm_layer)
        self.head_cls = nn.Conv2d(sem_ch, num_classes, 1)
        self.affemb_head = AffEmbHead(inst_ch, emb_dim, norm_layer)
        self.latent_head = nn.Conv2d(inst_ch, latent_dim, 1)
        self.sdf_head = nn.Conv2d(latent_dim, 1, 1)
        self.sdf_logvar_head = nn.Conv2d(inst_ch, 1, 1)
        self.out_pos_scale = out_pos_scale

    def make_up(self, in_ch: int, out_ch: int, norm_layer: Callable[[int], nn.Module]) -> nn.Module:
        """Create a 2x upsampling block with optional pixel shuffle."""
        if self.use_pixelshuffle:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.SiLU(True),
            )
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv3x3(in_ch, out_ch),
            norm_layer(out_ch),
            nn.SiLU(True),
        )

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

    def forward(
        self,
        ctx_1x: torch.Tensor,
        mask_1x: torch.Tensor,
        point_map_1x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run FPN-style head on full resolution features."""
        out_hw = ctx_1x.shape[-2:]
        if self.head_downsample > 1:
            low_hw = (max(1, out_hw[0] // self.head_downsample), max(1, out_hw[1] // self.head_downsample))
            ctx_1x = F.interpolate(ctx_1x, size=low_hw, mode="bilinear", align_corners=False)
            mask_1x = F.interpolate(mask_1x, size=low_hw, mode="bilinear", align_corners=False)
            point_map_1x = F.interpolate(point_map_1x, size=low_hw, mode="bilinear", align_corners=False)

        x4_in = torch.cat((ctx_1x, mask_1x, point_map_1x), dim=1)
        detail = x4_in - self.detail_pool(x4_in)
        e1 = self.enc1(x4_in)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        e6 = self.down5(e5)

        b = self.bridge(e6)

        u5 = self.up5(b)
        if u5.shape[-2:] != e5.shape[-2:]:
            u5 = F.interpolate(u5, size=e5.shape[-2:], mode="bilinear", align_corners=False)
        d5 = self.dec5(torch.cat((u5, e5), dim=1))

        u4 = self.up4(d5)
        if u4.shape[-2:] != e4.shape[-2:]:
            u4 = F.interpolate(u4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat((u4, e4), dim=1))

        u3 = self.up3(d4)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat((u3, e3), dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((u2, e2), dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((u1, e1), dim=1))
        detail = self.detail_proj(detail)
        x = x + detail * self.detail_gate(x)
        x_geo = self.neck_geo(x)
        x_sem = self.neck_sem(x)
        x_inst = self.neck_inst(x)

        out_posz = self.head_posz(self.head_posz_pre(x_geo))
        cls_logits = self.head_cls(self.head_cls_pre(x_sem))
        affemb_out = self.affemb_head(x_inst)
        latent_map = self.latent_head(x_inst)
        sdf_map = self.sdf_head(latent_map)
        sdf_logvar = self.sdf_logvar_head(x_inst)

        if self.head_downsample > 1:
            out_posz = F.interpolate(out_posz, size=out_hw, mode="bilinear", align_corners=False)
            cls_logits = F.interpolate(cls_logits, size=out_hw, mode="bilinear", align_corners=False)
            aff_logits = F.interpolate(affemb_out["aff_logits"], size=out_hw, mode="bilinear", align_corners=False)
            emb = F.interpolate(affemb_out["emb"], size=out_hw, mode="bilinear", align_corners=False)
            latent_map = F.interpolate(latent_map, size=out_hw, mode="bilinear", align_corners=False)
            sdf_map = F.interpolate(sdf_map, size=out_hw, mode="bilinear", align_corners=False)
            sdf_logvar = F.interpolate(sdf_logvar, size=out_hw, mode="bilinear", align_corners=False)
            emb = F.normalize(emb, dim=1, eps=1e-6)
            affemb_out = {"aff_logits": aff_logits, "emb": emb}

        mu_raw = out_posz[:, 0:3]
        lv_raw = out_posz[:, 3:6]
        log_z = mu_raw[:, 2:3] * self.out_pos_scale
        mu_pos = torch.cat([mu_raw[:, 0:2], log_z], dim=1)
        lv_pos = lv_raw.clamp(-8.0, 4.0)
        sdf_logvar = sdf_logvar.clamp(-8.0, 4.0)

        if self.latent_l2_norm:
            latent_map = F.normalize(latent_map, dim=1, eps=1e-6)

        return {
            "pos_mu": mu_pos,
            "pos_logvar": lv_pos,
            "cls_logits": cls_logits,
            "latent_map": latent_map,
            "sdf_map": sdf_map,
            "sdf_logvar": sdf_logvar,
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

    def __init__(self, context_ch: int, up_factor: int = 4, groups: int = 16) -> None:
        super().__init__()
        self.s = int(up_factor)
        self.mask_head = nn.Sequential(
            conv3x3(context_ch, context_ch),
            nn.GroupNorm(groups, context_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(context_ch, 9 * (self.s * self.s), kernel_size=1, bias=True),
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
        head_base_ch: int = 96,
        head_ch_scale: float = 1.35,
        head_downsample: int = 4,
        point_map_norm_mean: Optional[Sequence[float]] = None,
        point_map_norm_std: Optional[Sequence[float]] = None,
        point_map_norm_eps: float = 1e-6,
        sdf_pose_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.lookup_mode = lookup_mode.lower()
        self.hidden_ch = int(hidden_ch)
        self.context_ch = int(context_ch)
        if self.context_ch % 4 != 0:
            raise ValueError("context_ch must be divisible by 4 to build context_1x")
        self.context_1x_ch = self.context_ch // 4

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
            out_ch_1x=self.context_1x_ch,
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
                ctx_ch=self.context_1x_ch,
                use_pixelshuffle=True,
                rot_repr=rot_repr,
                emb_dim=emb_dim,
                head_base_ch=head_base_ch,
                head_ch_scale=head_ch_scale,
                head_downsample=head_downsample,
            )
        self.upsampler = ConvexUpsampler(context_ch=context_ch, up_factor=4, groups=16)
        self.use_ctx1x_for_upsample = True
        up_guidance_in_ch = self.context_ch + (self.context_1x_ch * 16)
        self.up_guidance_fuse = nn.Sequential(
            nn.Conv2d(up_guidance_in_ch, context_ch, kernel_size=1, bias=False),
            nn.GroupNorm(16, context_ch),
            nn.SiLU(inplace=True),
        )
        self.point_map_norm_eps = float(point_map_norm_eps)
        self.use_point_map_norm = point_map_norm_mean is not None and point_map_norm_std is not None
        if self.use_point_map_norm:
            mean = torch.as_tensor(point_map_norm_mean, dtype=torch.float32)
            std = torch.as_tensor(point_map_norm_std, dtype=torch.float32)
            if mean.numel() != 3 or std.numel() != 3:
                raise ValueError("point_map_norm_mean/std must have 3 values for (X/Z, Y/Z, logZ).")
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1).clamp_min(self.point_map_norm_eps)
            self.register_buffer("point_map_norm_mean", mean)
            self.register_buffer("point_map_norm_std", std)
        else:
            self.point_map_norm_mean = None
            self.point_map_norm_std = None

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
        disp_init: Optional[torch.Tensor] = None,
        *,
        Wk_1_4: Optional[torch.Tensor] = None,
        wfg_1_4: Optional[torch.Tensor] = None,
        min_px: int = 10,
        min_wsum: float = 1e-6,
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
        depth_1_4 = point_map_cur[:, 2:3]
        if self.use_ctx1x_for_upsample:
            ctx1x = stuff["context_1x"]  # (B, context_ch/16, H4*4, W4*4)
            # (B, context_ch, H4, W4)
            ctx1x_s2d = F.pixel_unshuffle(ctx1x, 4)
            # fuse (B, context_ch + context_ch, H4, W4) -> (B, context_ch, H4, W4)
            h_up = self.up_guidance_fuse(torch.cat([context0, ctx1x_s2d], dim=1))
        else:
            h_up = context0

        # use fused guidance for convex upsampling mask
        disp_1x = self.upsampler(disp_preds[-1], h_up, scale_factor=4.0)
        disp_log_var_1x = self.upsampler(disp_log_var_preds[-1], h_up, scale_factor=1.0) + (2.0 * math.log(4.0))
        point_map_1x, _, point_map_conf_1x = disparity_to_pointmap_from_Kpair_with_conf(
            disp_1x,
            disp_log_var_1x,
            K_pair_1x=K_pair_1x,
            baseline_mm=baseline_mm,
            downsample=1,
            eps=1e-6,
        )
        point_map_1x_norm = normalize_point_map(
            point_map_1x,
            eps=self.point_map_norm_eps,
            mean=self.point_map_norm_mean,
            std=self.point_map_norm_std,
        )

        head_out = self.pose_head(
            ctx_1x=stuff["context_1x"],
            mask_1x=point_map_conf_1x,
            point_map_1x=point_map_1x_norm,
        )
        pos_mu_norm = head_out["pos_mu"]
        pos_logvar_norm = head_out["pos_logvar"]
        pos_mu = denormalize_pos_mu(pos_mu_norm)
        pos_logvar = denormalize_pos_logvar(pos_logvar_norm, pos_mu_norm)

        if torch.jit.is_scripting():
            out = torch.jit.annotate(Dict[str, torch.Tensor], {})
            out["featL_1_4"] = featL
            out["featR_1_4"] = featR
            out["hidden0"] = hidden
            out["context0"] = context0
            out["context_1x"] = stuff["context_1x"]
            out["depth_1_4"] = depth_1_4
            out["left_mask_1_4"] = point_map_conf
            out["point_map_conf"] = point_map_conf
            out["disp_1x"] = disp_1x
            out["disp_log_var_1x"] = disp_log_var_1x
            out["point_map_1x"] = point_map_1x
            out["point_map_conf_1x"] = point_map_conf_1x
            out["sem_logits"] = head_out["cls_logits"]
            out["pos_mu"] = pos_mu
            out["pos_logvar"] = pos_logvar
            out["pos_mu_norm"] = pos_mu_norm
            out["pos_logvar_norm"] = pos_logvar_norm
            out["mask_logits"] = head_out["mask_logits"]
            out["center_logits"] = head_out["center_logits"]
            out["rot_mat"] = head_out["rot_mat"]
            out["rot_logvar_theta"] = head_out["rot_logvar_theta"]
            out["aff_logits"] = head_out["aff_logits"]
            out["emb"] = head_out["emb"]
            if Wk_1_4 is not None and wfg_1_4 is not None:
                pos_map_pred = pos_mu_to_pointmap(pos_mu, K_pair_1x[:, 0], downsample=1)
                r_pred, t_pred, v_pred, _, _ = rot_utils.pose_from_maps_auto(
                    rot_map=head_out["rot_mat"],
                    pos_map=pos_map_pred,
                    Wk_1_4=Wk_1_4,
                    wfg=wfg_1_4,
                    min_px=min_px,
                    min_wsum=min_wsum,
                )
                out["pose_R"] = r_pred
                out["pose_t"] = t_pred
                out["pose_valid"] = v_pred
            return out

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

        if Wk_1_4 is not None and wfg_1_4 is not None:
            pos_map_pred = pos_mu_to_pointmap(pos_mu, K_pair_1x[:, 0], downsample=1)
            r_pred, t_pred, v_pred, _, _ = rot_utils.pose_from_maps_auto(
                rot_map=head_out["rot_mat"],
                pos_map=pos_map_pred,
                Wk_1_4=Wk_1_4,
                wfg=wfg_1_4,
                min_px=min_px,
                min_wsum=min_wsum,
            )
            out["pose_R"] = r_pred
            out["pose_t"] = t_pred
            out["pose_valid"] = v_pred
        return out


class PanopticStereoMultiHeadLatent(PanopticStereoMultiHead):
    """Panoptic stereo model with latent map head (no rotation outputs)."""

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
        latent_dim: int = 16,
        latent_l2_norm: bool = True,
        head_base_ch: int = 96,
        head_ch_scale: float = 1.35,
        head_downsample: int = 4,
        point_map_norm_mean: Optional[Sequence[float]] = None,
        point_map_norm_std: Optional[Sequence[float]] = None,
        point_map_norm_eps: float = 1e-6,
        sdf_pose_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            levels=levels,
            norm_layer=norm_layer,
            l2_normalize_feature=l2_normalize_feature,
            use_ctx_aspp=use_ctx_aspp,
            lookup_mode=lookup_mode,
            radius_w=radius_w,
            radius_h=radius_h,
            hidden_ch=hidden_ch,
            context_ch=context_ch,
            num_classes=num_classes,
            rot_repr=rot_repr,
            emb_dim=emb_dim,
            use_dummy_head=False,
            head_base_ch=head_base_ch,
            head_ch_scale=head_ch_scale,
            head_downsample=head_downsample,
            point_map_norm_mean=point_map_norm_mean,
            point_map_norm_std=point_map_norm_std,
            point_map_norm_eps=point_map_norm_eps,
        )
        self.pose_head = LiteFPNMultiTaskHeadNoRotWithAffEmbLatent(
            norm_layer=norm_layer,
            num_classes=num_classes,
            ctx_ch=self.context_1x_ch,
            use_pixelshuffle=True,
            emb_dim=emb_dim,
            latent_dim=latent_dim,
            latent_l2_norm=latent_l2_norm,
            head_base_ch=head_base_ch,
            head_ch_scale=head_ch_scale,
            head_downsample=head_downsample,
            out_pos_scale=1.0,
        )
        sdf_pose_cfg = sdf_pose_cfg or {}
        self.sdf_pose_net = SDFPoseDeltaNet(
            in_ch=int(sdf_pose_cfg.get("in_ch", 2)),
            base_ch=int(sdf_pose_cfg.get("base_ch", 16)),
            num_down=int(sdf_pose_cfg.get("num_down", 4)),
            hidden_ch=int(sdf_pose_cfg.get("hidden_ch", 128)),
            out_scale_rot=float(sdf_pose_cfg.get("out_scale_rot", 0.5)),
            out_scale_trans=float(sdf_pose_cfg.get("out_scale_trans", 0.02)),
        )
        self.sdf_decoder = SDFVolumeDecoder(
            latent_dim=latent_dim,
            base_ch=int(sdf_pose_cfg.get("decoder_base_ch", 32)),
            base_res=int(sdf_pose_cfg.get("decoder_base_res", 4)),
        )

    def forward(
        self,
        stereo: torch.Tensor,
        K_pair_1x: torch.Tensor,
        baseline_mm: Union[float, torch.Tensor],
        iters: int = 8,
        disp_init: Optional[torch.Tensor] = None,
        *,
        Wk_1_4: Optional[torch.Tensor] = None,
        wfg_1_4: Optional[torch.Tensor] = None,
        min_px: int = 10,
        min_wsum: float = 1e-6,
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
        depth_1_4 = point_map_cur[:, 2:3]
        if self.use_ctx1x_for_upsample:
            ctx1x = stuff["context_1x"]
            ctx1x_s2d = F.pixel_unshuffle(ctx1x, 4)
            h_up = self.up_guidance_fuse(torch.cat([context0, ctx1x_s2d], dim=1))
        else:
            h_up = context0

        disp_1x = self.upsampler(disp_preds[-1], h_up, scale_factor=4.0)
        disp_log_var_1x = self.upsampler(disp_log_var_preds[-1], h_up, scale_factor=1.0) + (2.0 * math.log(4.0))
        point_map_1x, _, point_map_conf_1x = disparity_to_pointmap_from_Kpair_with_conf(
            disp_1x,
            disp_log_var_1x,
            K_pair_1x=K_pair_1x,
            baseline_mm=baseline_mm,
            downsample=1,
            eps=1e-6,
        )
        point_map_1x_norm = normalize_point_map(
            point_map_1x,
            eps=self.point_map_norm_eps,
            mean=self.point_map_norm_mean,
            std=self.point_map_norm_std,
        )

        head_out = self.pose_head(
            ctx_1x=stuff["context_1x"],
            mask_1x=point_map_conf_1x,
            point_map_1x=point_map_1x_norm,
        )
        pos_mu_norm = head_out["pos_mu"]
        pos_logvar_norm = head_out["pos_logvar"]
        pos_mu = denormalize_pos_mu(pos_mu_norm)
        pos_logvar = denormalize_pos_logvar(pos_logvar_norm, pos_mu_norm)

        if torch.jit.is_scripting():
            out = torch.jit.annotate(Dict[str, torch.Tensor], {})
            out["featL_1_4"] = featL
            out["featR_1_4"] = featR
            out["hidden0"] = hidden
            out["context0"] = context0
            out["context_1x"] = stuff["context_1x"]
            out["depth_1_4"] = depth_1_4
            out["left_mask_1_4"] = point_map_conf
            out["point_map_conf"] = point_map_conf
            out["disp_1x"] = disp_1x
            out["disp_log_var_1x"] = disp_log_var_1x
            out["point_map_1x"] = point_map_1x
            out["point_map_conf_1x"] = point_map_conf_1x
            out["sem_logits"] = head_out["cls_logits"]
            out["pos_mu"] = pos_mu
            out["pos_logvar"] = pos_logvar
            out["pos_mu_norm"] = pos_mu_norm
            out["pos_logvar_norm"] = pos_logvar_norm
            out["cls_logits"] = head_out["cls_logits"]
            out["aff_logits"] = head_out["aff_logits"]
            out["emb"] = head_out["emb"]
            out["latent_map"] = head_out["latent_map"]
            out["sdf_map"] = head_out["sdf_map"]
            out["sdf_logvar"] = head_out["sdf_logvar"]
            return out

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

    def predict_sdf_pose_delta(self, sdf_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict pose deltas from paired SDF volumes."""
        return self.sdf_pose_net(sdf_pairs)

def pos_mu_to_pointmap(
    pos_mu: torch.Tensor,
    K_left_1x: torch.Tensor,
    downsample: int = 4,
) -> torch.Tensor:
    """Convert (dx, dy, Z) or (Z) to an XYZ point map at 1/4 resolution."""
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

    fx = K_left_1x[:, 0, 0].view(B, 1, 1, 1)
    fy = K_left_1x[:, 1, 1].view(B, 1, 1, 1)
    cx = K_left_1x[:, 0, 2].view(B, 1, 1, 1)
    cy = K_left_1x[:, 1, 2].view(B, 1, 1, 1)

    u_c = u + dx
    v_c = v + dy
    X = (u_c - cx) / fx * z
    Y = (v_c - cy) / fy * z
    return torch.cat([X, Y, z], dim=1)
