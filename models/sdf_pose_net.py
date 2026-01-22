#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDF pair encoder and decoder utilities."""

import torch
import torch.nn as nn


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm with a safe group count."""
    groups = max(1, min(max_groups, num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SDFPoseDeltaNet(nn.Module):
    """Predict pose delta from paired SDF volumes."""

    def __init__(
        self,
        in_ch: int = 2,
        base_ch: int = 16,
        num_down: int = 4,
        hidden_ch: int = 128,
        out_scale_rot: float = 0.5,
        out_scale_trans: float = 0.02,
    ) -> None:
        super().__init__()
        if num_down < 1:
            raise ValueError("num_down must be >= 1")

        ch = int(base_ch)
        layers = []
        for i in range(num_down):
            in_c = in_ch if i == 0 else ch
            out_c = ch if i == 0 else min(ch * 2, 256)
            layers.extend(
                [
                    nn.Conv3d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                    _make_gn(out_c),
                    nn.SiLU(inplace=True),
                ]
            )
            ch = out_c

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(ch, int(hidden_ch)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_ch), 6),
        )
        self.out_scale_rot = float(out_scale_rot)
        self.out_scale_trans = float(out_scale_trans)

    def forward(self, sdf_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sdf_pair: (N, 2, D, H, W) concatenated SDF volumes.

        Returns:
            rotvec: (N, 3) rotation vectors (axis-angle).
            t_delta: (N, 3) translation deltas in camera frame.
        """
        x = self.encoder(sdf_pair)
        x = self.pool(x).flatten(1)
        out = self.head(x)
        rotvec = torch.tanh(out[:, 0:3]) * self.out_scale_rot
        t_delta = torch.tanh(out[:, 3:6]) * self.out_scale_trans
        return rotvec, t_delta


class SDFVolumeDecoder(nn.Module):
    """Decode a latent vector into a dense SDF volume."""

    def __init__(
        self,
        latent_dim: int = 16,
        base_ch: int = 32,
        base_res: int = 4,
    ) -> None:
        super().__init__()
        if base_res < 2:
            raise ValueError("base_res must be >= 2")
        self.latent_dim = int(latent_dim)
        self.base_ch = int(base_ch)
        self.base_res = int(base_res)
        self.fc = nn.Linear(self.latent_dim, self.base_ch * (self.base_res ** 3))
        self.block = nn.Sequential(
            nn.Conv3d(self.base_ch, self.base_ch, 3, padding=1, bias=False),
            _make_gn(self.base_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.base_ch, 1, 1),
        )

    def forward(self, latent: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
        """
        Args:
            latent: (N, latent_dim)
            out_shape: (D, H, W)

        Returns:
            sdf_vol: (N, 1, D, H, W)
        """
        if latent.numel() == 0:
            return latent.new_empty((0, 1, *out_shape))
        x = self.fc(latent)
        x = x.view(latent.size(0), self.base_ch, self.base_res, self.base_res, self.base_res)
        x = self.block(x)
        x = nn.functional.interpolate(x, size=out_shape, mode="trilinear", align_corners=True)
        return x
