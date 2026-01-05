"""Lightweight stereo network for disparity and optional pose estimation."""

from typing import Dict

import torch
import torch.nn as nn


class SimpleStereoNet(nn.Module):
    """Small CNN that estimates disparity and optionally a 6D pose.

    Args:
        in_channels: Number of input channels (expects concatenated left/right RGB).
        features: Base channel width for the encoder/decoder.
        predict_pose: If ``True``, also outputs a pose vector ``(rx, ry, rz, tx, ty, tz, conf)``.
    """

    def __init__(self, in_channels: int = 6, features: int = 32, predict_pose: bool = True):
        super().__init__()
        self.predict_pose: bool = bool(predict_pose)
        c: int = int(features)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2*c, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 4*c, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_disp = nn.Sequential(
            nn.ConvTranspose2d(4*c, 2*c, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*c, c, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        if self.predict_pose:
            # Global pooling + MLP head for pose
            self.pose_head: nn.Module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(4*c, 2*c),
                nn.ReLU(inplace=True),
                nn.Linear(2*c, 7),
            )
        else:
            self.pose_head = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass.

        Args:
            x: Stereo pair tensor shaped ``(B, 6, H, W)`` with left/right RGB stacked.

        Returns:
            Dict[str, torch.Tensor]: ``{"disp": disp}`` and, if enabled, ``{"pose": pose}``.
        """
        feat = self.encoder(x)
        disp = self.decoder_disp(feat)

        out = torch.jit.annotate(Dict[str, torch.Tensor], {"disp": disp})
        if self.predict_pose:
            pose = self.pose_head(feat)
            out["pose"] = pose
        return out
