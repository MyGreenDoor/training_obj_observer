import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleStereoNet(nn.Module):
    """
    A tiny CNN that predicts a disparity map (1xHxW) and, optionally, a 6D pose vector per image (rx, ry, rz, tx, ty, tz).
    Inputs: (B, 6, H, W) concatenated left/right RGB
    """
    def __init__(self, in_channels=6, features=32, predict_pose=True):
        super().__init__()
        self.predict_pose = predict_pose
        c = features

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
            self.pose_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(4*c, 2*c),
                nn.ReLU(inplace=True),
                nn.Linear(2*c, 7),
            )

    def forward(self, x):
        feat = self.encoder(x)
        disp = self.decoder_disp(feat)
        # Non-negative disparity
        # disp = F.relu(disp)

        out = {"disp": disp}
        if self.predict_pose:
            pose = self.pose_head(feat)
            out["pose"] = pose
        return out
