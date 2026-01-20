from typing import Optional

import torch


def normalize_point_map(
    point_map: torch.Tensor,
    eps: float = 1e-6,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize XYZ point map to (X/Z, Y/Z, logZ), optionally standardizing by mean/std."""
    z = point_map[:, 2:3].clamp_min(eps)
    xz = point_map[:, 0:1] / z
    yz = point_map[:, 1:2] / z
    logz = torch.log(z)
    point_map_norm = torch.cat([xz, yz, logz], dim=1)
    if mean is None or std is None:
        return point_map_norm
    mean = mean.to(device=point_map.device, dtype=point_map.dtype)
    std = std.to(device=point_map.device, dtype=point_map.dtype).clamp_min(eps)
    return (point_map_norm - mean) / std


def denormalize_point_map(
    point_map_norm: torch.Tensor,
    eps: float = 1e-6,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recover XYZ point map from (X/Z, Y/Z, logZ), optionally de-standardizing by mean/std."""
    if mean is not None and std is not None:
        mean = mean.to(device=point_map_norm.device, dtype=point_map_norm.dtype)
        std = std.to(device=point_map_norm.device, dtype=point_map_norm.dtype).clamp_min(eps)
        point_map_norm = point_map_norm * std + mean
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
