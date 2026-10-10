from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] <= 2 or mask.shape[-2] <= 2:
        return mask
    p1 = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    skeleton = F.relu(mask - soft_open(mask))
    for _ in range(max(int(iterations) - 1, 0)):
        mask = soft_erode(mask)
        delta = F.relu(mask - soft_open(mask))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0, 1)


def skeleton_distance_weights(
    targets: torch.Tensor,
    skeleton_iterations: int = 10,
    radius: int = 8,
    sigma: float = 5.0,
    gain: float = 4.0,
) -> torch.Tensor:
    target_skeleton = soft_skeletonize(targets.float(), iterations=skeleton_iterations).detach()
    weights = torch.ones_like(target_skeleton)
    if radius <= 0 or gain <= 0:
        return weights
    covered = target_skeleton.clamp(0, 1)
    weights = weights + gain * covered
    sigma = max(float(sigma), 1e-6)
    for distance in range(1, int(radius) + 1):
        dilated = soft_dilate(covered)
        ring = F.relu(dilated - covered).clamp(0, 1)
        weights = weights + gain * math.exp(-distance / sigma) * ring
        covered = torch.maximum(covered, dilated)
    return weights.detach()


class SkeletonDistanceLoss(nn.Module):
    def __init__(self, skeleton_iterations: int = 10, radius: int = 8,
                 sigma: float = 5.0, gain: float = 4.0):
        super().__init__()
        self.skeleton_iterations = skeleton_iterations
        self.radius = radius
        self.sigma = sigma
        self.gain = gain

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().clamp(0, 1)
        weights = skeleton_distance_weights(
            targets,
            skeleton_iterations=self.skeleton_iterations,
            radius=self.radius,
            sigma=self.sigma,
            gain=self.gain,
        )
        # 本仓库损失管线吃的是 logits（bce_dice 用 BCEWithLogitsLoss），
        # 因此这里用 with_logits 版本；mamba2seg 原版吃的是概率（BCE）。
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        return (bce * weights).sum() / weights.sum().clamp_min(1.0)
