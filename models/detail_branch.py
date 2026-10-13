"""高分辨率细节支路（detail branch）。

从原始输入出发、保持较高分辨率、直达预测头前的轻量支路，为 MFS 提供
原生像素级细节（主通路 512x512 输入在 patch embedding 处即压到 64x64
token，MFS 的最细输入也是 64x64 上采样而来）。

结构（输入 (B,3,512,512) → 输出 (B,16,128,128)）：
    Conv3x3(3→16, stride=2) + GroupNorm + SiLU
    → Conv3x3(16→16, stride=2) + GroupNorm + SiLU
"""
import torch
import torch.nn as nn


class DetailBranch(nn.Module):
    def __init__(self, in_channels=3, channels=16, gn_groups=4):
        super().__init__()
        assert channels % gn_groups == 0, 'GroupNorm groups 需整除通道数'
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(gn_groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x
