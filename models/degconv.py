"""DEGConv（方向引导边缘门控卷积），参照 MixerCSeg (arXiv:2603.01361) 第 3.4 节。

插在每级 encoder stage 输出之后、进入 MFS 之前，为特征注入
"方向先验 + 门控边缘增强"。对单级输入特征 F ∈ R^{B×C×H×W}：

1) 空间分块：F 划分为 2×2 不重叠 view（各 H/2×W/2）；H、W 先 reflect-pad
   到 16 的倍数（保证 view 可被 8×8 cell 整除），输出前裁回原尺寸。
2) 方向嵌入（每个 view 内，全程 no_grad，作为先验特征不需要梯度）：
   通道均值 → Sobel 求 dx/dy → θ = arctan2(dy,dx) 映射到 [0,π) →
   8×8 cell 内统计 n=18 bin 方向直方图（按 bin 中心角加权）→ p ∈ R^{n×h/8×w/8}
   → ε = avgpool(Norm(ReLU(Conv3x3(Conv1x1(p)))))，维度 C。
3) EdgeConv：PW(C→C/2) → 1×7 与 7×1 条形卷积并联 → concat → DW3×3 → PW(C→C)。
4) 门控融合：g = sigmoid(EdgeConv(F_view + ε))；F'_view = g ⊙ EdgeConv(F_view)。
5) view 拼回原空间排布，再过一层独立 EdgeConv 后处理（消除分块边界错位）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS = 18      # [0, π) 均匀划分的方向 bin 数
CELL = 8         # 直方图 cell 尺寸（8×8）
VIEWS = 2        # 空间分块档位（2×2 = 4 个 view）


class EdgeConv(nn.Module):
    """轻量边缘卷积：PW 降维 → 条形卷积并联 → DW → PW 恢复。"""

    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.pw1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.strip_h = nn.Conv2d(mid, mid, kernel_size=(1, 7), padding=(0, 3))
        self.strip_v = nn.Conv2d(mid, mid, kernel_size=(7, 1), padding=(3, 0))
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                            groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([self.strip_h(x), self.strip_v(x)], dim=1)
        x = self.dw(x)
        return self.pw2(x)


class DirectionEmbed(nn.Module):
    """方向先验直方图 → C 维嵌入（Conv1x1 → Conv3x3 → ReLU → Norm → avgpool）。

    调用方在 no_grad 上下文中使用（先验特征不需要梯度）。
    """

    def __init__(self, channels, n_bins=N_BINS):
        super().__init__()
        self.conv1 = nn.Conv2d(n_bins, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(max(1, channels // 16), channels)

    def forward(self, p):
        x = self.conv2(F.relu(self.conv1(p)))
        x = self.norm(x)
        return x.mean(dim=(2, 3), keepdim=True)


class DEGConv(nn.Module):
    """方向引导边缘门控卷积（单级特征版本，输入输出同形状）。"""

    def __init__(self, channels, n_bins=N_BINS, cell=CELL, views=VIEWS):
        super().__init__()
        self.channels = channels
        self.n_bins = n_bins
        self.cell = cell
        self.views = views
        self.dir_embed = DirectionEmbed(channels, n_bins)
        # 门控与内容共用同一个 EdgeConv（论文形式：同一算子作用两次）。
        self.edge = EdgeConv(channels)
        # view 拼回后的后处理 EdgeConv（独立参数，消除分块边界错位）。
        self.edge_post = EdgeConv(channels)
        # Sobel 算子（固定先验，注册为 buffer 随设备迁移，不参与训练）。
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('sobel_x', kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('sobel_y', ky.view(1, 1, 3, 3), persistent=False)
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) * (math.pi / n_bins)
        self.register_buffer('bin_centers', centers, persistent=False)

    def _to_views(self, x):
        """(B,C,H,W) → (B*V*V, C, H/V, W/V)，view 行优先排列。"""
        b, c, h, w = x.shape
        v = self.views
        x = x.reshape(b, c, v, h // v, v, w // v)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(b * v * v, c, h // v, w // v)

    def _from_views(self, x, b, h, w):
        """_to_views 的逆变换。"""
        v = self.views
        c = x.shape[1]
        x = x.reshape(b, v, v, c, h // v, w // v)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    def _direction_histogram(self, view):
        """方向先验直方图 p：(B*V*V, n_bins, h/cell, w/cell)。纯张量操作。"""
        bv, _, h, w = view.shape
        hc, wc = h // self.cell, w // self.cell
        gray = view.mean(dim=1, keepdim=True)
        dx = F.conv2d(gray, self.sobel_x, padding=1)
        dy = F.conv2d(gray, self.sobel_y, padding=1)
        theta = torch.remainder(torch.atan2(dy, dx), math.pi)  # [0, π)
        bin_idx = (theta * (self.n_bins / math.pi)).long().clamp_(0, self.n_bins - 1)
        # 每个 cell 内按 bin 计数（scatter_add），再按 cell 内像素数归一、
        # 以 bin 中心角加权，得到方向先验向量。
        cells = bin_idx.reshape(bv, hc, self.cell, wc, self.cell) \
                       .permute(0, 1, 3, 2, 4).reshape(bv, hc * wc, -1)
        hist = torch.zeros(bv, hc * wc, self.n_bins,
                           device=view.device, dtype=torch.float32)
        hist.scatter_add_(2, cells, torch.ones_like(cells, dtype=torch.float32))
        hist = hist / cells.shape[-1]
        p = hist * self.bin_centers.view(1, 1, -1)
        return p.reshape(bv, hc, wc, self.n_bins).permute(0, 3, 1, 2).contiguous()

    def _direction_embedding(self, view):
        """方向嵌入 ε：(B*V*V, C, 1, 1)。全程 no_grad（先验不需要梯度）。"""
        with torch.no_grad():
            p = self._direction_histogram(view)
            return self.dir_embed(p)

    def forward(self, x):
        b, _, h, w = x.shape
        # H、W pad 到 views*cell 的倍数（reflect），输出前裁回。
        mult = self.views * self.cell
        pad_h = (-h) % mult
        pad_w = (-w) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        hp, wp = x.shape[2], x.shape[3]

        view = self._to_views(x)
        eps = self._direction_embedding(view)
        g = torch.sigmoid(self.edge(view + eps))
        view = g * self.edge(view)
        out = self._from_views(view, b, hp, wp)
        out = self.edge_post(out)

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out
