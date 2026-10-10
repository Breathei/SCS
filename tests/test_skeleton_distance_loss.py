"""
SkeletonDistanceLoss 验收测试。

a) 1px 宽水平直线 GT：权重图 min=1.0，骨架处=1+gain=5.0，
   距骨架第 d 环权重=1+gain*exp(-d/sigma)，radius 环以外=1.0；
b) 全零 GT：权重全 1，SDL 退化为普通 BCE（与
   F.binary_cross_entropy_with_logits 均值一致，容差 1e-5）；
c) pred.requires_grad=True，loss.backward() 后 pred.grad 非零，
   且 weights 无梯度（detach）；
d) batch 内一半空 mask 一半有裂缝：无 NaN，loss 有限。

注意：本仓库管线传进损失的是 logits（bce_dice 用 BCEWithLogitsLoss），
因此 models/skeleton_loss.py 的 BCE 项为 with_logits 口径，测试同口径。

用法: pytest tests/test_skeleton_distance_loss.py
      python tests/test_skeleton_distance_loss.py   # 直接运行亦可
"""
import sys
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/xby/SCSegamba')
from models.skeleton_loss import SkeletonDistanceLoss, skeleton_distance_weights

GAIN = 4.0
SIGMA = 5.0
RADIUS = 8


def make_line_gt(h=64, w=64, row=32):
    """1px 宽水平直线 GT，避开图像边缘（端点效应留给边距）。"""
    gt = torch.zeros(1, 1, h, w)
    gt[0, 0, row, 8:-8] = 1.0
    return gt


def test_a_line_weight_values():
    gt = make_line_gt()
    weights = skeleton_distance_weights(gt)
    row = 32
    margin = slice(16, -16)          # 远离线的两个端点

    # 最小值 = 1.0（远背景）
    assert weights.min().item() == 1.0
    # 骨架处 = 1 + gain = 5.0
    assert torch.allclose(weights[0, 0, row, margin],
                          torch.full_like(weights[0, 0, row, margin], 1.0 + GAIN))
    # 第 d 环（上下两侧对称）= 1 + gain * exp(-d / sigma)
    for d in range(1, RADIUS + 1):
        expected = 1.0 + GAIN * math.exp(-d / SIGMA)
        for r in (row - d, row + d):
            assert torch.allclose(weights[0, 0, r, margin],
                                  torch.full_like(weights[0, 0, r, margin], expected),
                                  atol=1e-6), f"ring {d} mismatch"
    # radius 环以外 = 1.0
    for r in (row - RADIUS - 1, row + RADIUS + 1, row - 20, row + 20):
        assert torch.all(weights[0, 0, r, margin] == 1.0), f"outside radius at row {r}"


def test_b_empty_gt_degenerates_to_bce():
    torch.manual_seed(0)
    preds = torch.randn(2, 1, 64, 64)
    targets = torch.zeros(2, 1, 64, 64)

    loss_fn = SkeletonDistanceLoss()
    sdl = loss_fn(preds, targets)
    ref = F.binary_cross_entropy_with_logits(preds, targets)  # mean reduction

    weights = skeleton_distance_weights(targets)
    assert torch.all(weights == 1.0)
    assert torch.isfinite(sdl)
    assert abs(sdl.item() - ref.item()) < 1e-5


def test_c_gradient_flows_only_to_pred():
    torch.manual_seed(0)
    preds = torch.randn(1, 1, 64, 64, requires_grad=True)
    targets = make_line_gt()

    loss_fn = SkeletonDistanceLoss()
    loss = loss_fn(preds, targets)
    loss.backward()

    assert preds.grad is not None
    assert preds.grad.abs().sum().item() > 0
    assert torch.isfinite(preds.grad).all()

    # 权重图无梯度：不要求梯度、无计算图
    weights = skeleton_distance_weights(targets)
    assert not weights.requires_grad
    assert weights.grad_fn is None


def test_d_mixed_batch_no_nan():
    torch.manual_seed(0)
    preds = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.zeros(2, 1, 64, 64)
    targets[1, 0, 32, 8:-8] = 1.0      # 第二个样本有裂缝，第一个全背景

    loss_fn = SkeletonDistanceLoss()
    loss = loss_fn(preds, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(preds.grad).all()
    assert preds.grad.abs().sum().item() > 0


if __name__ == '__main__':
    test_a_line_weight_values()
    print('PASS a) line GT weight values')
    test_b_empty_gt_degenerates_to_bce()
    print('PASS b) empty GT degenerates to BCE')
    test_c_gradient_flows_only_to_pred()
    print('PASS c) gradient flows only to pred')
    test_d_mixed_batch_no_nan()
    print('PASS d) mixed batch no NaN')
    print('ALL PASS')
