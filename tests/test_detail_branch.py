"""高分辨率细节支路（detail branch）测试。

默认关闭时行为与现状逐 bit 一致；开启 + 恒等/零初始化时初始前向与关闭
完全一致（等价起点），detail 通路从零开始学习。
"""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from models.detail_branch import DetailBranch  # noqa: E402


def _ns(**overrides):
    ns = argparse.Namespace(
        device='cpu', discretization='euler', lam_nheads=8,
        trap_lambda=0.5, trap_boundary='euler', trap_lambda_per_dir=False,
        scan_routing='none', scan_param_order='current',
        asr_beta=0.1, asr_route_grid=1, asr_v2_beta=0.25,
        BCELoss_ratio=0.83, DiceLoss_ratio=0.17,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


class TestDetailBranch(unittest.TestCase):

    def test_default_off_bit_identical(self):
        # a) 默认关闭：旧 Namespace（无 detail_branch 属性）与显式 False
        #    同 seed 下 state_dict、MFS 前向逐 bit 一致；且不包含支路模块。
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(detail_branch=False))

        sd_legacy = model_legacy.state_dict()
        sd_off = model_off.state_dict()
        self.assertEqual(list(sd_legacy.keys()), list(sd_off.keys()))
        for k in sd_legacy:
            self.assertTrue(torch.equal(sd_legacy[k], sd_off[k]), k)
        self.assertFalse(any('detail' in k for k in sd_legacy))
        self.assertIsNone(model_legacy.detail_branch)
        self.assertIsNone(model_off.detail_branch)

        g = torch.Generator().manual_seed(2)
        inputs = [torch.randn(1, ch, s, s, generator=g)
                  for ch, s in ((128, 8), (64, 16), (32, 32), (16, 64))]
        model_legacy.eval()
        model_off.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model_legacy.MFS(inputs),
                                        model_off.MFS(inputs)))

    @unittest.skipUnless(torch.cuda.is_available(), 'scan kernel 需要 CUDA')
    def test_on_zero_init_matches_off_forward(self):
        # b) 开启 + 恒等/零初始化：首次前向输出 == 关闭时输出（容差 1e-6）。
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(detail_branch=False))
        torch.manual_seed(0)
        model_on, _ = build_model(_ns(detail_branch=True))
        # backbone/MFS 的 RNG 消耗在 DetailBranch 之前，两者权重应一致。
        off_sd, on_sd = model_off.state_dict(), model_on.state_dict()
        for k in off_sd:
            self.assertTrue(torch.equal(off_sd[k], on_sd[k]), k)

        model_off.eval().cuda()
        model_on.eval().cuda()
        x = torch.randn(1, 3, 512, 512,
                        generator=torch.Generator().manual_seed(3)).cuda()
        with torch.no_grad():
            y_off = model_off(x)
            y_on = model_on(x)
        self.assertEqual(y_on.shape, y_off.shape)
        self.assertTrue(torch.allclose(y_on, y_off, atol=1e-6),
                        (y_on - y_off).abs().max().item())

    @unittest.skipUnless(torch.cuda.is_available(), 'scan kernel 需要 CUDA')
    def test_on_forward_shape_and_finite_grads(self):
        # c) 开启后输出 (B,1,512,512)，DetailBranch 与 backbone 梯度有限。
        torch.manual_seed(0)
        model, _ = build_model(_ns(detail_branch=True))
        model.train().cuda()
        x = torch.randn(1, 3, 512, 512,
                        generator=torch.Generator().manual_seed(4)).cuda()
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 512, 512))
        self.assertTrue(torch.isfinite(y).all())
        target = (torch.rand_like(y) > 0.97).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertIsNotNone(model.detail_branch.conv1.weight.grad)
        self.assertIsNotNone(model.detail_fuse.weight.grad)
        # detail 融合权重第一步即获得非零梯度。
        self.assertGreater(model.detail_fuse.weight.grad.abs().sum().item(), 0.0)
        backbone_grads = [p.grad for n, p in model.backbone.named_parameters()
                          if p.grad is not None]
        self.assertTrue(any(g.abs().sum().item() > 0 for g in backbone_grads))

    def test_param_overhead_below_10k(self):
        # d) 支路新增参数量 < 10k，并打印完整对照。
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(detail_branch=False))
        torch.manual_seed(0)
        model_on, _ = build_model(_ns(detail_branch=True))
        branch_params = (_num_params(model_on.detail_branch)
                         + _num_params(model_on.detail_fuse))
        total_off = _num_params(model_off)
        total_on = _num_params(model_on)
        print(f'DetailBranch params: {branch_params} '
              f'(branch={_num_params(model_on.detail_branch)}, '
              f'fuse={_num_params(model_on.detail_fuse)}); '
              f'total off={total_off}, on={total_on}, '
              f'delta={total_on - total_off}')
        self.assertLess(branch_params, 10_000)
        self.assertEqual(total_on - total_off, branch_params)

    def test_cli_flag_propagates(self):
        # e) --detail_branch 打开后模型确实包含 DetailBranch。
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertFalse(args.detail_branch)
        args = parser.parse_args(['--detail_branch'])
        self.assertTrue(args.detail_branch)

        torch.manual_seed(0)
        model, _ = build_model(_ns(detail_branch=True))
        self.assertIsInstance(model.detail_branch, DetailBranch)
        self.assertIsNotNone(model.detail_fuse)

    def test_legacy_checkpoint_strict_loads_default(self):
        # f) 旧 checkpoint（无支路）在默认配置下 strict-load 正常。
        torch.manual_seed(0)
        legacy_model, _ = build_model(_ns())
        legacy_sd = {k: v.clone() for k, v in legacy_model.state_dict().items()}

        torch.manual_seed(123)
        fresh_model, _ = build_model(_ns())
        fresh_model.load_state_dict(legacy_sd, strict=True)
        for k, v in fresh_model.state_dict().items():
            self.assertTrue(torch.equal(v, legacy_sd[k]), k)


if __name__ == '__main__':
    unittest.main()
