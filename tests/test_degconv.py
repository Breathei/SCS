"""DEGConv（方向引导边缘门控卷积，--degconv）测试。

默认关闭时与现状逐 bit 一致；开启时 4 级 stage 输出各经一个独立 DEGConv
实例后再进 MFS。方向嵌入分支全程 no_grad（先验不需要梯度）。
"""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from models.degconv import DEGConv, EdgeConv  # noqa: E402


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


class TestDEGConv(unittest.TestCase):

    def test_default_off_bit_identical(self):
        # a) 回归：旧 Namespace（无 degconv 属性）与显式 False 同 seed 下
        #    state_dict、MFS 前向逐 bit 一致；不包含任何 degconv 模块。
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(degconv=False))
        sd_legacy, sd_off = model_legacy.state_dict(), model_off.state_dict()
        self.assertEqual(list(sd_legacy.keys()), list(sd_off.keys()))
        for k in sd_legacy:
            self.assertTrue(torch.equal(sd_legacy[k], sd_off[k]), k)
        self.assertFalse(any('degconv' in k for k in sd_legacy))
        self.assertIsNone(model_legacy.degconvs)
        self.assertIsNone(model_off.degconvs)

    @unittest.skipUnless(torch.cuda.is_available(), 'scan kernel 需要 CUDA')
    def test_default_off_forward_bit_identical_cuda(self):
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(degconv=False))
        model_legacy.eval().cuda()
        model_off.eval().cuda()
        x = torch.randn(1, 3, 512, 512,
                        generator=torch.Generator().manual_seed(3)).cuda()
        with torch.no_grad():
            self.assertTrue(torch.equal(model_legacy(x), model_off(x)))

    def test_shape_and_odd_size_and_view_roundtrip(self):
        # b) 形状：4 级真实 (通道, 分辨率) 下输入输出同形状；奇数尺寸经
        #    reflect-pad 后输出裁回原尺寸；view 拆分/拼回是完全可逆的。
        for c, s in ((128, 64), (64, 128), (32, 256), (16, 512)):
            m = DEGConv(c)
            x = torch.randn(2, c, s, s, generator=torch.Generator().manual_seed(c))
            y = m(x)
            self.assertEqual(y.shape, x.shape, (c, s))
            self.assertTrue(torch.isfinite(y).all())
        m = DEGConv(32)
        for s in (61, 47):
            x = torch.randn(2, 32, s, s + 3,
                            generator=torch.Generator().manual_seed(s))
            y = m(x)
            self.assertEqual(y.shape, x.shape, s)

        m = DEGConv(8)
        x = torch.arange(2 * 8 * 30 * 46, dtype=torch.float32).reshape(2, 8, 30, 46)
        v = m._to_views(x)
        back = m._from_views(v, 2, 30, 46)
        self.assertTrue(torch.equal(back, x))
        # view 内容确实是 2×2 空间分块（行优先）。
        self.assertTrue(torch.equal(v[0], x[0, :, :15, :23]))
        self.assertTrue(torch.equal(v[1], x[0, :, :15, 23:]))
        self.assertTrue(torch.equal(v[2], x[0, :, 15:, :23]))
        self.assertTrue(torch.equal(v[3], x[0, :, 15:, 23:]))

    def test_direction_prior_has_no_grad(self):
        # c) 先验无梯度：ε requires_grad=False，DirectionEmbed 参数无梯度；
        #    EdgeConv（门控/内容/后处理）参数梯度正常非零。
        torch.manual_seed(0)
        m = DEGConv(32)
        view = torch.randn(2, 32, 32, 32,
                           generator=torch.Generator().manual_seed(1))
        eps = m._direction_embedding(view)
        self.assertFalse(eps.requires_grad)
        self.assertEqual(eps.shape, (2, 32, 1, 1))

        x = torch.randn(2, 32, 32, 32,
                        generator=torch.Generator().manual_seed(2),
                        requires_grad=True)
        y = m(x)
        y.sum().backward()
        for name, p in m.named_parameters():
            if name.startswith('dir_embed.'):
                self.assertTrue(p.grad is None or p.grad.abs().sum().item() == 0.0,
                                name)
            else:
                self.assertIsNotNone(p.grad, name)
                self.assertGreater(p.grad.abs().sum().item(), 0.0, name)
        self.assertIsNotNone(x.grad)

    def test_param_overhead(self):
        # d) 参数量增量 < +0.5M，打印各级明细。
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(degconv=False))
        torch.manual_seed(0)
        model_on, _ = build_model(_ns(degconv=True))
        per_level = {c: _num_params(d)
                     for c, d in zip((128, 64, 32, 16), model_on.degconvs)}
        delta = _num_params(model_on) - _num_params(model_off)
        print('DEGConv params per level: '
              + ', '.join(f'C{c}={n}' for c, n in per_level.items())
              + f'; total delta={delta} ({delta / 1e6:.3f}M)')
        self.assertEqual(delta, sum(per_level.values()))
        self.assertLess(delta, 500_000)

    def test_cli_flag_propagates(self):
        # e) --degconv 打开后模型确实包含 4 个独立 DEGConv 实例。
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertFalse(args.degconv)
        args = parser.parse_args(['--degconv'])
        self.assertTrue(args.degconv)

        torch.manual_seed(0)
        model, _ = build_model(_ns(degconv=True))
        self.assertEqual(len(model.degconvs), 4)
        for c, d in zip((128, 64, 32, 16), model.degconvs):
            self.assertIsInstance(d, DEGConv)
            self.assertEqual(d.channels, c)
            self.assertIsInstance(d.edge, EdgeConv)
            self.assertIsInstance(d.edge_post, EdgeConv)
        # 各级参数独立（不共享存储）。
        ptrs = {id(p) for d in model.degconvs for p in d.parameters()}
        self.assertEqual(len(ptrs), sum(1 for d in model.degconvs
                                        for _ in d.parameters()))


if __name__ == '__main__':
    unittest.main()
