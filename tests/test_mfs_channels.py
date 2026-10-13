"""MFS 通道数参数化测试。

--mfs_channels（默认 8）控制 MFS 解码头宽度；默认档必须与改动前逐 bit
一致（state_dict key、参数值、前向数值全不变）。
"""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from models.MFS import MFS  # noqa: E402


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


def _mfs_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(1, 128, 8, 8, generator=g),
        torch.randn(1, 64, 16, 16, generator=g),
        torch.randn(1, 32, 32, 32, generator=g),
        torch.randn(1, 16, 64, 64, generator=g),
    ]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


class TestMfsChannels(unittest.TestCase):

    def test_parser_default_is_8(self):
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertEqual(args.mfs_channels, 8)

    def test_default_8_bit_identical_to_legacy(self):
        # a) 旧 Namespace（无 mfs_channels 属性，走 getattr 默认）与显式 8：
        #    同 seed 下 state_dict key/值、前向输出全部逐 bit 一致（容差 0）。
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_explicit, _ = build_model(_ns(mfs_channels=8))

        sd_legacy = model_legacy.state_dict()
        sd_explicit = model_explicit.state_dict()
        self.assertEqual(list(sd_legacy.keys()), list(sd_explicit.keys()))
        for k in sd_legacy:
            self.assertTrue(torch.equal(sd_legacy[k], sd_explicit[k]), k)

        model_legacy.eval()
        model_explicit.eval()
        # 全模型前向依赖 CUDA scan kernel，CPU 不可跑；本次改动只触及 MFS，
        # 对 MFS 模块做逐 bit 前向对比即可覆盖“前向数值全不变”。
        inputs = _mfs_inputs(seed=2)
        with torch.no_grad():
            y_legacy = model_legacy.MFS(inputs)
            y_explicit = model_explicit.MFS(inputs)
        self.assertTrue(torch.equal(y_legacy, y_explicit))

    def test_widths_forward_shape_and_finite_grads(self):
        # b) 8/12/16 三档前向 shape 正确、梯度有限。
        for width in (8, 12, 16):
            torch.manual_seed(0)
            mfs = MFS(width)
            inputs = [t.requires_grad_(True) for t in _mfs_inputs()]
            out = mfs(inputs)
            self.assertEqual(out.shape, (1, 1, 64, 64), f'width={width}')
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            for t in inputs:
                self.assertTrue(torch.isfinite(t.grad).all(), f'width={width}')
            for name, p in mfs.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),
                                    f'width={width} {name}')

    def test_param_count_monotonic(self):
        # c) 三档参数量单调递增。
        counts = {}
        for width in (8, 12, 16):
            torch.manual_seed(0)
            counts[width] = _num_params(MFS(width))
        print(f'MFS param counts: 8ch={counts[8]}, '
              f'12ch={counts[12]}, 16ch={counts[16]}')
        self.assertLess(counts[8], counts[12])
        self.assertLess(counts[12], counts[16])

    def test_legacy_checkpoint_strict_loads(self):
        # d) 旧 checkpoint（无 mfs_channels 时代保存的 state_dict）在默认
        #    配置下 strict-load 正常。
        torch.manual_seed(0)
        legacy_model, _ = build_model(_ns())
        legacy_sd = {k: v.clone() for k, v in legacy_model.state_dict().items()}

        torch.manual_seed(123)  # 不同初始化，确保加载真正生效
        fresh_model, _ = build_model(_ns())
        fresh_model.load_state_dict(legacy_sd, strict=True)
        fresh_sd = fresh_model.state_dict()
        for k in legacy_sd:
            self.assertTrue(torch.equal(fresh_sd[k], legacy_sd[k]), k)


if __name__ == '__main__':
    unittest.main()
