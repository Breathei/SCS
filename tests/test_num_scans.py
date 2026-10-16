"""扫描剪枝（--num_scans 1/2/4）测试。

固定语义顺序：[横向蛇形, 纵向蛇形, 主对角蛇形, 副对角蛇形]，剪枝保留前 N 条。
默认 4 必须与改动前逐 bit 一致；1/2 时被剪掉的路径不生成排列、不做 SSM
前向、不参与融合；三路共享同一 SSM，参数量不随 num_scans 变化。
"""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D  # noqa: E402


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


def _make_savss2d(num_scans, d_model=32, seed=0):
    torch.manual_seed(seed)
    return SAVSS_2D(d_model=d_model, num_scans=num_scans,
                    default_hw_shape=(8, 8))


class TestNumScans(unittest.TestCase):

    def test_default_4_bit_identical(self):
        # a) 回归：旧 Namespace（无 num_scans 属性）与显式 4 同 seed 下
        #    state_dict 逐 bit 一致；CUDA 上整模型前向 torch.equal。
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_4, _ = build_model(_ns(num_scans=4))
        sd_legacy, sd_4 = model_legacy.state_dict(), model_4.state_dict()
        self.assertEqual(list(sd_legacy.keys()), list(sd_4.keys()))
        for k in sd_legacy:
            self.assertTrue(torch.equal(sd_legacy[k], sd_4[k]), k)

    @unittest.skipUnless(torch.cuda.is_available(), 'scan kernel 需要 CUDA')
    def test_default_4_forward_bit_identical_cuda(self):
        torch.manual_seed(0)
        model_legacy, _ = build_model(_ns())
        torch.manual_seed(0)
        model_4, _ = build_model(_ns(num_scans=4))
        model_legacy.eval().cuda()
        model_4.eval().cuda()
        x = torch.randn(1, 3, 512, 512,
                        generator=torch.Generator().manual_seed(3)).cuda()
        with torch.no_grad():
            y_legacy = model_legacy(x)
            y_4 = model_4(x)
        self.assertTrue(torch.equal(y_legacy, y_4))

    def test_pruned_scan_count_and_order(self):
        # b) 剪枝正确性：N=2/1 时实际参与 SSM 前向的路径数为 2/1，
        #    且保留的排列与 4 向 sass() 的前 N 条完全一致（横+纵 / 仅横）。
        hw = (8, 8)
        for n in (2, 1):
            m = _make_savss2d(n)
            full_orders, full_inv, full_dirs = m.sass(hw, n_dirs=4)
            orders, inv, dirs = m.sass(hw, n_dirs=n)
            self.assertEqual(len(orders), n)
            self.assertEqual(orders, full_orders[:n])
            self.assertEqual(inv, full_inv[:n])
            self.assertEqual(dirs, full_dirs[:n])

            # spy 替换 scan 函数，统计 forward 中实际的 SSM 调用次数。
            calls = []

            def spy_scan(u, dt, A, B, C, D, **kwargs):
                calls.append(tuple(u.shape))
                return torch.zeros_like(u)

            m._get_scan_fn = lambda: spy_scan
            x = torch.randn(1, hw[0] * hw[1], 32,
                            generator=torch.Generator().manual_seed(1))
            out = m(x, hw_shape=hw)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all())
            print(f'num_scans={n}: actual SSM scan calls = {len(calls)} '
                  f'(kept orders == full sass[:{n}])')
            self.assertEqual(len(calls), n)

        # 默认 4 路的调用次数 sanity check。
        m4 = _make_savss2d(4)
        calls = []
        m4._get_scan_fn = lambda: lambda u, dt, A, B, C, D, **kw: (
            calls.append(1), torch.zeros_like(u))[1]
        x = torch.randn(1, 64, 32, generator=torch.Generator().manual_seed(1))
        m4(x, hw_shape=hw)
        self.assertEqual(len(calls), 4)

    def test_routing_modes_reject_pruning(self):
        # b') 路由模式按 4 路分支设计，与剪枝组合必须显式报错而非静默出错。
        for routing in ('asr', 'asr_scan', 'asr_scan_v2'):
            with self.assertRaises(AssertionError):
                SAVSS_2D(d_model=32, num_scans=2, scan_routing=routing)

    def test_param_counts_identical(self):
        # c) 参数量核对：SSM 权重共享，4/2/1 三种配置总参数量必须完全不变。
        counts = {}
        for n in (4, 2, 1):
            torch.manual_seed(0)
            model, _ = build_model(_ns(num_scans=n))
            counts[n] = _num_params(model)
        print(f'param counts: num_scans=4 -> {counts[4]}, '
              f'num_scans=2 -> {counts[2]}, num_scans=1 -> {counts[1]}')
        self.assertEqual(counts[4], counts[2])
        self.assertEqual(counts[4], counts[1])

    def test_cli_propagates_to_layers(self):
        # d) CLI：默认 4；--num_scans 2 传入后每个 SAVSS_2D 层都是 2。
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertEqual(args.num_scans, 4)
        args = parser.parse_args(['--num_scans', '2'])
        self.assertEqual(args.num_scans, 2)

        torch.manual_seed(0)
        model, _ = build_model(_ns(num_scans=2))
        layers = [m for m in model.modules() if isinstance(m, SAVSS_2D)]
        self.assertEqual(len(layers), 4)
        for m in layers:
            self.assertEqual(m.num_scans, 2)


if __name__ == '__main__':
    unittest.main()
