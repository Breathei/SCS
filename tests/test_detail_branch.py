"""高分辨率细节支路（detail branch）测试。

最终形态（硬编码在 --detail_branch 开关内）：
    delta  = Conv1x1(16→C, 权重偏置全零)(upsample(detail))
    fused  = mfs_feat + delta
    refine = PWConv1x1(C→C, 零初始化) ∘ DWConv3x3(C→C, groups=C)
    fused  = fused + refine(fused)
    随后接原有 dropout → linear_pred。

默认关闭时行为与现状逐 bit 一致；开启 + 零初始化时初始前向与关闭完全
一致（等价起点），detail 通路以残差修正方式从零开始学习。
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
        # b) 开启 + 零初始化：首次前向输出 == 关闭时输出（容差 1e-6）。
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(detail_branch=False))
        torch.manual_seed(0)
        model_on, _ = build_model(_ns(detail_branch=True))
        # backbone/MFS 的 RNG 消耗在 DetailBranch 之前，两者权重应一致。
        off_sd, on_sd = model_off.state_dict(), model_on.state_dict()
        for k in off_sd:
            self.assertTrue(torch.equal(off_sd[k], on_sd[k]), k)
        # delta 投影与 refine 输出层必须全零初始化（等价起点的硬要求）。
        self.assertEqual(model_on.detail_delta.weight.abs().sum().item(), 0.0)
        self.assertEqual(model_on.detail_delta.bias.abs().sum().item(), 0.0)
        self.assertEqual(model_on.detail_refine_pw.weight.abs().sum().item(), 0.0)
        self.assertEqual(model_on.detail_refine_pw.bias.abs().sum().item(), 0.0)

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
    def test_on_forward_shape_and_two_step_grads(self):
        # c) 输出 (B,1,512,512) 且梯度有限。
        #    首 step：delta 投影与 refine_pw 获得非零梯度；DetailBranch 内部
        #    与 refine_dw 被零权重隔离，允许接近零。第二 step 起必须非零。
        torch.manual_seed(0)
        model, _ = build_model(_ns(detail_branch=True))
        model.train().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(1, 3, 512, 512,
                        generator=torch.Generator().manual_seed(4)).cuda()

        def one_step():
            optimizer.zero_grad()
            y = model(x)
            self.assertEqual(y.shape, (1, 1, 512, 512))
            self.assertTrue(torch.isfinite(y).all())
            target = (torch.rand_like(y) > 0.97).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y, target)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            return loss

        # step 0：零权重之后的层（delta、refine_pw）立即有非零梯度。
        loss0 = one_step()
        self.assertGreater(model.detail_delta.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.detail_refine_pw.weight.grad.abs().sum().item(), 0.0)
        optimizer.step()

        # step 1：零权重已离开 0，支路内部与 refine_dw 必须收到非零梯度。
        loss1 = one_step()
        self.assertGreater(model.detail_branch.conv1.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.detail_branch.conv2.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.detail_refine_dw.weight.grad.abs().sum().item(), 0.0)
        backbone_grads = [p.grad for n, p in model.backbone.named_parameters()
                          if p.grad is not None]
        self.assertTrue(any(g.abs().sum().item() > 0 for g in backbone_grads))
        self.assertTrue(torch.isfinite(loss0) and torch.isfinite(loss1))

    def test_param_overhead_below_4000(self):
        # d) 支路新增参数量 < 4000（C=8 时），并打印完整对照。
        torch.manual_seed(0)
        model_off, _ = build_model(_ns(detail_branch=False))
        torch.manual_seed(0)
        model_on, _ = build_model(_ns(detail_branch=True))
        parts = {
            'DetailBranch': _num_params(model_on.detail_branch),
            'detail_delta': _num_params(model_on.detail_delta),
            'detail_refine_dw': _num_params(model_on.detail_refine_dw),
            'detail_refine_pw': _num_params(model_on.detail_refine_pw),
        }
        branch_params = sum(parts.values())
        total_off = _num_params(model_off)
        total_on = _num_params(model_on)
        print('detail branch params: '
              + ', '.join(f'{k}={v}' for k, v in parts.items())
              + f'; total={branch_params}; '
              + f'model off={total_off}, on={total_on}, '
              + f'delta={total_on - total_off}')
        self.assertLess(branch_params, 4000)
        self.assertEqual(total_on - total_off, branch_params)

    def test_cli_flag_propagates(self):
        # e) --detail_branch 打开后模型确实包含 DetailBranch 及残差融合层。
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertFalse(args.detail_branch)
        args = parser.parse_args(['--detail_branch'])
        self.assertTrue(args.detail_branch)

        torch.manual_seed(0)
        model, _ = build_model(_ns(detail_branch=True))
        self.assertIsInstance(model.detail_branch, DetailBranch)
        self.assertIsNotNone(model.detail_delta)
        self.assertIsNotNone(model.detail_refine_dw)
        self.assertIsNotNone(model.detail_refine_pw)
        self.assertEqual(model.detail_refine_dw.groups, model.mfs_channels)

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
