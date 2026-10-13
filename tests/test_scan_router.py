"""Focused tests for the scan-aware ASR router."""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import compute_asr_v2_coverage_loss, compute_balancing_loss
from models import build_model  # noqa: F401,E402
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import (
    SAVSS_2D,
    ScanRouter,
    ScanRouterV2,
)


class TestScanRouter(unittest.TestCase):
    def test_scan_router_zero_init_and_grid_shapes(self):
        router = ScanRouter(32)
        context = torch.randn(2, 32, 3, 5)
        branches = torch.randn(2, 4, 32, 3, 5)
        for grid in (1, 2, 4):
            w, w_grid = router.scan_weights(context, branches, (3, 5), grid)
            self.assertEqual(w.shape, (2, 4, 3, 5))
            self.assertEqual(w_grid.shape, (2, 4, grid, grid))
            self.assertTrue(torch.allclose(w, torch.full_like(w, 0.25)))
            self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones_like(w[:, 0])))
            self.assertTrue(torch.allclose(w_grid.sum(dim=1), torch.ones_like(w_grid[:, 0])))

    def test_branch_descriptor_has_gradient(self):
        router = ScanRouter(32)
        context = torch.randn(2, 32, 4, 4)
        branches = torch.randn(2, 4, 32, 4, 4, requires_grad=True)
        with torch.no_grad():
            router.fc2.weight.normal_(0.0, 0.1)
            router.fc2.bias.zero_()
        w, _ = router.scan_weights(context, branches, (4, 4), 2)
        loss = (w[:, 0:1] * branches[:, 0]).mean()
        loss.backward()
        self.assertIsNotNone(branches.grad)
        self.assertTrue(torch.isfinite(branches.grad).all())
        self.assertGreater(router.fc2.weight.grad.abs().sum().item(), 0.0)

    def test_tiny_asr_scan_forward_backward(self):
        for grid in (1, 2):
            module = SAVSS_2D(
                d_model=32,
                d_state=3,
                expand=1,
                scan_routing='asr_scan',
                asr_beta=0.1,
                asr_route_grid=grid,
            )
            captured = []

            def spy_scan(u, delta, A, B, C, D, **kwargs):
                captured.append(u)
                return u

            module._get_scan_fn = lambda: spy_scan
            x = torch.randn(2, 15, 32, requires_grad=True)
            y = module(x, (3, 5))
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(module._route_w.shape, (2, 4, grid, grid))
            self.assertTrue(torch.allclose(
                module._route_w.sum(dim=1),
                torch.ones_like(module._route_w[:, 0]),
            ))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(len(captured), 4)

    def test_zero_init_asr_scan_matches_sum(self):
        torch.manual_seed(9)
        module = SAVSS_2D(
            d_model=32,
            d_state=3,
            expand=1,
            scan_routing='asr_scan',
            asr_beta=0.1,
            asr_route_grid=2,
        ).eval()
        captured = []

        def spy_scan(u, delta, A, B, C, D, **kwargs):
            captured.append(u)
            return u

        module._get_scan_fn = lambda: spy_scan
        x = torch.randn(1, 15, 32)
        with torch.no_grad():
            y = module(x, (3, 5))
            orders, inverse_orders, _ = module.sass((3, 5))
            expected = sum(
                u.permute(0, 2, 1)[:, inverse_orders[i], :]
                for i, u in enumerate(captured)
            )
            expected = expected * module.act(module.in_proj(x).chunk(2, dim=-1)[1])
            expected = module.out_proj(expected)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-5))

    def test_balancing_is_per_layer_and_soft(self):
        class Fake(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self._route_w = weights

        first = Fake(torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]]))
        second = Fake(torch.full((1, 4, 2, 2), 0.25))
        model = torch.nn.ModuleList([first, second])
        loss = compute_balancing_loss(model)
        # First layer contributes 4, second contributes 1; average is 2.5.
        self.assertAlmostEqual(loss.item(), 2.5, places=6)


class TestScanRouterV2(unittest.TestCase):
    def test_parser_defaults(self):
        from main import get_args_parser
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertEqual(args.scan_routing, 'none')
        self.assertEqual(args.scan_param_order, 'current')
        self.assertEqual(args.asr_v2_beta, 0.25)
        self.assertEqual(args.asr_router_lr_mult, 1.0)
        self.assertEqual(args.asr_v2_coverage_coef, 0.01)
        self.assertEqual(args.asr_v2_coverage_cap, 0.6)
        self.assertEqual(args.asr_v2_balance_warmup_ratio, 0.1)
        self.assertFalse(args.asr_router_diag)

    def test_zero_init_uniform_and_grid_shapes(self):
        router = ScanRouterV2(32)
        context = torch.randn(2, 32, 3, 5)
        branches = torch.randn(2, 4, 64, 3, 5)  # E = 2 * C
        for grid in (1, 2, 4):
            w, w_grid = router.scan_weights(context, branches, (3, 5), grid)
            self.assertEqual(w.shape, (2, 4, 3, 5))
            self.assertEqual(w_grid.shape, (2, 4, grid, grid))
            self.assertTrue(torch.allclose(w, torch.full_like(w, 0.25)))
            self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones_like(w[:, 0])))
            self.assertTrue(torch.allclose(
                w_grid.sum(dim=1), torch.ones_like(w_grid[:, 0])
            ))

    def test_head_receives_gradient_at_zero_init(self):
        torch.manual_seed(0)
        router = ScanRouterV2(32)
        context = torch.randn(2, 32, 4, 4)
        branches = torch.randn(2, 4, 64, 4, 4)
        w, _ = router.scan_weights(context, branches, (4, 4), 2)
        # Asymmetric per-direction target makes dL/dlogits branch-dependent.
        target = torch.tensor([0.7, 0.1, 0.1, 0.1]).view(1, 4, 1, 1)
        loss = (w * target).mean()
        loss.backward()
        self.assertIsNotNone(router.head.weight.grad)
        self.assertGreater(router.head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(router.head.bias.grad.abs().sum().item(), 0.0)
        # At zero head init, gradient w.r.t. upstream projections is blocked
        # (d logits/d feat = head.weight = 0); after one optimizer step the
        # head is non-zero and the descriptor path receives gradient.
        opt = torch.optim.SGD(router.parameters(), lr=0.1)
        opt.step()
        router.zero_grad()
        w, _ = router.scan_weights(context, branches, (4, 4), 2)
        (w * target).mean().backward()
        self.assertGreater(router.desc_proj.weight.grad.abs().sum().item(), 0.0)

    def test_route_weights_respond_to_branch_contrast(self):
        torch.manual_seed(0)
        router = ScanRouterV2(32)
        # Give the head a non-zero init so logits can differ across branches.
        with torch.no_grad():
            router.head.weight.normal_(0.0, 0.1)
        context = torch.randn(1, 32, 4, 4)
        branches = torch.randn(1, 4, 64, 4, 4)
        # Make branch 0 strongly distinct from the other three.
        branches[:, 0] = branches[:, 0] * 4.0 + 3.0
        w, w_grid = router.scan_weights(context, branches, (4, 4), 2)
        per_dir = w_grid.mean(dim=(0, 2, 3))
        self.assertFalse(torch.allclose(
            per_dir, torch.full_like(per_dir, 0.25), atol=1e-3
        ))

    def test_tiny_v2_forward_backward(self):
        for grid in (1, 2):
            module = SAVSS_2D(
                d_model=32,
                d_state=3,
                expand=1,
                scan_routing='asr_scan_v2',
                asr_v2_beta=0.25,
                asr_route_grid=grid,
            )
            captured = []

            def spy_scan(u, delta, A, B, C, D, **kwargs):
                captured.append(u)
                # 分支间必须有差异，否则路由在数学上是 no-op：
                # 分支相同时 dL/dlogit ≡ 0，head 不可能获得梯度。
                return u * (0.5 + 0.5 * len(captured))

            module._get_scan_fn = lambda: spy_scan
            x = torch.randn(2, 15, 32, requires_grad=True)
            y = module(x, (3, 5))
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(module._route_w.shape, (2, 4, grid, grid))
            self.assertTrue(torch.allclose(
                module._route_w.sum(dim=1),
                torch.ones_like(module._route_w[:, 0]),
            ))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(len(captured), 4)
            # head 在零初始化下也必须立刻获得非零 task 梯度。
            self.assertGreater(
                module.router_v2.head.weight.grad.abs().sum().item(), 0.0
            )

    def test_identical_branches_grads_stay_finite(self):
        # 退化情形：spy scan 原样返回 u → 四个分支完全相同 → adv≡0、rms≈0。
        # sqrt 的 eps 必须保证反传数值有限（历史上 sqrt(0) backward 的 inf
        # 与零初始化 head 的 0 上游梯度相乘产生 NaN）。此时 dL/dlogit≡0，
        # head 梯度为 0 是数学上的正确结果，不作非零断言。
        for grid in (1, 2):
            module = SAVSS_2D(
                d_model=32,
                d_state=3,
                expand=1,
                scan_routing='asr_scan_v2',
                asr_v2_beta=0.25,
                asr_route_grid=grid,
            )
            module._get_scan_fn = lambda: (lambda u, delta, A, B, C, D, **kw: u)
            x = torch.randn(2, 15, 32, requires_grad=True)
            y = module(x, (3, 5))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for name, p in module.named_parameters():
                if p.grad is not None:
                    self.assertTrue(
                        torch.isfinite(p.grad).all(),
                        f'non-finite grad in {name} (grid={grid})',
                    )

    def test_zero_init_v2_matches_sum(self):
        torch.manual_seed(9)
        module = SAVSS_2D(
            d_model=32,
            d_state=3,
            expand=1,
            scan_routing='asr_scan_v2',
            asr_v2_beta=0.25,
            asr_route_grid=2,
        ).eval()
        captured = []

        def spy_scan(u, delta, A, B, C, D, **kwargs):
            captured.append(u)
            return u

        module._get_scan_fn = lambda: spy_scan
        x = torch.randn(1, 15, 32)
        with torch.no_grad():
            y = module(x, (3, 5))
            orders, inverse_orders, _ = module.sass((3, 5))
            expected = sum(
                u.permute(0, 2, 1)[:, inverse_orders[i], :]
                for i, u in enumerate(captured)
            )
            expected = expected * module.act(module.in_proj(x).chunk(2, dim=-1)[1])
            expected = module.out_proj(expected)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-5))

    def test_v2_beta_zero_matches_sum_with_nonzero_head(self):
        torch.manual_seed(3)
        module = SAVSS_2D(
            d_model=32,
            d_state=3,
            expand=1,
            scan_routing='asr_scan_v2',
            asr_v2_beta=0.0,
            asr_route_grid=2,
        ).eval()
        # 非零 head 使权重不再均匀；beta=0 必须严格退回等和 baseline。
        with torch.no_grad():
            module.router_v2.head.weight.normal_(0.0, 0.2)
            module.router_v2.head.bias.normal_(0.0, 0.1)
        captured = []

        def spy_scan(u, delta, A, B, C, D, **kwargs):
            captured.append(u)
            return u

        module._get_scan_fn = lambda: spy_scan
        x = torch.randn(1, 15, 32)
        with torch.no_grad():
            y = module(x, (3, 5))
            orders, inverse_orders, _ = module.sass((3, 5))
            expected = sum(
                u.permute(0, 2, 1)[:, inverse_orders[i], :]
                for i, u in enumerate(captured)
            )
            expected = expected * module.act(module.in_proj(x).chunk(2, dim=-1)[1])
            expected = module.out_proj(expected)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-5))

    def test_coverage_loss_one_sided_cap(self):
        class FakeV2(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.scan_routing = 'asr_scan_v2'
                self._route_w = weights

        # 均匀分布：所有方向占比 0.25 < cap=0.6 → 无惩罚。
        uniform = torch.nn.ModuleList([
            FakeV2(torch.full((2, 4, 2, 2), 0.25))
        ])
        loss = compute_asr_v2_coverage_loss(uniform, coverage_cap=0.6)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

        # 完全塌缩到方向 0：p=[1,0,0,0] → relu(1-0.6)^2 = 0.16。
        collapsed = torch.nn.ModuleList([
            FakeV2(torch.tensor(
                [[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]]
            ).expand(2, 4, 1, 1).contiguous())
        ])
        loss = compute_asr_v2_coverage_loss(collapsed, coverage_cap=0.6)
        self.assertAlmostEqual(loss.item(), 0.16, places=6)

        # 占比 0.7 > cap：relu(0.7-0.6)^2 = 0.01（其余方向 0.1 不罚）。
        shifted = torch.full((1, 4, 1, 1), 0.1)
        shifted[0, 0, 0, 0] = 0.7
        model = torch.nn.ModuleList([FakeV2(shifted)])
        loss = compute_asr_v2_coverage_loss(model, coverage_cap=0.6)
        self.assertAlmostEqual(loss.item(), 0.01, places=6)

    def test_balancing_loss_skips_v2_modules(self):
        class FakeV2(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.scan_routing = 'asr_scan_v2'
                self._route_w = weights

        model = torch.nn.ModuleList([
            FakeV2(torch.full((1, 4, 1, 1), 1.0))
        ])
        # V2 模块不参与 legacy balancing loss；模型中没有其它路由模块 → None。
        self.assertIsNone(compute_balancing_loss(model))

    def test_build_model_forwards_v2_config(self):
        args = argparse.Namespace(
            device='cpu', discretization='euler', lam_nheads=8,
            trap_lambda=0.5, trap_boundary='euler', trap_lambda_per_dir=False,
            scan_routing='asr_scan_v2', scan_param_order='current',
            asr_beta=0.1, asr_route_grid=2, asr_v2_beta=0.3,
            BCELoss_ratio=0.83, DiceLoss_ratio=0.17,
        )
        model, _ = build_model(args)
        modules = [m for m in model.modules() if isinstance(m, SAVSS_2D)]
        self.assertGreater(len(modules), 0)
        for m in modules:
            self.assertEqual(m.scan_routing, 'asr_scan_v2')
            self.assertEqual(m.asr_v2_beta, 0.3)
            self.assertEqual(m.asr_route_grid, 2)
            self.assertTrue(hasattr(m, 'router_v2'))
            self.assertFalse(hasattr(m, 'router'))
        v2_keys = [k for k in model.state_dict() if 'router_v2' in k]
        self.assertEqual(len(v2_keys), 6 * len(modules))  # 3 Linear * (w,b)
        legacy_keys = [k for k in model.state_dict()
                       if ('router.' in k or k.startswith('router.'))
                       and 'router_v2' not in k]
        self.assertEqual(legacy_keys, [])

    def test_router_diag_entry_detached(self):
        module = SAVSS_2D(
            d_model=32,
            d_state=3,
            expand=1,
            scan_routing='asr_scan_v2',
            asr_v2_beta=0.25,
            asr_route_grid=2,
        ).train()
        module._get_scan_fn = lambda: (lambda u, delta, A, B, C, D, **kw: u)
        SAVSS_2D._router_diag_active = True
        SAVSS_2D._router_diag_step = 7
        try:
            x = torch.randn(1, 15, 32)
            module(x, (3, 5))
            entry = module._router_diag_entry
            self.assertIsNotNone(entry)
            self.assertEqual(entry['step'], 7)
            self.assertEqual(entry['route_grid'], 2)
            self.assertEqual(len(entry['w_mean']), 4)
            self.assertEqual(len(entry['argmax_freq']), 4)
            self.assertEqual(len(entry['branch_rms']), 4)
            self.assertGreaterEqual(entry['entropy_mean'], 0.0)
            self.assertGreaterEqual(entry['residual_ratio'], 0.0)
        finally:
            SAVSS_2D._router_diag_active = False


if __name__ == '__main__':
    unittest.main()
