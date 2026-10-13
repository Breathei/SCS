"""
Tests for scan_param_order=current/aligned in SAVSS_2D.

The aligned mode reorders dt/B/C with the same SASS order used by u. The
legacy current mode must keep the historical parameter order unchanged.
"""
import argparse
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser  # noqa: E402
from models import build_model  # noqa: E402,F401
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D  # noqa: E402


class TestScanParamOrder(unittest.TestCase):

    def test_parser_default_is_current(self):
        parser = argparse.ArgumentParser(parents=[get_args_parser()])
        args = parser.parse_args([])
        self.assertEqual(args.scan_param_order, 'current')

    def test_module_rejects_invalid_order(self):
        with self.assertRaises(AssertionError):
            SAVSS_2D(d_model=32, d_state=4, expand=1, scan_param_order='bad')

    def test_scan_param_helper_current_and_aligned(self):
        module = SAVSS_2D(d_model=32, d_state=3, expand=1)
        H, W = 3, 5
        orders, _, directions = module.sass((H, W))
        batch_size, E, N, L = 2, 32, 3, H * W

        dt = torch.arange(batch_size * E * L, dtype=torch.float32).reshape(batch_size, E, L)
        B = torch.arange(batch_size * N * L, dtype=torch.float32).reshape(batch_size, N, L)
        C = (2000 + torch.arange(batch_size * N * L, dtype=torch.float32)).reshape(batch_size, N, L)
        direction_Bs = [
            torch.arange(batch_size * N * L, dtype=torch.float32).reshape(batch_size, N, L) + 10 * i
            for i, _ in enumerate(directions)
        ]

        for order, dB in zip(orders, direction_Bs):
            module.scan_param_order = 'current'
            dt_dir, B_dir, C_dir = module._scan_params_for_order(order, dt, B, C, dB)
            self.assertTrue(torch.equal(dt_dir, dt))
            self.assertTrue(torch.allclose(B_dir, B + dB))
            self.assertTrue(torch.equal(C_dir, C))

            module.scan_param_order = 'aligned'
            dt_dir, B_dir, C_dir = module._scan_params_for_order(order, dt, B, C, dB)
            self.assertTrue(torch.equal(dt_dir, dt[:, :, order]))
            self.assertTrue(torch.allclose(B_dir, B[:, :, order] + dB))
            self.assertTrue(torch.equal(C_dir, C[:, :, order]))
            self.assertTrue(torch.allclose(B_dir - B[:, :, order], dB))
            self.assertTrue(dt_dir.is_contiguous())
            self.assertTrue(B_dir.is_contiguous())
            self.assertTrue(C_dir.is_contiguous())

    def test_forward_passes_matching_parameters_to_scan(self):
        H, W = 3, 5
        batch_size, d_model = 1, 32
        x = torch.randn(batch_size, H * W, d_model)

        for scan_param_order in ('current', 'aligned'):
            module = SAVSS_2D(
                d_model=d_model,
                d_state=3,
                expand=1,
                scan_param_order=scan_param_order,
            ).eval()
            captured = []
            expected = []
            original_helper = module._scan_params_for_order

            def capture_params(order, dt, B, C, dB):
                params = original_helper(order, dt, B, C, dB)
                expected.append((order, *(p.detach().clone() for p in params)))
                return params

            module._scan_params_for_order = capture_params

            def spy_scan(u, delta, A, B, C, D, **kwargs):
                captured.append((u.detach().clone(), delta.detach().clone(),
                                 B.detach().clone(), C.detach().clone()))
                return u

            module._get_scan_fn = lambda: spy_scan
            with torch.no_grad():
                output = module(x, (H, W))

            self.assertEqual(output.shape, x.shape)
            self.assertEqual(len(captured), 4)
            self.assertEqual(len(expected), 4)
            for (_, dt_expected, B_expected, C_expected), (
                    u_actual, dt_actual, B_actual, C_actual) in zip(expected, captured):
                self.assertTrue(torch.equal(dt_actual, dt_expected))
                self.assertTrue(torch.equal(B_actual, B_expected))
                self.assertTrue(torch.equal(C_actual, C_expected))
                self.assertEqual(u_actual.shape, dt_actual.shape)

    def test_build_model_defaults_old_namespace_to_current(self):
        args = argparse.Namespace(
            device='cpu', discretization='euler', lam_nheads=8,
            trap_lambda=0.5, trap_boundary='euler', trap_lambda_per_dir=False,
            scan_routing='none', BCELoss_ratio=0.83, DiceLoss_ratio=0.17,
        )
        model, _ = build_model(args)
        modules = [m for m in model.modules() if isinstance(m, SAVSS_2D)]
        self.assertGreater(len(modules), 0)
        self.assertTrue(all(m.scan_param_order == 'current' for m in modules))

    def test_build_model_forwards_aligned_to_all_savss2d(self):
        args = argparse.Namespace(
            device='cpu', discretization='euler', lam_nheads=8,
            trap_lambda=0.5, trap_boundary='euler', trap_lambda_per_dir=False,
            scan_routing='none', scan_param_order='aligned',
            BCELoss_ratio=0.83, DiceLoss_ratio=0.17,
        )
        model, _ = build_model(args)
        modules = [m for m in model.modules() if isinstance(m, SAVSS_2D)]
        self.assertGreater(len(modules), 0)
        self.assertTrue(all(m.scan_param_order == 'aligned' for m in modules))

    def test_current_and_aligned_state_dict_keys_match(self):
        torch.manual_seed(0)
        current = SAVSS_2D(d_model=32, d_state=4, expand=1, scan_param_order='current')
        aligned = SAVSS_2D(d_model=32, d_state=4, expand=1, scan_param_order='aligned')
        self.assertEqual(set(current.state_dict().keys()), set(aligned.state_dict().keys()))
        aligned.load_state_dict(current.state_dict(), strict=True)

    def test_v2_does_not_change_scan_params(self):
        """asr_scan_v2 只消费已恢复的 y_scan；传给 kernel 的 dt/B/C 必须与
        _scan_params_for_order 的输出完全一致（current 和 aligned 都一样）。"""
        H, W = 3, 5
        batch_size, d_model = 1, 32
        x = torch.randn(batch_size, H * W, d_model)

        for scan_param_order in ('current', 'aligned'):
            module = SAVSS_2D(
                d_model=d_model,
                d_state=3,
                expand=1,
                scan_routing='asr_scan_v2',
                scan_param_order=scan_param_order,
                asr_v2_beta=0.25,
                asr_route_grid=2,
            ).eval()
            captured = []
            expected = []
            original_helper = module._scan_params_for_order

            def capture_params(order, dt, B, C, dB):
                params = original_helper(order, dt, B, C, dB)
                expected.append((order, *(p.detach().clone() for p in params)))
                return params

            module._scan_params_for_order = capture_params

            def spy_scan(u, delta, A, B, C, D, **kwargs):
                captured.append((u.detach().clone(), delta.detach().clone(),
                                 B.detach().clone(), C.detach().clone()))
                return u

            module._get_scan_fn = lambda: spy_scan
            with torch.no_grad():
                output = module(x, (H, W))

            self.assertEqual(output.shape, x.shape)
            self.assertEqual(len(captured), 4)
            self.assertEqual(len(expected), 4)
            for (_, dt_expected, B_expected, C_expected), (
                    u_actual, dt_actual, B_actual, C_actual) in zip(expected, captured):
                self.assertTrue(torch.equal(dt_actual, dt_expected))
                self.assertTrue(torch.equal(B_actual, B_expected))
                self.assertTrue(torch.equal(C_actual, C_expected))
                self.assertEqual(u_actual.shape, dt_actual.shape)


if __name__ == '__main__':
    unittest.main()
