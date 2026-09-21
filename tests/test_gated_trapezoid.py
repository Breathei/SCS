"""
Unit tests for the single-parameter learnable trapezoidal selective scan.

The original gated formulation had two scalars (gate g and lambda λ), but
math shows they only appear as g(1-λ). This file now tests the simplified
mode where g is fixed to 1.0 and only λ is learnable.
"""
import sys
import unittest

import torch

# Import the project's model entry point first to avoid circular imports.
from models import build_model  # noqa: F401
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
    selective_scan_trapezoidal_fn,
    selective_scan_euler_pytorch_fn,
)
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal_triton import (
    selective_scan_trapezoidal_triton_fn,
)
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D


class TestLearnableTrapezoidalFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_inputs(self, B=2, E=8, N=4, L=16, requires_grad=False):
        torch.manual_seed(42)
        u = torch.randn(B, E, L, device=self.device, requires_grad=requires_grad)
        delta = torch.randn(B, E, L, device=self.device, requires_grad=requires_grad) * 0.1
        A = -torch.rand(E, N, device=self.device, requires_grad=requires_grad)
        Bmat = torch.randn(B, N, L, device=self.device, requires_grad=requires_grad) * 0.1
        C = torch.randn(B, N, L, device=self.device, requires_grad=requires_grad) * 0.1
        D = torch.randn(E, device=self.device, requires_grad=requires_grad)
        delta_bias = torch.randn(E, device=self.device, requires_grad=requires_grad) * 0.1
        return u, delta, A, Bmat, C, D, delta_bias

    def test_lam_one_equals_euler(self):
        """λ=1 with gate=1 must reduce to the Euler recurrence."""
        u, d, A, B, C, D, db = self._make_inputs()
        y_euler = selective_scan_euler_pytorch_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True
        )
        y_lam_one = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=1.0, gate=1.0, boundary="euler"
        )
        self.assertTrue(torch.allclose(y_euler, y_lam_one, atol=1e-6, rtol=1e-5))

    def test_lam_half_equals_original_trapezoid(self):
        """λ=0.5 with gate=1 must equal the standard trapezoidal recurrence."""
        u, d, A, B, C, D, db = self._make_inputs()
        y_trap = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=0.5
        )
        y_lam_half = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=0.5, gate=1.0, boundary="euler"
        )
        self.assertTrue(torch.allclose(y_trap, y_lam_half, atol=1e-6, rtol=1e-5))

    def test_lam_gradient_nonzero(self):
        """The scalar λ must receive a non-zero, finite gradient."""
        u, d, A, B, C, D, db = self._make_inputs(requires_grad=True)
        lam = torch.tensor(0.5, device=self.device, requires_grad=True)
        y = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=lam, gate=1.0, boundary="euler"
        )
        y.sum().backward()
        self.assertIsNotNone(lam.grad)
        self.assertTrue(torch.isfinite(lam.grad))
        self.assertTrue(lam.grad.abs().item() > 0.0)

    def test_various_shapes(self):
        """λ=1/0.5 equivalence on multiple (B, E, N, L) combinations."""
        configs = [(1, 4, 2, 8), (2, 16, 8, 32), (3, 8, 4, 7)]
        for B, E, N, L in configs:
            u, d, A, Bmat, C, D, db = self._make_inputs(B, E, N, L)
            y_euler = selective_scan_euler_pytorch_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True
            )
            y_lam_one = selective_scan_trapezoidal_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True,
                lam=1.0, gate=1.0, boundary="euler"
            )
            y_trap = selective_scan_trapezoidal_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True,
                lam=0.5
            )
            y_lam_half = selective_scan_trapezoidal_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True,
                lam=0.5, gate=1.0, boundary="euler"
            )
            self.assertTrue(torch.allclose(y_euler, y_lam_one, atol=1e-5, rtol=1e-4),
                            f'lam=1 failed for shape {(B,E,N,L)}')
            self.assertTrue(torch.allclose(y_trap, y_lam_half, atol=1e-5, rtol=1e-4),
                            f'lam=0.5 failed for shape {(B,E,N,L)}')


@unittest.skipUnless(torch.cuda.is_available(), "Triton tests require CUDA")
class TestLearnableTrapezoidalTriton(unittest.TestCase):

    def _make_inputs(self, B=2, E=8, N=4, L=16, requires_grad=False):
        torch.manual_seed(123)
        u = torch.randn(B, E, L, device='cuda', requires_grad=requires_grad)
        delta = torch.randn(B, E, L, device='cuda', requires_grad=requires_grad) * 0.1
        A = -torch.rand(E, N, device='cuda', requires_grad=requires_grad)
        Bmat = torch.randn(B, N, L, device='cuda', requires_grad=requires_grad) * 0.1
        C = torch.randn(B, N, L, device='cuda', requires_grad=requires_grad) * 0.1
        D = torch.randn(E, device='cuda', requires_grad=requires_grad)
        delta_bias = torch.randn(E, device='cuda', requires_grad=requires_grad) * 0.1
        return u, delta, A, Bmat, C, D, delta_bias

    def test_triton_lam_one_equals_euler(self):
        """Triton λ=1, gate=1 must match the PyTorch Euler reference."""
        u, d, A, B, C, D, db = self._make_inputs()
        y_ref = selective_scan_euler_pytorch_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True
        )
        gate = torch.tensor(1.0, device='cuda')
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=1.0, gate=gate, boundary="euler", gated=True
        )
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-4))

    def test_triton_lam_half_equals_trapezoid(self):
        """Triton λ=0.5, gate=1 must match the PyTorch trapezoid reference."""
        u, d, A, B, C, D, db = self._make_inputs()
        y_ref = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True, lam=0.5
        )
        gate = torch.tensor(1.0, device='cuda')
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=0.5, gate=gate, boundary="euler", gated=True
        )
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-4))

    def test_triton_learnable_lambda_gradient_matches_reference(self):
        """Triton learnable scalar λ gradient must match PyTorch reference."""
        u, d, A, B, C, D, db = self._make_inputs(requires_grad=True)
        gate = torch.tensor(1.0, device='cuda')
        lam_tri = torch.tensor(0.5, device='cuda', requires_grad=True)
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=lam_tri, gate=gate, boundary="euler", gated=True
        )
        y_tri.sum().backward()
        tri_grad = lam_tri.grad.detach().clone()

        lam_ref = torch.tensor(0.5, device='cuda', requires_grad=True)
        y_ref = selective_scan_trapezoidal_fn(
            u.detach(), d.detach(), A.detach(), B.detach(), C.detach(), D.detach(),
            delta_bias=db.detach(), delta_softplus=True,
            lam=lam_ref, gate=1.0, boundary="euler"
        )
        y_ref.sum().backward()
        ref_grad = lam_ref.grad.detach().clone()

        self.assertTrue(torch.allclose(tri_grad, ref_grad, atol=1e-4, rtol=1e-3))

    def test_triton_zero_prev_boundary(self):
        """Triton zero_prev boundary must match PyTorch reference."""
        u, d, A, B, C, D, db = self._make_inputs()
        gate = torch.tensor(1.0, device='cuda')
        y_ref = selective_scan_trapezoidal_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=0.5, gate=gate, boundary="zero_prev"
        )
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, d, A, B, C, D, delta_bias=db, delta_softplus=True,
            lam=0.5, gate=gate, boundary="zero_prev", gated=True
        )
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-4))

    def test_triton_various_shapes(self):
        """Single-parameter mode works across multiple shapes."""
        configs = [(1, 4, 2, 8), (2, 16, 8, 32), (3, 8, 4, 7)]
        for B, E, N, L in configs:
            u, d, A, Bmat, C, D, db = self._make_inputs(B, E, N, L)
            gate = torch.tensor(1.0, device='cuda')
            y_ref = selective_scan_trapezoidal_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True,
                lam=0.5, gate=gate, boundary="euler"
            )
            y_tri = selective_scan_trapezoidal_triton_fn(
                u, d, A, Bmat, C, D, delta_bias=db, delta_softplus=True,
                lam=0.5, gate=gate, boundary="euler", gated=True
            )
            self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-4, rtol=1e-3),
                            f'Triton single-param failed for shape {(B,E,N,L)}')


class TestLearnableTrapezoidalSAVSS2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_module(self, **kwargs):
        defaults = dict(d_model=32, d_state=4, expand=2)
        defaults.update(kwargs)
        return SAVSS_2D(**defaults).to(self.device)

    def test_module_has_trainable_lambda(self):
        """gated_trapezoid module must expose a trainable trap_lambda_logit."""
        m = self._make_module(discretization='gated_trapezoid', trap_lambda=0.5)
        self.assertTrue(hasattr(m, 'trap_lambda_logit'))
        self.assertIsInstance(m.trap_lambda_logit, torch.nn.Parameter)
        self.assertTrue(m.trap_lambda_logit.requires_grad)
        self.assertEqual(m.trap_lambda_logit.shape, torch.Size([]))
        self.assertFalse(hasattr(m, 'trap_gate_logit'))

    def test_module_lambda_is_sigmoid(self):
        """sigmoid(logit(0.5)) should be close to 0.5 (default init)."""
        m = self._make_module(discretization='gated_trapezoid', trap_lambda=0.5)
        lam = torch.sigmoid(m.trap_lambda_logit)
        self.assertTrue(0.0 < lam.item() < 1.0)
        self.assertAlmostEqual(lam.item(), 0.5, places=5)

    def test_module_runs_all_sass_paths(self):
        """gated_trapezoid forward must work for various HxW (4 SASS paths)."""
        for H, W in [(4, 4), (3, 5), (2, 7), (8, 8)]:
            m = self._make_module(discretization='gated_trapezoid')
            L = H * W
            x = torch.randn(2, L, 32, device=self.device, requires_grad=True)
            y = m(x, (H, W))
            self.assertEqual(y.shape, (2, L, 32))
            self.assertFalse(hasattr(m, 'last_trap_gate'))
            self.assertIsNotNone(m.last_trap_lambda)
            self.assertTrue(0.0 < m.last_trap_lambda.item() < 1.0)
            y.sum().backward()
            self.assertIsNotNone(m.trap_lambda_logit.grad)
            self.assertTrue(torch.isfinite(m.trap_lambda_logit.grad))
            self.assertTrue(m.trap_lambda_logit.grad.abs().item() > 0.0)

    def test_lam_half_matches_trapezoid_module(self):
        """Module with λ=0.5 should behave like the fixed trapezoid module."""
        torch.manual_seed(0)
        H, W = 4, 4
        L = H * W
        x = torch.randn(2, L, 32, device=self.device)

        m_gated = self._make_module(discretization='gated_trapezoid', trap_lambda=0.5)
        # Force λ = 0.5 (sigmoid(0) = 0.5); gate is fixed to 1.0 internally.
        m_gated.trap_lambda_logit.data.fill_(0.0)
        with torch.no_grad():
            y_lam_half = m_gated(x, (H, W))

        m_trap = self._make_module(discretization='trapezoidal_fixed', trap_lambda=0.5)
        # Align all shared parameters; exclude the gated-only lambda logit.
        sd = {k: v for k, v in m_gated.state_dict().items()
              if not k.startswith('trap_lambda_logit')}
        m_trap.load_state_dict(sd, strict=False)
        with torch.no_grad():
            y_trap = m_trap(x, (H, W))

        diff = (y_lam_half - y_trap).abs().max().item()
        rel = diff / (y_trap.abs().mean().item() + 1e-6)
        self.assertLess(rel, 1e-2, f'relative diff too large: {rel}')


if __name__ == '__main__':
    unittest.main()
