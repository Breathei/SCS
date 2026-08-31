'''
Unit tests for SAVSS_2D trapezoidal scan and SASS inverse recovery.
'''

import sys
import unittest

import torch

# Import the project's model entry point first to avoid the circular import
# that occurs when importing SAVSS_layer before models/__init__.py has finished.
from models import build_model  # noqa: F401
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
    selective_scan_trapezoidal_fn,
    selective_scan_euler_pytorch_fn,
)
try:
    from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal_triton import (
        selective_scan_trapezoidal_triton_fn,
    )
    _TRITON_AVAILABLE = True
except Exception:
    selective_scan_trapezoidal_triton_fn = None
    _TRITON_AVAILABLE = False
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D


class TestSelectiveScanTrapezoidal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_inputs(self, batch_size=2, E=8, N=4, L=16, dtype=torch.float32, requires_grad=False):
        torch.manual_seed(42)
        u = torch.randn(batch_size, E, L, dtype=dtype, device=self.device, requires_grad=requires_grad)
        delta = torch.randn(batch_size, E, L, dtype=dtype, device=self.device, requires_grad=requires_grad) * 0.1
        A = -torch.rand(E, N, dtype=dtype, device=self.device, requires_grad=requires_grad)
        B = torch.randn(batch_size, N, L, dtype=dtype, device=self.device, requires_grad=requires_grad) * 0.1
        C = torch.randn(batch_size, N, L, dtype=dtype, device=self.device, requires_grad=requires_grad) * 0.1
        D = torch.randn(E, dtype=dtype, device=self.device, requires_grad=requires_grad)
        delta_bias = torch.randn(E, dtype=dtype, device=self.device, requires_grad=requires_grad) * 0.1
        return u, delta, A, B, C, D, delta_bias

    def _reference_trapezoidal(self, u, delta, A, B, C, D, delta_bias, lam=0.5):
        """Naive reference implementation matching the mathematical recurrence."""
        if delta_bias is not None:
            delta = delta + delta_bias.view(1, -1, 1)
        delta = torch.nn.functional.softplus(delta)
        Bb, E, L = u.shape
        N = A.shape[1]
        y = torch.empty_like(u)
        h = torch.zeros(Bb, E, N, dtype=u.dtype, device=u.device)

        B_t = B.transpose(1, 2)  # (B, L, N)
        C_t = C.transpose(1, 2)  # (B, L, N)

        for t in range(L):
            d_t = delta[:, :, t].unsqueeze(-1)
            u_t = u[:, :, t].unsqueeze(-1)
            b_t = B_t[:, t, :].unsqueeze(1)  # (B, 1, N)
            c_t = C_t[:, t, :].unsqueeze(1)  # (B, 1, N)
            A_bar_t = torch.exp(d_t * A.view(1, E, N))
            P_t = u_t * b_t  # (B, E, N)

            if t == 0:
                h = d_t * P_t
            else:
                P_prev = u[:, :, t - 1].unsqueeze(-1) * B_t[:, t - 1, :].unsqueeze(1)
                h = A_bar_t * h \
                    + (1 - lam) * d_t * A_bar_t * P_prev \
                    + lam * d_t * P_t

            y_t = (h * c_t).sum(dim=-1) + D.view(1, -1) * u[:, :, t]
            y[:, :, t] = y_t
        return y

    def test_formula_small_manual(self):
        """Compare against a naive hand-rolled reference on tiny inputs."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs(batch_size=1, E=2, N=2, L=4)
        lam = 0.5
        y_ref = self._reference_trapezoidal(u, delta, A, B, C, D, delta_bias, lam=lam)
        y = selective_scan_trapezoidal_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=lam
        )
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4))

    def test_output_shape(self):
        """Output shape should be (B, E, L); last_state shape (B, E, N)."""
        B, E, N, L = 2, 16, 8, 32
        u, delta, A, Bmat, C, D, delta_bias = self._make_inputs(B, E, N, L)
        y, h = selective_scan_trapezoidal_fn(
            u, delta, A, Bmat, C, D, delta_bias=delta_bias,
            delta_softplus=True, return_last_state=True, lam=0.5
        )
        self.assertEqual(y.shape, (B, E, L))
        self.assertEqual(h.shape, (B, E, N))

    def test_gradient_flow(self):
        """All inputs should receive non-None, finite gradients."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs(
            batch_size=1, E=4, N=4, L=8, requires_grad=True
        )
        y = selective_scan_trapezoidal_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=0.5
        )
        loss = y.sum()
        # Inputs may be non-leaf (e.g. scaled after creation); retain_grad()
        # so their .grad fields are populated by backward().
        for tensor in [u, delta, A, B, C, D, delta_bias]:
            tensor.retain_grad()
        loss.backward()
        for name, tensor in [
            ('u', u), ('delta', delta), ('A', A), ('B', B), ('C', C), ('D', D), ('delta_bias', delta_bias)
        ]:
            self.assertIsNotNone(tensor.grad, f'{name} has no gradient')
            self.assertTrue(torch.isfinite(tensor.grad).all(), f'{name} gradient has non-finite values')

    def test_euler_degradation_consistency(self):
        """lam=1.0 should match the pure-PyTorch Euler helper exactly."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs(batch_size=2, E=8, N=4, L=16)
        y_euler = selective_scan_euler_pytorch_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True
        )
        y_lam1 = selective_scan_trapezoidal_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=1.0
        )
        self.assertTrue(torch.allclose(y_euler, y_lam1, atol=1e-6, rtol=1e-5))


class TestSASSInverse(unittest.TestCase):

    def _get_sass(self, hw_shape):
        # Any small config works; sass only needs hw_shape.
        module = SAVSS_2D(d_model=16, d_state=4, expand=2).eval()
        return module.sass(hw_shape)

    def test_inverse_recovery(self):
        """x[:, o, :][:, inv_order, :] should recover x for all four orders."""
        for H, W in [(4, 4), (3, 5), (8, 8), (2, 7)]:
            orders, inverse_orders, _ = self._get_sass((H, W))
            L = H * W
            x = torch.arange(L).float().view(1, L, 1).expand(2, L, 4)
            for o, inv_o in zip(orders, inverse_orders):
                x_scan = x[:, o, :]
                x_rec = x_scan[:, inv_o, :]
                self.assertTrue(
                    torch.equal(x, x_rec),
                    f'Inverse failed for shape ({H}, {W})'
                )

    def test_o3_is_permutation(self):
        """o3_inverse should be a permutation of 0..L-1 after the bug fix."""
        for H, W in [(4, 4), (3, 5), (8, 8)]:
            orders, inverse_orders, _ = self._get_sass((H, W))
            o3_inv = inverse_orders[2]
            self.assertEqual(sorted(o3_inv), list(range(H * W)))

    def test_orders_are_permutations(self):
        """All four scan orders should be permutations of 0..L-1."""
        for H, W in [(4, 4), (3, 5), (8, 8)]:
            orders, _, _ = self._get_sass((H, W))
            L = H * W
            for idx, o in enumerate(orders):
                self.assertEqual(len(o), L, f'order {idx} wrong length for ({H},{W})')
                self.assertEqual(sorted(o), list(range(L)), f'order {idx} not a permutation')


@unittest.skipUnless(_TRITON_AVAILABLE and torch.cuda.is_available(),
                     'Triton kernel requires CUDA')
class TestSelectiveScanTrapezoidalTriton(unittest.TestCase):

    def _make_inputs(self, batch_size=2, E=8, N=4, L=16, requires_grad=False):
        torch.manual_seed(42)
        u = torch.randn(batch_size, E, L, dtype=torch.float32, device='cuda',
                        requires_grad=requires_grad)
        delta = torch.randn(batch_size, E, L, dtype=torch.float32, device='cuda',
                            requires_grad=requires_grad) * 0.1
        A = -torch.rand(E, N, dtype=torch.float32, device='cuda', requires_grad=requires_grad)
        B = torch.randn(batch_size, N, L, dtype=torch.float32, device='cuda',
                        requires_grad=requires_grad) * 0.1
        C = torch.randn(batch_size, N, L, dtype=torch.float32, device='cuda',
                        requires_grad=requires_grad) * 0.1
        D = torch.randn(E, dtype=torch.float32, device='cuda', requires_grad=requires_grad)
        delta_bias = torch.randn(E, dtype=torch.float32, device='cuda',
                                 requires_grad=requires_grad) * 0.1
        return u, delta, A, B, C, D, delta_bias

    def test_forward_matches_pytorch_lam_half(self):
        """Triton forward should match the pure-PyTorch reference for lam=0.5."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs()
        y_ref = selective_scan_trapezoidal_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=0.5
        )
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=0.5
        )
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-4, rtol=1e-3))

    def test_forward_matches_pytorch_lam_one(self):
        """lam=1.0 Triton should match the pure-PyTorch trapezoidal (Euler) reference."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs()
        y_ref = selective_scan_trapezoidal_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=1.0
        )
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=1.0
        )
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-4, rtol=1e-3))

    def test_backward_matches_pytorch(self):
        """Triton backward gradients should match the pure-PyTorch reference."""
        u, delta, A, B, C, D, delta_bias = self._make_inputs(requires_grad=True)
        dout = torch.randn_like(u)

        def clone_grad(tensor):
            return tensor.detach().clone().requires_grad_(True)

        u_ref = clone_grad(u)
        delta_ref = clone_grad(delta)
        A_ref = clone_grad(A)
        B_ref = clone_grad(B)
        C_ref = clone_grad(C)
        D_ref = clone_grad(D)
        delta_bias_ref = clone_grad(delta_bias)

        y_ref = selective_scan_trapezoidal_fn(
            u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref,
            delta_bias=delta_bias_ref, delta_softplus=True, lam=0.5
        )
        y_ref.backward(dout)

        u_tri = clone_grad(u)
        delta_tri = clone_grad(delta)
        A_tri = clone_grad(A)
        B_tri = clone_grad(B)
        C_tri = clone_grad(C)
        D_tri = clone_grad(D)
        delta_bias_tri = clone_grad(delta_bias)

        y_tri = selective_scan_trapezoidal_triton_fn(
            u_tri, delta_tri, A_tri, B_tri, C_tri, D_tri,
            delta_bias=delta_bias_tri, delta_softplus=True, lam=0.5
        )
        y_tri.backward(dout)

        for name, g_ref, g_tri in [
            ('u', u_ref.grad, u_tri.grad),
            ('delta', delta_ref.grad, delta_tri.grad),
            ('A', A_ref.grad, A_tri.grad),
            ('B', B_ref.grad, B_tri.grad),
            ('C', C_ref.grad, C_tri.grad),
            ('D', D_ref.grad, D_tri.grad),
            ('delta_bias', delta_bias_ref.grad, delta_bias_tri.grad),
        ]:
            self.assertIsNotNone(g_tri, f'{name} has no Triton gradient')
            self.assertTrue(torch.isfinite(g_tri).all(),
                            f'{name} Triton gradient has non-finite values')
            diff = (g_ref - g_tri).abs()
            self.assertTrue(
                diff.max() < 1e-3,
                f'{name} gradient max diff {diff.max().item():.6e} exceeds 1e-3'
            )

    def test_euler_degradation_via_mamba(self):
        """lam=1.0 Triton should be close to the fused mamba_ssm Euler kernel."""
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except Exception:
            self.skipTest('mamba_ssm selective_scan_fn not available')

        u, delta, A, B, C, D, delta_bias = self._make_inputs()
        y_mamba = selective_scan_fn(
            u, delta, A, B, C, D, z=None,
            delta_bias=delta_bias, delta_softplus=True, return_last_state=False
        )
        y_tri = selective_scan_trapezoidal_triton_fn(
            u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True, lam=1.0
        )
        self.assertTrue(torch.allclose(y_mamba, y_tri, atol=1e-3, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
