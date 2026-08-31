"""Quick gradient check for Triton trapezoidal selective scan vs PyTorch reference."""
import sys
sys.path.insert(0, '/home/xby/SCSegamba')

import torch

from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
    selective_scan_trapezoidal_fn,
)
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal_triton import (
    selective_scan_trapezoidal_triton_fn,
)


def make_inputs(batch_size, E, N, L, dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    u = torch.randn(batch_size, E, L, dtype=dtype, device='cuda', requires_grad=True)
    delta = torch.randn(batch_size, E, L, dtype=dtype, device='cuda', requires_grad=True) * 0.1
    A = -torch.rand(E, N, dtype=dtype, device='cuda', requires_grad=True)
    B = torch.randn(batch_size, N, L, dtype=dtype, device='cuda', requires_grad=True) * 0.1
    C = torch.randn(batch_size, N, L, dtype=dtype, device='cuda', requires_grad=True) * 0.1
    D = torch.randn(E, dtype=dtype, device='cuda', requires_grad=True)
    delta_bias = torch.randn(E, dtype=dtype, device='cuda', requires_grad=True) * 0.1
    return u, delta, A, B, C, D, delta_bias


def compare(lam=0.5, batch_size=2, E=8, N=4, L=16):
    u, delta, A, Bmat, C, D, delta_bias = make_inputs(batch_size, E, N, L)

    # Reference PyTorch.
    u_ref = u.detach().clone().requires_grad_(True)
    delta_ref = delta.detach().clone().requires_grad_(True)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = Bmat.detach().clone().requires_grad_(True)
    C_ref = C.detach().clone().requires_grad_(True)
    D_ref = D.detach().clone().requires_grad_(True)
    delta_bias_ref = delta_bias.detach().clone().requires_grad_(True)

    y_ref = selective_scan_trapezoidal_fn(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref,
        delta_bias=delta_bias_ref, delta_softplus=True, lam=lam
    )
    dout = torch.randn_like(y_ref)
    y_ref.backward(dout)

    # Triton.
    u_tri = u.detach().clone().requires_grad_(True)
    delta_tri = delta.detach().clone().requires_grad_(True)
    A_tri = A.detach().clone().requires_grad_(True)
    B_tri = Bmat.detach().clone().requires_grad_(True)
    C_tri = C.detach().clone().requires_grad_(True)
    D_tri = D.detach().clone().requires_grad_(True)
    delta_bias_tri = delta_bias.detach().clone().requires_grad_(True)

    y_tri = selective_scan_trapezoidal_triton_fn(
        u_tri, delta_tri, A_tri, B_tri, C_tri, D_tri,
        delta_bias=delta_bias_tri, delta_softplus=True, lam=lam
    )
    y_tri.backward(dout)

    print(f"\nlam={lam} B={batch_size} E={E} N={N} L={L}")
    print(f"y diff max: {(y_ref - y_tri).abs().max().item():.6e}")
    for name, g_ref, g_tri in [
        ('u', u_ref.grad, u_tri.grad),
        ('delta', delta_ref.grad, delta_tri.grad),
        ('A', A_ref.grad, A_tri.grad),
        ('B', B_ref.grad, B_tri.grad),
        ('C', C_ref.grad, C_tri.grad),
        ('D', D_ref.grad, D_tri.grad),
        ('delta_bias', delta_bias_ref.grad, delta_bias_tri.grad),
    ]:
        diff = (g_ref - g_tri).abs()
        print(f"{name} grad diff max: {diff.max().item():.6e}  mean: {diff.mean().item():.6e}")


if __name__ == '__main__':
    compare(lam=1.0, batch_size=2, E=8, N=4, L=16)
    compare(lam=0.5, batch_size=2, E=8, N=4, L=16)
    compare(lam=0.5, batch_size=1, E=4, N=2, L=4)
