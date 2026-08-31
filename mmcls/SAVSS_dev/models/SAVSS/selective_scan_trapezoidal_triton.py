'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

Fused Triton kernel for exponential-trapezoidal selective scan.

Tensor layout is identical to mamba_ssm.ops.selective_scan_interface.selective_scan_fn:

    u, delta : (B, E, L)
    A        : (E, N)
    B, C     : (B, N, L)
    D        : (E,)
    delta_bias : (E,)

Recurrence (exponential-trapezoidal, lam=0.5 by default):

    h_t = exp(Δ_t A) h_{t-1}
          + (1-λ) Δ_t exp(Δ_t A) B_{t-1} x_{t-1}
          + λ Δ_t B_t x_t
    y_t = C_t^T h_t + D x_t

where Δ_t = softplus(delta_t + delta_bias).
At t=0 the previous term is missing, so the first step falls back to Euler:

    h_0 = Δ_0 B_0 x_0

This keeps lam=1.0 exactly equivalent to the standard Euler discretization.
'''

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@triton.jit
def _softplus(x):
    """Numerically stable softplus."""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _sigmoid(x):
    """Sigmoid implemented via softplus."""
    return tl.where(x > 0.0,
                    1.0 / (1.0 + tl.exp(-x)),
                    tl.exp(x) / (1.0 + tl.exp(x)))


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _selective_scan_trapezoidal_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    delta_bias_ptr,
    y_ptr,
    h_ptr,
    B_batch,
    E,
    L,
    N,
    u_stride_batch,
    u_stride_e,
    u_stride_l,
    delta_stride_batch,
    delta_stride_e,
    delta_stride_l,
    A_stride_e,
    A_stride_n,
    B_stride_batch,
    B_stride_n,
    B_stride_l,
    C_stride_batch,
    C_stride_n,
    C_stride_l,
    D_stride_e,
    delta_bias_stride_e,
    y_stride_batch,
    y_stride_e,
    y_stride_l,
    h_stride_batch,
    h_stride_e,
    h_stride_l,
    h_stride_n,
    lam: tl.constexpr,
    delta_softplus: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One CUDA block per (batch, inner-dim) sequence; sequential scan over L."""
    pid = tl.program_id(0)
    b = pid // E
    e = pid % E
    if b >= B_batch:
        return

    n_offs = tl.arange(0, BLOCK_N)
    mask_n = n_offs < N

    # Load per-channel constants.
    A_e = tl.load(A_ptr + e * A_stride_e + n_offs * A_stride_n,
                  mask=mask_n, other=0.0).to(tl.float32)
    D_e = tl.load(D_ptr + e * D_stride_e).to(tl.float32)
    if delta_softplus:
        delta_bias_e = tl.load(delta_bias_ptr + e * delta_bias_stride_e).to(tl.float32)
    else:
        delta_bias_e = 0.0

    one_minus_lam = 1.0 - lam

    # Base pointers for this (b, e).
    u_base = u_ptr + b * u_stride_batch + e * u_stride_e
    delta_base = delta_ptr + b * delta_stride_batch + e * delta_stride_e
    B_base = B_ptr + b * B_stride_batch
    C_base = C_ptr + b * C_stride_batch
    y_base = y_ptr + b * y_stride_batch + e * y_stride_e
    h_base = h_ptr + b * h_stride_batch + e * h_stride_e

    h = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(L):
        # Load scalar inputs for time t.
        u_t = tl.load(u_base + t * u_stride_l).to(tl.float32)
        delta_t = tl.load(delta_base + t * delta_stride_l).to(tl.float32)
        if delta_softplus:
            dt = _softplus(delta_t + delta_bias_e)
        else:
            dt = delta_t + delta_bias_e

        B_t = tl.load(B_base + n_offs * B_stride_n + t * B_stride_l,
                      mask=mask_n, other=0.0).to(tl.float32)
        C_t = tl.load(C_base + n_offs * C_stride_n + t * C_stride_l,
                      mask=mask_n, other=0.0).to(tl.float32)

        P_t = u_t * B_t

        if t == 0:
            h = dt * P_t
        else:
            u_prev = tl.load(u_base + (t - 1) * u_stride_l).to(tl.float32)
            B_prev = tl.load(B_base + n_offs * B_stride_n + (t - 1) * B_stride_l,
                            mask=mask_n, other=0.0).to(tl.float32)
            P_prev = u_prev * B_prev
            A_bar = tl.exp(dt * A_e)
            h = A_bar * h + one_minus_lam * dt * A_bar * P_prev + lam * dt * P_t

        # Save hidden state for backward.
        tl.store(h_base + t * h_stride_l + n_offs * h_stride_n,
                 h, mask=mask_n)

        # Output projection.
        y_t = tl.sum(h * C_t) + D_e * u_t
        tl.store(y_base + t * y_stride_l, y_t)


# --------------------------------------------------------------------------- #
# Backward kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _selective_scan_trapezoidal_bwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    delta_bias_ptr,
    h_ptr,
    dout_ptr,
    du_ptr,
    ddelta_ptr,
    dA_per_block_ptr,
    dB_ptr,
    dC_ptr,
    dD_per_block_ptr,
    ddelta_bias_per_block_ptr,
    B_batch,
    E,
    L,
    N,
    u_stride_batch,
    u_stride_e,
    u_stride_l,
    delta_stride_batch,
    delta_stride_e,
    delta_stride_l,
    A_stride_e,
    A_stride_n,
    B_stride_batch,
    B_stride_n,
    B_stride_l,
    C_stride_batch,
    C_stride_n,
    C_stride_l,
    D_stride_e,
    delta_bias_stride_e,
    ddelta_bias_per_block_stride_batch,
    ddelta_bias_per_block_stride_e,
    dA_per_block_stride_batch,
    dA_per_block_stride_e,
    dA_per_block_stride_n,
    dD_per_block_stride_batch,
    dD_per_block_stride_e,
    h_stride_batch,
    h_stride_e,
    h_stride_l,
    h_stride_n,
    dout_stride_batch,
    dout_stride_e,
    dout_stride_l,
    du_stride_batch,
    du_stride_e,
    du_stride_l,
    ddelta_stride_batch,
    ddelta_stride_e,
    ddelta_stride_l,
    dB_stride_batch,
    dB_stride_n,
    dB_stride_l,
    dC_stride_batch,
    dC_stride_n,
    dC_stride_l,
    lam: tl.constexpr,
    delta_softplus: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward scan.  One block per (batch, e); scan t from L-1 down to 0."""
    pid = tl.program_id(0)
    b = pid // E
    e = pid % E
    if b >= B_batch:
        return

    n_offs = tl.arange(0, BLOCK_N)
    mask_n = n_offs < N

    A_e = tl.load(A_ptr + e * A_stride_e + n_offs * A_stride_n,
                  mask=mask_n, other=0.0).to(tl.float32)
    D_e = tl.load(D_ptr + e * D_stride_e).to(tl.float32)
    if delta_softplus:
        delta_bias_e = tl.load(delta_bias_ptr + e * delta_bias_stride_e).to(tl.float32)
    else:
        delta_bias_e = 0.0

    one_minus_lam = 1.0 - lam

    u_base = u_ptr + b * u_stride_batch + e * u_stride_e
    delta_base = delta_ptr + b * delta_stride_batch + e * delta_stride_e
    B_base = B_ptr + b * B_stride_batch
    C_base = C_ptr + b * C_stride_batch
    h_base = h_ptr + b * h_stride_batch + e * h_stride_e
    dout_base = dout_ptr + b * dout_stride_batch + e * dout_stride_e
    du_base = du_ptr + b * du_stride_batch + e * du_stride_e
    ddelta_base = ddelta_ptr + b * ddelta_stride_batch + e * ddelta_stride_e
    dB_base = dB_ptr + b * dB_stride_batch
    dC_base = dC_ptr + b * dC_stride_batch
    ddelta_bias_per_block_base = ddelta_bias_per_block_ptr + b * ddelta_bias_per_block_stride_batch + e * ddelta_bias_per_block_stride_e
    dA_per_block_base = dA_per_block_ptr + b * dA_per_block_stride_batch + e * dA_per_block_stride_e
    dD_per_block_base = dD_per_block_ptr + b * dD_per_block_stride_batch + e * dD_per_block_stride_e

    # Adjoint variable.  At the start of an iteration for time t, dh holds
    # adj_{t+1}.  We compute adj_t, use it for gradients, then shift it into dh
    # for the next (earlier) time step.
    dh = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dA = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dD = 0.0
    ddelta_bias = 0.0

    # a_prev/dt_prev correspond to a_{t+1}/dt_{t+1} for the current t.
    a_prev = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dt_prev = 0.0

    for t in range(L - 1, -1, -1):
        u_t = tl.load(u_base + t * u_stride_l).to(tl.float32)
        delta_t = tl.load(delta_base + t * delta_stride_l).to(tl.float32)
        if delta_softplus:
            dt_raw = delta_t + delta_bias_e
            dt = _softplus(dt_raw)
            dt_prime = _sigmoid(dt_raw)
        else:
            dt_raw = delta_t + delta_bias_e
            dt = dt_raw
            dt_prime = 1.0

        B_t = tl.load(B_base + n_offs * B_stride_n + t * B_stride_l,
                      mask=mask_n, other=0.0).to(tl.float32)
        C_t = tl.load(C_base + n_offs * C_stride_n + t * C_stride_l,
                      mask=mask_n, other=0.0).to(tl.float32)
        h_t = tl.load(h_base + t * h_stride_l + n_offs * h_stride_n,
                      mask=mask_n, other=0.0).to(tl.float32)

        P_t = u_t * B_t
        A_bar = tl.exp(dt * A_e)

        dout_t = tl.load(dout_base + t * dout_stride_l).to(tl.float32)

        # adj_t = dout_t * C_t + a_{t+1} * adj_{t+1}
        # a_prev/dt_prev hold a_{t+1}/dt_{t+1} from the previous iteration.
        adj_t = dout_t * C_t + a_prev * dh

        # Gradients w.r.t. u_t and B_t.
        # Because u_t/B_t appear directly in h_t and (as the previous input)
        # directly in h_{t+1}, we accumulate both contributions.  The t=0 step
        # uses an Euler first step (full dt_0, no lam weighting).
        if t == 0:
            du_t = D_e * dout_t + dt * tl.sum(B_t * adj_t)
            dB_t = dt * u_t * adj_t
        else:
            du_t = D_e * dout_t + lam * dt * tl.sum(B_t * adj_t)
            dB_t = lam * dt * u_t * adj_t
        if t < L - 1:
            du_t += one_minus_lam * dt_prev * tl.sum(B_t * a_prev * dh)
            dB_t += one_minus_lam * dt_prev * u_t * a_prev * dh

        # Gradient w.r.t. C_t.
        dC_t = dout_t * h_t

        # Gradient w.r.t. A_e and delta_t (direct effects on h_t only).
        if t == 0:
            ddelta_t = dt_prime * tl.sum(adj_t * P_t)
        else:
            h_prev = tl.load(h_base + (t - 1) * h_stride_l + n_offs * h_stride_n,
                            mask=mask_n, other=0.0).to(tl.float32)
            u_prev = tl.load(u_base + (t - 1) * u_stride_l).to(tl.float32)
            B_prev = tl.load(B_base + n_offs * B_stride_n + (t - 1) * B_stride_l,
                            mask=mask_n, other=0.0).to(tl.float32)
            P_prev = u_prev * B_prev

            # dA contribution at step t.
            dA += adj_t * A_bar * (h_prev + one_minus_lam * dt * P_prev) * dt

            # ddelta_t.
            term = A_bar * A_e * (h_prev + one_minus_lam * dt * P_prev) \
                + one_minus_lam * A_bar * P_prev \
                + lam * P_t
            ddelta_t = dt_prime * tl.sum(adj_t * term)

        # Accumulate dD and ddelta_bias.
        dD += dout_t * u_t
        ddelta_bias += ddelta_t

        # Store per-time gradients.  dB and dC need atomic adds because
        # multiple e-blocks contribute to the same (b, n, t) index.
        tl.store(du_base + t * du_stride_l, du_t)
        tl.store(ddelta_base + t * ddelta_stride_l, ddelta_t)
        tl.atomic_add(dB_base + n_offs * dB_stride_n + t * dB_stride_l,
                      dB_t, mask=mask_n)
        tl.atomic_add(dC_base + n_offs * dC_stride_n + t * dC_stride_l,
                      dC_t, mask=mask_n)

        # Move to previous time step.
        dh = adj_t
        a_prev = A_bar
        dt_prev = dt

    # Store per-channel accumulated gradients.
    tl.store(dA_per_block_base + n_offs * dA_per_block_stride_n,
             dA, mask=mask_n)
    tl.store(dD_per_block_base, dD)
    # Per-block partial sum for delta_bias; Python side sums over batch.
    tl.store(ddelta_bias_per_block_base, ddelta_bias)


# --------------------------------------------------------------------------- #
# Autograd function
# --------------------------------------------------------------------------- #
class SelectiveScanTrapezoidalTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus, lam):
        B_batch, E, L = u.shape
        N = A.shape[1]
        device = u.device
        dtype = u.dtype

        # Make sure inputs are contiguous and on CUDA.
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        if delta_bias is not None:
            delta_bias = delta_bias.contiguous()

        y = torch.empty(B_batch, E, L, dtype=dtype, device=device)
        h = torch.empty(B_batch, E, L, N, dtype=torch.float32, device=device)

        grid = (B_batch * E,)
        BLOCK_N = triton.next_power_of_2(N)

        # Triton kernels cannot accept None pointers; use zero tensors instead.
        D_kernel = D if D is not None else torch.zeros(E, dtype=dtype, device=device)
        delta_bias_kernel = delta_bias if delta_bias is not None else torch.zeros(E, dtype=dtype, device=device)

        _selective_scan_trapezoidal_fwd_kernel[grid](
            u, delta, A, B, C, D_kernel, delta_bias_kernel, y, h,
            B_batch, E, L, N,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            D_kernel.stride(0),
            delta_bias_kernel.stride(0),
            y.stride(0), y.stride(1), y.stride(2),
            h.stride(0), h.stride(1), h.stride(2), h.stride(3),
            lam=lam,
            delta_softplus=delta_softplus,
            BLOCK_N=BLOCK_N,
        )

        ctx.delta_softplus = delta_softplus
        ctx.lam = lam
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, h)
        return y

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, h = ctx.saved_tensors
        B_batch, E, L = u.shape
        N = A.shape[1]
        device = u.device
        dtype = u.dtype

        dout = dout.contiguous()

        du = torch.empty_like(u)
        ddelta = torch.empty_like(delta)
        dA_per_block = torch.empty(B_batch, E, N, dtype=torch.float32, device=device)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dD_per_block = torch.empty(B_batch, E, dtype=torch.float32, device=device)
        ddelta_bias_per_block = torch.empty(B_batch, E, dtype=torch.float32, device=device)

        grid = (B_batch * E,)
        BLOCK_N = triton.next_power_of_2(N)

        D_kernel = D if D is not None else torch.zeros(E, dtype=dtype, device=device)
        delta_bias_kernel = delta_bias if delta_bias is not None else torch.zeros(E, dtype=dtype, device=device)

        _selective_scan_trapezoidal_bwd_kernel[grid](
            u, delta, A, B, C, D_kernel, delta_bias_kernel, h, dout,
            du, ddelta, dA_per_block, dB, dC, dD_per_block, ddelta_bias_per_block,
            B_batch, E, L, N,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            D_kernel.stride(0),
            delta_bias_kernel.stride(0),
            ddelta_bias_per_block.stride(0), ddelta_bias_per_block.stride(1),
            dA_per_block.stride(0), dA_per_block.stride(1), dA_per_block.stride(2),
            dD_per_block.stride(0), dD_per_block.stride(1),
            h.stride(0), h.stride(1), h.stride(2), h.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2),
            du.stride(0), du.stride(1), du.stride(2),
            ddelta.stride(0), ddelta.stride(1), ddelta.stride(2),
            dB.stride(0), dB.stride(1), dB.stride(2),
            dC.stride(0), dC.stride(1), dC.stride(2),
            lam=ctx.lam,
            delta_softplus=ctx.delta_softplus,
            BLOCK_N=BLOCK_N,
        )

        dA = dA_per_block.sum(dim=0)
        dD = dD_per_block.sum(dim=0)
        ddelta_bias = ddelta_bias_per_block.sum(dim=0)

        # dD should match D's dtype if D exists.
        if D is not None:
            dD = dD.to(D.dtype)
        else:
            dD = None

        dA = dA.to(A.dtype)
        ddelta_bias = ddelta_bias.to(delta_bias.dtype) if delta_bias is not None else None

        # Gradients for arguments:
        # u, delta, A, B, C, D, delta_bias, delta_softplus, lam
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def selective_scan_trapezoidal_triton_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=True,
    return_last_state=False,
    lam=0.5,
):
    """
    Fused Triton implementation of exponential-trapezoidal selective scan.

    Parameters match `selective_scan_trapezoidal_fn` exactly so it can be used
    as a drop-in replacement inside SAVSS_2D.
    """
    if z is not None:
        # Gating is handled outside this function, but accept the argument
        # for signature compatibility.
        pass

    if u.device.type != "cuda":
        raise RuntimeError(
            "selective_scan_trapezoidal_triton_fn requires CUDA tensors. "
            "Use selective_scan_trapezoidal_fn on CPU."
        )

    y = SelectiveScanTrapezoidalTriton.apply(
        u, delta, A, B, C, D, delta_bias, delta_softplus, lam
    )

    if return_last_state:
        # last_state is not directly returned by the Triton kernel; fall back
        # to computing it from the saved hidden states if requested.
        # h shape: (B, E, L, N); last_state = h[:, :, -1, :]
        raise NotImplementedError("return_last_state is not yet supported by the Triton kernel.")

    return y
