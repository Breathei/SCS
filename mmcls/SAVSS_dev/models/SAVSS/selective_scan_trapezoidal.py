'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

Pure-PyTorch reference implementations of selective scan discretizations.

These functions are intended as a correctness reference and as a fallback when
mamba_ssm's fused kernel is unavailable.  They implement the same tensor layout
as mamba_ssm.ops.selective_scan_interface.selective_scan_fn:

    u, delta : (B, E, L)
    A        : (E, N)
    B, C     : (B, N, L)
    D        : (E,)
    delta_bias : (E,)

Two discretizations are provided:

1. exponential-Euler (lam=1.0):
       h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t

2. Mamba-3 style exponential-trapezoidal (lam=0.5 by default):
       h_t = exp(Δ_t A) h_{t-1}
             + (1-λ_t) Δ_t exp(Δ_t A) B_{t-1} x_{t-1}
             + λ_t Δ_t B_t x_t

3. Gated trapezoidal (trainable gate g, default init → 0):
       h_t = exp(Δ_t A) h_{t-1}
             + Δ_t B_t x_t
             + g * (1-λ_t) * Δ_t * (exp(Δ_t A) B_{t-1} x_{t-1} - B_t x_t)

   gate=0 recovers Euler; gate=1 recovers the original trapezoidal rule.

In all cases the output is:
       y_t = C_t^T h_t + D x_t

The first step (t=0) has no previous input.  Two boundary modes are
supported:

    boundary="euler"    : h_0 = Δ_0 B_0 x_0  (matches mamba_ssm fused kernel)
    boundary="zero_prev": h_0 = Δ_0 [B_0 x_0 + g*(1-λ_0)*(0 - B_0 x_0)]

lam may be a float scalar (fixed) or a tensor of shape (B, L, NH) with one
data-dependent mixing weight per token per head; channel e uses head
index e // (E // NH), so E must be divisible by NH.
'''

import torch
import torch.nn.functional as F


def selective_scan_trapezoidal_fn(
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
    gate=1.0,
    boundary="euler",
    gated=False,
):
    """
    Pure-PyTorch exponential-trapezoidal selective scan.

    Parameters
    ----------
    u : torch.Tensor
        Input sequence, shape (B, E, L).
    delta : torch.Tensor
        Raw delta (before bias/softplus), shape (B, E, L).
    A : torch.Tensor
        Continuous state matrix (diagonal), shape (E, N).
    B, C : torch.Tensor
        Input-dependent weights, shape (B, N, L).
    D : torch.Tensor, optional
        Skip connection scale, shape (E,).
    z : torch.Tensor, optional
        Kept for signature compatibility with mamba_ssm; gating is done
        outside this function.
    delta_bias : torch.Tensor, optional
        Bias added to delta before softplus, shape (E,).
    delta_softplus : bool
        Whether to apply softplus to delta after adding the bias.
    return_last_state : bool
        If True, return (y, last_state).
    lam : float or torch.Tensor
        Trapezoidal mixing weight.  A float scalar gives a fixed weight
        (lam=1.0 recovers Euler; lam=0.5 gives the standard trapezoidal
        rule).  A tensor of shape (B, L, NH) gives a data-dependent,
        per-token weight for each of NH heads; channel e uses head
        e // (E // NH).
    gate : float or torch.Tensor
        Deprecated.  The gate is kept only for compatibility with older
        call sites and tests; in the single-parameter mode it is fixed to
        1.0 and only `lam` is learnable.
    boundary : {"euler", "zero_prev"}
        How to handle the first time step where P_{t-1} is undefined.
    gated : bool
        Ignored.  Present only for signature compatibility with the Triton
        kernel, which uses the same keyword.

    Returns
    -------
    y : torch.Tensor
        Output sequence, shape (B, E, L).
    last_state : torch.Tensor, optional
        Final hidden state, shape (B, E, N); returned if return_last_state=True.
    """
    if z is not None:
        # This function does not apply SiLU gating; that is done in SAVSS_2D.
        # We accept the argument only to keep the call signature drop-in.
        pass

    B_batch, E, L = u.shape
    N = A.shape[1]
    dtype = u.dtype
    device = u.device

    # Δ preprocessing: add bias and optional softplus.
    if delta_bias is not None:
        delta_processed = delta + delta_bias.view(1, -1, 1)
    else:
        delta_processed = delta
    if delta_softplus:
        delta_processed = F.softplus(delta_processed)

    # Discretized state decay: (B, E, L, N).
    # A is negative, delta is positive -> A_bar in (0, 1].
    A_bar = torch.exp(delta_processed.unsqueeze(-1) * A.view(1, E, 1, N))

    # P_t = B_t * x_t, shape (B, E, L, N).
    # B is (B, N, L); bring L to the front, unsqueeze E, multiply by u (B, L, E, 1).
    B_t = B.transpose(1, 2)                       # (B, L, N)
    u_t = u.transpose(1, 2)                       # (B, L, E)
    P = u_t.unsqueeze(-1) * B_t.unsqueeze(2)      # (B, L, E, N)
    P = P.permute(0, 2, 1, 3).contiguous()        # (B, E, L, N)

    # Prepare output.
    y = torch.empty(B_batch, E, L, dtype=dtype, device=device)
    h = torch.zeros(B_batch, E, N, dtype=dtype, device=device)

    # lam: fixed scalar, 0-dim tensor (learnable scalar), or (B, L, NH) tensor
    # expanded to (B, E, L) so that channel e uses head e // (E // NH).
    lam_is_data_dependent = torch.is_tensor(lam) and lam.dim() == 3
    if lam_is_data_dependent:
        NH = lam.shape[-1]
        if lam.shape[0] != B_batch or lam.shape[1] != L:
            raise ValueError(
                f"tensor lam must have shape (B, L, NH)=({B_batch}, {L}, NH), "
                f"got {tuple(lam.shape)}"
            )
        if E % NH != 0:
            raise ValueError(f"E={E} must be divisible by NH={NH}")
        lam_full = lam.permute(0, 2, 1).repeat_interleave(E // NH, dim=1)

    for t in range(L):
        d_t = delta_processed[:, :, t].unsqueeze(-1)          # (B, E, 1)
        P_t = P[:, :, t, :]                         # (B, E, N)
        A_bar_t = A_bar[:, :, t, :]                 # (B, E, N)
        if lam_is_data_dependent:
            lam_t = lam_full[:, :, t].unsqueeze(-1)  # (B, E, 1)
        else:
            lam_t = lam

        if t == 0:
            # No previous input; choose boundary condition.
            if boundary == "euler":
                # Matches the mamba_ssm fused kernel boundary.
                h = d_t * P_t
            else:  # boundary == "zero_prev": assume P_{-1} = 0
                h = d_t * (P_t + gate * (1.0 - lam_t) * (-P_t))
        else:
            P_prev = P[:, :, t - 1, :]              # (B, E, N)
            # Euler term + gated trapezoidal correction.
            h = A_bar_t * h + d_t * P_t \
                + gate * (1.0 - lam_t) * d_t * (A_bar_t * P_prev - P_t)

        # y_t = C_t^T h_t + D * u_t
        C_now = C[:, :, t]                          # (B, N)
        y_t = (h * C_now.unsqueeze(1)).sum(dim=-1)  # (B, E)
        if D is not None:
            y_t = y_t + D.view(1, -1) * u[:, :, t]  # (B, E)
        y[:, :, t] = y_t

    if return_last_state:
        return y, h
    return y


def selective_scan_euler_pytorch_fn(
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
):
    """
    Pure-PyTorch exponential-Euler selective scan.

    This is exactly the lam=1.0 specialisation of
    :func:`selective_scan_trapezoidal_fn`, provided as a convenience fallback
    when mamba_ssm's fused kernel is unavailable.
    """
    return selective_scan_trapezoidal_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        lam=1.0,
    )
