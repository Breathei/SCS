'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

本文件实现了 SAVSS（Selective Scan-based Vision State Space）模型的核心层：
1. SAVSS_2D: 2D 选择性扫描 / 状态空间模块，沿图像的 4 个方向做选择性扫描（SS2D 机制），
   将 2D 视觉 token 序列化后调用 mamba_ssm 的 selective_scan_fn 进行状态空间建模。
2. SAVSS_Layer: 由 LayerNorm、GBC 卷积、SAVSS_2D、PAF 融合、DropPath、残差连接等组成的完整层，
   是 SAVSS backbone 的基本组成单元。

依赖：mamba_ssm（提供 selective_scan_fn 与 RMSNorm）。
'''
import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from models.GBC import GBC, BottConv
from models.PAF import PAF
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
    _SCAN_DIAG_ENABLED,
    _compute_scan_diag,
)


def _mamba_install_message():
    return (
        "mamba-ssm CUDA extension is unavailable. Install or rebuild "
        "mamba-ssm and causal-conv1d for the active PyTorch/CUDA stack. "
        "For RTX 50-series / Blackwell GPUs, use a recent CUDA 12.8+ "
        "PyTorch wheel and set TORCH_CUDA_ARCH_LIST=\"12.0\" when "
        "building from source."
    )


# 构建 RMSNorm。RMSNorm 是 Mamba 系列模型常用的归一化层，
# 相比 LayerNorm 只计算输入的均方根进行缩放，通常与 selective scan 配合使用。
def _build_rms_norm(embed_dims):
    try:
        from mamba_ssm.ops.triton.layernorm import RMSNorm
    except Exception as exc:
        raise RuntimeError(
            "mamba-ssm Triton RMSNorm is unavailable. Either install a "
            "mamba-ssm/Triton build matching the active CUDA stack or set "
            "use_rms_norm=False in the SAVSS layer config. " +
            _mamba_install_message()
        ) from exc
    return RMSNorm(embed_dims)

# SAVSS_2D: 2D 选择性扫描模块（对应 SS2D 思想）。
# 将 HxW 的图像 token 沿 4 个不同方向展平成一维序列，分别输入 Mamba 的 selective scan，
# 再把 4 个方向的扫描结果逆序拼回 2D，从而在不破坏空间结构的前提下建模全局关系。
class SAVSS_2D(nn.Module):
    # 诊断日志全局开关（由 engine.py 在每 200 个 training step 触发一次）。
    _diag_active = False
    _diag_step = -1
    # asr_scan_v2 路由诊断开关（同样由 engine.py 周期性触发）。
    _router_diag_active = False
    _router_diag_step = -1
    _router_diag_entry = None

    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False,
            conv_bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
            discretization="euler",
            lam_nheads=8,
            trap_lambda=0.5,
            trap_boundary="euler",
            trap_lambda_per_dir=False,
            scan_routing='none',
            scan_param_order='current',
            asr_beta=0.1,
            asr_route_grid=1,
            asr_v2_beta=0.25,
            num_scans=4,
    ):
        super().__init__()
        # 保存状态空间模型超参数。
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        # 经过 expand 后的隐藏维度，selective scan 的内部通道数。
        self.d_inner = int(self.expand * self.d_model)
        # dt（离散化步长）的投影秩，auto 时取 d_model/16 向上取整。
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        # 4 个扫描方向：横向蛇形、纵向蛇形、主对角线蛇形、反对角线蛇形。
        self.n_directions = 4

        self.scan_routing = scan_routing
        assert self.scan_routing in ('none', 'asr', 'asr_scan', 'asr_scan_v2'), \
            f"Unsupported scan_routing: {self.scan_routing}"
        self.scan_param_order = scan_param_order
        assert self.scan_param_order in ('current', 'aligned'), \
            f"Unsupported scan_param_order: {self.scan_param_order}"
        self.asr_beta = float(asr_beta)
        assert 0.0 <= self.asr_beta <= 1.0, \
            f"asr_beta must be in [0, 1], got {self.asr_beta}"
        self.asr_route_grid = int(asr_route_grid)
        assert self.asr_route_grid in (1, 2, 4), \
            f"asr_route_grid must be one of (1, 2, 4), got {self.asr_route_grid}"
        self.asr_v2_beta = float(asr_v2_beta)
        assert 0.0 <= self.asr_v2_beta <= 1.0, \
            f"asr_v2_beta must be in [0, 1], got {self.asr_v2_beta}"

        # 扫描剪枝：只保留 sass() 固定语义顺序 [横, 纵, 主对角, 副对角] 的
        # 前 num_scans 条路径；被剪掉的路径不生成排列、不做 SSM 前向、
        # 不参与融合。默认 4 与原实现逐 bit 一致。
        self.num_scans = int(num_scans)
        assert self.num_scans in (1, 2, 4), \
            f"num_scans must be one of (1, 2, 4), got {self.num_scans}"
        if self.scan_routing != 'none':
            # 路由融合的 4 路权重/分支索引按 4 条扫描设计，剪枝组合未定义。
            assert self.num_scans == 4, \
                f"scan_routing={self.scan_routing} requires num_scans=4"

        self.discretization = discretization
        assert self.discretization in (
            "euler", "trapezoid", "trapezoidal_fixed", "trapezoidal_data", "gated_trapezoid"
        ), f"Unsupported discretization: {self.discretization}"

        # 数据依赖的梯形混合权重 lam：
        # lam = sigmoid(lam_proj(x))，逐 token、逐 head，形状 (B, L, lam_nheads)。
        # 权重与偏置均零初始化 → 初始 lam ≡ sigmoid(0) = 0.5，
        # 训练起点与已验证的固定 lam=0.5 版本完全一致（梯度仍可使权重离开 0）。
        self.lam_nheads = lam_nheads
        if self.discretization == "trapezoidal_data":
            assert self.d_inner % lam_nheads == 0, \
                f"d_inner={self.d_inner} must be divisible by lam_nheads={lam_nheads}"
            self.lam_proj = nn.Linear(d_model, lam_nheads, bias=True)
            nn.init.zeros_(self.lam_proj.weight)
            nn.init.zeros_(self.lam_proj.bias)

        # Gated trapezoid 配置。
        self.trap_lambda = float(trap_lambda)
        self.trap_boundary = trap_boundary
        self.trap_lambda_per_dir = trap_lambda_per_dir
        assert self.trap_boundary in ("euler", "zero_prev"), \
            f"Unsupported trap_boundary: {self.trap_boundary}"
        if self.discretization == "gated_trapezoid":
            # gate g 已移除：数学上 g 与 λ 冗余，固定 g=1 即退化为单参数梯形。
            lam_init = float(trap_lambda)
            eps = 1e-6
            lam_init = max(eps, min(1.0 - eps, lam_init))
            lam_logit = math.log(lam_init / (1.0 - lam_init))
            if self.trap_lambda_per_dir:
                # 4 条扫描方向各自学习一个 λ。
                # 索引顺序与 sass() 返回的 4 个方向一致：
                #   0 = 横向蛇形，1 = 纵向蛇形，2 = 主对角蛇形，3 = 副对角蛇形。
                self.trap_lambda_logit = nn.Parameter(
                    torch.full((4,), lam_logit, dtype=torch.float32)
                )
                self.last_lam_per_dir = None
            else:
                # 标量可学习 λ（梯形混合权重），用 sigmoid 约束在 (0,1)。
                self.trap_lambda_logit = nn.Parameter(
                    torch.tensor(lam_logit, dtype=torch.float32)
                )
                self.last_trap_lambda = None

        # 自适应扫描路由器（Adaptive Scan Router）。
        # 只在 asr / asr_scan / asr_scan_v2 模式下实例化；none 模式下与原始代码完全等价。
        # asr_scan_v2 使用独立的 router_v2 namespace，便于 checkpoint sanitizer 区分版本。
        if self.scan_routing in ('asr', 'asr_scan'):
            self.router = ScanRouter(self.d_model)
            self.last_route_w = None
        elif self.scan_routing == 'asr_scan_v2':
            self.router_v2 = ScanRouterV2(self.d_model)
            self.last_route_w = None

        # Layer Scale 可学习系数，用于训练初期的稳定。
        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # in_proj: 把输入特征投影到 2*d_inner（一半给 x，一半给 z 门控）。
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 局部卷积：用 BottConv 对 x 做 2D 局部上下文建模，保持空间结构。
        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        # x_proj: 把卷积后的 x 投影到 dt_rank + 2*d_state，用于生成 dt、B、C。
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        # dt_proj: 把 dt 从 dt_rank 投影到 d_inner，并为每个通道学习独立的离散化步长。
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # 初始化 dt_proj 的权重与偏置，使离散化步长在合理范围内。
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 状态矩阵 A：用 1~d_state 的对数参数化，负指数后得到稳定的连续状态矩阵。
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D: 跳跃连接（skip connection）的可学习系数。
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        # out_proj: 将 d_inner 映射回 d_model，输出与输入同维度。
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        # direction_Bs: 为 4 个扫描方向分别学习一组 B 的偏移（+1 是保留一个默认/全零方向）。
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def _get_scan_fn(self):
        """根据 discretization 选择 scan 函数。"""
        if self.discretization in (
            "trapezoid", "trapezoidal_fixed", "trapezoidal_data", "gated_trapezoid"
        ):
            # 优先使用 Triton kernel；编译失败时回退到纯 PyTorch。
            try:
                from .selective_scan_trapezoidal_triton import selective_scan_trapezoidal_triton_fn
                return selective_scan_trapezoidal_triton_fn
            except Exception:
                from .selective_scan_trapezoidal import selective_scan_trapezoidal_fn
                return selective_scan_trapezoidal_fn
        # euler: 优先使用 mamba_ssm 的 fused kernel。
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            return selective_scan_fn
        except Exception:
            # 无 mamba_ssm 时的纯 PyTorch Euler 回退。
            from .selective_scan_trapezoidal import selective_scan_euler_pytorch_fn
            return selective_scan_euler_pytorch_fn

    # sass = Self-Adaptive Scan Sequence
    # 根据当前 (H, W) 生成蛇形扫描序列及其逆序，用于把 2D token 展平成 1D。
    # n_dirs 控制只生成固定语义顺序 [横, 纵, 主对角, 副对角] 的前 n_dirs 条
    # （扫描剪枝：被剪掉的方向连排列都不生成）；默认 4 与原行为完全一致。
    def sass(self, hw_shape, n_dirs=4):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        # 方向 1：横向蛇形扫描（从底行开始左右往返）。
        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        # 方向 2：纵向蛇形扫描（从左上角开始上下往返）。
        if n_dirs >= 2:
            i, j = 0, 0
            i_d = "down"
            while j < W:
                assert i_d in ["down", "up"]
                idx = i * W + j
                o2_inverse[idx] = len(o2)
                o2.append(idx)
                if i_d == "down":
                    if i < H - 1:
                        i = i + 1
                        d2.append(4)
                    else:
                        j = j + 1
                        d2.append(1)
                        i_d = "up"
                else:
                    if i > 0:
                        i = i - 1
                        d2.append(3)
                    else:
                        j = j + 1
                        d2.append(1)
                        i_d = "down"
            d2 = [0] + d2[:-1]

        # 方向 3：主对角线蛇形扫描（沿 \ 方向对角线往返）。
        if n_dirs >= 3:
            for diag in range(H + W - 1):
                if diag % 2 == 0:
                    for i in range(min(diag + 1, H)):
                        j = diag - i
                        if j < W:
                            idx = i * W + j
                            o3.append(idx)
                            o3_inverse[idx] = len(o3) - 1
                            d3.append(1 if j == diag else 4)
                else:
                    for j in range(min(diag + 1, W)):
                        i = diag - j
                        if i < H:
                            idx = i * W + j
                            o3.append(idx)
                            o3_inverse[idx] = len(o3) - 1
                            d3.append(4 if i == diag else 1)
            d3 = [0] + d3[:-1]

        # 方向 4：反对角线蛇形扫描（沿 / 方向对角线往返，列做镜像）。
        if n_dirs >= 4:
            for diag in range(H + W - 1):
                if diag % 2 == 0:
                    for i in range(min(diag + 1, H)):
                        j = diag - i
                        if j < W:
                            idx = i * W + (W - j - 1)
                            o4.append(idx)
                            o4_inverse[idx] = len(o4) - 1
                            d4.append(1 if j == diag else 4)
                else:
                    for j in range(min(diag + 1, W)):
                        i = diag - j
                        if i < H:
                            idx = i * W + (W - j - 1)
                            o4.append(idx)
                            o4_inverse[idx] = len(o4) - 1
                            d4.append(4 if i == diag else 1)
            d4 = [0] + d4[:-1]

        # 返回：n_dirs 条扫描顺序、对应的逆序（用于恢复 2D）、方向编码。
        # n_dirs=4（默认）时与原返回完全一致。
        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4))[:n_dirs], \
            (tuple(o1_inverse), tuple(o2_inverse),
             tuple(o3_inverse), tuple(o4_inverse))[:n_dirs], \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))[:n_dirs]

    def _scan_params_for_order(self, order, dt, B, C, dB):
        if self.scan_param_order == 'aligned':
            # dB is already built in the current scan order from sass().
            dt_dir = dt[:, :, order].contiguous()
            B_dir = (B[:, :, order] + dB).contiguous()
            C_dir = C[:, :, order].contiguous()
        else:
            dt_dir = dt
            B_dir = (B + dB).contiguous()
            C_dir = C
        return dt_dir, B_dir, C_dir

    # SAVSS_2D 前向传播：完成 "投影 → 局部卷积 → 生成 dt/B/C → 4 向 selective scan → 融合 → 输出"。
    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        if L != H * W:
            raise ValueError(
                f"SAVSS_2D received L={L}, but hw_shape={hw_shape} "
                f"implies {H * W} tokens"
            )
        E = self.d_inner

        # 保留 SAVSS_2D 的输入，用于 ASR 路由器（asr 模式下根据原始特征图预测 4 向权重）。
        x_in = x

        # conv_state / ssm_state 为 Mamba 状态保留位，当前版本未使用 last_state。
        conv_state, ssm_state = None, None
        # 0) 数据依赖 lam / gated trapezoid 可学习 λ：
        #    在 in_proj 之前从 x 计算，与固定 lam 路径互斥。
        lam = None
        if self.discretization == "trapezoidal_data":
            lam = torch.sigmoid(self.lam_proj(x))
            # detach 副本供训练监控（lam 分布/随训练变化的检查）。
            self.last_lam = lam.detach()

        lam_per_dir = None
        if self.discretization == "gated_trapezoid":
            # 仅保留可学习 λ；gate 固定为 1.0（与 λ 冗余，退化为单参数梯形）。
            if self.trap_lambda_per_dir:
                # 每条扫描方向一个 λ，形状 (4,)。
                lam_per_dir = torch.sigmoid(self.trap_lambda_logit)
                self.last_lam_per_dir = lam_per_dir.detach().clone()
            else:
                lam = torch.sigmoid(self.trap_lambda_logit)
                self.last_trap_lambda = lam.detach().clone()
        # 1) 输入投影：x 被切成两段，分别作为主干 x 和门控 z。
        xz = self.in_proj(x)
        # 2) 状态矩阵 A：参数化存储为 log，前向时取负指数。
        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=-1)
        # 3) 把序列恢复成 2D 做局部卷积，再展平回序列。
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 4) 从卷积后的特征生成离散化步长 dt、输入相关矩阵 B 和 C。
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        assert self.activation in ["silu", "swish"]

        # 5) 生成扫描顺序与方向编码，并为每个方向构造 direction-aware 的 B 偏移。
        #    扫描剪枝：只生成/遍历前 num_scans 条路径（默认 4 = 原 4 向）。
        orders, inverse_orders, directions = self.sass(hw_shape, n_dirs=self.num_scans)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        # 6) 对保留的 num_scans 个方向分别调用 selective_scan_fn：
        #    先按 order 重排 x → scan → 按 inv_order 逆序恢复 → 得到该方向的全局响应。
        #    数据依赖 lam 时，lam 与 x 一样按各方向的 order 重排后传入。
        scan_fn = self._get_scan_fn()
        y_scan = []
        diag_entries = []
        for dir_idx, (o, inv_order, dB) in enumerate(
            zip(orders, inverse_orders, direction_Bs)
        ):
            scan_kwargs = {}
            if self.discretization in ("trapezoid", "trapezoidal_fixed"):
                scan_kwargs["lam"] = self.trap_lambda
            elif self.discretization == "trapezoidal_data":
                scan_kwargs["lam"] = lam[:, o, :].contiguous()
            elif self.discretization == "gated_trapezoid":
                # gate 已固定为 1.0，仅 λ 可学习。
                if self.trap_lambda_per_dir:
                    scan_kwargs["lam"] = lam_per_dir[dir_idx]
                else:
                    scan_kwargs["lam"] = lam
                scan_kwargs["gate"] = torch.tensor(
                    1.0, device=x.device, dtype=torch.float32
                )
                scan_kwargs["boundary"] = self.trap_boundary
                scan_kwargs["gated"] = True

            u_dir = x_conv[:, o, :].permute(0, 2, 1).contiguous()
            dt_dir, B_dir, C_dir = self._scan_params_for_order(o, dt, B, C, dB)
            y_scan.append(
                scan_fn(
                    u_dir,
                    dt_dir,
                    A,
                    B_dir,
                    C_dir,
                    self.D.float(),
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                    **scan_kwargs,
                ).permute(0, 2, 1)[:, inv_order, :]
            )

            # 诊断日志：仅在 SCAN_DIAG=1 且当前被触发时计算，不影响前向结果和梯度。
            if SAVSS_2D._diag_active and self.training:
                d = _compute_scan_diag(
                    u_dir, dt_dir, A, B_dir,
                    D=self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    lam=scan_kwargs.get("lam", 1.0),
                    gate=scan_kwargs.get("gate", 1.0),
                    boundary=scan_kwargs.get("boundary", "euler"),
                )
                if d is not None:
                    diag_entries.append(d)

        # 诊断日志：当前层 4 个方向的指标取平均，供 engine.py 写入 scan_diag.jsonl。
        self._diag_entry = None
        if SAVSS_2D._diag_active and self.training and diag_entries:
            self._diag_entry = {
                "step": int(SAVSS_2D._diag_step),
                "mean_delta": float(sum(d["mean_delta"] for d in diag_entries) / len(diag_entries)),
                "euler_norm": float(sum(d["euler_norm"] for d in diag_entries) / len(diag_entries)),
                "corr_norm": float(sum(d["corr_norm"] for d in diag_entries) / len(diag_entries)),
                "ratio": float(sum(d["ratio"] for d in diag_entries) / len(diag_entries)),
                "adj_cosine": float(sum(d["adj_cosine"] for d in diag_entries) / len(diag_entries)),
                "h_norm": float(sum(d["h_norm"] for d in diag_entries) / len(diag_entries)),
            }

        # 7) 4 向扫描结果融合，并用 SiLU 门控 z 进行调制，最后投影回 d_model。
        if self.scan_routing == 'asr':
            # legacy ASR：每张图一个 4 路权重，保留旧 checkpoint/forward 语义。
            router_input = x_in.reshape(batch_size, H, W, self.d_model) \
                              .permute(0, 3, 1, 2).contiguous()
            w = self.router(router_input)              # (B, 4)
            self._route_w = w                           # 供 balancing loss 使用
            self.last_route_w = w.detach().clone()      # 供 epoch 日志使用

            # y = 4 * sum_k( w[:,k] * y_scan[k] ) * act(z)
            y_2d = sum(
                w[:, k].view(batch_size, 1, 1, 1) *
                y_scan[k].reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
                for k in range(4)
            )
            y = (4.0 * y_2d).permute(0, 2, 3, 1).reshape(batch_size, L, E)
        elif self.scan_routing == 'asr_scan':
            router_input = x_in.reshape(batch_size, H, W, self.d_model) \
                              .permute(0, 3, 1, 2).contiguous()
            branch_maps = torch.stack([
                y_k.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
                for y_k in y_scan
            ], dim=1)
            w, w_grid = self.router.scan_weights(
                router_input, branch_maps, (H, W), self.asr_route_grid
            )
            self._route_w = w_grid
            self.last_route_w = w_grid.detach().mean(dim=(-2, -1)).clone()

            y_route = sum(
                w[:, k:k + 1] * branch_maps[:, k]
                for k in range(4)
            ) * 4.0
            y_base = branch_maps.sum(dim=1)
            y_2d = y_base + self.asr_beta * (y_route - y_base)
            y = y_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)
        elif self.scan_routing == 'asr_scan_v2':
            # V2：branch-advantage 描述子 + 零初始化 logit head，
            # 初始与等和 baseline 严格等价，residual 强度用独立的 asr_v2_beta。
            router_input = x_in.reshape(batch_size, H, W, self.d_model) \
                              .permute(0, 3, 1, 2).contiguous()
            branch_maps = torch.stack([
                y_k.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
                for y_k in y_scan
            ], dim=1)
            w, w_grid = self.router_v2.scan_weights(
                router_input, branch_maps, (H, W), self.asr_route_grid
            )
            self._route_w = w_grid
            self.last_route_w = w_grid.detach().mean(dim=(-2, -1)).clone()

            y_route = sum(
                w[:, k:k + 1] * branch_maps[:, k]
                for k in range(4)
            ) * 4.0
            y_base = branch_maps.sum(dim=1)
            y_2d = y_base + self.asr_v2_beta * (y_route - y_base)
            y = y_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

            # 路由诊断：仅在 engine 触发时生成，全部 detach，不保留 autograd graph。
            if SAVSS_2D._router_diag_active and self.training:
                self._router_diag_entry = self._compute_router_diag(
                    w_grid, branch_maps, y_route, y_base
                )
            else:
                self._router_diag_entry = None
        else:
            # none 模式：保持原始等权求和，计算路径与参数量完全不变。
            self._route_w = None
            self.last_route_w = None
            y = sum(y_scan)
        y = y * self.act(z)
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

    @torch.no_grad()
    def _compute_router_diag(self, w_grid, branch_maps, y_route, y_base):
        """生成 asr_scan_v2 的单层路由诊断（全 detached 标量/列表）。"""
        eps = 1e-12
        probs = w_grid.detach().float()
        entropy = -(probs * (probs + eps).log()).sum(dim=1)
        argmax_idx = probs.reshape(probs.size(0), probs.size(1), -1).argmax(dim=1)
        freq = [
            float((argmax_idx == k).float().mean().item())
            for k in range(self.n_directions)
        ]
        return {
            "step": int(SAVSS_2D._router_diag_step),
            "route_grid": self.asr_route_grid,
            "beta": self.asr_v2_beta,
            "entropy_mean": float(entropy.mean().item()),
            "entropy_norm_mean": float((entropy / math.log(4.0)).mean().item()),
            "w_mean": probs.mean(dim=(0, 2, 3)).tolist(),
            "w_std_mean": float(probs.std(dim=1).mean().item()),
            "w_min": float(probs.min().item()),
            "w_max": float(probs.max().item()),
            "argmax_freq": freq,
            "branch_rms": [
                float(branch_maps[:, k].detach().float().pow(2).mean().sqrt().item())
                for k in range(self.n_directions)
            ],
            "residual_ratio": float(
                ((y_route - y_base).detach().float().norm() /
                 (y_base.detach().float().norm() + eps)).item()
            ),
        }


class ScanRouter(nn.Module):
    """自适应扫描路由器。

    输入：SAVSS_2D 的输入特征图 x，shape (B, C, H, W)
    输出：对 4 条扫描路径的 softmax 权重，shape (B, 4)。

    最后一个 Linear 层零初始化，使得初始 logits=0 → softmax 输出精确为 1/4，
    从而保证 asr 模式初始状态与 none 模式的等权求和逐元素等价。
    """
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 4)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        b = x.size(0)
        v = self.gap(x).view(b, -1)                 # (B, C)
        logits = self.fc2(self.relu(self.fc1(v)))   # (B, 4)
        return F.softmax(logits, dim=-1)            # (B, 4)

    def scan_weights(self, context, branch_maps, hw_shape, route_grid=1):
        """Return spatial scan weights and their coarse-grid counterpart.

        ``context`` has shape ``(B, C, H, W)`` and ``branch_maps`` has shape
        ``(B, 4, E, H, W)``.  The branch channels are reduced from E to C by
        averaging the expand groups, so the legacy router projections can be
        reused without adding a large descriptor projection.
        """
        batch_size, channels, height, width = context.shape
        _, n_branches, inner_channels, _, _ = branch_maps.shape
        if n_branches != 4 or channels != self.fc1.in_features:
            raise ValueError(
                f"Expected context (B, {self.fc1.in_features}, H, W) and "
                f"four branches, got context={tuple(context.shape)}"
            )
        if inner_channels % channels != 0:
            raise ValueError(
                f"Branch channels {inner_channels} must be divisible by "
                f"router channels {channels}"
            )

        route_grid = int(route_grid)
        if route_grid not in (1, 2, 4):
            raise ValueError(f"route_grid must be one of (1, 2, 4), got {route_grid}")

        pool = nn.AdaptiveAvgPool2d((route_grid, route_grid))
        context_grid = pool(context)
        context_hidden = self.relu(
            self.fc1(context_grid.permute(0, 2, 3, 1))
        )
        context_logits = self.fc2(context_hidden).permute(0, 3, 1, 2)

        branch_grid = pool(
            branch_maps.reshape(batch_size * n_branches, inner_channels,
                                height, width)
        ).reshape(batch_size, n_branches, inner_channels, route_grid, route_grid)
        expand = inner_channels // channels
        branch_desc = branch_grid.reshape(
            batch_size, n_branches, expand, channels, route_grid, route_grid
        ).mean(dim=2)
        branch_hidden = self.relu(
            self.fc1(branch_desc.permute(0, 1, 3, 4, 2))
        )
        # Each scan branch uses the corresponding row of fc2.  The operation
        # is deliberately bias-free here; context_logits already contains the
        # shared fc2 bias exactly once.
        branch_logits = torch.einsum(
            'bkrqh,kh->bkrq', branch_hidden, self.fc2.weight
        )
        grid_logits = context_logits + branch_logits
        full_logits = F.interpolate(
            grid_logits, size=(height, width), mode='bilinear',
            align_corners=False
        )
        return F.softmax(full_logits, dim=1), F.softmax(grid_logits, dim=1)


class ScanRouterV2(nn.Module):
    """Branch-advantage 扫描路由器（asr_scan_v2）。

    与 legacy ScanRouter 的区别：branch 描述子不再直接对每个扫描分支的输出
    做平均（方向差异会在 pooling 时正负抵消），而是使用相对 4 分支均值的
    signed advantage（branch_k - base）及其 magnitude：

    1. branch maps 先按 expand 组 mean 降到 C 通道；
    2. 计算 base = mean_k(branch_k)，advantage = branch_k - base；
    3. 在 coarse route grid 上 pooling，按 grid cell 做 RMS 归一化
       （防止某一层/某个方向的绝对扫描尺度直接决定路由），
       同时保留 log1p(RMS) 标量作为幅度通道；
    4. context 与每个 branch 的 descriptor 分别过共享 projection 后相加，
       经四路 logit head 得到 coarse logits，再上采样并 softmax。

    logit head 零初始化 → 初始权重精确为 1/4，残差融合起点与等和 baseline
    严格等价；同时 head 直接从 branch contrast 获得非零 task gradient，
    不受"最终层为零导致整条路径梯度被阻断"的影响。
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        self.channels = channels
        hidden = hidden or max(4, channels // 4)
        # +1：每个 branch 每格的 log1p(RMS) 幅度标量。
        self.desc_proj = nn.Linear(channels + 1, hidden)
        self.ctx_proj = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def scan_weights(self, context, branch_maps, hw_shape, route_grid=1):
        """返回 full-resolution 与 coarse-grid 的 4 路 softmax 权重。

        ``context`` 形状 (B, C, H, W)，``branch_maps`` 形状 (B, 4, E, H, W)，
        其中 E 必须能被 C 整除（E = expand * C）。
        """
        batch_size, channels, height, width = context.shape
        _, n_branches, inner_channels, _, _ = branch_maps.shape
        if n_branches != 4 or channels != self.channels:
            raise ValueError(
                f"Expected context (B, {self.channels}, H, W) and "
                f"four branches, got context={tuple(context.shape)}"
            )
        if inner_channels % channels != 0:
            raise ValueError(
                f"Branch channels {inner_channels} must be divisible by "
                f"router channels {channels}"
            )
        route_grid = int(route_grid)
        if route_grid not in (1, 2, 4):
            raise ValueError(f"route_grid must be one of (1, 2, 4), got {route_grid}")

        expand = inner_channels // channels
        # (B, 4, E, H, W) -> 按 expand 组 mean -> (B, 4, C, H, W)
        branch = branch_maps.reshape(
            batch_size, n_branches, expand, channels, height, width
        ).mean(dim=2)
        base = branch.mean(dim=1, keepdim=True)
        adv = branch - base                                   # signed advantage
        adv_grid = F.adaptive_avg_pool2d(
            adv.reshape(batch_size * n_branches, channels, height, width),
            (route_grid, route_grid),
        ).reshape(batch_size, n_branches, channels, route_grid, route_grid)
        # 按 grid cell 在通道维做 RMS 归一化，剥离绝对尺度；幅度单独成通道。
        # 注意 eps 必须放在 sqrt 内部：各分支完全相同（adv≡0）时 sqrt 在 0 处的
        # 梯度为 inf，与上游零梯度相乘会产生 NaN（0*inf=nan）。
        ms = adv_grid.pow(2).mean(dim=2, keepdim=True)
        rms = torch.sqrt(ms + 1e-8)
        signed = adv_grid / rms
        magnitude = torch.log1p(rms)
        desc = torch.cat([signed, magnitude], dim=2)          # (B,4,C+1,R,R)

        desc_feat = self.relu(
            self.desc_proj(desc.permute(0, 1, 3, 4, 2))
        )                                                      # (B,4,R,R,hidden)
        context_grid = F.adaptive_avg_pool2d(context, (route_grid, route_grid))
        ctx_feat = self.relu(
            self.ctx_proj(context_grid.permute(0, 2, 3, 1))
        )                                                      # (B,R,R,hidden)
        feat = desc_feat + ctx_feat.unsqueeze(1)
        grid_logits = self.head(feat).squeeze(-1)              # (B,4,R,R)
        full_logits = F.interpolate(
            grid_logits, size=(height, width), mode='bilinear',
            align_corners=False
        )
        return F.softmax(full_logits, dim=1), F.softmax(grid_logits, dim=1)


# SAVSS_Layer: 完整的一个 SAVSS 层，通常堆叠多次构成 backbone。
# 包含：归一化 → GBC 卷积 → SAVSS_2D 全局扫描 → PAF 融合 → GroupNorm → DropPath → 残差。
class SAVSS_Layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
    ):

        super(SAVSS_Layer, self).__init__()
        # 把当前层的通道数写入 mamba_cfg，供 SAVSS_2D 使用。
        mamba_cfg.update({'d_model': embed_dims})
        if use_rms_norm:
            self.norm = _build_rms_norm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        # 可选的深度可分离卷积分支，用于进一步融合局部信息。
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )

        # 核心：2D 选择性扫描模块。
        self.SAVSS_2D = SAVSS_2D(**mamba_cfg)
        # DropPath（Stochastic Depth），用于训练深层网络的正则化。
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        # 后续用于维度对齐与融合的辅助层。
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)
        self.GBC_C = GBC(embed_dims)
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        if L != H * W:
            raise ValueError(
                f"SAVSS_Layer received L={L}, but hw_shape={hw_shape} "
                f"implies {H * W} tokens"
            )
        # 把 (B, L, C) 序列恢复成 (B, C, H, W) 以便做 2D 卷积。
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 先经过两轮 GBC 卷积做局部特征增强。
        for i in range(2):
            x = self.GBC_C(x)

        # 重新展平为序列，过 LayerNorm 后输入 SAVSS_2D 做全局扫描。
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))
        b, l, c = mixed_x.shape
        h, w = hw_shape
        if l != h * w:
            raise ValueError(
                f"SAVSS_Layer mixed output L={l}, but hw_shape={hw_shape} "
                f"implies {h * w} tokens"
            )
        # PAF：把原始局部特征 x 与全局扫描特征 mixed_x 进行融合。
        mixed_x = self.PAF_256(x.permute(0, 2, 1).reshape(b, c, h, w),
                               mixed_x.permute(0, 2, 1).reshape(b, c, h, w))
        mixed_x = self.GN_256(mixed_x).reshape(b, c, h * w).permute(0, 2, 1)

        # 若启用 dwconv，再补一轮 GBC 卷积局部细化。
        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.GBC_C(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)

        # 残差分支：对融合后的特征做 GroupNorm + Linear，再与主分支相加。
        mixed_x_res = self.linear_256(self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))
        return mixed_x + mixed_x_res
