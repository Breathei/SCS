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
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from models.GBC import GBC, BottConv
from models.PAF import PAF


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
    # 根据当前 (H, W) 生成 4 条蛇形扫描序列及其逆序，用于把 2D token 展平成 1D。
    def sass(self, hw_shape):
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

        # 返回：4 条扫描顺序、对应的逆序（用于恢复 2D）、以及每个位置的方向编码。
        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    # SAVSS_2D 前向传播：完成 "投影 → 局部卷积 → 生成 dt/B/C → 4 向 selective scan → 融合 → 输出"。
    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

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

        # 5) 生成 4 向扫描顺序与方向编码，并为每个方向构造 direction-aware 的 B 偏移。
        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        # 6) 对 4 个方向分别调用 selective_scan_fn：
        #    先按 order 重排 x → scan → 按 inv_order 逆序恢复 → 得到该方向的全局响应。
        #    数据依赖 lam 时，lam 与 x 一样按各方向的 order 重排后传入。
        scan_fn = self._get_scan_fn()
        y_scan = []
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
            y_scan.append(
                scan_fn(
                    x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                    dt,
                    A,
                    (B + dB).contiguous(),
                    C,
                    self.D.float(),
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                    **scan_kwargs,
                ).permute(0, 2, 1)[:, inv_order, :]
            )

        # 7) 4 向扫描结果相加，并用 SiLU 门控 z 进行调制，最后投影回 d_model。
        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

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
        H = W = int(math.sqrt(L))
        # 把 (B, L, C) 序列恢复成 (B, C, H, W) 以便做 2D 卷积。
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 先经过两轮 GBC 卷积做局部特征增强。
        for i in range(2):
            x = self.GBC_C(x)

        # 重新展平为序列，过 LayerNorm 后输入 SAVSS_2D 做全局扫描。
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))
        b, l, c = mixed_x.shape
        h = w = int(math.sqrt(l))
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
