"""
数值正确性验证：梯形选择性扫描前向（只读测试，不修改任何被测代码）。

测试1：λ=1 退化等价性 —— discretization='euler' vs 'trapezoidal_fixed'+trap_lambda=1.0，
       同一模型权重、同一输入，比较最终分割输出，阈值 1e-5。
       附加诊断：强制梯形走纯 PyTorch fallback，与 euler 的纯 PyTorch fallback 对比
       （两者应调用同一个函数同一组参数，期望逐位一致），用于区分"结构性错误"
       与"Triton 浮点误差"。
测试2：Triton 核 vs 纯 PyTorch 参考 —— trapezoidal_fixed + trap_lambda=0.45，
       同一模型同一输入，强制两条路径，阈值 1e-3（>1e-2 判定 Triton 前向有 bug）。
测试3：λ=0 退化检查 —— 纯 PyTorch 实现 vs 手写朴素 for 循环（独立重算），阈值 1e-5。

运行：python tests/test_trapezoidal_forward_validation.py
"""
import argparse
import os
import sys
import unittest

import torch
import torch.nn.functional as F

# 仓库根目录加入 sys.path，使脚本可从任意工作目录运行。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 先导入项目模型入口以避免循环导入（与 tests/ 下既有测试保持同一模式）。
from models import build_model  # noqa: F401
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
    selective_scan_trapezoidal_fn,
)
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal_triton import (
    selective_scan_trapezoidal_triton_fn,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512          # 与训练输入一致
SEED_WEIGHTS = 20260830
SEED_INPUT = 1234


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_full_model(discretization, trap_lambda):
    """以固定种子构建完整分割模型（backbone + MFS decoder），eval 模式。"""
    seed_all(SEED_WEIGHTS)
    args = argparse.Namespace(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        discretization=discretization,
        lam_nheads=8,
        trap_lambda=trap_lambda,
        trap_boundary='euler',
        trap_lambda_per_dir=False,
        scan_routing='none',
        BCELoss_ratio=0.83,
        DiceLoss_ratio=0.17,
    )
    model, _ = build_model(args)
    model = model.to(DEVICE).eval()
    return model


def force_scan_fn(model, fn):
    """在测试进程内 monkeypatch 所有 SAVSS_2D 实例的 scan 函数（不改任何文件）。"""
    n = 0
    for m in model.modules():
        if isinstance(m, SAVSS_2D):
            m._get_scan_fn = lambda: fn
            n += 1
    assert n > 0, "model contains no SAVSS_2D modules"
    return n


def resolved_scan_fns(model):
    """返回模型中各自 SAVSS_2D 自动解析到的 scan 函数名（用于报告实际路径）。"""
    names = set()
    for m in model.modules():
        if isinstance(m, SAVSS_2D):
            fn = m._get_scan_fn()
            names.add(f"{fn.__module__}.{fn.__name__}")
    return sorted(names)


def make_input():
    seed_all(SEED_INPUT)
    return torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)


@torch.no_grad()
def run_model(model, x):
    return model(x)


def compare_tensors(name_a, a, name_b, b, topk=10):
    """返回 (max_abs_err, mean_abs_err, rel_err, 报告字符串)。"""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = (a - b).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    scale = b.abs().max().item()
    rel_err = max_err / scale if scale > 0 else float('inf')
    lines = [
        f"  max|{name_a} - {name_b}| = {max_err:.3e}",
        f"  mean|diff|          = {mean_err:.3e}",
        f"  max|{name_b}|        = {scale:.3e}  (相对误差 {rel_err:.3e})",
    ]
    if max_err > 0:
        flat = diff.flatten()
        k = min(topk, flat.numel())
        vals, idxs = torch.topk(flat, k)
        lines.append(f"  误差最大的前 {k} 个元素：")
        af, bf = a.flatten(), b.flatten()
        for v, i in zip(vals.tolist(), idxs.tolist()):
            coord = list(torch.unravel_index(torch.tensor(i), a.shape))
            lines.append(
                f"    idx {coord}: {name_a}={af[i].item(): .6e}  "
                f"{name_b}={bf[i].item(): .6e}  |diff|={v:.3e}"
            )
    return max_err, mean_err, rel_err, "\n".join(lines)


class Test1LamOneEulerEquivalence(unittest.TestCase):
    """λ=1 时梯形应退化为欧拉：最终分割输出最大绝对误差 < 1e-5。"""

    THRESHOLD = 1e-5

    @classmethod
    def setUpClass(cls):
        cls.x = make_input()
        cls.model_euler = build_full_model('euler', trap_lambda=0.5)
        cls.model_trap = build_full_model('trapezoidal_fixed', trap_lambda=1.0)
        print("\n[测试1] euler 路径自动解析到: "
              f"{resolved_scan_fns(cls.model_euler)}")
        print("[测试1] trapezoidal_fixed 路径自动解析到: "
              f"{resolved_scan_fns(cls.model_trap)}")

    def test_lam1_equals_euler_autopath(self):
        y_euler = run_model(self.model_euler, self.x)
        y_trap = run_model(self.model_trap, self.x)
        max_err, _, _, report = compare_tensors("y_euler", y_euler, "y_trap(lam=1)", y_trap)
        print(f"\n[测试1] 自动路径对比（euler vs trapezoidal_fixed+lam=1.0）:\n{report}")
        # 诊断：强制梯形走纯 PyTorch fallback，隔离结构性错误 vs Triton 浮点误差。
        n = force_scan_fn(self.model_trap, selective_scan_trapezoidal_fn)
        y_trap_ref = run_model(self.model_trap, self.x)
        max_err_ref, _, _, report_ref = compare_tensors(
            "y_euler", y_euler, "y_trap_pytorch(lam=1)", y_trap_ref)
        print(f"[测试1-诊断] 强制纯 PyTorch 路径（{n} 个 SAVSS_2D 被 patch）:\n{report_ref}")
        # 诊断2：euler 也强制走纯 PyTorch fallback（selective_scan_euler_pytorch_fn
        # 就是 trapezoidal_fn(lam=1.0)），此时两条路径调用同一函数同一组参数，
        # 结构正确则必须逐位一致（max err == 0.0）。
        from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import (
            selective_scan_euler_pytorch_fn,
        )
        force_scan_fn(self.model_euler, selective_scan_euler_pytorch_fn)
        y_euler_ref = run_model(self.model_euler, self.x)
        max_err_pp, _, _, report_pp = compare_tensors(
            "y_euler_pytorch", y_euler_ref, "y_trap_pytorch(lam=1)", y_trap_ref)
        print(f"[测试1-诊断2] 双方都强制纯 PyTorch 路径（应逐位一致）:\n{report_pp}")
        if max_err_pp == 0.0 and max_err > 0.0:
            print("[测试1-诊断2] 纯 PyTorch 双方逐位一致 → 梯形结构正确；"
                  "自动路径误差来自 mamba_ssm fused 核与参考实现之间的 fp32 数值差异。")
        self.assertLess(
            max_err, self.THRESHOLD,
            f"λ=1 退化等价性失败：max abs err = {max_err:.3e} >= {self.THRESHOLD}")


class Test2TritonVsPytorchRef(unittest.TestCase):
    """trapezoidal_fixed + lam=0.45：Triton 核 vs 纯 PyTorch 参考，阈值 1e-3。"""

    THRESHOLD = 1e-3
    BUG_THRESHOLD = 1e-2

    @classmethod
    def setUpClass(cls):
        cls.x = make_input()
        cls.model = build_full_model('trapezoidal_fixed', trap_lambda=0.45)

    def test_triton_matches_pytorch_ref(self):
        force_scan_fn(self.model, selective_scan_trapezoidal_triton_fn)
        y_triton = run_model(self.model, self.x)
        force_scan_fn(self.model, selective_scan_trapezoidal_fn)
        y_ref = run_model(self.model, self.x)
        max_err, _, rel_err, report = compare_tensors(
            "y_triton", y_triton, "y_pytorch_ref", y_ref)
        print(f"\n[测试2] trapezoidal_fixed+lam=0.45, Triton vs 纯PyTorch:\n{report}")
        if max_err > self.BUG_THRESHOLD:
            self.fail(f"Triton 前向疑似有 bug：max abs err = {max_err:.3e} > 1e-2")
        self.assertLess(
            max_err, self.THRESHOLD,
            f"Triton vs PyTorch 参考误差超阈值：{max_err:.3e} >= {self.THRESHOLD} "
            f"(相对误差 {rel_err:.3e})")


class Test3LamZeroDegeneracy(unittest.TestCase):
    """λ=0：h_t = exp(ΔA)h_{t-1} + Δ·exp(ΔA)·B_{t-1}x_{t-1}，与独立手写循环对比。"""

    THRESHOLD = 1e-5

    @staticmethod
    def naive_lam0_scan(u, delta, A, B, C, D, delta_bias, dtype=None):
        """独立重算：不复用被测实现的任何中间表达式。

        边界与被测实现的 boundary="euler" 一致：h_0 = Δ_0 B_0 x_0；
        t>=1 时 h_t = exp(Δ_t A) · (h_{t-1} + Δ_t B_{t-1} x_{t-1})。
        """
        if dtype is not None:
            u, delta, A, B, C, D, delta_bias = (
                t.to(dtype) for t in (u, delta, A, B, C, D, delta_bias))
        dt = F.softplus(delta + delta_bias.view(1, -1, 1))      # (B, E, L)
        Bb, E, L = u.shape
        N = A.shape[1]
        h = torch.zeros(Bb, E, N, dtype=u.dtype, device=u.device)
        y = torch.empty(Bb, E, L, dtype=u.dtype, device=u.device)
        P_prev = None
        for t in range(L):
            dt_t = dt[:, :, t].unsqueeze(-1)                    # (B, E, 1)
            A_bar_t = torch.exp(dt_t * A.view(1, E, N))         # (B, E, N)
            if t == 0:
                h = dt_t * (u[:, :, t].unsqueeze(-1) * B[:, :, t].unsqueeze(1))
            else:
                h = A_bar_t * (h + dt_t * P_prev)
            P_prev = u[:, :, t].unsqueeze(-1) * B[:, :, t].unsqueeze(1)
            y_t = (h * C[:, :, t].unsqueeze(1)).sum(-1) + D.view(1, -1) * u[:, :, t]
            y[:, :, t] = y_t
        return y

    def _run_case(self, B=2, E=16, N=8, L=64, seed=7):
        torch.manual_seed(seed)
        u = torch.randn(B, E, L, device=DEVICE)
        delta = torch.randn(B, E, L, device=DEVICE) * 0.1
        A = -torch.rand(E, N, device=DEVICE) - 0.05
        Bmat = torch.randn(B, N, L, device=DEVICE) * 0.5
        C = torch.randn(B, N, L, device=DEVICE) * 0.5
        D = torch.randn(E, device=DEVICE)
        dbias = torch.randn(E, device=DEVICE) * 0.1

        y_impl = selective_scan_trapezoidal_fn(
            u, delta, A, Bmat, C, D=D, delta_bias=dbias,
            delta_softplus=True, lam=0.0, boundary="euler")
        y_naive = self.naive_lam0_scan(u, delta, A, Bmat, C, D, dbias)
        max_err, _, _, report = compare_tensors(
            "y_impl(lam=0)", y_impl, "y_naive_fp32", y_naive)
        print(f"\n[测试3] B={B} E={E} N={N} L={L} seed={seed}:\n{report}")

        # 诊断：fp64 朴素循环作为金标准，分别报告两者与金标准的误差。
        y_gold = self.naive_lam0_scan(u, delta, A, Bmat, C, D, dbias,
                                      dtype=torch.float64)
        err_impl_gold = (y_impl.double() - y_gold).abs().max().item()
        err_naive_gold = (y_naive.double() - y_gold).abs().max().item()
        print(f"  [诊断] vs fp64 金标准: impl={err_impl_gold:.3e}  "
              f"naive_fp32={err_naive_gold:.3e}")
        return max_err

    def test_lam0_short(self):
        self.assertLess(self._run_case(L=64, seed=7), self.THRESHOLD)

    def test_lam0_long(self):
        self.assertLess(self._run_case(L=512, seed=99), self.THRESHOLD)


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=True)
