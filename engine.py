'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Iterable
import json
import torch
import time
from tqdm import tqdm

# 注意：SAVSS_2D / _SCAN_DIAG_ENABLED 的导入必须放在函数内部（延迟导入）。
# 若在模块顶层导入，main.py 先 import engine 时会形成循环：
# engine → SAVSS_layer → models/__init__ → decoder → SAVSS.py → SAVSS_layer(未初始化完)


def _route_weights_as_samples(route_w):
    """Normalize global/grid route weights to (batch, samples, directions)."""
    if route_w.dim() == 2:
        return route_w.unsqueeze(1)
    if route_w.dim() == 4:
        return route_w.permute(0, 2, 3, 1).reshape(route_w.size(0), -1, route_w.size(1))
    raise ValueError(f"Unexpected route weight shape: {tuple(route_w.shape)}")


def compute_balancing_loss(model):
    """Compute per-layer soft route coverage loss.

    Each layer is reduced independently, so a collapsed layer cannot be hidden
    by a different layer.  The detached second factor keeps the loss scale
    stable while avoiding argmax tie bias at uniform initialization.

    Used by legacy ``asr`` / ``asr_scan`` only; ``asr_scan_v2`` uses the
    anti-collapse coverage loss below instead.
    """
    layer_losses = []
    for m in model.modules():
        if getattr(m, 'scan_routing', None) == 'asr_scan_v2':
            continue
        route_w = getattr(m, '_route_w', None)
        if route_w is None:
            continue
        samples = _route_weights_as_samples(route_w)
        p = samples.mean(dim=(0, 1))
        layer_losses.append(4.0 * (p * p.detach()).sum())
    if not layer_losses:
        return None
    return torch.stack(layer_losses).mean()


def compute_asr_v2_coverage_loss(model, coverage_cap=0.6):
    """Anti-collapse coverage loss for asr_scan_v2 (one-sided upper cap).

    Unlike the legacy uniformizing balancing loss (whose optimum is the
    uniform distribution), this only penalizes a direction whose per-layer
    average share exceeds ``coverage_cap``, so meaningful nonuniform routing
    is not pulled back to uniform.
    """
    layer_losses = []
    for m in model.modules():
        if getattr(m, 'scan_routing', None) != 'asr_scan_v2':
            continue
        route_w = getattr(m, '_route_w', None)
        if route_w is None:
            continue
        samples = _route_weights_as_samples(route_w)
        p = samples.mean(dim=(0, 1))
        layer_losses.append(torch.relu(p - coverage_cap).pow(2).sum())
    if not layer_losses:
        return None
    return torch.stack(layer_losses).mean()


class ModelEMA:
    """模型参数与 buffers 的指数滑动平均（EMA）。

    参数使用指数滑动平均；BatchNorm 等 running buffers 在每次更新时直接
    同步当前模型值，并保持整数 buffer（如 num_batches_tracked）的类型。
    EMA 不参与训练路径：前向/反向始终使用原始权重，shadow 仅用于 eval
    与 checkpoint 保存。

    更新规则：shadow = d * shadow + (1 - d) * param
    warmup：前 warmup_steps 步使用 d_t = min(decay, (1 + step) / (10 + step))。
    """

    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        if self.updates < self.warmup_steps:
            d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        else:
            d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)
        for n, b in model.named_buffers():
            # Buffers include integer counters as well as floating-point running
            # statistics, so copying is safer than applying the parameter EMA rule.
            self.buffers[n].copy_(b.detach())

    def state_dict(self):
        return {
            'shadow': {n: t.detach().clone() for n, t in self.shadow.items()},
            'buffers': {n: t.detach().clone() for n, t in self.buffers.items()},
            'updates': self.updates,
        }

    def load_state_dict(self, state):
        # 兼容纯权重 dict（无 'shadow' 包装）以及旧版仅保存参数的 EMA。
        if 'shadow' in state:
            self.updates = state.get('updates', 0)
            shadow_state = state['shadow']
            buffer_state = state.get('buffers', {})
        else:
            shadow_state = state
            buffer_state = {}
        for n, t in shadow_state.items():
            if n in self.shadow:
                self.shadow[n].copy_(t)
        for n, t in buffer_state.items():
            if n in self.buffers:
                self.buffers[n].copy_(t)

    @torch.no_grad()
    def apply_to(self, model):
        """把 EMA 状态写入 model，返回原始状态备份供 restore 使用。"""
        backup = {}
        for n, p in model.named_parameters():
            backup[n] = p.detach().clone()
            p.copy_(self.shadow[n])
        for n, b in model.named_buffers():
            backup[n] = b.detach().clone()
            b.copy_(self.buffers[n])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            p.copy_(backup[n])
        for n, b in model.named_buffers():
            b.copy_(backup[n])


def _write_scan_diag(model):
    """把当前 step 各 SAVSS_2D 层的诊断指标追加写入 scan_diag.jsonl。"""
    from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D  # 延迟导入，避免循环
    entries = []
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, SAVSS_2D):
            entry = getattr(m, '_diag_entry', None)
            if entry is not None:
                entry = dict(entry)
                entry['layer_idx'] = layer_idx
                entries.append(entry)
            m._diag_entry = None
            layer_idx += 1

    if not entries:
        return

    with open('scan_diag.jsonl', 'a', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')


def _write_router_diag(model):
    """把当前 step 各 SAVSS_2D 层的 asr_scan_v2 路由诊断写入 asr_router_diag.jsonl。

    必须在 backward() 之后调用，这样才能附带 router 各 projection 的梯度范数。
    """
    from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D  # 延迟导入，避免循环
    entries = []
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, SAVSS_2D):
            entry = getattr(m, '_router_diag_entry', None)
            if entry is not None:
                entry = dict(entry)
                entry['layer_idx'] = layer_idx
                router = getattr(m, 'router_v2', None)
                if router is not None:
                    for label, param in (
                        ('head_grad_norm', router.head.weight),
                        ('desc_proj_grad_norm', router.desc_proj.weight),
                        ('ctx_proj_grad_norm', router.ctx_proj.weight),
                    ):
                        if param.grad is not None:
                            entry[label] = float(
                                param.grad.detach().float().norm().item()
                            )
                entries.append(entry)
            m._router_diag_entry = None
            layer_idx += 1

    if not entries:
        return

    with open('asr_router_diag.jsonl', 'a', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     epoch: int, args = None, logger = None, ema: 'ModelEMA' = None):
    # 延迟导入（原因见模块顶部注释）；函数被调用时所有模块均已初始化完毕。
    from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_2D
    from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal import _SCAN_DIAG_ENABLED

    model.train()
    criterion.train()

    device = torch.device(args.device)
    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")

    bal_loss_sum = 0.0
    bal_loss_cnt = 0
    route_w_sums = {}
    route_w_counts = {}

    # asr_scan_v2：独立的 anti-collapse coverage loss（带 warmup + linear ramp），
    # 不使用 legacy 的均匀化 balancing loss；诊断开关只在 v2 模式下生效。
    scan_routing = getattr(args, 'scan_routing', 'none')
    asr_v2_mode = scan_routing == 'asr_scan_v2'
    asr_router_diag = bool(getattr(args, 'asr_router_diag', False)) and asr_v2_mode
    cover_coef = float(getattr(args, 'asr_v2_coverage_coef', 0.01))
    cover_cap = float(getattr(args, 'asr_v2_coverage_cap', 0.6))
    warmup_ratio = float(getattr(args, 'asr_v2_balance_warmup_ratio', 0.1))
    total_steps_est = max(1, int(getattr(args, 'epochs', 1)) * len(data_loader))
    cover_warmup_steps = int(total_steps_est * warmup_ratio)
    cover_sum = 0.0
    cover_cnt = 0
    cover_coef_sum = 0.0

    # SkeletonDistanceLoss：--skel_loss_coef>0 时才实例化并参与反传；
    # coef=0（默认）不创建、不调用，训练路径与现状逐 bit 一致。
    # 权重图每个 batch 由 GT 即时生成（增广后 GT 每次都不同，禁止缓存），
    # 内部已 detach，梯度只流向 pred。
    skel_coef = float(getattr(args, 'skel_loss_coef', 0.0))
    skel_criterion = None
    if skel_coef > 0.0:
        from models.skeleton_loss import SkeletonDistanceLoss
        skel_criterion = SkeletonDistanceLoss()
    # 全局 training step 计数（跨 epoch 持续累加），供 SCAN_DIAG 触发与记录使用。
    global_step = getattr(train_one_epoch, '_global_step', 0)

    for i, data in enumerate(data_loader):
        samples = data['image'].to(device)
        targets = data['label'].to(device)
        # 边界加权 BCE 的逐像素权重图；--boundary_alpha=0（默认）时数据集不
        # 提供该键，weights=None，criterion 走与旧版完全一致的调用。
        weights = data.get('weight')
        if weights is not None:
            weights = weights.to(device)

        # 诊断日志：每 200 个 training step 触发一次，默认关闭不影响前向。
        diag_should_run = _SCAN_DIAG_ENABLED and (global_step % 200 == 0)
        if diag_should_run:
            SAVSS_2D._diag_active = True
            SAVSS_2D._diag_step = global_step
        else:
            SAVSS_2D._diag_active = False

        # asr_scan_v2 路由诊断：同一节奏（每 200 step），只在 --asr_router_diag 时启用。
        router_diag_should_run = asr_router_diag and (global_step % 200 == 0)
        SAVSS_2D._router_diag_active = router_diag_should_run
        SAVSS_2D._router_diag_step = global_step

        output = model(samples)
        loss_final = criterion(output, targets.float(), weight=weights)

        if diag_should_run:
            _write_scan_diag(model)
            SAVSS_2D._diag_active = False

        if asr_v2_mode:
            # V2：one-sided anti-collapse coverage loss，前 warmup_steps 步系数为 0，
            # 随后同样长度线性 ramp 到目标系数；不用 legacy 的均匀化 balancing loss。
            L_cover = compute_asr_v2_coverage_loss(model, coverage_cap=cover_cap)
            if cover_warmup_steps <= 0:
                ramp = 1.0
            elif global_step < cover_warmup_steps:
                ramp = 0.0
            else:
                ramp = min(1.0, (global_step - cover_warmup_steps) / cover_warmup_steps)
            coef_eff = cover_coef * ramp
            if L_cover is not None:
                cover_sum += L_cover.item()
                cover_cnt += 1
                cover_coef_sum += coef_eff
                total_loss = loss_final + coef_eff * L_cover
            else:
                total_loss = loss_final
            bal_loss_sum += 0.0
            bal_loss_cnt += 1
        else:
            L_bal = compute_balancing_loss(model)
            if L_bal is not None:
                total_loss = loss_final + args.bal_loss_coef * L_bal
                bal_loss_sum += L_bal.item()
                bal_loss_cnt += 1
            else:
                total_loss = loss_final
                bal_loss_sum += 0.0
                bal_loss_cnt += 1

        # 收集每层的 4 路平均权重，供 epoch 日志使用（所有路由模式共用）。
        for name, m in model.named_modules():
            if hasattr(m, 'last_route_w') and m.last_route_w is not None:
                rw = m.last_route_w
                if rw.dim() == 4:
                    rw = rw.mean(dim=(-2, -1))
                if name not in route_w_sums:
                    route_w_sums[name] = torch.zeros(4, device=rw.device)
                    route_w_counts[name] = 0
                route_w_sums[name] += rw.sum(dim=0)
                route_w_counts[name] += rw.size(0)

        if skel_criterion is not None:
            total_loss = total_loss + skel_coef * skel_criterion(output, targets)

        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | loss final -> {loss_final_str} | lr -> {l}")

        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)
        optimizer.zero_grad()
        total_loss.backward()
        if router_diag_should_run:
            # backward 之后才能拿到 router 各 projection 的梯度范数。
            _write_router_diag(model)
            SAVSS_2D._router_diag_active = False
        optimizer.step()
        if ema is not None:
            # EMA 只更新 shadow，不改动模型权重，不影响训练路径。
            ema.update(model)
        global_step += 1

    train_one_epoch._global_step = global_step
    pbar.close()

    avg_bal_loss = bal_loss_sum / bal_loss_cnt if bal_loss_cnt else 0.0
    route_w_avgs = {}
    for name in route_w_sums:
        route_w_avgs[name] = (route_w_sums[name] / route_w_counts[name]).tolist()

    result = {'bal_loss': avg_bal_loss, 'route_w_avgs': route_w_avgs}
    if asr_v2_mode:
        result['coverage_loss'] = cover_sum / cover_cnt if cover_cnt else 0.0
        result['coverage_coef_eff'] = cover_coef_sum / cover_cnt if cover_cnt else 0.0
    return result
