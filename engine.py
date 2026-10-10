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


def compute_balancing_loss(model):
    """ASR 负载均衡损失。

    L_bal = 4 * sum_k( w_bar_k * f_k )
    - w_bar_k：路径 k 的 soft 平均权重（对 batch 和层求平均，可微）。
    - f_k：路径 k 被“选中”的频率（argmax 的 detached one-hot 平均）。
    """
    route_ws = []
    for m in model.modules():
        if hasattr(m, '_route_w') and m._route_w is not None:
            route_ws.append(m._route_w)
    if not route_ws:
        return None

    all_w = torch.cat(route_ws, dim=0)          # (N, 4)
    w_bar = all_w.mean(dim=0)                    # (4,)

    max_idx = all_w.argmax(dim=-1)               # (N,)
    f_onehot = torch.zeros_like(all_w)
    f_onehot.scatter_(1, max_idx.unsqueeze(1), 1.0)
    f = f_onehot.detach().mean(dim=0)            # (4,)

    return 4.0 * (w_bar * f).sum()


class ModelEMA:
    """模型参数的指数滑动平均（EMA）。

    只跟踪 named_parameters（GN 只有 weight/bias 参数、无 running buffer，
    无需特殊处理）。EMA 不参与训练路径：前向/反向始终使用原始权重，
    shadow 仅用于 eval 与 checkpoint 保存。

    更新规则：shadow = d * shadow + (1 - d) * param
    warmup：前 warmup_steps 步使用 d_t = min(decay, (1 + step) / (10 + step))。
    """

    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        if self.updates < self.warmup_steps:
            d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        else:
            d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self):
        return {
            'shadow': {n: t.detach().clone() for n, t in self.shadow.items()},
            'updates': self.updates,
        }

    def load_state_dict(self, state):
        # 兼容纯权重 dict（无 'shadow' 包装）的存档。
        if 'shadow' in state:
            self.updates = state.get('updates', 0)
            state = state['shadow']
        for n, t in state.items():
            if n in self.shadow:
                self.shadow[n].copy_(t)

    @torch.no_grad()
    def apply_to(self, model):
        """把 shadow 权重写入 model，返回原始权重备份（eval 完后用 restore 恢复）。"""
        backup = {}
        for n, p in model.named_parameters():
            backup[n] = p.detach().clone()
            p.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            p.copy_(backup[n])


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

        output = model(samples)
        loss_final = criterion(output, targets.float(), weight=weights)

        if diag_should_run:
            _write_scan_diag(model)
            SAVSS_2D._diag_active = False

        L_bal = compute_balancing_loss(model)
        if L_bal is not None:
            total_loss = loss_final + args.bal_loss_coef * L_bal
            bal_loss_sum += L_bal.item()
            bal_loss_cnt += 1

            # 收集每层的 4 路平均权重，供 epoch 日志使用。
            for name, m in model.named_modules():
                if hasattr(m, 'last_route_w') and m.last_route_w is not None:
                    rw = m.last_route_w                      # (B, 4)
                    if name not in route_w_sums:
                        route_w_sums[name] = torch.zeros(4, device=rw.device)
                        route_w_counts[name] = 0
                    route_w_sums[name] += rw.sum(dim=0)
                    route_w_counts[name] += rw.size(0)
        else:
            total_loss = loss_final
            bal_loss_sum += 0.0
            bal_loss_cnt += 1

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
        optimizer.step()
        if ema is not None:
            # EMA 只读参数做滑动平均，不改动模型权重，不影响训练路径。
            ema.update(model)
        global_step += 1

    train_one_epoch._global_step = global_step
    pbar.close()

    avg_bal_loss = bal_loss_sum / bal_loss_cnt if bal_loss_cnt else 0.0
    route_w_avgs = {}
    for name in route_w_sums:
        route_w_avgs[name] = (route_w_sums[name] / route_w_counts[name]).tolist()

    return {'bal_loss': avg_bal_loss, 'route_w_avgs': route_w_avgs}
