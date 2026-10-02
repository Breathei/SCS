"""
ASR 验收测试：梯度检查。

asr 模式下跑 2 个 training step，确认：
1. 路由器最后一层（fc2）权重梯度非零、无 NaN。
2. 所有参数无 NaN。
"""
import argparse
import sys

import torch

sys.path.insert(0, '/home/xby/SCSegamba')
from main import get_args_parser
from models import build_model
from engine import compute_balancing_loss


def strip_stale(state_dict):
    return {k: v for k, v in state_dict.items() if 'trap_gate_logit' not in k}


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    args.phase = 'train'
    args.batch_size = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.scan_routing = 'asr'
    args.bal_loss_coef = 0.01

    model, criterion = build_model(args)
    ckpt = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(strip_stale(ckpt.get('model', ckpt)), strict=False)
    model.train().to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(0)
    x = torch.randn(1, 3, args.load_height, args.load_width).to(args.device)
    t = torch.randint(0, 2, (1, 1, args.load_height, args.load_width)).float().to(args.device)

    for step in range(2):
        optimizer.zero_grad()
        out = model(x)
        main_loss = criterion(out, t)
        L_bal = compute_balancing_loss(model)
        if L_bal is None:
            L_bal = torch.tensor(0.0, device=args.device)
        loss = main_loss + args.bal_loss_coef * L_bal
        loss.backward()

        router = model.backbone.layers[0].SAVSS_2D.router
        g = router.fc2.weight.grad
        assert g is not None, "Router last-layer gradient is None"
        assert not torch.isnan(g).any(), "NaN gradient in router last layer"
        assert g.abs().max() > 0.0, "Router last-layer gradient is all zero"

        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN parameter: {name}"

        optimizer.step()
        print(
            f"step {step}: loss={loss.item():.4f}, "
            f"router.fc2 grad max={g.abs().max():.6e}"
        )

    print("PASS")


if __name__ == '__main__':
    main()
