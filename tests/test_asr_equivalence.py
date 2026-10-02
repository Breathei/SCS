"""
ASR 验收测试 Gate A：
asr 模式在零初始化下应与 none 模式逐元素等价。

加载官方发布的 SCSegamba 权重（strict=False），固定一张输入，
比较 scan_routing='asr' 与 scan_routing='none' 的输出最大绝对误差 < 1e-5。
"""
import argparse
import sys

import torch

sys.path.insert(0, '/home/xby/SCSegamba')
from main import get_args_parser
from models import build_model


def strip_stale(state_dict):
    return {k: v for k, v in state_dict.items() if 'trap_gate_logit' not in k}


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    args.phase = 'test'
    args.batch_size = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args_none = argparse.Namespace(**vars(args))
    args_none.scan_routing = 'none'
    model_none, _ = build_model(args_none)

    args_asr = argparse.Namespace(**vars(args))
    args_asr.scan_routing = 'asr'
    model_asr, _ = build_model(args_asr)

    ckpt = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    sd = strip_stale(ckpt.get('model', ckpt))

    # none 模式结构与权重完全一致；asr 模式多出 router（零初始化），用 strict=False。
    model_none.load_state_dict(sd, strict=True)
    model_asr.load_state_dict(sd, strict=False)

    model_none.eval().to(args.device)
    model_asr.eval().to(args.device)

    torch.manual_seed(42)
    x = torch.randn(1, 3, args.load_height, args.load_width).to(args.device)

    with torch.no_grad():
        out_none = model_none(x)
        out_asr = model_asr(x)

    diff = (out_none - out_asr).abs().max().item()
    print(f"max abs error (asr vs none): {diff:.2e}")
    assert diff < 1e-5, f"Equivalence test failed: {diff}"
    print("PASS")


if __name__ == '__main__':
    main()
