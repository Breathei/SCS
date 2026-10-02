"""
ASR 验收测试：参数量报告。

比较 scan_routing='none' 与 scan_routing='asr' 的参数量，
新增参数应等于 4 层 ScanRouter 的参数量：
    4 * (C * (C//4) + (C//4) * 4 + C//4 + 4)
对于 Crack 配置 C=256，即为 4 * (256*64 + 64*4 + 64 + 4) = 66,832。
"""
import argparse
import sys

sys.path.insert(0, '/home/xby/SCSegamba')
from main import get_args_parser
from models import build_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    args.phase = 'test'
    args.batch_size = 1
    args.device = 'cpu'

    args_none = argparse.Namespace(**vars(args))
    args_none.scan_routing = 'none'
    args_asr = argparse.Namespace(**vars(args))
    args_asr.scan_routing = 'asr'

    model_none, _ = build_model(args_none)
    model_asr, _ = build_model(args_asr)

    n_none = count_params(model_none)
    n_asr = count_params(model_asr)
    added = n_asr - n_none

    expected = 0
    for m in model_asr.modules():
        if hasattr(m, 'router'):
            expected += sum(p.numel() for p in m.router.parameters())

    pct = 100.0 * added / n_none if n_none else 0.0
    print(f"none params:  {n_none:,}")
    print(f"asr params:   {n_asr:,}")
    print(f"added params: {added:,} ({pct:.4f}%)")
    print(f"expected:     {expected:,}")
    assert added == expected, "Added param count does not match router formula"
    print("PASS")


if __name__ == '__main__':
    main()
