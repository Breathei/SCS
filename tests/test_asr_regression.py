"""
ASR 验收测试 Gate D：回归测试。

scan_routing='none' 模式下，用官方权重在 DeepCrack 测试划分上推理，
比较 mIoU 与未改动代码的基线（差异 < 0.0001）。

先运行 --save_baseline 保存基线，之后代码改动后运行 --baseline_json 做对比。
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/xby/SCSegamba')
from datasets import create_dataset
from eval.evaluate import cal_mIoU_metrics
from main import get_args_parser
from models import build_model


def strip_stale(state_dict):
    return {k: v for k, v in state_dict.items() if 'trap_gate_logit' not in k}


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument('--save_baseline', type=str, default=None,
                        help='Path to save baseline mIoU JSON')
    parser.add_argument('--baseline_json', type=str, default=None,
                        help='Path to baseline mIoU JSON to compare against')
    args = parser.parse_args()

    args.phase = 'test'
    args.batch_size = 1
    args.scan_routing = 'none'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _ = build_model(args)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(strip_stale(ckpt.get('model', ckpt)), strict=True)
    model.to(args.device).eval()

    test_dl = create_dataset(args)
    preds, gts = [], []
    with torch.no_grad():
        for data in test_dl:
            x = data['image'].to(args.device)
            target = data['label'].to(args.device, dtype=torch.int64)
            out = model(x)

            target = target[0, 0].cpu().numpy()
            out = out[0, 0].cpu().numpy()

            if target.max() > 0:
                target = 255 * (target / target.max())
            if out.max() > 0:
                out = 255 * (out / out.max())

            preds.append(out)
            gts.append(target)

    miou = cal_mIoU_metrics(preds, gts)
    print(f"mIoU = {miou:.6f}")

    if args.save_baseline:
        with open(args.save_baseline, 'w') as f:
            json.dump({'miou': miou}, f)
        print(f"Saved baseline to {args.save_baseline}")
    elif args.baseline_json:
        with open(args.baseline_json) as f:
            baseline = json.load(f)['miou']
        diff = abs(miou - baseline)
        print(f"Baseline mIoU = {baseline:.6f}, diff = {diff:.6e}")
        assert diff < 1e-4, f"Regression too large: {diff}"
        print("PASS")
    else:
        print("No baseline provided; just reported mIoU")


if __name__ == '__main__':
    main()
